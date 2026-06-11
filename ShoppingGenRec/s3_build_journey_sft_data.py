"""Step 3: Unified Journey SFT Data Builder (Event2Journey + Profile2Journey)

Reads step6 merged output (Ranked TSV) and id2meta to build SFT training
data.  Products are resolved to 7-slot text IDs (TIDs), filtered through
multiple diversity layers, and assembled into instruction/input/output format.

Pipeline position:
  step5 (ANN search) → step6 (LLM ranker) → **s3 (SFT data builder)**

Two tasks via --task:
  event2journey:
    Input: user event history → Output: predicted shopping journeys with TIDs
  profile2journey:
    Input: shopping profile + recent events → Output: predicted shopping journeys

Diversity filtering pipeline (per journey):
  Layer 0: Pre-TID Jaccard word similarity on raw product text
           → remove near-duplicate products before TID resolution
  Layer 1 (optional): Pre-TID Cosine embedding similarity
           → remove semantically similar products (--use_embedding to enable)
  Layer 2: TID-level hard dedup (7-word overlap threshold)
  Layer 3: Greedy diversity reranking (word overlap + brand/seller penalty)

Input:
  Step6 merged ranked TSV (*_Ranked.tsv):
    UserId \\t ReadableUserEvents \\t ShoppingProfile \\t
    JourneyWithProducts \\t RankedJourneys

  id2meta.json:
    { ItemId: {title, description, categories, summary_words, ...} }

Usage:
  python s3_build_journey_sft_data.py --task event2journey \\
      --ranked_journey_file /path/to/*_Ranked.tsv \\
      --id2meta_file ./processed/id2meta.json \\
      --output_dir ./sft_data

  python s3_build_journey_sft_data.py --task profile2journey \\
      --ranked_journey_file /path/to/*_Ranked.tsv \\
      --id2meta_file ./processed/id2meta.json \\
      --output_dir ./sft_data
"""

import os
import csv
import json
import re
import sys
import random
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_EVENTS = 500
DEFAULT_MAX_RECENT_EVENTS = 500
DEFAULT_MAX_PRODUCTS = 20
DEFAULT_MIN_PRODUCTS = 8
DEFAULT_MIN_AVG_PRODUCTS = 8
DEFAULT_MIN_JOURNEYS = 1
DEFAULT_MAX_JOURNEYS = None
DEFAULT_KEEP_EMPTY_RATIO = 1
DEFAULT_COUNT_RATIO = 0.6
DEFAULT_DUP_THRESHOLD = 6
DEFAULT_JACCARD_THRESHOLD = 0.7
DEFAULT_COSINE_THRESHOLD = 0.98


# =============================================================================
# JSON Helpers
# =============================================================================

def _fix_backslash_json(text):
    """Iteratively fix backslash-escaped quotes and parse as JSON."""
    if not text or not text.strip():
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    _BS = chr(92)
    _Q = chr(34)
    _PH = "\x00_ESC_Q_\x00"
    cur = text
    for _ in range(3):
        if _BS + _Q not in cur:
            break
        cur = cur.replace(_BS + _BS + _Q, _PH)
        cur = cur.replace(_BS + _Q, _Q)
        cur = cur.replace(_PH, _BS + _Q)
        try:
            return json.loads(cur)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _clean_profile_json(raw):
    """Unescape multi-layer escaped profile JSON.

    Returns clean JSON string or original if unparseable.
    """
    if not raw or not raw.strip():
        return raw
    try:
        obj = json.loads(raw)
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False, separators=(',', ': '))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    text = raw
    for _ in range(3):
        text = text.replace('\\\\', '\x00__BS__\x00')
        text = text.replace('\\"', '"')
        text = text.replace('\x00__BS__\x00', '\\')
        try:
            obj = json.loads(text)
            if isinstance(obj, (dict, list)):
                return json.dumps(obj, ensure_ascii=False, separators=(',', ': '))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return raw


# =============================================================================
# Input Loading: Step6 Ranked TSV
# =============================================================================

def _parse_event_lines(events_text):
    """Parse ReadableUserEvents from #N#-separated text into event list."""
    if not events_text or not events_text.strip():
        return []
    text = events_text.replace("#N#", "\n")
    events = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+\s*\|\s*(.*)", line)
        if m:
            event = m.group(1).strip()
            if event:
                events.append(event)
    return events


def load_from_ranked_tsv(filepath):
    """Load shopping journey data from step6 merged Ranked TSV.

    Ranked TSV columns (from step6 run_merge):
      UserId, ReadableUserEvents, ShoppingProfile,
      JourneyWithProducts, RankedJourneys

    RankedJourneys JSON structure:
      {"ContinuedJourneys": [{
        "JourneyType": "...", "Title": "...", "Description": "...",
        "ConversationStarter": "...", "WhyAmISeeingThis": "...",
        "Products": [{"Rank":1, "OfferId":"...", "Title":"...",
                      "Seller":"...", "Price":"...", "Brand":"...",
                      "Category":"...", "OriginalQuery":"..."}],
        "RankingSummary": {...}
      }]}

    Returns:
        Dict: {UserId: {
          user_shopping_events: [str],
          user_profile: str,
          journeys: [{
            title, description, conversation_starter, reason,
            journey_type, product_ids: [str],
            products_info: {OfferId: {Title, Seller, Price, Brand, Category}}
          }]
        }}
    """
    print(f"  Loading ranked TSV: {filepath}")
    shopping_data = {}
    total_rows = 0
    parse_fail = 0

    with open(filepath, "r", encoding="utf-8", buffering=8 << 20) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_rows += 1
            uid = (row.get("UserId") or "").strip()
            if not uid:
                continue

            # Parse events
            events_text = row.get("ReadableUserEvents", "")
            events = _parse_event_lines(events_text)

            # Parse profile
            profile_raw = row.get("ShoppingProfile", "")
            profile = _clean_profile_json(profile_raw)

            # Parse RankedJourneys
            ranked_raw = (row.get("RankedJourneys") or "").strip()
            ranked_obj = _fix_backslash_json(ranked_raw)
            if ranked_obj is None or "ContinuedJourneys" not in ranked_obj:
                parse_fail += 1
                # Still store user with empty journeys if they have events
                if events:
                    shopping_data[uid] = {
                        "user_shopping_events": events,
                        "user_profile": profile,
                        "journeys": [],
                    }
                continue

            journeys = []
            for j in ranked_obj["ContinuedJourneys"]:
                if not isinstance(j, dict):
                    continue

                products = j.get("Products", [])
                product_ids = []
                products_info = {}
                for p in products:
                    if not isinstance(p, dict):
                        continue
                    oid = str(p.get("OfferId", "")).strip()
                    if not oid or oid in products_info:
                        continue
                    product_ids.append(oid)
                    products_info[oid] = {
                        "Title": p.get("Title", ""),
                        "Seller": p.get("Seller", ""),
                        "Price": p.get("Price", ""),
                        "Brand": p.get("Brand", ""),
                        "Category": p.get("Category", ""),
                        "OriginalQuery": p.get("OriginalQuery", ""),
                    }

                cs = j.get("ConversationStarter", "")
                if isinstance(cs, list):
                    cs = cs[0] if cs else ""

                journeys.append({
                    "title": j.get("Title", ""),
                    "description": j.get("Description", ""),
                    "conversation_starter": cs,
                    "reason": j.get("WhyAmISeeingThis", ""),
                    "journey_type": j.get("JourneyType", "explicit"),
                    "product_ids": product_ids,
                    "products_info": products_info,
                    "ranking_summary": j.get("RankingSummary", {}),
                })

            shopping_data[uid] = {
                "user_shopping_events": events,
                "user_profile": profile,
                "journeys": journeys,
            }

            if total_rows % 100000 == 0:
                print(f"    ... {total_rows:,} rows", flush=True)

    total_j = sum(len(v["journeys"]) for v in shopping_data.values())
    print(f"    Rows: {total_rows:,}, parse failures: {parse_fail:,}")
    print(f"    Users: {len(shopping_data):,}, journeys: {total_j:,}")
    return shopping_data


# =============================================================================
# Time Normalization
# =============================================================================

def _normalize_time_expr(match):
    """Normalize a single time expression to days or hours."""
    text = match.group(0)
    parts = re.findall(r'(\d+)\s*(month|week|day|hour|minute|second)s?', text,
                       re.IGNORECASE)
    if not parts:
        return text
    total_hours = 0
    total_minutes = 0
    for num_str, unit in parts:
        num = int(num_str)
        u = unit.lower()
        if u == 'month':
            total_hours += num * 30 * 24
        elif u == 'week':
            total_hours += num * 7 * 24
        elif u == 'day':
            total_hours += num * 24
        elif u == 'hour':
            total_hours += num
        elif u == 'minute':
            total_minutes += num
    total_days = total_hours // 24
    if total_days > 0:
        return f"{total_days} days ago"
    elif total_hours > 0:
        return f"{total_hours} hours ago"
    elif total_minutes > 0:
        return f"{total_minutes} minutes ago"
    return "0 minutes ago"


def normalize_event_times(text):
    """Normalize all time expressions (weeks/months -> days) in event text."""
    pattern = r'(?:\d+\s*(?:month|week|day|hour|minute|second)s?\s*)+ago'
    return re.sub(pattern, _normalize_time_expr, text, flags=re.IGNORECASE)


# =============================================================================
# Pre-TID Diversity Filtering (Jaccard + optional Cosine)
# =============================================================================

def _product_text(p_info):
    """Build text for Jaccard/embedding: Title | Brand | Seller | Price."""
    parts = []
    for field in ("Title", "Brand", "Seller", "Price"):
        val = p_info.get(field, "")
        if val:
            parts.append(str(val))
    return " | ".join(parts)


def _jaccard_words(text_a, text_b):
    """Word-level Jaccard similarity."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def pre_tid_diversity_filter(product_ids, products_info,
                             jaccard_thresh=DEFAULT_JACCARD_THRESHOLD,
                             cosine_thresh=DEFAULT_COSINE_THRESHOLD,
                             embedding_map=None):
    """Filter near-duplicate products BEFORE TID resolution.

    Greedy rank-preserving filter: iterate through products in rank order,
    keep a product only if it is sufficiently different from all already-kept
    products (by Jaccard word similarity and optionally cosine embedding
    similarity).

    Returns:
        Tuple of (filtered_product_ids, n_removed_jaccard, n_removed_cosine).
    """
    if len(product_ids) <= 1:
        return product_ids, 0, 0

    # Pre-compute text representations
    texts = []
    for pid in product_ids:
        p_info = products_info.get(pid, {})
        texts.append(_product_text(p_info))
    token_sets = [set(t.lower().split()) for t in texts]

    # Pre-load embeddings if available
    emb_list = None
    use_cosine = embedding_map is not None
    if use_cosine:
        emb_list = [embedding_map.get(pid) for pid in product_ids]

    kept_indices = [0]  # always keep rank-1 product
    n_removed_jaccard = 0
    n_removed_cosine = 0

    for i in range(1, len(product_ids)):
        is_diverse = True

        for j in kept_indices:
            # Jaccard check
            if token_sets[i] and token_sets[j]:
                inter = token_sets[i] & token_sets[j]
                union = token_sets[i] | token_sets[j]
                jac = len(inter) / len(union) if union else 0.0
                if jac >= jaccard_thresh:
                    n_removed_jaccard += 1
                    is_diverse = False
                    break

            # Cosine check (only if Jaccard passed)
            if (use_cosine and emb_list[i] is not None
                    and emb_list[j] is not None):
                cos = float(np.dot(emb_list[i], emb_list[j]))
                if cos >= cosine_thresh:
                    n_removed_cosine += 1
                    is_diverse = False
                    break

        if is_diverse:
            kept_indices.append(i)

    filtered_pids = [product_ids[i] for i in kept_indices]
    return filtered_pids, n_removed_jaccard, n_removed_cosine


def load_or_generate_embeddings(all_product_ids, all_products_info,
                                embedding_model_path, cache_dir,
                                batch_size=1024, max_length=512):
    """Load cached embeddings, generate missing ones, update cache.

    Returns: dict of OfferId -> np.ndarray (L2-normalized).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "product_embeddings_cache.npz")

    # Load existing cache
    cached_map = {}
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        cached_ids = list(data["offer_ids"])
        cached_embs = data["embeddings"]
        for i, oid in enumerate(cached_ids):
            cached_map[str(oid)] = cached_embs[i]
        print(f"    Loaded embedding cache: {len(cached_map):,} products")

    unique_ids = list(set(all_product_ids))
    missing_ids = [oid for oid in unique_ids if oid not in cached_map]

    if missing_ids:
        print(f"    Need embeddings for {len(missing_ids):,} new products "
              f"({len(unique_ids) - len(missing_ids):,} cached)")

        texts = []
        valid_ids = []
        for oid in missing_ids:
            p = all_products_info.get(oid, {})
            text = _product_text(p)
            if text.strip():
                texts.append(text)
                valid_ids.append(oid)

        if texts:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer

            num_gpus = (torch.cuda.device_count()
                        if torch.cuda.is_available() else 0)
            device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
            print(f"    Loading embedding model ({num_gpus} GPUs) ...")
            model = AutoModel.from_pretrained(
                embedding_model_path,
                torch_dtype=torch.float16 if num_gpus > 0 else torch.float32,
                trust_remote_code=True,
            ).to(device)
            if num_gpus > 1:
                model = torch.nn.DataParallel(model)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(
                embedding_model_path, trust_remote_code=True
            )

            all_embs = []
            n_batches = (len(texts) + batch_size - 1) // batch_size
            for b_idx in tqdm(range(n_batches), desc="    Embedding",
                              mininterval=30):
                start = b_idx * batch_size
                batch = texts[start:start + batch_size]
                inputs = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    last_hidden = (outputs.last_hidden_state
                                   if hasattr(outputs, "last_hidden_state")
                                   else outputs[0])
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(
                        last_hidden.size()).float()
                    embs = (torch.sum(last_hidden * mask, 1)
                            / torch.clamp(mask.sum(1), min=1e-9))
                    embs = F.normalize(embs, p=2, dim=1)
                    all_embs.append(embs.cpu().numpy().astype(np.float32))

            new_embs = np.vstack(all_embs)
            for i, oid in enumerate(valid_ids):
                cached_map[oid] = new_embs[i]
            print(f"    Generated {len(valid_ids):,} new embeddings")

            del model, tokenizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Save updated cache
        all_ids = list(cached_map.keys())
        all_embs_arr = np.vstack([cached_map[oid] for oid in all_ids])
        np.savez_compressed(cache_file,
                            offer_ids=np.array(all_ids),
                            embeddings=all_embs_arr)
        print(f"    Updated cache: {len(all_ids):,} products -> {cache_file}")
    else:
        print(f"    All {len(unique_ids):,} products found in cache")

    return cached_map


# =============================================================================
# TID Resolution & Post-TID Diversity
# =============================================================================

# Module-level TID cache: avoid re-parsing summary_words for the same OfferId
_tid_cache = {}


def get_item_tid(item_id, id2meta):
    """Get the text ID (7 summary words) for an item.

    Prefers summary_words_norm if available, falls back to summary_words.
    Results are cached in _tid_cache for performance.

    Returns:
        Tuple of (list of 7 summary words, used_norm: bool), or (None, False)
        if not found or invalid.
    """
    cached = _tid_cache.get(item_id)
    if cached is not None:
        return cached

    if item_id not in id2meta:
        _tid_cache[item_id] = (None, False)
        return None, False
    meta = id2meta[item_id]

    used_norm = False
    summary_words = meta.get("summary_words_norm")
    if summary_words and len(summary_words) >= 7 and "" not in summary_words:
        used_norm = True
    else:
        summary_words = meta.get("summary_words", [])

    if not summary_words or "" in summary_words:
        _tid_cache[item_id] = (None, False)
        return None, False
    valid_words = [
        word.replace("[", "").replace("]", "")
        for word in summary_words
        if word and word.strip()
    ]
    if len(valid_words) < 7:
        _tid_cache[item_id] = (None, False)
        return None, False
    result = (valid_words[:7], used_norm)
    _tid_cache[item_id] = result
    return result


def resolve_journey_tids(journey, id2meta, max_products=None):
    """Resolve a journey's product_ids to text IDs.

    Resolves ALL products to TIDs (no truncation). The max_products cap
    is applied AFTER diversity filtering (Layer 3) to preserve candidates
    that survive hard dedup and greedy reranking.

    Returns dict with product_tids, product_ids, or None if no products
    could be resolved.
    """
    product_tids = []
    resolved_pids = []
    norm_count = 0
    for pid in journey.get("product_ids", []):
        tid, used_norm = get_item_tid(pid, id2meta)
        if tid is not None:
            product_tids.append(tid)
            resolved_pids.append(pid)
            if used_norm:
                norm_count += 1

    if not product_tids:
        return None

    return {
        "title": journey.get("title", ""),
        "description": journey.get("description", ""),
        "conversation_starter": journey.get("conversation_starter", ""),
        "reason": journey.get("reason", ""),
        "journey_type": journey.get("journey_type", "explicit"),
        "product_tids": product_tids,
        "product_ids": resolved_pids,
        "norm_count": norm_count,
    }


def _get_brand_seller(pid, id2meta):
    """Extract (brand, seller) from id2meta for a product id."""
    meta = id2meta.get(pid)
    if meta is None:
        return ("", "")
    attrs = meta.get("attributes", {})
    brand = (attrs.get("Brand", "").strip().lower()
             if isinstance(attrs.get("Brand"), str) else "")
    seller = (attrs.get("Seller", "").strip().lower()
              if isinstance(attrs.get("Seller"), str) else "")
    return (brand, seller)


def diversify_journey_products(product_tids, product_ids, id2meta,
                               dup_threshold=DEFAULT_DUP_THRESHOLD,
                               max_products=None, dynamic_threshold=True):
    """Diversify products via TID-level hard dedup + greedy reranking.

    Stage 1 — Hard dedup: remove products whose 7-word TID overlap
    >= dup_threshold with any already-selected product.

    Stage 2 — Greedy diversity reranking: iteratively pick the candidate
    with the lowest effective score (word overlap + brand/seller penalty).

    Returns:
        Tuple (deduped_tids, deduped_pids, num_removed_dedup).
    """
    n = len(product_tids)
    if n <= 1:
        return product_tids, product_ids, 0

    if dynamic_threshold and max_products is not None:
        dup_threshold = (dup_threshold if n < max_products // 2
                         else max(dup_threshold - 1, 1))

    word_sets = [set(tid) for tid in product_tids]

    # Stage 1: Hard dedup
    keep_mask = [True] * n
    num_removed = 0
    for i in range(1, n):
        if not keep_mask[i]:
            continue
        for j in range(i):
            if not keep_mask[j]:
                continue
            overlap = len(word_sets[i] & word_sets[j])
            if overlap >= dup_threshold:
                keep_mask[i] = False
                num_removed += 1
                break

    cand_tids = [product_tids[i] for i in range(n) if keep_mask[i]]
    cand_pids = [product_ids[i] for i in range(n) if keep_mask[i]]
    cand_sets = [word_sets[i] for i in range(n) if keep_mask[i]]

    if len(cand_tids) <= 1:
        return cand_tids, cand_pids, num_removed

    # Pre-fetch brand/seller
    cand_bs = [_get_brand_seller(pid, id2meta) for pid in cand_pids]

    # Stage 2: Greedy diversity reranking
    selected_idx = [0]
    remaining = set(range(1, len(cand_tids)))

    while remaining:
        best_i = None
        best_score = float('inf')

        for i in remaining:
            max_ov = max(len(cand_sets[i] & cand_sets[s]) for s in selected_idx)

            b_i, s_i = cand_bs[i]
            brand_penalty = 0
            if b_i or s_i:
                for s in selected_idx:
                    b_s, s_s = cand_bs[s]
                    if ((b_i and b_s and b_i == b_s)
                            or (s_i and s_s and s_i == s_s)):
                        brand_penalty = 1
                        break

            effective_score = max_ov + brand_penalty
            if effective_score < best_score:
                best_score = effective_score
                best_i = i

        selected_idx.append(best_i)
        remaining.remove(best_i)

    reranked_tids = [cand_tids[i] for i in selected_idx]
    reranked_pids = [cand_pids[i] for i in selected_idx]
    return reranked_tids, reranked_pids, num_removed


# =============================================================================
# Instruction / Input / Output Builders
# =============================================================================

def build_output_json(resolved_journeys):
    """Build the structured JSON output string for SFT training.

    Output format:
    {"ContinuedJourneys":[{"JourneyType":"...","Title":"...",
     "Description":"...","ConversationStarter":"...",
     "Reason":"...","ProductTIDs":[["a","b",...],...]},...]}
    """
    continued = []
    for j in resolved_journeys:
        continued.append({
            "JourneyType": j.get("journey_type", "explicit"),
            "Title": j["title"],
            "Description": j.get("description", ""),
            "ConversationStarter": j.get("conversation_starter", ""),
            "Reason": j["reason"],
            "ProductTIDs": j["product_tids"],
        })
    return json.dumps({"ContinuedJourneys": continued}, ensure_ascii=False)


def create_instruction(task, num_journeys, min_products_in_user,
                       count_ratio=DEFAULT_COUNT_RATIO):
    """Create instruction text.

    Returns:
        Tuple of (instruction_text, has_count, prompt_line).
    """
    has_count = num_journeys > 0 and random.random() < count_ratio

    if task == "event2journey":
        if has_count:
            opening = (f"Based on the user's shopping event history, predict "
                       f"{num_journeys} shopping journey(s) the user is likely to pursue.")
        else:
            opening = ("Based on the user's shopping event history, predict "
                       "an appropriate number of shopping journey(s) the user is likely to pursue.")
    else:
        if has_count:
            opening = (f"Based on the user's shopping profile and shopping event history, predict "
                       f"{num_journeys} shopping journey(s) the user is likely to pursue.")
        else:
            opening = ("Based on the user's shopping profile and shopping event history, predict "
                       "an appropriate number of shopping journey(s) the user is likely to pursue.")

    product_text = f"at least {min_products_in_user}"

    instruction = (
        f"{opening}"
        f" Each journey has a JourneyType ('explicit' or 'related'),"
        f" a short engaging Title,"
        f" a Description (2-3 sentences in personal-shopper tone highlighting"
        f" why this journey fits the user and what value exploring it brings),"
        f" a ConversationStarter (a natural first-person opening"
        f" that resumes the shopping journey),"
        f" a Reason (explains which user signals triggered this journey),"
        f" and {product_text} recommended products as text IDs (7 slots each)."
        f" Products within each journey must be diverse:"
        f" cover different sellers, brands, styles, use cases, and subcategories."
        f' Output JSON:'
        f' {{"ContinuedJourneys":[{{"JourneyType":"...","Title":"...",'
        f'"Description":"...","ConversationStarter":"...",'
        f'"Reason":"...",'
        f'"ProductTIDs":[["s1","s2","s3","s4","s5","s6","s7"],...]}},...]}}.'
    )

    if has_count:
        jword = "journey" if num_journeys == 1 else "journeys"
        prompt_line = (f"Predict the user's shopping journeys, "
                       f"exactly {num_journeys} {jword}, "
                       f"at least {min_products_in_user} products in each journey:")
    else:
        prompt_line = (f"Predict an appropriate number of shopping journeys, "
                       f"at least {min_products_in_user} products in each journey:")

    return instruction, has_count, prompt_line


def build_input_text(task, user_events, max_events,
                     profile_text=None, max_recent_events=None,
                     prompt_line=None):
    """Build input text based on task type.

    Returns:
        Tuple of (input_text, num_events_used).
    """
    final_prompt = prompt_line or "Predict the user's shopping journeys:"
    if task == "event2journey":
        events = user_events[:max_events]
        lines = ["User Event History:"]
        for idx, event in enumerate(events, 1):
            event = normalize_event_times(event)
            if len(event) > 150:
                event = event[:150] + "..."
            lines.append(f"{idx} | {event}")
        lines.append("")
        lines.append(final_prompt)
        return "\n".join(lines), len(events)
    else:  # profile2journey
        n = max_recent_events or DEFAULT_MAX_RECENT_EVENTS
        recent = user_events[:n]
        lines = [
            "User Shopping Profile:",
            profile_text or "",
            "",
            "Recent Shopping Events:",
        ]
        for idx, event in enumerate(recent, 1):
            event = normalize_event_times(event)
            if len(event) > 150:
                event = event[:150] + "..."
            lines.append(f"{idx} | {event}")
        lines.append("")
        lines.append(final_prompt)
        return "\n".join(lines), len(recent)


# =============================================================================
# Save
# =============================================================================

def save_sft_data(sft_data, output_file):
    """Save SFT data (full and training versions).

    Full version (with metadata): <name>_full.json
    Training version (instruction/input/output only): <name>.json
    """
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    full_file = output_file.replace(".json", "_full.json")
    with open(full_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    full_mb = os.path.getsize(full_file) / (1024 * 1024)
    print(f"Full data saved: {full_file} ({full_mb:.1f} MB)")

    training_data = [
        {
            "instruction": s["instruction"],
            "input": s["input"],
            "output": s["output"],
        }
        for s in sft_data
    ]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    train_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Training data saved: {output_file} ({train_mb:.1f} MB)")


def export_vis_jsonl(vis_data, output_file):
    """Export visualization data as step8-compatible JSONL.

    Each product retains its real metadata (Title, Seller, Brand, Price)
    from step6 ranker output, plus the resolved TID from s3 filtering.
    This gives an accurate view of what s3 kept after all diversity layers.

    Args:
        vis_data: List of step8-compatible user dicts (from create_sft_data).
        output_file: Path to write the JSONL file.
    """
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_users = 0
    n_journeys = 0
    n_products = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for user_record in vis_data:
            f.write(json.dumps(user_record, ensure_ascii=False) + "\n")
            n_users += 1
            for j in user_record.get("journeys", []):
                n_journeys += 1
                n_products += len(j.get("products", []))

    file_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Step8 vis JSONL saved: {output_file} ({file_mb:.1f} MB)")
    print(f"  {n_users} users, {n_journeys} journeys, "
          f"{n_products} products (real metadata + TID)")


# =============================================================================
# Main Pipeline
# =============================================================================

def create_sft_data(
    task,
    shopping_journey_data,
    id2meta,
    embedding_map=None,
    jaccard_threshold=DEFAULT_JACCARD_THRESHOLD,
    cosine_threshold=DEFAULT_COSINE_THRESHOLD,
    max_events=DEFAULT_MAX_EVENTS,
    max_recent_events=DEFAULT_MAX_RECENT_EVENTS,
    max_products=DEFAULT_MAX_PRODUCTS,
    min_products=DEFAULT_MIN_PRODUCTS,
    min_avg_products=DEFAULT_MIN_AVG_PRODUCTS,
    min_journeys=DEFAULT_MIN_JOURNEYS,
    max_journeys=DEFAULT_MAX_JOURNEYS,
    keep_empty_ratio=DEFAULT_KEEP_EMPTY_RATIO,
    count_ratio=DEFAULT_COUNT_RATIO,
    dup_threshold=DEFAULT_DUP_THRESHOLD,
):
    """Create SFT data for the specified task.

    Filtering pipeline per journey:
      1. Pre-TID: Jaccard + optional Cosine on raw product text
      2. TID resolution: OfferId -> 7-word summary
      3. Post-TID: hard dedup (word overlap) + greedy reranking
      4. min_products filter

    Returns:
        Tuple of (sft_data, vis_data):
          sft_data: List of SFT sample dicts.
          vis_data: List of step8-compatible user dicts for visualization.
    """
    sft_data = []
    vis_data = []
    skip_reasons = defaultdict(int)
    total_entries = len(shopping_journey_data)

    # ---- Per-journey pipeline stats (ALL journeys, unified tracking) ----
    # Each journey gets a tuple: (s0_original, s1_after_jaccard, s2_after_tid,
    #                              s3_after_dedup, status)
    # status: "kept", "no_tid", "min_products"
    journey_pipeline = []
    total_pre_tid_removed_jaccard = 0
    total_pre_tid_removed_cosine = 0
    total_tid_dedup_removed = 0

    # ---- User-level stats ----
    users_low_avg_products = 0
    users_half_filtered = 0
    users_below_min_journeys = 0
    empty_journey_total = 0
    empty_journey_kept = 0

    # ---- Output sample stats ----
    event_counts = []
    journey_counts = []
    product_counts = []         # per-journey final product count (output)
    instruction_with_count = 0
    instruction_without_count = 0
    min_product_values_in_instruction = []

    # ---- Input-side stats (before s3 filtering, for comparison) ----
    input_journey_counts = []       # journeys per user from step6
    input_product_counts = []       # products per journey from step6
    input_type_dist = defaultdict(int)

    # ---- Norm TID stats ----
    total_norm_tids = 0
    total_resolved_tids = 0

    for user_id, entry in tqdm(
        shopping_journey_data.items(),
        desc=f"Building {task} SFT data",
        mininterval=30, maxinterval=60,
    ):
        user_events = entry.get("user_shopping_events", [])
        journeys = entry.get("journeys", [])

        if not user_events:
            skip_reasons["no_user_events"] += 1
            continue

        if task == "profile2journey":
            user_profile = _clean_profile_json(entry.get("user_profile", ""))
            if not user_profile:
                skip_reasons["no_profile"] += 1
                continue
        else:
            user_profile = None  # not used for event2journey

        # ---- User-level pre-filter: average resolvable products ----
        if journeys:
            resolvable_counts = []
            for j in journeys:
                n_resolvable = sum(
                    1 for pid in j.get("product_ids", [])
                    if get_item_tid(pid, id2meta)[0] is not None
                )
                resolvable_counts.append(n_resolvable)
            avg_resolvable = sum(resolvable_counts) / len(resolvable_counts)
            if avg_resolvable < min_avg_products:
                users_low_avg_products += 1
                skip_reasons["low_avg_products"] += 1
                continue

        # ---- Per-journey filtering pipeline ----
        resolved_journeys = []
        journeys_before_filter = 0

        for journey in journeys:
            orig_count = len(journey.get("product_ids", []))

            # Layer 0+1: Pre-TID diversity (Jaccard + optional Cosine)
            pre_pids = journey.get("product_ids", [])
            products_info = journey.get("products_info", {})

            filtered_pids, n_jac, n_cos = pre_tid_diversity_filter(
                pre_pids, products_info,
                jaccard_thresh=jaccard_threshold,
                cosine_thresh=cosine_threshold,
                embedding_map=embedding_map,
            )
            total_pre_tid_removed_jaccard += n_jac
            total_pre_tid_removed_cosine += n_cos
            n_after_jaccard = len(filtered_pids)

            # Update product_ids for TID resolution
            journey_filtered = dict(journey)
            journey_filtered["product_ids"] = filtered_pids

            # Layer 2: TID resolution (no truncation — resolve all candidates)
            resolved = resolve_journey_tids(journey_filtered, id2meta)
            if resolved is None:
                journey_pipeline.append(
                    (orig_count, n_after_jaccard, 0, 0, "no_tid"))
                continue
            journeys_before_filter += 1
            n_after_tid = len(resolved["product_tids"])

            # Layer 3: Post-TID diversity (hard dedup + greedy reranking)
            div_tids, div_pids, n_removed = diversify_journey_products(
                resolved["product_tids"], resolved["product_ids"], id2meta,
                dup_threshold=dup_threshold,
                max_products=max_products,
            )
            resolved["product_tids"] = div_tids
            resolved["product_ids"] = div_pids
            total_tid_dedup_removed += n_removed
            n_after_dedup = len(div_tids)

            # Final cap: truncate to max_products AFTER all diversity filtering
            if max_products and n_after_dedup > max_products:
                div_tids = div_tids[:max_products]
                div_pids = div_pids[:max_products]
                resolved["product_tids"] = div_tids
                resolved["product_ids"] = div_pids
                n_after_dedup = len(div_tids)

            # min_products filter AFTER all diversity steps
            if n_after_dedup < min_products:
                journey_pipeline.append(
                    (orig_count, n_after_jaccard, n_after_tid, n_after_dedup,
                     "min_products"))
                continue

            journey_pipeline.append(
                (orig_count, n_after_jaccard, n_after_tid, n_after_dedup,
                 "kept"))
            # Recount norm TIDs based on final kept product_ids
            final_norm = sum(
                1 for pid in resolved["product_ids"]
                if get_item_tid(pid, id2meta)[1]
            )
            total_norm_tids += final_norm
            total_resolved_tids += n_after_dedup
            # Carry products_info for visualization export
            resolved["products_info"] = products_info
            resolved["orig_product_count"] = orig_count
            resolved["ranking_summary"] = journey.get("ranking_summary", {})
            resolved_journeys.append(resolved)

        # Check if >= 50% of journeys were filtered out
        if journeys_before_filter > 0 and resolved_journeys:
            filtered_count = journeys_before_filter - len(resolved_journeys)
            if filtered_count >= journeys_before_filter / 2:
                users_half_filtered += 1
                skip_reasons["half_journeys_filtered"] += 1
                continue

        # Handle empty journeys
        if not resolved_journeys:
            if not journeys:
                empty_journey_total += 1
                if random.random() >= keep_empty_ratio:
                    skip_reasons["empty_journeys_sampled_out"] += 1
                    continue
                empty_journey_kept += 1
            else:
                skip_reasons["all_journeys_filtered"] += 1
                continue

        # Journey subsampling removed — keep all journeys to preserve
        # the LLM's original ordering (explicit → related).

        # min_journeys check
        if resolved_journeys and len(resolved_journeys) < min_journeys:
            users_below_min_journeys += 1
            skip_reasons["below_min_journeys"] += 1
            continue

        # Build instruction / input / output
        final_num_journeys = len(resolved_journeys)
        if resolved_journeys:
            min_products_in_user = min(
                len(j["product_tids"]) for j in resolved_journeys
            )
        else:
            min_products_in_user = min_products

        instruction, has_count, prompt_line = create_instruction(
            task, final_num_journeys, min_products_in_user, count_ratio,
        )

        if task == "event2journey":
            input_text, num_events_used = build_input_text(
                task, user_events, max_events,
                prompt_line=prompt_line,
            )
        else:
            input_text, num_events_used = build_input_text(
                task, user_events, max_events,
                profile_text=user_profile,
                max_recent_events=max_recent_events,
                prompt_line=prompt_line,
            )

        if has_count:
            instruction_with_count += 1
        else:
            instruction_without_count += 1
        min_product_values_in_instruction.append(min_products_in_user)
        output_text = build_output_json(resolved_journeys)

        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "user_id": user_id,
                "task": task,
                "num_events": num_events_used,
                "num_journeys": final_num_journeys,
                "num_products_per_journey": [
                    len(j["product_tids"]) for j in resolved_journeys
                ],
            },
        }
        sft_data.append(sample)

        # Track input-side stats for this user (before s3 filtering)
        input_journey_counts.append(len(journeys))
        for j_in in journeys:
            input_product_counts.append(len(j_in.get("product_ids", [])))
            input_type_dist[j_in.get("journey_type", "explicit")] += 1

        # Build visualization record with real product info + TID
        vis_journeys = []
        for j in resolved_journeys:
            vis_products = []
            for pi, (pid, tid) in enumerate(zip(j["product_ids"], j["product_tids"])):
                p_info = j.get("products_info", {}).get(pid, {})
                vis_products.append({
                    "global_offer_id": pid,
                    "Title": p_info.get("Title", ""),
                    "Seller": p_info.get("Seller", ""),
                    "OriginalPrice": p_info.get("Price", ""),
                    "Brand": p_info.get("Brand", ""),
                    "Gender": "",
                    "AgeGroup": "",
                    "OriginalQuery": p_info.get("OriginalQuery", ""),
                    "TID": tid,
                    "Rank": pi + 1,
                    "ImageUrl": "",
                    "OfferUrl": "",
                })
            vis_journeys.append({
                "journeyType": j.get("journey_type", "explicit"),
                "title": j["title"],
                "description": j.get("description", ""),
                "conversationStarter": j.get("conversation_starter", ""),
                "reason": j.get("reason", ""),
                "products": vis_products,
                "stats": {
                    "totalCandidates": j.get("ranking_summary", {}).get(
                        "totalCandidates",
                        j.get("orig_product_count", len(vis_products))),
                    "selectedCount": len(vis_products),
                    "filteredCount": max(0,
                        j.get("ranking_summary", {}).get(
                            "totalCandidates",
                            j.get("orig_product_count", len(vis_products)))
                        - len(vis_products)),
                    "step6SelectedCount": j.get("orig_product_count",
                                                 len(vis_products)),
                },
            })

        # Reconstruct profile and events for vis
        vis_profile = {}
        if task == "profile2journey" and user_profile:
            try:
                vis_profile = json.loads(user_profile) if isinstance(user_profile, str) else user_profile
            except (json.JSONDecodeError, TypeError):
                pass
        vis_events = "\n".join(
            f"{i+1} | {ev}" for i, ev in enumerate(user_events)
        )
        vis_data.append({
            "stableid": user_id,
            "userShoppingProfile": vis_profile,
            "recentShoppingEvents": vis_events,
            "journeys": vis_journeys,
        })

        event_counts.append(num_events_used)
        journey_counts.append(final_num_journeys)
        for j in resolved_journeys:
            n = len(j["product_tids"])
            product_counts.append(n)

    # =========================================================================
    # Comprehensive Statistics (organized by pipeline stage)
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"Data Statistics ({task})")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Section 1: Per-Journey Product Pipeline
    # ------------------------------------------------------------------
    if journey_pipeline:
        s0 = np.array([t[0] for t in journey_pipeline])
        s1 = np.array([t[1] for t in journey_pipeline])
        s2 = np.array([t[2] for t in journey_pipeline])
        s3 = np.array([t[3] for t in journey_pipeline])
        statuses = [t[4] for t in journey_pipeline]
        n_no_tid = statuses.count("no_tid")
        n_min_prod = statuses.count("min_products")
        n_kept = statuses.count("kept")

        has_tid_mask = s2 > 0
        s2_valid = s2[has_tid_mask]
        s3_valid = s3[has_tid_mask]

        print(f"\n--- Section 1: Per-Journey Product Pipeline "
              f"({len(journey_pipeline):,} journeys) ---")
        print(f"  Stage 0  From ranker:          "
              f"Mean={s0.mean():>5.1f}  Min={s0.min():>3}  "
              f"Max={s0.max():>3}  Total={s0.sum():>10,}")
        jac_info = f"Jaccard >= {jaccard_threshold}: -{total_pre_tid_removed_jaccard:,}"
        if embedding_map is not None:
            jac_info += f", Cosine >= {cosine_threshold}: -{total_pre_tid_removed_cosine:,}"
        else:
            jac_info += " (Cosine: OFF)"
        print(f"      ↓ {jac_info}")
        print(f"  Stage 1  After Jaccard filter:  "
              f"Mean={s1.mean():>5.1f}  Min={s1.min():>3}  "
              f"Max={s1.max():>3}  Total={s1.sum():>10,}")
        print(f"      ↓ OfferId → TID resolve (all candidates, no cap)")
        if len(s2_valid) > 0:
            print(f"  Stage 2  After TID resolution:  "
                  f"Mean={s2_valid.mean():>5.1f}  Min={s2_valid.min():>3}  "
                  f"Max={s2_valid.max():>3}  Total={s2_valid.sum():>10,}  "
                  f"({len(s2_valid):,} journeys)")
        if n_no_tid > 0:
            print(f"      ✗ {n_no_tid:,} journeys dropped (0 TIDs resolved)")
        print(f"      ↓ TID hard dedup (overlap >= {dup_threshold}/7): "
              f"-{total_tid_dedup_removed:,} products")
        if len(s3_valid) > 0:
            print(f"  Stage 3  After TID dedup:       "
                  f"Mean={s3_valid.mean():>5.1f}  Min={s3_valid.min():>3}  "
                  f"Max={s3_valid.max():>3}  Total={s3_valid.sum():>10,}")
        print(f"      ↓ Final cap: top {max_products} per journey")
        if n_min_prod > 0:
            print(f"      ✗ {n_min_prod:,} journeys dropped "
                  f"(< {min_products} products)")
        print(f"  Result:  {n_kept:,} journeys kept / "
              f"{len(journey_pipeline):,} total")

    # ------------------------------------------------------------------
    # Section 2: User-Level Filtering
    # ------------------------------------------------------------------
    print(f"\n--- Section 2: User-Level Filtering ---")
    print(f"  Total users in data:             {total_entries:>10,}")
    if task == "profile2journey":
        profile_count = sum(
            1 for e in shopping_journey_data.values()
            if e.get("user_profile")
        )
        print(f"  With user_profile:               {profile_count:>10,}")
    print(f"  Dropped (no events):             "
          f"{skip_reasons.get('no_user_events', 0):>10,}")
    if task == "profile2journey":
        print(f"  Dropped (no profile):            "
              f"{skip_reasons.get('no_profile', 0):>10,}")
    print(f"  Dropped (low avg products < {min_avg_products}):"
          f"     {users_low_avg_products:>10,}")
    print(f"  Dropped (>= 50% journeys filtered):"
          f"  {users_half_filtered:>10,}")
    print(f"  Dropped (all journeys filtered): "
          f"{skip_reasons.get('all_journeys_filtered', 0):>10,}")
    print(f"  Dropped (< {min_journeys} journey after filter):"
          f"   {users_below_min_journeys:>10,}")
    if empty_journey_total > 0:
        print(f"  Empty journey users:              "
              f"{empty_journey_total:>10,}  "
              f"(kept {empty_journey_kept}, "
              f"ratio={keep_empty_ratio:.1f})")
    print(f"  Final SFT samples (users):       {len(sft_data):>10,}")

    # ------------------------------------------------------------------
    # Section 3: Input / Output Statistics (before vs after s3 filtering)
    # ------------------------------------------------------------------
    print(f"\n--- Section 3: Input / Output Statistics ({len(sft_data):,} users) ---")

    total_instructions = instruction_with_count + instruction_without_count
    if total_instructions > 0:
        print(f"  Instruction with count:    {instruction_with_count:>8,} "
              f"({instruction_with_count / total_instructions * 100:.0f}%)"
              f"   without: {instruction_without_count:,} "
              f"({instruction_without_count / total_instructions * 100:.0f}%)")
    if min_product_values_in_instruction:
        arr = np.array(min_product_values_in_instruction)
        print(f"  'at least N' products:     "
              f"Min={arr.min()}, Max={arr.max()}, Mean={arr.mean():.1f}")

    if event_counts:
        arr = np.array(event_counts)
        print(f"  Events/sample:   Mean={arr.mean():>6.1f}  "
              f"P50={int(np.percentile(arr, 50)):>5}  "
              f"P90={int(np.percentile(arr, 90)):>5}  "
              f"Max={arr.max():>5}")

    # Journeys per user: input vs output
    if input_journey_counts and journey_counts:
        in_arr = np.array(input_journey_counts)
        out_arr = np.array(journey_counts)
        print(f"  Journeys/user:")
        print(f"    Input  (step6): Mean={in_arr.mean():>5.1f}  "
              f"P50={int(np.percentile(in_arr, 50)):>4}  "
              f"Max={in_arr.max():>4}  "
              f"Total={in_arr.sum():>6,}")
        print(f"    Output (s3):    Mean={out_arr.mean():>5.1f}  "
              f"P50={int(np.percentile(out_arr, 50)):>4}  "
              f"Max={out_arr.max():>4}  "
              f"Total={out_arr.sum():>6,}")

    # Products per journey: input vs output
    if input_product_counts and product_counts:
        in_arr = np.array(input_product_counts)
        out_arr = np.array(product_counts)
        print(f"  Products/journey:")
        print(f"    Input  (step6): Mean={in_arr.mean():>5.1f}  "
              f"Min={in_arr.min():>3}  Max={in_arr.max():>3}  "
              f"Total={in_arr.sum():>8,}")
        print(f"    Output (s3):    Mean={out_arr.mean():>5.1f}  "
              f"Min={out_arr.min():>3}  Max={out_arr.max():>3}  "
              f"Total={out_arr.sum():>8,}")

    # Journey types: input vs output
    if sft_data:
        output_type_dist = defaultdict(int)
        for s in sft_data:
            try:
                out_obj = json.loads(s["output"])
                for j in out_obj.get("ContinuedJourneys", []):
                    output_type_dist[j.get("JourneyType", "explicit")] += 1
            except (json.JSONDecodeError, TypeError):
                pass
        all_types = sorted(set(list(input_type_dist.keys()) +
                               list(output_type_dist.keys())),
                           key=lambda t: -(input_type_dist.get(t, 0) +
                                           output_type_dist.get(t, 0)))
        total_in = sum(input_type_dist.values()) or 1
        total_out = sum(output_type_dist.values()) or 1
        print(f"  Journey types:")
        for jt in all_types:
            ic = input_type_dist.get(jt, 0)
            oc = output_type_dist.get(jt, 0)
            print(f"    {jt:12s}  Input: {ic:>5,} ({ic/total_in*100:.0f}%)  "
                  f"Output: {oc:>5,} ({oc/total_out*100:.0f}%)")

    if total_resolved_tids > 0:
        print(f"  TID source:      summary_words_norm "
              f"{total_norm_tids:,}/{total_resolved_tids:,} "
              f"({total_norm_tids / total_resolved_tids * 100:.1f}%)")

    return sft_data, vis_data


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Journey SFT Data Builder (event2journey / profile2journey). "
                    "Reads step6 ranked output + id2meta, resolves products to "
                    "TIDs with multi-layer diversity filtering."
    )
    parser.add_argument(
        "--task", type=str, default="profile2journey",
        choices=["event2journey", "profile2journey"],
        help="Task type (default: profile2journey)",
    )

    # Input files
    parser.add_argument(
        "--ranked_journey_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260528/raw_data_IDB/ranker_output_full/"
                "UserEvents_clean_combined_full_journey_with_products_Ranked.tsv",
        help="Path to step6 merged *_Ranked.tsv "
             "(columns: UserId, ReadableUserEvents, ShoppingProfile, "
             "JourneyWithProducts, RankedJourneys)",
    )
    parser.add_argument(
        "--id2meta_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260528/processed_IDB/"
                "id2meta_with_norm.json",
        help="Path to id2meta JSON from s1_generate_tid",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260528/sft_data_IDB/",
        help="Output directory",
    )

    # Event/input controls
    parser.add_argument(
        "--max_events", type=int, default=DEFAULT_MAX_EVENTS,
        help=f"Max events for event2journey (default: {DEFAULT_MAX_EVENTS})",
    )
    parser.add_argument(
        "--max_recent_events", type=int, default=DEFAULT_MAX_RECENT_EVENTS,
        help=f"Max recent events for profile2journey "
             f"(default: {DEFAULT_MAX_RECENT_EVENTS})",
    )

    # Journey/product controls
    parser.add_argument(
        "--max_products_per_journey", type=int, default=DEFAULT_MAX_PRODUCTS,
        help=f"Max products per journey (default: {DEFAULT_MAX_PRODUCTS})",
    )
    parser.add_argument(
        "--min_products_per_journey", type=int, default=DEFAULT_MIN_PRODUCTS,
        help=f"Min products per journey; below are dropped "
             f"(default: {DEFAULT_MIN_PRODUCTS})",
    )
    parser.add_argument(
        "--min_avg_products", type=int, default=DEFAULT_MIN_AVG_PRODUCTS,
        help=f"Min average resolvable products per journey for a user "
             f"(default: {DEFAULT_MIN_AVG_PRODUCTS})",
    )
    parser.add_argument(
        "--min_journeys", type=int, default=DEFAULT_MIN_JOURNEYS,
        help=f"Min journeys per user after filtering "
             f"(default: {DEFAULT_MIN_JOURNEYS})",
    )
    parser.add_argument(
        "--max_journeys", type=int, default=DEFAULT_MAX_JOURNEYS,
        help=f"Max journeys per sample "
             f"(default: {DEFAULT_MAX_JOURNEYS})",
    )
    parser.add_argument(
        "--keep_empty_ratio", type=float, default=DEFAULT_KEEP_EMPTY_RATIO,
        help=f"Fraction of zero-journey users to keep "
             f"(default: {DEFAULT_KEEP_EMPTY_RATIO})",
    )
    parser.add_argument(
        "--count_ratio", type=float, default=DEFAULT_COUNT_RATIO,
        help=f"Prob of including journey count in instruction "
             f"(default: {DEFAULT_COUNT_RATIO})",
    )
    parser.add_argument(
        "--dup_threshold", type=int, default=DEFAULT_DUP_THRESHOLD,
        help=f"Min TID word overlap for near-duplicate "
             f"(default: {DEFAULT_DUP_THRESHOLD})",
    )

    # Pre-TID diversity
    parser.add_argument(
        "--jaccard_threshold", type=float, default=DEFAULT_JACCARD_THRESHOLD,
        help=f"Jaccard word similarity threshold for pre-TID dedup "
             f"(default: {DEFAULT_JACCARD_THRESHOLD})",
    )
    parser.add_argument(
        "--cosine_threshold", type=float, default=DEFAULT_COSINE_THRESHOLD,
        help=f"Cosine embedding similarity threshold for pre-TID dedup "
             f"(default: {DEFAULT_COSINE_THRESHOLD})",
    )
    parser.add_argument(
        "--use_embedding", action="store_true", default=False,
        help="Enable cosine embedding similarity for pre-TID diversity "
             "(default: off, Jaccard only)",
    )
    parser.add_argument(
        "--embedding_model", type=str,
        default="/scratch/workspaceblobstore/users/xiaoyukou/ckpts/"
                "Qwen3-Embedding-0.6B",
        help="Path to embedding model for cosine diversity",
    )
    parser.add_argument(
        "--embedding_cache_dir", type=str, default="",
        help="Directory for embedding cache (default: same as output_dir)",
    )

    # General
    parser.add_argument(
        "--seed", type=int, default=43,
        help="Random seed (default: 43)",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Debug mode: only process a subset of users",
    )
    parser.add_argument(
        "--debug_sample_size", type=int, default=100,
        help="Number of users to sample in debug mode (default: 100)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    task = args.task
    # --use_embedding explicitly enables; otherwise default off
    use_embedding = args.use_embedding

    # =========================================================================
    # Step 1: Load input files
    # =========================================================================
    print("=" * 70)
    print(f"Step 1: Loading input files (task={task})")
    print("=" * 70)

    shopping_data = load_from_ranked_tsv(args.ranked_journey_file)
    print(f"    Entries: {len(shopping_data):,}")

    # Debug mode: subsample users
    if args.debug:
        all_keys = list(shopping_data.keys())
        sample_n = min(args.debug_sample_size, len(all_keys))
        sampled_keys = random.sample(all_keys, sample_n)
        shopping_data = {k: shopping_data[k] for k in sampled_keys}
        print(f"    [DEBUG] Subsampled to {len(shopping_data):,} users")

    # For profile2journey, check that user_profile exists
    if task == "profile2journey":
        profile_count = sum(
            1 for e in shopping_data.values() if e.get("user_profile")
        )
        print(f"    Entries with user_profile: {profile_count:,}")
        if profile_count == 0:
            print("ERROR: No entries have 'user_profile'.", file=sys.stderr)
            sys.exit(1)

    print(f"  Loading id2meta: {args.id2meta_file}")
    with open(args.id2meta_file, "r", encoding="utf-8") as f:
        raw = f.read().rstrip('\x00')
        id2meta = json.loads(raw)
    print(f"    Items: {len(id2meta):,}")

    # Quick coverage check
    all_pids = set()
    all_products_info = {}  # for embedding generation
    for entry in shopping_data.values():
        for j in entry.get("journeys", []):
            all_pids.update(j.get("product_ids", []))
            if j.get("products_info"):
                all_products_info.update(j["products_info"])
    found = sum(1 for pid in all_pids if pid in id2meta)
    has_tid = 0
    has_norm = 0
    for pid in all_pids:
        tid, used_norm = get_item_tid(pid, id2meta)
        if tid is not None:
            has_tid += 1
            if used_norm:
                has_norm += 1
    print(f"    Distinct product IDs in journeys: {len(all_pids):,}")
    print(f"    Found in id2meta: {found:,} "
          f"({found / max(len(all_pids), 1) * 100:.1f}%)")
    print(f"    With valid TID: {has_tid:,} "
          f"({has_tid / max(len(all_pids), 1) * 100:.1f}%)")
    print(f"    Using summary_words_norm: {has_norm:,} "
          f"({has_norm / max(has_tid, 1) * 100:.1f}% of valid TIDs)")

    # =========================================================================
    # Step 1b: Generate embeddings (if enabled)
    # =========================================================================
    embedding_map = None
    if use_embedding:
        print()
        print("=" * 70)
        print("Step 1b: Generating product embeddings for cosine diversity")
        print(f"  Model: {args.embedding_model}")
        print("=" * 70)
        cache_dir = args.embedding_cache_dir or args.output_dir
        embedding_map = load_or_generate_embeddings(
            list(all_pids), all_products_info,
            args.embedding_model, cache_dir,
        )
        print(f"  Embedding map: {len(embedding_map):,} products loaded")
    else:
        print(f"\n  Cosine embedding: OFF (use --use_embedding to enable)")

    # =========================================================================
    # Step 2: Build SFT data
    # =========================================================================
    print()
    print("=" * 70)
    print(f"Step 2: Building {task} SFT data")
    print(f"  max_events = {args.max_events}")
    if task == "profile2journey":
        print(f"  max_recent_events = {args.max_recent_events}")
    print(f"  max_products_per_journey = {args.max_products_per_journey}")
    print(f"  min_products_per_journey = {args.min_products_per_journey}")
    print(f"  min_avg_products = {args.min_avg_products}")
    print(f"  min_journeys = {args.min_journeys}")
    print(f"  max_journeys = {args.max_journeys}")
    print(f"  keep_empty_ratio = {args.keep_empty_ratio}")
    print(f"  count_ratio = {args.count_ratio}")
    print(f"  dup_threshold = {args.dup_threshold}")
    print(f"  jaccard_threshold = {args.jaccard_threshold}")
    print(f"  cosine_threshold = {args.cosine_threshold}")
    print(f"  use_embedding = {use_embedding}")
    print(f"  seed = {args.seed}")
    print("=" * 70)

    sft_data, vis_data = create_sft_data(
        task=task,
        shopping_journey_data=shopping_data,
        id2meta=id2meta,
        embedding_map=embedding_map,
        jaccard_threshold=args.jaccard_threshold,
        cosine_threshold=args.cosine_threshold,
        max_events=args.max_events,
        max_recent_events=args.max_recent_events,
        max_products=args.max_products_per_journey,
        min_products=args.min_products_per_journey,
        min_avg_products=args.min_avg_products,
        min_journeys=args.min_journeys,
        max_journeys=args.max_journeys,
        keep_empty_ratio=args.keep_empty_ratio,
        count_ratio=args.count_ratio,
        dup_threshold=args.dup_threshold,
    )

    # =========================================================================
    # Step 3: Save output
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Saving output")
    print("=" * 70)

    output_file = os.path.join(args.output_dir, f"{task}_sft.json")
    save_sft_data(sft_data, output_file)

    # Export step8-compatible JSONL for HTML visualization (real products + TID)
    vis_jsonl = os.path.join(args.output_dir, f"{task}_sft_for_vis.jsonl")
    export_vis_jsonl(vis_data, vis_jsonl)
    print(f"\n  To visualize, run:")
    print(f"  python cook_journey_data/step8_generate_html.py "
          f"--input {vis_jsonl} --results_dir {args.output_dir} "
          f"--item_json <path/to/item.json> --skip_rerank")

    # =========================================================================
    # Step 4: Show example cases
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("Example cases (first 3):")
    print(f"{'=' * 70}")
    for idx, sample in enumerate(sft_data[:3]):
        meta = sample["metadata"]
        print(f"\n--- Example {idx + 1} ---")
        print(f"  User ID:        {meta['user_id']}")
        print(f"  Task:           {meta['task']}")
        print(f"  Num events:     {meta['num_events']}")
        print(f"  Num journeys:   {meta['num_journeys']}")
        print(f"  Products/j:     {meta['num_products_per_journey']}")
        print(f"  Instruction:    {sample['instruction'][:200]}...")
        input_lines = sample["input"].split("\n")
        max_show = 12
        print(f"  Input (first {max_show} lines):")
        for line in input_lines[:max_show]:
            print(f"    {line[:150]}")
        if len(input_lines) > max_show:
            print(f"    ... ({len(input_lines) - max_show} more lines)")
        try:
            out_obj = json.loads(sample["output"])
            cj = out_obj.get("ContinuedJourneys", [])
            if not cj:
                print(f"  Output:         (empty journeys)")
            else:
                print(f"  Output ({len(cj)} journeys):")
                for ji, journey in enumerate(cj[:3], 1):
                    title = journey.get("Title", "")
                    reason = journey.get("Reason", "")
                    tids = journey.get("ProductTIDs", [])
                    print(f"    Journey {ji}: {title}")
                    print(f"      Reason: {reason[:120]}")
                    print(f"      Products: {len(tids)} TIDs")
                    if tids:
                        print(f"        First: {tids[0]}")
                if len(cj) > 3:
                    print(f"    ... ({len(cj) - 3} more journeys)")
        except json.JSONDecodeError:
            print(f"  Output:         {sample['output'][:200]}...")

    print(f"\n{'=' * 70}")
    print("Done!")


if __name__ == "__main__":
    main()
