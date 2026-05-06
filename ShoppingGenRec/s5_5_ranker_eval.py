"""Step 5.5: Journey Ranker Evaluation

Evaluates a trained ranker model against LLM ground truth from the ranked
TSV test data.

Pipeline:
  1. Load test TSV (UserId, Profile, JourneyIndex, Journey, OUTPUT).
  2. Sample N rows with valid LLM OUTPUT as ground truth.
  3. Build ranker prompts (instruction + profile + journey).
  4. Run vLLM inference to get SLM predictions.
  5. Compare SLM vs LLM across multiple metrics:
     - JSON format correctness
     - Filter rate & coverage
     - Ranking quality (overlap, NDCG, brand/retailer hit, diversity)
     - Filter quality (safety, blocklist, gender)

Usage:
    python s5_5_ranker_eval.py \\
        --model_path /path/to/checkpoint \\
        --test_file /path/to/batch_00004.tsv \\
        --output_dir ./eval_results/ranker/ \\
        --sample_n 500
"""

import os
import re
import csv
import json
import sys
import time
import random
import argparse
import math
from collections import defaultdict

import numpy as np

csv.field_size_limit(sys.maxsize)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Constants
# =============================================================================

# V1: full output format (JourneyType, Title, ..., Products with all fields)
RANKER_INSTRUCTION_V1 = (
    "Rank and filter candidate products for a shopping journey. "
    "Pool all products from all queries, apply filters, output one ranked list.\n\n"
    "Filters (in order):\n"
    "- Safety: exclude weapons, medical, tobacco, alcohol, adult content\n"
    "- Relevance: must match journey title, description, and original query\n"
    "- Gender: zero tolerance for mismatches in gender-sensitive categories "
    "(clothing, shoes, accessories, jewelry, bags, watches, beauty)\n"
    "- Seller: exclude blocklisted sellers (shein, temu, ebay, aliexpress, "
    "wish, dhgate). Prefer brand stores > specialized retailers > general retailers\n"
    "- Price: soft signal, not a hard filter\n"
    "- Diversity: collapse near-duplicates, prefer variety\n\n"
    "Use the user's shopping profile as ranking signals: "
    "shoppingGenderPreference (hard gender filter), brandPreferences, "
    "retailerPreferences, priceSensitivity, fashionStyle (soft boosts).\n\n"
    "Input products may include optional Category, Brand, Gender fields — "
    "use them as hints for ranking but do not include them in output. "
    'Output JSON: {"JourneyType":"...","Title":"...","Description":"...",'
    '"ConversationStarter":"...","WhyAmISeeingThis":"...",'
    '"Products":[{"Rank":1,"OfferId":"...","Title":"...","Seller":"...",'
    '"Price":"...","OriginalQuery":"..."},...]}\n\n'
    "Now rank the products."
)

# V2: compact output format (Products with Rank+OfferId only)
RANKER_INSTRUCTION_V2 = (
    "Rank and filter candidate products for a shopping journey. "
    "Pool all products from all queries, apply filters, output one ranked list.\n\n"
    "Filters: Safety (exclude weapons, medical, tobacco, alcohol, adult) > "
    "Relevance (must match journey title and query) > "
    "Gender (zero tolerance in clothing/shoes/accessories/jewelry/bags/watches/beauty) > "
    "Seller (exclude shein/temu/ebay/aliexpress/wish/dhgate; prefer brand stores > specialized > general) > "
    "Price (soft signal) > Diversity (collapse near-duplicates).\n\n"
    "Use the user's shopping profile as ranking signals.\n\n"
    'Output JSON: {"Products":[{"Rank":1,"OfferId":"..."},...]}'
    "\n\nNow rank the products."
)

RANKER_INSTRUCTIONS = {
    "v1": RANKER_INSTRUCTION_V1,
    "v2": RANKER_INSTRUCTION_V2,
}

SELLER_BLOCKLIST = {
    "shein", "temu", "ebay", "aliexpress", "wish", "dhgate",
    "lightinthebox", "global sources", "alibaba",
}

SAFETY_KEYWORDS = [
    "weapon", "firearm", "gun", "ammunition", "ammo", "knife",
    "tobacco", "cigarette", "vaping", "vape", "e-cigarette",
    "alcohol", "beer", "wine", "liquor", "whiskey", "vodka", "rum",
    "adult", "racy", "lingerie", "sexy costume",
    "drug", "controlled substance", "prescription",
    "supplement", "cbd", "thc", "cannabis", "marijuana",
    "funeral", "casket", "urn", "memorial stone",
    "hunting decoy", "hunting blind", "hunting call",
]

GENDER_SENSITIVE_CATEGORIES_KEYWORDS = [
    "clothing", "shoes", "dress", "skirt", "blouse", "bra",
    "lingerie", "swimwear", "jewelry", "watches", "bags",
    "handbag", "purse", "accessories", "fragrance", "perfume",
    "beauty", "cosmetic", "makeup", "suit", "tie", "underwear",
]


# =============================================================================
# Data Loading
# =============================================================================

def _unescape_json(text):
    """Try to parse JSON, progressively unescaping if needed."""
    for _ in range(3):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            text = text.replace('\\"', '"')
    return None


def load_test_tsv(filepath, max_read=None):
    """Load ranked TSV, return list of row dicts with parsed fields."""
    print(f"  Loading: {filepath}")
    rows = []
    total = 0
    parse_ok = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            if max_read and total > max_read:
                break

            uid = row.get("UserId", "").strip()
            ji = row.get("JourneyIndex", "").strip()
            profile_raw = row.get("Profile", "")
            journey_raw = row.get("Journey", "")
            output_raw = row.get("OUTPUT", "")

            # Parse OUTPUT (LLM ground truth)
            gt = _unescape_json(output_raw)
            if gt is None or not isinstance(gt, dict):
                continue
            gt_prods = gt.get("Products", [])
            if not gt_prods:
                continue

            # Parse Journey
            journey = _unescape_json(journey_raw)
            if journey is None or not isinstance(journey, dict):
                continue
            queries = journey.get("Queries", [])
            if not queries:
                continue

            # Parse Profile
            profile = _unescape_json(profile_raw)
            if profile is None:
                profile = {}
            # Wrap if needed
            if "userShoppingProfile" not in profile:
                profile = {"userShoppingProfile": profile}

            # Count input products
            n_input = sum(len(q.get("Products", [])) for q in queries)

            parse_ok += 1
            rows.append({
                "UserId": uid,
                "JourneyIndex": ji,
                "profile": profile,
                "journey": journey,
                "gt_output": gt,
                "gt_products": gt_prods,
                "n_input_products": n_input,
                "n_gt_products": len(gt_prods),
                "profile_raw": profile_raw,
                "journey_raw": journey_raw,
                "output_raw": output_raw,
            })

    print(f"    Read {total:,} rows, {parse_ok:,} valid ({parse_ok/max(total,1)*100:.1f}%)")
    return rows


# =============================================================================
# Prompt Building
# =============================================================================

def build_ranker_prompt(profile_json, journey_json):
    """Build the ranker input text: User Profile + Journey."""
    lines = [
        "User Shopping Profile:",
        json.dumps(profile_json, ensure_ascii=False),
        "",
        "Journey:",
        json.dumps(journey_json, ensure_ascii=False),
    ]
    return "\n".join(lines)


# =============================================================================
# JSON Parsing
# =============================================================================

def parse_ranker_output(raw):
    """Parse ranker JSON output from raw text."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # Strip think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip OUTPUT tags
    text = re.sub(r"</?OUTPUT>", "", text).strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Find JSON object
    bs = text.find("{")
    if bs == -1:
        return None
    depth = 0
    be = -1
    for i in range(bs, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                be = i
                break
    cand = text[bs:be + 1] if be != -1 else text[bs:]
    try:
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def enrich_products(parsed_output, offerid_to_product):
    """Normalize parsed output: extract Rank+OfferId, join back with input
    candidates to get full product info (Title, Seller, Price, Brand, etc.).

    Works for both v1 (full output) and v2 (compact Rank+OfferId only).
    Returns a list of enriched product dicts.
    """
    if parsed_output is None:
        return []
    prods = parsed_output.get("Products", [])
    if not prods:
        return []

    enriched = []
    seen_ids = set()
    for p in prods:
        oid = str(p.get("OfferId", ""))
        if not oid or oid in seen_ids:
            continue
        seen_ids.add(oid)

        rank = p.get("Rank", len(enriched) + 1)
        # Start with input candidate info
        base = dict(offerid_to_product.get(oid, {}))
        # Override with output fields (Rank always from output)
        base["Rank"] = rank
        base["OfferId"] = oid
        # If output has Title/Seller/Price (v1), use those as well
        for field in ("Title", "Seller", "Price", "OriginalQuery"):
            if p.get(field):
                base[field] = p[field]
        enriched.append(base)
    return enriched


# =============================================================================
# Metrics Computation
# =============================================================================

def _normalize_seller(s):
    """Normalize seller name for blocklist checking."""
    s = s.lower().strip()
    for suffix in [".com", " official", " official store", " store"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s


def _dcg(relevances, k=None):
    """Compute DCG@k."""
    if k:
        relevances = relevances[:k]
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def _ndcg(pred_ids, gt_ids, k=None):
    """Compute NDCG@k. GT order defines ideal ranking."""
    gt_set = set(gt_ids)

    # Relevance: 1 if in GT, 0 otherwise (binary)
    pred_rels = [1.0 if pid in gt_set else 0.0 for pid in pred_ids]
    # Ideal: all GT items first
    ideal_rels = [1.0] * min(len(gt_ids), k or len(gt_ids))

    dcg = _dcg(pred_rels, k)
    idcg = _dcg(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0


def _brand_hit(prods, brand_prefs):
    """Fraction of products matching any brand preference."""
    if not brand_prefs or not prods:
        return None
    hits = 0
    for p in prods:
        brand_norm = _normalize_seller(p.get("Brand", ""))
        title_norm = _normalize_seller(p.get("Title", ""))
        seller_norm = _normalize_seller(p.get("Seller", ""))
        if any(bp in brand_norm or bp in title_norm or bp in seller_norm
               for bp in brand_prefs):
            hits += 1
    return hits / len(prods)


def _brand_coverage(prods, brand_prefs):
    """Fraction of brand preferences covered by products."""
    if not brand_prefs:
        return None
    covered = set()
    for p in prods:
        brand_norm = _normalize_seller(p.get("Brand", ""))
        title_norm = _normalize_seller(p.get("Title", ""))
        seller_norm = _normalize_seller(p.get("Seller", ""))
        for bp in brand_prefs:
            if bp in brand_norm or bp in title_norm or bp in seller_norm:
                covered.add(bp)
    return len(covered) / len(brand_prefs)


def _retailer_hit(prods, retailer_prefs):
    """Fraction of products matching any retailer preference."""
    if not retailer_prefs or not prods:
        return None
    hits = 0
    for p in prods:
        seller_norm = _normalize_seller(p.get("Seller", ""))
        if any(rp in seller_norm for rp in retailer_prefs):
            hits += 1
    return hits / len(prods)


def _retailer_coverage(prods, retailer_prefs):
    """Fraction of retailer preferences covered by products."""
    if not retailer_prefs:
        return None
    covered = set()
    for p in prods:
        seller_norm = _normalize_seller(p.get("Seller", ""))
        for rp in retailer_prefs:
            if rp in seller_norm:
                covered.add(rp)
    return len(covered) / len(retailer_prefs)


def _seller_diversity(prods):
    """Ratio of unique sellers to total products."""
    if not prods:
        return 0.0
    sellers = set(_normalize_seller(p.get("Seller", "")) for p in prods)
    return len(sellers) / len(prods)


def _brand_diversity(prods):
    """Ratio of unique brands to total products (enriched products have Brand)."""
    if not prods:
        return 0.0
    brands = set()
    for p in prods:
        brand = p.get("Brand", "") or p.get("Seller", "")
        brands.add(_normalize_seller(brand))
    return len(brands) / len(prods)


# Pre-compile safety keyword regex with word boundaries
_SAFETY_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in SAFETY_KEYWORDS) + r')\b',
    re.IGNORECASE,
)


def _count_safety(prods):
    """Count products with safety-violating keywords in title (word-boundary match)."""
    count = 0
    for p in prods:
        title = p.get("Title", "")
        if _SAFETY_PATTERN.search(title):
            count += 1
    return count


# Pre-compile blocklist regex with word boundaries
_BLOCKLIST_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in SELLER_BLOCKLIST) + r')\b',
    re.IGNORECASE,
)


def _count_blocklist(prods):
    """Count products from blocklisted sellers (word-boundary match)."""
    count = 0
    for p in prods:
        seller = p.get("Seller", "").lower().strip()
        if _BLOCKLIST_PATTERN.search(seller):
            count += 1
    return count


# =============================================================================
# Diversity Metrics
# =============================================================================

def _product_text(p):
    """Build text representation for a product (for Jaccard & embedding)."""
    parts = []
    for field in ("Title", "Brand", "Seller", "Price"):
        val = p.get(field, "")
        if val:
            parts.append(str(val))
    return " | ".join(parts)


def _jaccard_words(text_a, text_b):
    """Word-level Jaccard similarity between two strings."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _mean_pairwise_jaccard(prods):
    """Mean pairwise Jaccard similarity among products in a sample."""
    if len(prods) < 2:
        return None
    texts = [_product_text(p) for p in prods]
    total_sim = 0.0
    n_pairs = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total_sim += _jaccard_words(texts[i], texts[j])
            n_pairs += 1
    return total_sim / n_pairs if n_pairs > 0 else 0.0


def _mean_pairwise_cosine(embeddings):
    """Mean pairwise cosine similarity from an (N, dim) embedding matrix.
    Assumes embeddings are L2-normalized."""
    if embeddings is None or len(embeddings) < 2:
        return None
    sim_matrix = embeddings @ embeddings.T
    n = len(embeddings)
    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n, k=1)
    return float(np.mean(sim_matrix[triu_indices]))


def load_or_generate_embeddings(offer_ids, product_info, embedding_model_path,
                                cache_dir, batch_size=1024, max_length=512):
    """Load cached embeddings for OfferIds, generate missing ones, update cache.

    Args:
        offer_ids: list of OfferId strings to get embeddings for
        product_info: dict of OfferId -> product dict (with Title, Brand, etc.)
        embedding_model_path: path to embedding model
        cache_dir: directory for product_embeddings_cache.npz
        batch_size: batch size for embedding generation
        max_length: max token length

    Returns:
        dict of OfferId -> np.ndarray (L2-normalized embedding)
    """
    cache_file = os.path.join(cache_dir, "product_embeddings_cache.npz")
    cached_ids = []
    cached_embs = None

    # Load existing cache
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        cached_ids = list(data["offer_ids"])
        cached_embs = data["embeddings"]
        print(f"    Loaded embedding cache: {len(cached_ids):,} products")

    cached_map = {}
    for i, oid in enumerate(cached_ids):
        cached_map[str(oid)] = cached_embs[i]

    # Find missing
    unique_ids = list(set(offer_ids))
    missing_ids = [oid for oid in unique_ids if oid not in cached_map]

    if missing_ids:
        print(f"    Need embeddings for {len(missing_ids):,} new products "
              f"({len(unique_ids) - len(missing_ids):,} cached)")

        # Prepare texts
        texts = []
        valid_ids = []
        for oid in missing_ids:
            p = product_info.get(oid, {})
            text = _product_text(p)
            if text.strip():
                texts.append(text)
                valid_ids.append(oid)

        if texts:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from tqdm import tqdm

            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
            print(f"    Loading embedding model ({num_gpus} GPUs available) ...")
            model = AutoModel.from_pretrained(
                embedding_model_path,
                torch_dtype=torch.float16 if num_gpus > 0 else torch.float32,
                trust_remote_code=True,
            ).to(device)
            if num_gpus > 1:
                model = torch.nn.DataParallel(model)
                print(f"    Using DataParallel across {num_gpus} GPUs")
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(
                embedding_model_path, trust_remote_code=True
            )

            all_embs = []
            n_batches = (len(texts) + batch_size - 1) // batch_size
            for start in tqdm(range(0, len(texts), batch_size),
                              total=n_batches, desc="    Embedding",
                              mininterval=30):
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
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                    all_embs.append(embs.cpu().numpy().astype(np.float32))

            new_embs = np.vstack(all_embs)
            for i, oid in enumerate(valid_ids):
                cached_map[oid] = new_embs[i]
            print(f"    Generated {len(valid_ids):,} new embeddings")

            # Cleanup GPU memory
            del model, tokenizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Save updated cache
        all_ids = list(cached_map.keys())
        all_embs = np.vstack([cached_map[oid] for oid in all_ids])
        np.savez_compressed(cache_file,
                            offer_ids=np.array(all_ids),
                            embeddings=all_embs)
        print(f"    Updated cache: {len(all_ids):,} products -> {cache_file}")
    else:
        print(f"    All {len(unique_ids):,} products found in cache")

    return cached_map


def compute_metrics(samples):
    """Compute all evaluation metrics.

    Each sample has: gt_output, slm_output, profile, journey, input products.
    """
    # Accumulators
    n = len(samples)
    gt_json_ok = 0
    gt_has_products = 0
    gt_rank_valid = 0
    slm_json_ok = 0
    slm_has_products = 0
    slm_rank_valid = 0

    gt_filter_rates = []
    slm_filter_rates = []

    precisions = []
    recalls = []
    f1s = []
    ndcg_10s = []
    ndcg_15s = []
    ndcg_20s = []

    gt_brand_hits = []
    slm_brand_hits = []
    gt_retailer_hits = []
    slm_retailer_hits = []

    gt_brand_coverages = []
    slm_brand_coverages = []
    gt_retailer_coverages = []
    slm_retailer_coverages = []

    gt_seller_divs = []
    slm_seller_divs = []
    gt_brand_divs = []
    slm_brand_divs = []

    # Filter quality
    gt_safety_violations = 0
    slm_safety_violations = 0
    gt_blocklist_violations = 0
    slm_blocklist_violations = 0
    gt_total_prods = 0
    slm_total_prods = 0

    # Users with brand/retailer prefs (for denominator)
    n_with_brand_prefs = 0
    n_with_retailer_prefs = 0

    # OfferId quality (hallucination & duplicates)
    gt_samples_with_hallucination = 0
    slm_samples_with_hallucination = 0
    gt_samples_with_duplicates = 0
    slm_samples_with_duplicates = 0
    gt_n_hallucinated = 0
    slm_n_hallucinated = 0
    gt_n_duplicated = 0
    slm_n_duplicated = 0

    for s in samples:
        gt = s["gt_output"]
        slm = s["slm_output"]
        profile = s["profile"]
        journey = s["journey"]
        n_input = s["n_input_products"]

        gt_prods = gt.get("Products", []) if gt else []
        sp = profile.get("userShoppingProfile", {})
        brand_prefs = set(_normalize_seller(b) for b in sp.get("brandPreferences", []) if b.strip())
        retailer_prefs = set(_normalize_seller(r) for r in sp.get("retailerPreferences", []) if r.strip())

        # --- JSON Correctness (both GT and SLM) ---
        # GT
        if gt is not None and isinstance(gt, dict):
            gt_json_ok += 1
            if gt_prods:
                gt_has_products += 1
                gt_ranks = [p.get("Rank", 0) for p in gt_prods]
                if gt_ranks == list(range(1, len(gt_ranks) + 1)):
                    gt_rank_valid += 1
        # SLM
        if slm is not None:
            slm_json_ok += 1
            slm_prods = slm.get("Products", [])
            if slm_prods:
                slm_has_products += 1
                ranks = [p.get("Rank", 0) for p in slm_prods]
                if ranks == list(range(1, len(ranks) + 1)):
                    slm_rank_valid += 1
        else:
            slm_prods = []

        # --- OfferId Quality (hallucination & duplicates) ---
        input_ids = s.get("input_offerids", set())
        gt_raw_ids = s.get("gt_raw_offerids", [])
        slm_raw_ids = s.get("slm_raw_offerids", [])

        if gt_raw_ids and input_ids:
            gt_hall = sum(1 for oid in gt_raw_ids if oid and oid not in input_ids)
            if gt_hall > 0:
                gt_samples_with_hallucination += 1
                gt_n_hallucinated += gt_hall
            gt_seen = set()
            gt_dup = 0
            for oid in gt_raw_ids:
                if oid in gt_seen:
                    gt_dup += 1
                gt_seen.add(oid)
            if gt_dup > 0:
                gt_samples_with_duplicates += 1
                gt_n_duplicated += gt_dup

        if slm_raw_ids:
            if input_ids:
                slm_hall = sum(1 for oid in slm_raw_ids if oid and oid not in input_ids)
                if slm_hall > 0:
                    slm_samples_with_hallucination += 1
                    slm_n_hallucinated += slm_hall
            slm_seen = set()
            slm_dup = 0
            for oid in slm_raw_ids:
                if oid in slm_seen:
                    slm_dup += 1
                slm_seen.add(oid)
            if slm_dup > 0:
                slm_samples_with_duplicates += 1
                slm_n_duplicated += slm_dup

        # --- Filter Rate ---
        if n_input > 0:
            gt_filter_rates.append(len(gt_prods) / n_input)
            slm_filter_rates.append(len(slm_prods) / n_input)


        # --- Overlap (Precision, Recall, F1) ---
        gt_ids = [str(p.get("OfferId", "")) for p in gt_prods]
        slm_ids = [str(p.get("OfferId", "")) for p in slm_prods]
        gt_set = set(gt_ids)
        slm_set = set(slm_ids)
        overlap = gt_set & slm_set
        p = len(overlap) / max(len(slm_set), 1)
        r = len(overlap) / max(len(gt_set), 1)
        f = 2 * p * r / max(p + r, 1e-9)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        # --- NDCG ---
        if gt_ids:
            ndcg_10s.append(_ndcg(slm_ids, gt_ids, k=10))
            ndcg_15s.append(_ndcg(slm_ids, gt_ids, k=15))
            ndcg_20s.append(_ndcg(slm_ids, gt_ids, k=20))

        # --- Brand Preference Hit Rate (normalized comparison) ---
        if brand_prefs:
            n_with_brand_prefs += 1
            bh_gt = _brand_hit(gt_prods, brand_prefs)
            bh_slm = _brand_hit(slm_prods, brand_prefs)
            if bh_gt is not None:
                gt_brand_hits.append(bh_gt)
            if bh_slm is not None:
                slm_brand_hits.append(bh_slm)
            bc_gt = _brand_coverage(gt_prods, brand_prefs)
            bc_slm = _brand_coverage(slm_prods, brand_prefs)
            if bc_gt is not None:
                gt_brand_coverages.append(bc_gt)
            if bc_slm is not None:
                slm_brand_coverages.append(bc_slm)

        # --- Retailer Preference Hit Rate (normalized comparison) ---
        if retailer_prefs:
            n_with_retailer_prefs += 1
            rh_gt = _retailer_hit(gt_prods, retailer_prefs)
            rh_slm = _retailer_hit(slm_prods, retailer_prefs)
            if rh_gt is not None:
                gt_retailer_hits.append(rh_gt)
            if rh_slm is not None:
                slm_retailer_hits.append(rh_slm)
            rc_gt = _retailer_coverage(gt_prods, retailer_prefs)
            rc_slm = _retailer_coverage(slm_prods, retailer_prefs)
            if rc_gt is not None:
                gt_retailer_coverages.append(rc_gt)
            if rc_slm is not None:
                slm_retailer_coverages.append(rc_slm)

        # --- Seller Diversity (normalized) ---
        gt_seller_divs.append(_seller_diversity(gt_prods))
        slm_seller_divs.append(_seller_diversity(slm_prods))

        # --- Brand Diversity (enriched products have Brand field) ---
        gt_brand_divs.append(_brand_diversity(gt_prods))
        slm_brand_divs.append(_brand_diversity(slm_prods))

        # --- Filter Quality ---
        gt_total_prods += len(gt_prods)
        slm_total_prods += len(slm_prods)
        gt_safety_violations += _count_safety(gt_prods)
        slm_safety_violations += _count_safety(slm_prods)
        gt_blocklist_violations += _count_blocklist(gt_prods)
        slm_blocklist_violations += _count_blocklist(slm_prods)

    # Build results dict
    def _mean(arr):
        return float(np.mean(arr)) if arr else 0.0

    metrics = {
        "total_samples": n,
        "gt_json_parse_success": gt_json_ok,
        "gt_has_products": gt_has_products,
        "gt_rank_valid": gt_rank_valid,
        "slm_json_parse_success": slm_json_ok,
        "slm_has_products": slm_has_products,
        "slm_rank_valid": slm_rank_valid,
        "gt_total_products": gt_total_prods,
        "slm_total_products": slm_total_prods,
        "mean_gt_filter_rate": _mean(gt_filter_rates),
        "mean_slm_filter_rate": _mean(slm_filter_rates),
        "mean_gt_precision": 1.0,
        "mean_gt_recall": 1.0,
        "mean_gt_f1": 1.0,
        "mean_gt_ndcg_10": 1.0,
        "mean_gt_ndcg_15": 1.0,
        "mean_gt_ndcg_20": 1.0,
        "mean_precision": _mean(precisions),
        "mean_recall": _mean(recalls),
        "mean_f1": _mean(f1s),
        "mean_ndcg_10": _mean(ndcg_10s),
        "mean_ndcg_15": _mean(ndcg_15s),
        "mean_ndcg_20": _mean(ndcg_20s),
        "n_with_brand_prefs": n_with_brand_prefs,
        "mean_gt_brand_hit": _mean(gt_brand_hits),
        "mean_slm_brand_hit": _mean(slm_brand_hits),
        "mean_gt_brand_coverage": _mean(gt_brand_coverages),
        "mean_slm_brand_coverage": _mean(slm_brand_coverages),
        "n_with_retailer_prefs": n_with_retailer_prefs,
        "mean_gt_retailer_hit": _mean(gt_retailer_hits),
        "mean_slm_retailer_hit": _mean(slm_retailer_hits),
        "mean_gt_retailer_coverage": _mean(gt_retailer_coverages),
        "mean_slm_retailer_coverage": _mean(slm_retailer_coverages),
        "mean_gt_seller_diversity": _mean(gt_seller_divs),
        "mean_slm_seller_diversity": _mean(slm_seller_divs),
        "mean_gt_brand_diversity": _mean(gt_brand_divs),
        "mean_slm_brand_diversity": _mean(slm_brand_divs),
        "gt_safety_violations": gt_safety_violations,
        "slm_safety_violations": slm_safety_violations,
        "gt_blocklist_violations": gt_blocklist_violations,
        "slm_blocklist_violations": slm_blocklist_violations,
        "gt_samples_with_hallucination": gt_samples_with_hallucination,
        "gt_n_hallucinated_ids": gt_n_hallucinated,
        "slm_samples_with_hallucination": slm_samples_with_hallucination,
        "slm_n_hallucinated_ids": slm_n_hallucinated,
        "gt_samples_with_duplicates": gt_samples_with_duplicates,
        "gt_n_duplicated_ids": gt_n_duplicated,
        "slm_samples_with_duplicates": slm_samples_with_duplicates,
        "slm_n_duplicated_ids": slm_n_duplicated,
    }
    return metrics


# =============================================================================
# Display
# =============================================================================

def print_comparison(metrics):
    """Print side-by-side comparison table."""
    n = metrics["total_samples"]
    gt_tp = max(metrics["gt_total_products"], 1)
    slm_tp = max(metrics["slm_total_products"], 1)

    print(f"\n{'=' * 90}")
    print(f"  Ranker Evaluation: LLM (Ground Truth) vs SLM (Model)")
    print(f"{'=' * 90}")
    print(f"  # Evaluated samples: {n}")
    print()

    W1, W2, W3 = 50, 22, 22

    def _row(label, lv, sv):
        print(f"  {label:<{W1}s} {lv:>{W2}s} {sv:>{W3}s}")

    def _sep():
        print(f"  {'-' * W1} {'-' * W2} {'-' * W3}")

    def _cnt_pct(cnt, tot):
        return f"{cnt} ({cnt/max(tot,1)*100:.1f}%)"

    _row("Metric", "LLM (GT)", "SLM (Model)")
    _sep()

    # --- JSON Format Correctness ---
    print(f"\n  --- JSON Format Correctness ---")
    _row("Total samples", str(n), str(n))
    _row("JSON parse success",
         _cnt_pct(metrics['gt_json_parse_success'], n),
         _cnt_pct(metrics['slm_json_parse_success'], n))
    _row("Has Products array",
         _cnt_pct(metrics['gt_has_products'], n),
         _cnt_pct(metrics['slm_has_products'], n))
    _row("Valid Rank sequence (1,2,3...)",
         _cnt_pct(metrics['gt_rank_valid'], n),
         _cnt_pct(metrics['slm_rank_valid'], n))

    # Hallucination & Duplicate stats
    gt_hall_samples = metrics.get('gt_samples_with_hallucination', 0)
    slm_hall_samples = metrics.get('slm_samples_with_hallucination', 0)
    gt_hall_ids = metrics.get('gt_n_hallucinated_ids', 0)
    slm_hall_ids = metrics.get('slm_n_hallucinated_ids', 0)
    gt_dup_samples = metrics.get('gt_samples_with_duplicates', 0)
    slm_dup_samples = metrics.get('slm_samples_with_duplicates', 0)
    gt_dup_ids = metrics.get('gt_n_duplicated_ids', 0)
    slm_dup_ids = metrics.get('slm_n_duplicated_ids', 0)

    _row("Hallucinated OfferIds (user-level)",
         _cnt_pct(gt_hall_samples, n),
         _cnt_pct(slm_hall_samples, n))
    _row("Hallucinated OfferIds (offer-level)",
         f"{gt_hall_ids} ({gt_hall_ids/gt_tp*100:.2f}%)",
         f"{slm_hall_ids} ({slm_hall_ids/slm_tp*100:.2f}%)")
    _row("Duplicate OfferIds (user-level)",
         _cnt_pct(gt_dup_samples, n),
         _cnt_pct(slm_dup_samples, n))
    _row("Duplicate OfferIds (offer-level)",
         f"{gt_dup_ids} ({gt_dup_ids/gt_tp*100:.2f}%)",
         f"{slm_dup_ids} ({slm_dup_ids/slm_tp*100:.2f}%)")

    # --- Filter Rate & Coverage ---
    print(f"\n  --- Filter Rate & Coverage ---")
    _row("Total products",
         f"{metrics['gt_total_products']:,}",
         f"{metrics['slm_total_products']:,}")
    _row("Avg retention rate (output/input)",
         f"{metrics['mean_gt_filter_rate']*100:.1f}%",
         f"{metrics['mean_slm_filter_rate']*100:.1f}%")

    # --- Ranking Quality ---
    print(f"\n  --- Ranking Quality ---")
    _row("Product Overlap Precision",
         f"{metrics['mean_gt_precision']*100:.2f}%",
         f"{metrics['mean_precision']*100:.2f}%")
    _row("Product Overlap Recall",
         f"{metrics['mean_gt_recall']*100:.2f}%",
         f"{metrics['mean_recall']*100:.2f}%")
    _row("Product Overlap F1",
         f"{metrics['mean_gt_f1']*100:.2f}%",
         f"{metrics['mean_f1']*100:.2f}%")
    _row("NDCG@10",
         f"{metrics['mean_gt_ndcg_10']*100:.2f}%",
         f"{metrics['mean_ndcg_10']*100:.2f}%")
    _row("NDCG@15",
         f"{metrics['mean_gt_ndcg_15']*100:.2f}%",
         f"{metrics['mean_ndcg_15']*100:.2f}%")
    _row("NDCG@20",
         f"{metrics['mean_gt_ndcg_20']*100:.2f}%",
         f"{metrics['mean_ndcg_20']*100:.2f}%")

    nb = metrics.get('n_with_brand_prefs', 0)
    nr = metrics.get('n_with_retailer_prefs', 0)
    _row(f"Brand pref hit rate ({nb} users w/ prefs)",
         f"{metrics['mean_gt_brand_hit']*100:.2f}%",
         f"{metrics['mean_slm_brand_hit']*100:.2f}%")
    _row(f"Brand pref coverage ({nb} users w/ prefs)",
         f"{metrics['mean_gt_brand_coverage']*100:.2f}%",
         f"{metrics['mean_slm_brand_coverage']*100:.2f}%")
    _row(f"Retailer pref hit rate ({nr} users w/ prefs)",
         f"{metrics['mean_gt_retailer_hit']*100:.2f}%",
         f"{metrics['mean_slm_retailer_hit']*100:.2f}%")
    _row(f"Retailer pref coverage ({nr} users w/ prefs)",
         f"{metrics['mean_gt_retailer_coverage']*100:.2f}%",
         f"{metrics['mean_slm_retailer_coverage']*100:.2f}%")
    _row("Seller diversity (unique sellers/total)",
         f"{metrics['mean_gt_seller_diversity']*100:.1f}%",
         f"{metrics['mean_slm_seller_diversity']*100:.1f}%")
    _row("Brand diversity (unique brands/total)",
         f"{metrics['mean_gt_brand_diversity']*100:.1f}%",
         f"{metrics['mean_slm_brand_diversity']*100:.1f}%")
    gt_jac = metrics.get('mean_gt_jaccard_sim', 0)
    slm_jac = metrics.get('mean_slm_jaccard_sim', 0)
    _row("Near-dup similarity (Jaccard, lower=better)",
         f"{gt_jac:.4f}",
         f"{slm_jac:.4f}")
    gt_emb = metrics.get('mean_gt_emb_sim', 0)
    slm_emb = metrics.get('mean_slm_emb_sim', 0)
    if gt_emb > 0 or slm_emb > 0:
        _row("Near-dup similarity (embedding, lower=better)",
             f"{gt_emb:.4f}",
             f"{slm_emb:.4f}")

    # --- Filter Quality ---
    print(f"\n  --- Filter Quality ---")
    _row("Safety violations (keyword match)",
         f"{metrics['gt_safety_violations']} ({metrics['gt_safety_violations']/gt_tp*100:.2f}%)",
         f"{metrics['slm_safety_violations']} ({metrics['slm_safety_violations']/slm_tp*100:.2f}%)")
    _row("Blocklist seller violations",
         f"{metrics['gt_blocklist_violations']} ({metrics['gt_blocklist_violations']/gt_tp*100:.2f}%)",
         f"{metrics['slm_blocklist_violations']} ({metrics['slm_blocklist_violations']/slm_tp*100:.2f}%)")

    print(f"\n{'=' * 90}")


# =============================================================================
# vLLM Inference
# =============================================================================

def run_vllm_inference(prompts, model_path, num_gpus, gpu_mem, max_model_len,
                       max_tokens):
    """Run batched vLLM inference."""
    from vllm import LLM, SamplingParams

    print(f"\nInitializing vLLM ...")
    print(f"  Model: {model_path}")
    print(f"  TP: {num_gpus}, GPU mem: {gpu_mem}")
    print(f"  max_model_len: {max_model_len}, max_tokens: {max_tokens}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        trust_remote_code=True,
        seed=SEED,
        dtype="bfloat16",
        enforce_eager=True,
    )
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.7, top_p=0.8,
                        top_k=20)

    # Truncate
    _tok = llm.get_tokenizer()
    max_input = max_model_len - max_tokens
    truncated = 0
    for i, p in enumerate(prompts):
        tok_ids = _tok.encode(p)
        if len(tok_ids) > max_input:
            prompts[i] = _tok.decode(tok_ids[:max_input],
                                     skip_special_tokens=False)
            truncated += 1
    if truncated:
        print(f"  WARNING: Truncated {truncated}/{len(prompts)} prompts")

    print(f"  Running inference on {len(prompts)} prompts ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(prompts) / elapsed:.1f} items/s)")
    return [o.outputs[0].text.strip() for o in outputs], elapsed


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Journey Ranker Evaluation")
    p.add_argument(
        "--model_path", type=str,
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_full_ranker/checkpoint-650",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_full_ranker_v2_optimized_lr2e-5/checkpoint-1000",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_full_ranker_v3_lr1e5/checkpoint-2100",
    )
    p.add_argument(
        "--instruction_version", type=str, default="v2",
        choices=["v1", "v2"],
        help="v1: full output format, v2: compact Rank+OfferId only",
    )
    p.add_argument(
        "--test_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260424_JourneyRanker/sft_data/v1_500K_journey_ranker_sft_full.jsonl",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260424_JourneyRanker/sft_data/v2_500K_journey_ranker_sft_optimized.jsonl",
        help="Test data: either a ranked TSV or the SFT JSONL file",
    )
    p.add_argument("--output_dir", type=str, 
                   default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260424_JourneyRanker/evaluation_results",)
    p.add_argument("--output_file_name", type=str,
                   #default="ranker_eval_results_full_650.json",
                   #default="ranker_eval_results_v2_full_lr2e-5_1000.json",
                   default="ranker_eval_results_v3_full_lr1e-5_2100.json"
                   )
    p.add_argument("--sample_n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_gpus", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=20000)
    p.add_argument("--max_tokens", type=int, default=8000)
    p.add_argument("--max_read", type=int, default=50000,
                   help="Max rows to read from test file (for large TSV)")
    p.add_argument("--eval_only", action="store_true", default=False,
                   help="Skip inference, re-evaluate from existing detail JSONL")
    p.add_argument("--embedding_model", type=str,
                   default="/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3-Embedding-0.6B",
                   help="Path to embedding model for diversity metrics")
    p.add_argument("--skip_embedding_diversity", action="store_true",
                   default=False,
                   help="Skip embedding-based diversity (only compute Jaccard)")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # GPU (only needed for inference)
    if not args.eval_only and args.num_gpus is None:
        import torch
        args.num_gpus = max(
            torch.cuda.device_count() if torch.cuda.is_available() else 1, 1
        )

    # Output dir
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.test_file))
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Step 5.5: Journey Ranker Evaluation")
    print("=" * 70)
    print(f"  Model:      {args.model_path}")
    print(f"  Test file:  {args.test_file}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Instruction: {args.instruction_version}")
    print(f"  Sample N:   {args.sample_n}")
    if args.eval_only:
        print(f"  Mode:       eval_only (skip inference)")
    else:
        print(f"  GPUs:       {args.num_gpus}")
    print()

    # =========================================================================
    # Step 1: Load test data
    # =========================================================================
    print("Step 1: Loading test data ...")

    if args.test_file.endswith(".jsonl"):
        # Load from SFT JSONL format
        print("  Format: SFT JSONL")
        rows = []
        with open(args.test_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if args.max_read and line_num >= args.max_read:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                inp = d.get("input", "")
                out_raw = d.get("output", "")
                meta = d.get("metadata", {})

                # Parse output
                try:
                    gt = json.loads(out_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                gt_prods = gt.get("Products", [])
                if not gt_prods:
                    continue

                # Extract profile JSON from input
                profile = {}
                prof_match = re.search(
                    r'\{"userShoppingProfile":\{.*?\}\}', inp
                )
                if prof_match:
                    try:
                        profile = json.loads(prof_match.group(0))
                    except json.JSONDecodeError:
                        pass

                # Extract journey JSON from input
                journey = None
                j_start = inp.find('{"JourneyType"')
                if j_start >= 0:
                    depth = 0
                    for i in range(j_start, len(inp)):
                        if inp[i] == "{":
                            depth += 1
                        elif inp[i] == "}":
                            depth -= 1
                            if depth == 0:
                                try:
                                    journey = json.loads(inp[j_start:i + 1])
                                except json.JSONDecodeError:
                                    pass
                                break
                if journey is None:
                    continue

                n_input = sum(
                    len(q.get("Products", []))
                    for q in journey.get("Queries", [])
                )

                rows.append({
                    "UserId": meta.get("user_id", ""),
                    "JourneyIndex": meta.get("journey_index", ""),
                    "profile": profile,
                    "journey": journey,
                    "gt_output": gt,
                    "gt_products": gt_prods,
                    "n_input_products": n_input,
                    "n_gt_products": len(gt_prods),
                    "input_text": inp,
                })
        print(f"    Loaded {len(rows):,} valid samples from JSONL")

    elif args.test_file.endswith(".tsv"):
        # Load from ranked TSV
        print("  Format: Ranked TSV")
        rows = load_test_tsv(args.test_file, max_read=args.max_read)
    else:
        print(f"  ERROR: Unknown file format: {args.test_file}")
        sys.exit(1)

    if not rows:
        print("  ERROR: No valid samples found.")
        sys.exit(1)

    # =========================================================================
    # Step 2: Sample
    # =========================================================================
    print(f"\nStep 2: Sampling {args.sample_n} from {len(rows):,} ...")
    sample_n = min(args.sample_n, len(rows))
    sampled = random.sample(rows, sample_n)
    print(f"  Sampled {sample_n} rows")

    # =========================================================================
    # Step 3-4: Build prompts & run inference (or load existing results)
    # =========================================================================
    elapsed = 0.0

    if args.eval_only:
        # Load existing raw outputs from detail JSONL
        detail_path = os.path.join(
            args.output_dir,
            args.output_file_name.replace(".json", "_detail.jsonl"),
        )
        if not os.path.isfile(detail_path):
            print(f"  ERROR: Detail file not found: {detail_path}")
            print(f"  Cannot run --eval_only without existing results.")
            sys.exit(1)

        print(f"\nStep 3-4: Loading existing results (--eval_only) ...")
        print(f"  Detail file: {detail_path}")
        detail_rows = []
        with open(detail_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    detail_rows.append(json.loads(line))

        if len(detail_rows) != len(sampled):
            print(f"  WARNING: Detail file has {len(detail_rows)} rows "
                  f"but sampled {len(sampled)}.")
            print(f"  Make sure --seed, --sample_n, --test_file, "
                  f"--max_read match the original run.")

        raw_outputs = [r.get("slm_raw", "") for r in detail_rows]
        # Check for truncated slm_raw (old format used [:2000])
        n_truncated = sum(1 for raw in raw_outputs
                          if len(raw) >= 1999 and not raw.rstrip().endswith("}"))
        if n_truncated > 0:
            print(f"  WARNING: {n_truncated}/{len(raw_outputs)} slm_raw entries "
                  f"appear truncated (old detail format used [:2000]).")
            print(f"  Re-run without --eval_only to regenerate full outputs.")
        print(f"  Loaded {len(raw_outputs)} raw outputs")

    else:
        # Step 3: Build prompts
        print(f"\nStep 3: Building prompts ...")
        from transformers import AutoTokenizer
        tok_cfg = os.path.join(args.model_path, "tokenizer_config.json")
        if os.path.isfile(tok_cfg):
            with open(tok_cfg, "r") as f:
                tc = json.load(f)
            if tc.get("tokenizer_class") not in (
                "Qwen2Tokenizer", "PreTrainedTokenizerFast", None
            ):
                print(f"  [FIX] tokenizer_class "
                      f"'{tc.get('tokenizer_class')}' -> 'Qwen2Tokenizer'")
                tc["tokenizer_class"] = "Qwen2Tokenizer"
                with open(tok_cfg, "w") as f:
                    json.dump(tc, f, indent=2, ensure_ascii=False)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )

        ranker_instruction = RANKER_INSTRUCTIONS[args.instruction_version]
        print(f"  Using instruction version: {args.instruction_version}")

        prompts = []
        for s in sampled:
            # Build input text
            if "input_text" in s:
                user_content = ranker_instruction + "\n" + s["input_text"]
            else:
                input_text = build_ranker_prompt(s["profile"], s["journey"])
                user_content = ranker_instruction + "\n" + input_text

            msgs = [{"role": "user", "content": user_content}]
            formatted = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(formatted)

        print(f"  Built {len(prompts)} chat-formatted prompts")

        # Step 4: Run vLLM inference
        raw_outputs, elapsed = run_vllm_inference(
            prompts,
            model_path=args.model_path,
            num_gpus=args.num_gpus,
            gpu_mem=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
        )

        # Release vLLM GPU memory so embedding model can use it later
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Parse SLM outputs and enrich with input product info
    print(f"\nStep 5: Parsing & enriching SLM outputs ...")
    for i, raw in enumerate(raw_outputs):
        sampled[i]["slm_raw"] = raw
        parsed = parse_ranker_output(raw)
        sampled[i]["slm_parsed"] = parsed

        # Build OfferId -> full product dict from input journey
        offerid_to_product = {}
        for q in sampled[i]["journey"].get("Queries", []):
            for p in q.get("Products", []):
                oid = str(p.get("OfferId", ""))
                if oid:
                    prod = dict(p)
                    prod["OriginalQuery"] = q.get("Query", "")
                    offerid_to_product[oid] = prod

        # Save input OfferIds and raw output OfferIds for hallucination/duplicate checks
        # Extracted from parsed JSON (before enrichment/dedup), so we capture
        # all OfferIds including duplicates and hallucinated ones
        sampled[i]["input_offerids"] = set(offerid_to_product.keys())
        if parsed and parsed.get("Products"):
            sampled[i]["slm_raw_offerids"] = [
                str(p.get("OfferId", "")) for p in parsed["Products"]
                if p.get("OfferId")
            ]
        else:
            sampled[i]["slm_raw_offerids"] = []
        # GT raw OfferIds (before enrichment)
        gt_prods_raw = sampled[i].get("gt_products", [])
        sampled[i]["gt_raw_offerids"] = [
            str(p.get("OfferId", "")) for p in gt_prods_raw
            if p.get("OfferId")
        ]

        # Enrich: extract Rank+OfferId from output, join back with input
        sampled[i]["slm_enriched"] = enrich_products(parsed, offerid_to_product)
        # Also enrich GT output for consistent comparison
        sampled[i]["gt_enriched"] = enrich_products(
            sampled[i]["gt_output"], offerid_to_product
        )
        # Store enriched as the slm_output for metrics computation
        if sampled[i]["slm_enriched"]:
            sampled[i]["slm_output"] = {"Products": sampled[i]["slm_enriched"]}
        else:
            sampled[i]["slm_output"] = None
        # Update gt_output with enriched products
        sampled[i]["gt_output"] = {"Products": sampled[i]["gt_enriched"]}
        sampled[i]["gt_products"] = sampled[i]["gt_enriched"]

    parse_ok = sum(1 for s in sampled if s["slm_parsed"] is not None)
    enrich_ok = sum(1 for s in sampled if s["slm_enriched"])
    print(f"  JSON parse success: {parse_ok}/{sample_n} ({parse_ok/sample_n*100:.1f}%)")
    print(f"  Enriched with input info: {enrich_ok}/{sample_n}")

    # Save detail JSONL early so inference results survive later crashes
    detail_path = os.path.join(args.output_dir,
                               args.output_file_name.replace(".json",
                                                             "_detail.jsonl"))
    with open(detail_path, "w", encoding="utf-8") as f:
        for s in sampled:
            row = {
                "UserId": s["UserId"],
                "JourneyIndex": s["JourneyIndex"],
                "n_input_products": s["n_input_products"],
                "n_gt_products": s["n_gt_products"],
                "n_slm_products": len(s["slm_output"].get("Products", []))
                if s["slm_output"] else 0,
                "gt_product_ids": [p.get("OfferId", "")
                                   for p in s["gt_products"]],
                "slm_product_ids": [p.get("OfferId", "")
                                    for p in s["slm_output"].get("Products", [])]
                if s["slm_output"] else [],
                "slm_raw": s.get("slm_raw", ""),
                "input_product_ids": list(s.get("input_offerids", set())),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Detail JSONL saved early: {detail_path}")

    # =========================================================================
    # Step 5.5: Compute diversity metrics (Jaccard + optional embedding)
    # =========================================================================
    print(f"\nStep 5.5: Computing diversity metrics ...")

    # --- Jaccard diversity (always computed) ---
    gt_jaccard_sims = []
    slm_jaccard_sims = []
    for s in sampled:
        gt_prods = s["gt_output"].get("Products", []) if s["gt_output"] else []
        slm_prods = (s["slm_output"].get("Products", [])
                     if s["slm_output"] else [])
        jgt = _mean_pairwise_jaccard(gt_prods)
        jslm = _mean_pairwise_jaccard(slm_prods)
        if jgt is not None:
            gt_jaccard_sims.append(jgt)
        if jslm is not None:
            slm_jaccard_sims.append(jslm)
    print(f"  Jaccard: GT mean={np.mean(gt_jaccard_sims):.4f}, "
          f"SLM mean={np.mean(slm_jaccard_sims):.4f}")

    # --- Embedding diversity (optional) ---
    gt_emb_sims = []
    slm_emb_sims = []
    if not args.skip_embedding_diversity:
        try:
            # Collect all unique OfferIds that appear in GT or SLM outputs
            all_output_oids = set()
            product_info_map = {}  # OfferId -> product dict
            for s in sampled:
                for p in (s["gt_output"].get("Products", [])
                          if s["gt_output"] else []):
                    oid = p.get("OfferId", "")
                    if oid:
                        all_output_oids.add(oid)
                        product_info_map[oid] = p
                for p in (s["slm_output"].get("Products", [])
                          if s["slm_output"] else []):
                    oid = p.get("OfferId", "")
                    if oid:
                        all_output_oids.add(oid)
                        product_info_map[oid] = p

            print(f"  Unique OfferIds in outputs: {len(all_output_oids):,}")

            # Load/generate embeddings with cache
            emb_map = load_or_generate_embeddings(
                list(all_output_oids), product_info_map,
                args.embedding_model, args.output_dir,
            )

            # Compute per-sample mean pairwise cosine
            for s in sampled:
                gt_prods = (s["gt_output"].get("Products", [])
                            if s["gt_output"] else [])
                slm_prods = (s["slm_output"].get("Products", [])
                             if s["slm_output"] else [])

                # GT embeddings
                gt_embs = []
                for p in gt_prods:
                    oid = p.get("OfferId", "")
                    if oid in emb_map:
                        gt_embs.append(emb_map[oid])
                if len(gt_embs) >= 2:
                    gt_emb_sims.append(
                        _mean_pairwise_cosine(np.vstack(gt_embs)))

                # SLM embeddings
                slm_embs = []
                for p in slm_prods:
                    oid = p.get("OfferId", "")
                    if oid in emb_map:
                        slm_embs.append(emb_map[oid])
                if len(slm_embs) >= 2:
                    slm_emb_sims.append(
                        _mean_pairwise_cosine(np.vstack(slm_embs)))

            print(f"  Embedding: GT mean={np.mean(gt_emb_sims):.4f}, "
                  f"SLM mean={np.mean(slm_emb_sims):.4f}")
        except Exception as e:
            print(f"  WARNING: Embedding diversity computation failed: {e}")
            print(f"  Skipping embedding cosine similarity, keeping Jaccard only.")
            gt_emb_sims = []
            slm_emb_sims = []
    else:
        print(f"  Embedding diversity skipped (--skip_embedding_diversity)")

    # =========================================================================
    # Step 6: Compute metrics and print comparison
    # =========================================================================
    print(f"\nStep 6: Computing metrics ...")
    metrics = compute_metrics(sampled)
    metrics["inference_time_s"] = elapsed
    metrics["per_item_time_s"] = elapsed / sample_n

    # Add diversity metrics
    metrics["mean_gt_jaccard_sim"] = float(np.mean(gt_jaccard_sims)) if gt_jaccard_sims else 0.0
    metrics["mean_slm_jaccard_sim"] = float(np.mean(slm_jaccard_sims)) if slm_jaccard_sims else 0.0
    metrics["mean_gt_emb_sim"] = float(np.mean(gt_emb_sims)) if gt_emb_sims else 0.0
    metrics["mean_slm_emb_sim"] = float(np.mean(slm_emb_sims)) if slm_emb_sims else 0.0

    print_comparison(metrics)

    # =========================================================================
    # Step 7: Save results
    # =========================================================================
    output_path = os.path.join(args.output_dir, args.output_file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Metrics saved to: {output_path}")
    print(f"  Detail JSONL at: {detail_path}")

    if elapsed > 0:
        print(f"\n  Inference time: {elapsed:.1f}s "
              f"({sample_n / elapsed:.1f} items/s)")
    print("Done!")


if __name__ == "__main__":
    main()
