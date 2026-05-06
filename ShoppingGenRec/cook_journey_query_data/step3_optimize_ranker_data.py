"""Step 3: Optimize Ranker SFT Training Data

Reads the original ranker SFT JSONL and produces an optimized version with:
  1. Compact output: only [{"Rank":1,"OfferId":"xxx"},...]
  2. Shorter instruction (same rules, fewer tokens)
  3. Data cleaning: remove hallucinated/duplicate OfferIds, re-rank
  4. (Optional) Diversity filtering: remove near-duplicate products via
     Jaccard word similarity + cosine embedding similarity (OR logic)

Usage:
    # v2: compact + clean only
    python step3_optimize_ranker_data.py --no_diversity

    # v3: compact + clean + diversity (default)
    python step3_optimize_ranker_data.py

    # v3 without embedding (Jaccard only, fast)
    python step3_optimize_ranker_data.py --skip_embedding
"""

import os
import json
import re
import sys
import argparse
import random
from tqdm import tqdm

import numpy as np

SEED = 42
random.seed(SEED)

# =============================================================================
# Compact Instruction (shorter, same rules)
# =============================================================================

COMPACT_INSTRUCTION = (
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


# =============================================================================
# Diversity Helpers
# =============================================================================

def _product_text(p):
    """Build text for Jaccard/embedding: Title | Brand | Seller | Price."""
    parts = []
    for field in ("Title", "Brand", "Seller", "Price"):
        val = p.get(field, "")
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


def _extract_brands_from_input(inp):
    """Extract OfferId -> Brand mapping from input text by parsing journey JSON."""
    brands = {}
    j_start = inp.find('{"JourneyType"')
    if j_start < 0:
        return brands
    depth = 0
    for i in range(j_start, len(inp)):
        if inp[i] == '{':
            depth += 1
        elif inp[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    journey = json.loads(inp[j_start:i + 1])
                    for q in journey.get("Queries", []):
                        for p in q.get("Products", []):
                            oid = str(p.get("OfferId", ""))
                            if oid:
                                brands[oid] = p.get("Brand", "")
                except (json.JSONDecodeError, TypeError):
                    pass
                break
    return brands


def load_or_generate_embeddings(offer_ids, product_info, embedding_model_path,
                                cache_dir, batch_size=1024, max_length=512):
    """Load cached embeddings, generate missing ones, update cache.

    Compatible with s5_5_ranker_eval.py cache format (npz with offer_ids +
    embeddings arrays).

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
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer

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
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Optimize ranker SFT data: compact output + diversity"
    )
    p.add_argument(
        "--input_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260424_JourneyRanker/sft_data/"
                "v1_500K_journey_ranker_sft_full.jsonl",
    )
    p.add_argument(
        "--output_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260424_JourneyRanker/sft_data/"
                "v3_500K_journey_ranker_sft_diverse.jsonl",
    )
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--stats_only", action="store_true", default=False)
    p.add_argument("--stats_sample", type=int, default=10000)

    # Diversity
    p.add_argument("--no_diversity", action="store_true", default=False,
                   help="Disable diversity filtering (v2 mode)")
    p.add_argument("--jaccard_threshold", type=float, default=0.7,
                   help="Jaccard similarity threshold for diversity (default: 0.7)")
    p.add_argument("--cosine_threshold", type=float, default=0.98,
                   help="Cosine similarity threshold for diversity (default: 0.98)")
    p.add_argument("--skip_embedding", action="store_true", default=False,
                   help="Skip cosine embedding similarity (Jaccard only)")
    p.add_argument(
        "--embedding_model", type=str,
        default="/scratch/workspaceblobstore/users/xiaoyukou/ckpts/"
                "Qwen3-Embedding-0.6B",
    )
    p.add_argument(
        "--embedding_cache_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260424_JourneyRanker/sft_data/",
    )
    return p.parse_args()


# =============================================================================
# Core: Build compact output with cleaning + diversity
# =============================================================================

def build_compact_output(original_output_str, input_offer_ids,
                         input_brands, product_info_map,
                         embedding_map, jaccard_thresh, cosine_thresh,
                         use_diversity, use_cosine):
    """Convert full output to compact format with cleaning + diversity.

    Returns: (compact_json_str, stats_dict) or (None, stats_dict).
    """
    stats = {
        "n_hallucinated": 0, "n_duplicated": 0,
        "n_before_diversity": 0, "n_removed_jaccard": 0,
        "n_removed_cosine": 0, "n_after_diversity": 0,
        "removed_products": [],  # list of (product_text, reason, sim_value, kept_text)
    }

    try:
        obj = json.loads(original_output_str)
    except (json.JSONDecodeError, TypeError):
        return None, stats

    prods = obj.get("Products", [])
    if not prods:
        return None, stats

    # Step 1: Clean hallucinated + duplicate OfferIds
    seen_ids = set()
    cleaned = []
    for p in prods:
        oid = str(p.get("OfferId", ""))
        if oid not in input_offer_ids:
            stats["n_hallucinated"] += 1
            continue
        if oid in seen_ids:
            stats["n_duplicated"] += 1
            continue
        seen_ids.add(oid)
        # Build product info with Brand from input
        pinfo = {
            "OfferId": oid,
            "Title": p.get("Title", ""),
            "Seller": p.get("Seller", ""),
            "Price": p.get("Price", ""),
            "Brand": input_brands.get(oid, ""),
        }
        cleaned.append(pinfo)

    if not cleaned:
        return None, stats

    stats["n_before_diversity"] = len(cleaned)

    # Step 2: Diversity filtering (greedy, rank-order)
    if use_diversity and len(cleaned) > 1:
        # Pre-compute text representations and token sets
        texts = [_product_text(p) for p in cleaned]
        token_sets = [set(t.lower().split()) for t in texts]

        # Pre-load embeddings if available
        emb_list = None
        if use_cosine and embedding_map:
            emb_list = []
            for p in cleaned:
                oid = p["OfferId"]
                emb_list.append(embedding_map.get(oid))

        kept_indices = [0]  # always keep rank-1
        for i in range(1, len(cleaned)):
            is_diverse = True
            for j in kept_indices:
                # Jaccard check
                if token_sets[i] and token_sets[j]:
                    inter = token_sets[i] & token_sets[j]
                    union = token_sets[i] | token_sets[j]
                    jac = len(inter) / len(union) if union else 0.0
                    if jac >= jaccard_thresh:
                        stats["n_removed_jaccard"] += 1
                        stats["removed_products"].append((
                            texts[i], "jaccard", jac, texts[j]))
                        is_diverse = False
                        break

                # Cosine check
                if use_cosine and emb_list and emb_list[i] is not None and emb_list[j] is not None:
                    cos = float(np.dot(emb_list[i], emb_list[j]))
                    if cos >= cosine_thresh:
                        stats["n_removed_cosine"] += 1
                        stats["removed_products"].append((
                            texts[i], "cosine", cos, texts[j]))
                        is_diverse = False
                        break

            if is_diverse:
                kept_indices.append(i)

        cleaned = [cleaned[i] for i in kept_indices]

    stats["n_after_diversity"] = len(cleaned)

    if not cleaned:
        return None, stats

    # Step 3: Re-rank sequentially
    compact_prods = []
    for rank, p in enumerate(cleaned, start=1):
        compact_prods.append({"Rank": rank, "OfferId": p["OfferId"]})

    compact = {"Products": compact_prods}
    return json.dumps(compact, ensure_ascii=False), stats


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    use_diversity = not args.no_diversity
    use_cosine = use_diversity and not args.skip_embedding

    # Stats-only mode
    if args.stats_only:
        if not os.path.isfile(args.output_file):
            print(f"ERROR: File not found: {args.output_file}")
            sys.exit(1)
        compute_token_distribution(args.output_file, args.stats_sample)
        return

    print("=" * 70)
    print("Step 3: Optimize Ranker SFT Data"
          + (" + Diversity" if use_diversity else ""))
    print("=" * 70)
    print(f"  Input:  {args.input_file}")
    print(f"  Output: {args.output_file}")
    if use_diversity:
        print(f"  Diversity:  ON  (Jaccard >= {args.jaccard_threshold}"
              + (f" OR Cosine >= {args.cosine_threshold})" if use_cosine
                 else ", no embedding)"))
    else:
        print(f"  Diversity:  OFF")
    print()

    # Count input lines
    print("Counting input lines ...")
    total_lines = 0
    with open(args.input_file, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1
    print(f"  Total lines: {total_lines:,}")
    max_n = args.max_samples or total_lines

    # =================================================================
    # PASS 1: Collect product info for diversity (if enabled)
    # =================================================================
    product_info_map = {}  # OfferId -> {Title, Brand, Seller, Price}
    embedding_map = {}

    if use_diversity:
        print(f"\nPass 1: Collecting product info from {min(total_lines, max_n):,} samples ...")
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(
                tqdm(f, total=min(total_lines, max_n),
                     desc="Scanning", mininterval=30)
            ):
                if line_num >= max_n:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                inp = d.get("input", "")
                output_str = d.get("output", "")

                # Extract Brand from input
                input_brands = _extract_brands_from_input(inp)

                # Collect output product info
                try:
                    out = json.loads(output_str)
                    for p in out.get("Products", []):
                        oid = str(p.get("OfferId", ""))
                        if oid and oid not in product_info_map:
                            product_info_map[oid] = {
                                "Title": p.get("Title", ""),
                                "Seller": p.get("Seller", ""),
                                "Price": p.get("Price", ""),
                                "Brand": input_brands.get(oid, ""),
                            }
                except (json.JSONDecodeError, TypeError):
                    pass

        print(f"  Unique products in outputs: {len(product_info_map):,}")

        # Generate embeddings
        if use_cosine:
            print(f"\n  Generating embeddings ...")
            embedding_map = load_or_generate_embeddings(
                list(product_info_map.keys()),
                product_info_map,
                args.embedding_model,
                args.embedding_cache_dir,
            )

    # =================================================================
    # PASS 2: Process with compact + clean + diversity
    # =================================================================
    print(f"\nPass 2: Processing ...")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    converted = 0
    skipped = 0
    total_hall = 0
    total_dup = 0
    samples_with_hall = 0
    samples_with_dup = 0
    total_removed_jaccard = 0
    total_removed_cosine = 0
    total_before_div = 0
    total_after_div = 0
    samples_with_diversity_removal = 0
    product_counts_before = []
    product_counts_after = []
    diversity_examples = []  # collect examples for display

    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(
            tqdm(fin, total=min(total_lines, max_n),
                 desc="Converting", mininterval=30)
        ):
            if line_num >= max_n:
                break

            line = line.strip()
            if not line:
                continue

            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            original_output = d.get("output", "")
            inp = d.get("input", "")
            meta = d.get("metadata", {})

            # Extract OfferIds from input
            input_offer_ids = set(re.findall(r'"OfferId":\s*"(\d+)"', inp))

            # Extract Brands from input
            input_brands = _extract_brands_from_input(inp) if use_diversity else {}

            # Build compact output with cleaning + diversity
            compact_output, stats = build_compact_output(
                original_output, input_offer_ids, input_brands,
                product_info_map, embedding_map,
                args.jaccard_threshold, args.cosine_threshold,
                use_diversity, use_cosine,
            )
            if compact_output is None:
                skipped += 1
                continue

            # Accumulate stats
            total_hall += stats["n_hallucinated"]
            total_dup += stats["n_duplicated"]
            if stats["n_hallucinated"] > 0:
                samples_with_hall += 1
            if stats["n_duplicated"] > 0:
                samples_with_dup += 1
            total_removed_jaccard += stats["n_removed_jaccard"]
            total_removed_cosine += stats["n_removed_cosine"]
            total_before_div += stats["n_before_diversity"]
            total_after_div += stats["n_after_diversity"]
            product_counts_before.append(stats["n_before_diversity"])
            if stats["n_before_diversity"] > stats["n_after_diversity"]:
                samples_with_diversity_removal += 1
                # Collect examples (up to 10)
                if len(diversity_examples) < 10 and stats["removed_products"]:
                    diversity_examples.append({
                        "user_id": meta.get("user_id", "")[:12],
                        "before": stats["n_before_diversity"],
                        "after": stats["n_after_diversity"],
                        "removed": stats["removed_products"][:3],
                    })

            # Update metadata
            out_obj = json.loads(compact_output)
            n_out = len(out_obj.get("Products", []))
            meta["n_output_products"] = n_out
            product_counts_after.append(n_out)

            new_record = {
                "instruction": COMPACT_INSTRUCTION,
                "input": inp,
                "output": compact_output,
                "metadata": meta,
            }
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            converted += 1

    # =================================================================
    # Print Stats
    # =================================================================
    print(f"\n  Converted: {converted:,}")
    print(f"  Skipped:   {skipped:,}")

    total_ids = total_hall + total_dup + total_before_div
    print(f"\n  Data Cleaning Stats:")
    print(f"    Hallucinated OfferIds removed: {total_hall:,} "
          f"({total_hall / max(total_ids, 1) * 100:.2f}% of output OfferIds, "
          f"affecting {samples_with_hall:,}/{converted:,} samples "
          f"= {samples_with_hall / max(converted, 1) * 100:.1f}%)")
    print(f"    Duplicate OfferIds removed:    {total_dup:,} "
          f"({total_dup / max(total_ids, 1) * 100:.2f}% of output OfferIds, "
          f"affecting {samples_with_dup:,}/{converted:,} samples "
          f"= {samples_with_dup / max(converted, 1) * 100:.1f}%)")

    if use_diversity:
        n_removed_div = total_before_div - total_after_div
        print(f"\n  Diversity Filtering Stats:")
        print(f"    Products before diversity: {total_before_div:,}")
        print(f"    Products after diversity:  {total_after_div:,}")
        print(f"    Removed by diversity:      {n_removed_div:,} "
              f"({n_removed_div / max(total_before_div, 1) * 100:.1f}%)")
        print(f"      - by Jaccard >= {args.jaccard_threshold}: {total_removed_jaccard:,}")
        if use_cosine:
            print(f"      - by Cosine >= {args.cosine_threshold}:  {total_removed_cosine:,}")
        print(f"    Journeys affected: {samples_with_diversity_removal:,}/{converted:,} "
              f"({samples_with_diversity_removal / max(converted, 1) * 100:.1f}%)")

        # Before vs After product count distribution comparison
        pb = np.array(product_counts_before)
        pa_div = np.array(product_counts_after)
        print(f"\n  Product Count: Before vs After Diversity ({converted:,} samples):")
        print(f"    {'':>8s} {'Before':>10s} {'After':>10s} {'Delta':>10s}")
        print(f"    {'Min':>8s} {pb.min():>10d} {pa_div.min():>10d} {pa_div.min()-pb.min():>+10d}")
        print(f"    {'P25':>8s} {int(np.percentile(pb,25)):>10d} {int(np.percentile(pa_div,25)):>10d} "
              f"{int(np.percentile(pa_div,25))-int(np.percentile(pb,25)):>+10d}")
        print(f"    {'P50':>8s} {int(np.percentile(pb,50)):>10d} {int(np.percentile(pa_div,50)):>10d} "
              f"{int(np.percentile(pa_div,50))-int(np.percentile(pb,50)):>+10d}")
        print(f"    {'Mean':>8s} {pb.mean():>10.1f} {pa_div.mean():>10.1f} {pa_div.mean()-pb.mean():>+10.1f}")
        print(f"    {'P75':>8s} {int(np.percentile(pb,75)):>10d} {int(np.percentile(pa_div,75)):>10d} "
              f"{int(np.percentile(pa_div,75))-int(np.percentile(pb,75)):>+10d}")
        print(f"    {'P90':>8s} {int(np.percentile(pb,90)):>10d} {int(np.percentile(pa_div,90)):>10d} "
              f"{int(np.percentile(pa_div,90))-int(np.percentile(pb,90)):>+10d}")
        print(f"    {'Max':>8s} {pb.max():>10d} {pa_div.max():>10d} {pa_div.max()-pb.max():>+10d}")

        # Diversity examples
        if diversity_examples:
            print(f"\n  Diversity Filtering Examples (up to 10):")
            for idx, ex in enumerate(diversity_examples):
                print(f"\n    Example {idx+1} (user={ex['user_id']}): "
                      f"{ex['before']} -> {ex['after']} products "
                      f"(-{ex['before'] - ex['after']})")
                for removed_text, reason, sim_val, kept_text in ex["removed"]:
                    print(f"      REMOVED ({reason}={sim_val:.3f}):")
                    print(f"        - {removed_text[:90]}")
                    print(f"        ~ {kept_text[:90]}")

    # Product count distribution
    pa = np.array(product_counts_after)
    print(f"\n  Output Product Count Distribution ({converted:,} samples):")
    print(f"    Min={pa.min()}, P10={int(np.percentile(pa,10))}, "
          f"P25={int(np.percentile(pa,25))}, P50={int(np.percentile(pa,50))}, "
          f"Mean={pa.mean():.1f}, P75={int(np.percentile(pa,75))}, "
          f"P90={int(np.percentile(pa,90))}, Max={pa.max()}")
    for n in [3, 5, 8, 10, 12, 15, 20, 25]:
        cnt = int((pa <= n).sum())
        print(f"    <= {n}: {cnt:,} ({cnt/len(pa)*100:.1f}%)")

    # =================================================================
    # Token comparison (500 pairs)
    # =================================================================
    print(f"\n{'=' * 70}")
    print("Token Length Comparison (sampling 500 pairs)")
    print(f"{'=' * 70}")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3.5-9B",
            trust_remote_code=True,
        )
    except Exception:
        print("  [SKIP] Could not load tokenizer for comparison")
        compute_token_distribution(args.output_file, args.stats_sample)
        print("Done!")
        return

    old_samples = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            old_samples.append(json.loads(line))

    new_samples = []
    with open(args.output_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            new_samples.append(json.loads(line))

    n_compare = min(len(old_samples), len(new_samples))
    old_instr, old_inp, old_out, old_total = [], [], [], []
    new_instr, new_inp, new_out, new_total = [], [], [], []

    for i in range(n_compare):
        o = old_samples[i]
        n = new_samples[i]
        oi = len(tok.encode(o.get("instruction", "")))
        oip = len(tok.encode(o.get("input", "")))
        oo = len(tok.encode(o.get("output", "")))
        old_instr.append(oi); old_inp.append(oip); old_out.append(oo)
        old_total.append(oi + oip + oo)
        ni = len(tok.encode(n.get("instruction", "")))
        nip = len(tok.encode(n.get("input", "")))
        no_ = len(tok.encode(n.get("output", "")))
        new_instr.append(ni); new_inp.append(nip); new_out.append(no_)
        new_total.append(ni + nip + no_)

    W1, W2, W3, W4 = 20, 18, 18, 18
    def _row(label, old_arr, new_arr):
        om = np.mean(old_arr); nm = np.mean(new_arr)
        diff = nm - om
        sign = "+" if diff >= 0 else ""
        pct = diff / om * 100 if om > 0 else 0
        print(f"  {label:<{W1}s} {om:>{W2}.0f} {nm:>{W3}.0f} "
              f"{f'{sign}{diff:.0f} ({sign}{pct:.1f}%)':>{W4}s}")

    print(f"\n  {'Field':<{W1}s} {'Original (mean)':>{W2}s} "
          f"{'Compact (mean)':>{W3}s} {'Delta':>{W4}s}")
    print(f"  {'-' * W1} {'-' * W2} {'-' * W3} {'-' * W4}")
    _row("Instruction", old_instr, new_instr)
    _row("Input", old_inp, new_inp)
    _row("Output", old_out, new_out)
    _row("Total", old_total, new_total)

    savings = np.mean(old_total) - np.mean(new_total)
    print(f"\n  Avg savings: {savings:.0f} tokens/sample "
          f"({savings / np.mean(old_total) * 100:.1f}%)")
    print(f"  Output reduction: "
          f"{(np.mean(old_out) - np.mean(new_out)) / np.mean(old_out) * 100:.1f}%")

    file_size = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"\n  Output file: {args.output_file}")
    print(f"  File size: {file_size:.1f} MB")

    compute_token_distribution(args.output_file, args.stats_sample)
    print("Done!")


# =============================================================================
# Token Distribution
# =============================================================================

def compute_token_distribution(filepath, max_samples=5000):
    """Compute and print token length distribution of a JSONL file."""
    print(f"\n{'=' * 70}")
    print(f"Token Length Distribution: {os.path.basename(filepath)}")
    print(f"{'=' * 70}")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3.5-9B",
            trust_remote_code=True,
        )
    except Exception:
        print("  [SKIP] Could not load tokenizer")
        return

    instr_lens, input_lens, output_lens, total_lens, n_prods = [], [], [], [], []

    print(f"  Reading samples (max={max_samples if max_samples else 'all'}) ...")
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            il = len(tok.encode(d.get("instruction", "")))
            ipl = len(tok.encode(d.get("input", "")))
            ol = len(tok.encode(d.get("output", "")))
            instr_lens.append(il); input_lens.append(ipl); output_lens.append(ol)
            total_lens.append(il + ipl + ol)

            try:
                out_obj = json.loads(d.get("output", "{}"))
                n_prods.append(len(out_obj.get("Products", [])))
            except (json.JSONDecodeError, TypeError):
                pass

            count += 1
            if count % 1000 == 0:
                print(f"    {count:,} samples processed ...", flush=True)

    print(f"  Total samples analyzed: {count:,}")

    W = 20
    def _dist(name, arr):
        a = np.array(arr)
        print(f"  {name:<{W}s}  Min={a.min():>6}  P25={int(np.percentile(a,25)):>6}  "
              f"P50={int(np.percentile(a,50)):>6}  Mean={a.mean():>7.0f}  "
              f"P75={int(np.percentile(a,75)):>6}  P90={int(np.percentile(a,90)):>6}  "
              f"P95={int(np.percentile(a,95)):>6}  Max={a.max():>6}")

    print()
    _dist("Instruction", instr_lens)
    _dist("Input", input_lens)
    _dist("Output", output_lens)
    _dist("Total", total_lens)
    if n_prods:
        _dist("# Products", n_prods)

    ta = np.array(total_lens)
    buckets = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 20000]
    print(f"\n  Total token length bucket distribution:")
    prev = 0
    for b in buckets:
        cnt = int(((ta >= prev) & (ta < b)).sum())
        pct = cnt / len(ta) * 100
        bar = '#' * int(pct / 2)
        print(f"    {prev:>6}-{b:<6}: {cnt:>7,} ({pct:5.1f}%) {bar}")
        prev = b
    cnt = int((ta >= prev).sum())
    if cnt > 0:
        pct = cnt / len(ta) * 100
        print(f"    {prev:>6}+      : {cnt:>7,} ({pct:5.1f}%)")

    for cutoff in [8000, 10000, 12000, 16000]:
        over = int((ta > cutoff).sum())
        print(f"  Samples > {cutoff}: {over:,} ({over/len(ta)*100:.2f}%)")


if __name__ == "__main__":
    main()
