"""Analyze seller streak patterns in GT vs SLM outputs.

Checks if products from the same seller appear in consecutive runs
(e.g., seller A x5, then seller B x3, then seller C x4).
"""

import json
import re
import random
import sys
import csv
from collections import Counter
from itertools import groupby

csv.field_size_limit(sys.maxsize)

SEED = 42
random.seed(SEED)

DETAIL_FILE = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260424_JourneyRanker/evaluation_results/ranker_eval_results_v3_full_lr1e-5_150_detail.jsonl"
TEST_FILE = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260424_JourneyRanker/sft_data/v1_500K_journey_ranker_sft_full.jsonl"


def parse_journey_from_input(inp):
    j_start = inp.find('{"JourneyType"')
    if j_start < 0:
        return None
    depth = 0
    for i in range(j_start, len(inp)):
        if inp[i] == "{":
            depth += 1
        elif inp[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(inp[j_start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def build_offerid_map(journey):
    oid_map = {}
    for q in journey.get("Queries", []):
        query_text = q.get("Query", "")
        for p in q.get("Products", []):
            oid = str(p.get("OfferId", ""))
            if oid:
                prod = dict(p)
                prod["OriginalQuery"] = query_text
                oid_map[oid] = prod
    return oid_map


def get_seller_sequence(product_ids, oid_map):
    """Get ordered list of sellers for product IDs."""
    sellers = []
    for oid in product_ids:
        info = oid_map.get(str(oid), {})
        sellers.append(info.get("Seller", "???"))
    return sellers


def get_query_sequence(product_ids, oid_map):
    """Get ordered list of original queries for product IDs."""
    queries = []
    for oid in product_ids:
        info = oid_map.get(str(oid), {})
        queries.append(info.get("OriginalQuery", "???"))
    return queries


def compute_streaks(seq):
    """Compute consecutive runs in a sequence.
    Returns list of (value, length) tuples.
    """
    if not seq:
        return []
    return [(k, sum(1 for _ in g)) for k, g in groupby(seq)]


def streak_stats(streaks):
    """Compute stats from streaks."""
    if not streaks:
        return {"max_streak": 0, "mean_streak": 0, "n_streaks": 0, "n_items": 0}
    lengths = [l for _, l in streaks]
    return {
        "max_streak": max(lengths),
        "mean_streak": sum(lengths) / len(lengths),
        "n_streaks": len(streaks),
        "n_items": sum(lengths),
    }


def main():
    # Step 1: Load all detail rows
    print("Loading detail JSONL ...")
    details = []
    with open(DETAIL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                details.append(json.loads(line))
    print(f"  Loaded {len(details)} detail rows")

    # Build lookup keys
    lookup_keys = set()
    for d in details:
        key = (d["UserId"], str(d["JourneyIndex"]))
        lookup_keys.add(key)

    # Step 2: Scan test file for all matching rows
    print(f"Scanning test file for {len(lookup_keys)} matching rows ...")
    matched = {}
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num >= 50000:
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = d.get("metadata", {})
            uid = meta.get("user_id", "")
            ji = str(meta.get("journey_index", ""))
            key = (uid, ji)
            if key in lookup_keys:
                matched[key] = d
                if len(matched) == len(lookup_keys):
                    break
            if line_num % 10000 == 0:
                print(f"  Scanned {line_num:,} lines, found {len(matched)}/{len(lookup_keys)}")

    print(f"  Found {len(matched)}/{len(lookup_keys)} matches")

    # Step 3: Analyze all cases
    print(f"\nAnalyzing seller & query streaks across all {len(details)} cases ...\n")

    gt_seller_max_streaks = []
    slm_seller_max_streaks = []
    gt_query_max_streaks = []
    slm_query_max_streaks = []

    # Track cases with long streaks for printing examples
    long_streak_cases = []  # (max_streak, case_idx, "gt"/"slm")

    for idx, detail in enumerate(details):
        uid = detail["UserId"]
        ji = str(detail["JourneyIndex"])
        key = (uid, ji)
        test_row = matched.get(key)
        if test_row is None:
            continue

        journey = parse_journey_from_input(test_row.get("input", ""))
        if journey is None:
            continue

        oid_map = build_offerid_map(journey)

        gt_ids = detail.get("gt_product_ids", [])
        slm_ids = detail.get("slm_product_ids", [])

        # Seller streaks
        gt_sellers = get_seller_sequence(gt_ids, oid_map)
        slm_sellers = get_seller_sequence(slm_ids, oid_map)
        gt_s_streaks = compute_streaks(gt_sellers)
        slm_s_streaks = compute_streaks(slm_sellers)
        gt_s_stats = streak_stats(gt_s_streaks)
        slm_s_stats = streak_stats(slm_s_streaks)

        gt_seller_max_streaks.append(gt_s_stats["max_streak"])
        slm_seller_max_streaks.append(slm_s_stats["max_streak"])

        # Query streaks
        gt_queries = get_query_sequence(gt_ids, oid_map)
        slm_queries = get_query_sequence(slm_ids, oid_map)
        gt_q_streaks = compute_streaks(gt_queries)
        slm_q_streaks = compute_streaks(slm_queries)
        gt_q_stats = streak_stats(gt_q_streaks)
        slm_q_stats = streak_stats(slm_q_streaks)

        gt_query_max_streaks.append(gt_q_stats["max_streak"])
        slm_query_max_streaks.append(slm_q_stats["max_streak"])

        # Track long seller streaks
        if slm_s_stats["max_streak"] >= 5:
            long_streak_cases.append((slm_s_stats["max_streak"], idx, "slm",
                                      slm_s_streaks, slm_sellers, slm_ids, oid_map,
                                      gt_s_streaks, gt_sellers, gt_ids, detail))
        if gt_s_stats["max_streak"] >= 5:
            long_streak_cases.append((gt_s_stats["max_streak"], idx, "gt",
                                      gt_s_streaks, gt_sellers, gt_ids, oid_map,
                                      slm_s_streaks, slm_sellers, slm_ids, detail))

    # =========================================================================
    # Print aggregate statistics
    # =========================================================================
    import numpy as np

    print("=" * 90)
    print("SELLER STREAK ANALYSIS (consecutive same-seller products)")
    print("=" * 90)

    for label, max_streaks in [("GT (LLM)", gt_seller_max_streaks),
                                ("SLM (Model)", slm_seller_max_streaks)]:
        arr = np.array(max_streaks)
        print(f"\n  --- {label} ---")
        print(f"  Total cases:      {len(arr)}")
        print(f"  Max streak mean:  {arr.mean():.2f}")
        print(f"  Max streak median:{np.median(arr):.1f}")
        print(f"  Max streak max:   {arr.max()}")
        print(f"  Max streak min:   {arr.min()}")
        # Distribution of max streak lengths
        for threshold in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]:
            n = (arr >= threshold).sum()
            print(f"  Cases with max seller streak >= {threshold:2d}: "
                  f"{n:4d} ({n/len(arr)*100:.1f}%)")

    print(f"\n{'=' * 90}")
    print("QUERY STREAK ANALYSIS (consecutive products from same query)")
    print("=" * 90)

    for label, max_streaks in [("GT (LLM)", gt_query_max_streaks),
                                ("SLM (Model)", slm_query_max_streaks)]:
        arr = np.array(max_streaks)
        print(f"\n  --- {label} ---")
        print(f"  Max streak mean:  {arr.mean():.2f}")
        print(f"  Max streak median:{np.median(arr):.1f}")
        print(f"  Max streak max:   {arr.max()}")
        for threshold in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
            n = (arr >= threshold).sum()
            print(f"  Cases with max query streak >= {threshold:2d}: "
                  f"{n:4d} ({n/len(arr)*100:.1f}%)")

    # =========================================================================
    # Print example cases with long streaks
    # =========================================================================
    long_streak_cases.sort(key=lambda x: -x[0])

    print(f"\n{'=' * 90}")
    print(f"TOP EXAMPLES: Cases with longest seller streaks")
    print(f"{'=' * 90}")

    # Show top 15 SLM long streaks and top 15 GT long streaks
    shown_slm = 0
    shown_gt = 0
    for (max_s, case_idx, source, streaks, sellers, ids, oid_map,
         other_streaks, other_sellers, other_ids, detail) in long_streak_cases:

        if source == "slm" and shown_slm >= 15:
            continue
        if source == "gt" and shown_gt >= 15:
            continue

        if source == "slm":
            shown_slm += 1
        else:
            shown_gt += 1

        label = "SLM" if source == "slm" else "GT"
        other_label = "GT" if source == "slm" else "SLM"

        print(f"\n  --- Case #{case_idx} [{label} max seller streak = {max_s}] ---")
        print(f"  UserId: {detail['UserId']}")

        # Print the streak pattern
        print(f"  {label} seller streak pattern:")
        for seller, length in streaks:
            bar = "█" * length
            print(f"    {bar} {seller} x{length}")

        # Print the full product sequence with seller
        print(f"  {label} product sequence ({len(ids)} products):")
        for rank, oid in enumerate(ids, 1):
            info = oid_map.get(str(oid), {})
            seller = info.get("Seller", "???")
            title = info.get("Title", "???")[:60]
            query = info.get("OriginalQuery", "???")[:40]
            print(f"    Rank {rank:2d}: [{seller:30s}] {title}  (q: {query})")

        # Print the other side's streak pattern for comparison
        print(f"  {other_label} seller streak pattern:")
        for seller, length in other_streaks:
            bar = "█" * length
            print(f"    {bar} {seller} x{length}")

        if shown_slm >= 15 and shown_gt >= 15:
            break

    print("\nDone!")


if __name__ == "__main__":
    main()
