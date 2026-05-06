"""Analyze which products trigger safety keyword violations in ranker eval."""

import json
import re
import random
import sys
import numpy as np
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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

TEST_FILE = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260424_JourneyRanker/sft_data/v1_500K_journey_ranker_sft_full.jsonl"
SAMPLE_N = 500
MAX_READ = 50000

# Step 1: Load and sample (same as eval script)
print("Loading data...")
rows = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f):
        if line_num >= MAX_READ:
            break
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        out_raw = d.get("output", "")
        inp = d.get("input", "")
        try:
            gt = json.loads(out_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        gt_prods = gt.get("Products", [])
        if not gt_prods:
            continue

        # Extract journey from input
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

        # Build OfferId -> product from journey
        offerid_to_product = {}
        for q in journey.get("Queries", []):
            for p in q.get("Products", []):
                oid = str(p.get("OfferId", ""))
                if oid:
                    prod = dict(p)
                    prod["OriginalQuery"] = q.get("Query", "")
                    offerid_to_product[oid] = prod

        # Enrich GT products
        enriched_gt = []
        for p in gt_prods:
            oid = str(p.get("OfferId", ""))
            base = dict(offerid_to_product.get(oid, {}))
            base.update(p)
            enriched_gt.append(base)

        rows.append({
            "gt_products": enriched_gt,
            "offerid_to_product": offerid_to_product,
        })

print(f"Loaded {len(rows):,} valid samples")

# Sample same as eval
sampled = random.sample(rows, min(SAMPLE_N, len(rows)))
print(f"Sampled {len(sampled)} rows")

# Step 2: Find safety violations
print("\n" + "=" * 80)
print("SAFETY VIOLATION ANALYSIS (GT Products)")
print("=" * 80)

keyword_counter_old = Counter()
keyword_counter_new = Counter()
violation_examples_new = {}
total_prods = 0
total_violations_old = 0
total_violations_new = 0

# Precompile word-boundary patterns
_kw_patterns = {kw: re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in SAFETY_KEYWORDS}

for s in sampled:
    for p in s["gt_products"]:
        total_prods += 1
        title = p.get("Title", "")
        title_lower = title.lower()

        # OLD: substring match
        old_kws = [kw for kw in SAFETY_KEYWORDS if kw in title_lower]
        if old_kws:
            total_violations_old += 1
            for kw in old_kws:
                keyword_counter_old[kw] += 1

        # NEW: word-boundary match
        new_kws = [kw for kw in SAFETY_KEYWORDS if _kw_patterns[kw].search(title)]
        if new_kws:
            total_violations_new += 1
            for kw in new_kws:
                keyword_counter_new[kw] += 1
                if kw not in violation_examples_new:
                    violation_examples_new[kw] = []
                if len(violation_examples_new[kw]) < 5:
                    violation_examples_new[kw].append({
                        "OfferId": p.get("OfferId", ""),
                        "Title": p.get("Title", ""),
                        "Seller": p.get("Seller", ""),
                    })

print(f"\nTotal GT products: {total_prods}")
print(f"\n--- OLD (substring match) ---")
print(f"Total violations: {total_violations_old} ({total_violations_old/total_prods*100:.2f}%)")
for kw, cnt in keyword_counter_old.most_common():
    print(f"  {kw:<23s} {cnt:>8d}")

print(f"\n--- NEW (word-boundary match) ---")
print(f"Total violations: {total_violations_new} ({total_violations_new/total_prods*100:.2f}%)")
print(f"Reduction: {total_violations_old - total_violations_new} fewer false positives")
print()
print(f"{'Keyword':<25s} {'Count':>8s}")
print("-" * 40)
for kw, cnt in keyword_counter_new.most_common():
    print(f"  {kw:<23s} {cnt:>8d}")

print(f"\n--- Remaining Violations (word-boundary) ---")
for kw, cnt in keyword_counter_new.most_common():
    print(f"\n  [{kw}] ({cnt} matches):")
    for ex in violation_examples_new[kw]:
        title = ex['Title'][:120]
        print(f"    - {title}")
        print(f"      Seller: {ex['Seller']}, OfferId: {ex['OfferId']}")
