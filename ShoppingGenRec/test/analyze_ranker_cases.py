"""Analyze 10 random cases from ranker eval detail JSONL.

Joins detail results with the SFT test file to get full product info,
then prints GT vs SLM side-by-side comparison.
"""

import json
import re
import random
import sys
import csv

csv.field_size_limit(sys.maxsize)

SEED = 42
random.seed(SEED)

DETAIL_FILE = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260424_JourneyRanker/evaluation_results/ranker_eval_results_v3_full_lr1e-5_150_detail.jsonl"
TEST_FILE = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260424_JourneyRanker/sft_data/v1_500K_journey_ranker_sft_full.jsonl"

N_SAMPLES = 10


def parse_journey_from_input(inp):
    """Extract journey JSON from input text."""
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


def parse_profile_from_input(inp):
    """Extract user profile JSON from input text."""
    prof_match = re.search(r'\{"userShoppingProfile":\{.*?\}\}', inp)
    if prof_match:
        try:
            return json.loads(prof_match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def build_offerid_map(journey):
    """Build OfferId -> product dict from journey queries."""
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


def parse_slm_output(raw):
    """Parse SLM raw output into a dict."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</?OUTPUT>", "", text).strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
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


def main():
    # Step 1: Load detail file and pick 10 random rows
    print("=" * 80)
    print("Loading detail JSONL ...")
    details = []
    with open(DETAIL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                details.append(json.loads(line))
    print(f"  Loaded {len(details)} detail rows")

    sampled_indices = random.sample(range(len(details)), min(N_SAMPLES, len(details)))
    sampled = [details[i] for i in sampled_indices]
    print(f"  Randomly selected {len(sampled)} rows (indices: {sampled_indices})")

    # Build lookup keys for matching with test file
    # The eval script uses: seed=42, sample_n=500, max_read=50000
    # It loads up to max_read lines from the JSONL, then samples 500
    # We need to find matching rows by user_id + journey_index
    lookup_keys = set()
    for d in sampled:
        key = (d["UserId"], str(d["JourneyIndex"]))
        lookup_keys.add(key)
    print(f"  Need to find {len(lookup_keys)} unique (UserId, JourneyIndex) pairs")

    # Step 2: Scan test file to find matching rows
    print(f"\nScanning test file for matching rows (up to 50000 lines) ...")
    matched = {}  # (user_id, journey_index) -> test row
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
                print(f"  Scanned {line_num:,} lines, found {len(matched)}/{len(lookup_keys)} matches")

    print(f"  Found {len(matched)}/{len(lookup_keys)} matches")

    # Step 3: For each sampled detail row, join and print
    print("\n" + "=" * 100)
    print("CASE-BY-CASE ANALYSIS: GT vs SLM")
    print("=" * 100)

    for case_idx, detail in enumerate(sampled):
        uid = detail["UserId"]
        ji = str(detail["JourneyIndex"])
        key = (uid, ji)

        print(f"\n{'#' * 100}")
        print(f"CASE {case_idx + 1} / {len(sampled)}")
        print(f"  UserId: {uid}")
        print(f"  JourneyIndex: {ji}")
        print(f"  Input products: {detail['n_input_products']}")
        print(f"  GT products: {detail['n_gt_products']}")
        print(f"  SLM products: {detail['n_slm_products']}")
        print(f"{'#' * 100}")

        # Get test row
        test_row = matched.get(key)
        if test_row is None:
            print("  [WARNING] No matching test row found! Skipping.")
            continue

        # Parse journey from input
        inp = test_row.get("input", "")
        journey = parse_journey_from_input(inp)
        if journey is None:
            print("  [WARNING] Could not parse journey from input! Skipping.")
            continue

        # Parse profile
        profile = parse_profile_from_input(inp)
        sp = profile.get("userShoppingProfile", {})

        # Print user profile summary
        print(f"\n  --- User Profile ---")
        gender_pref = sp.get("shoppingGenderPreference", "N/A")
        brand_prefs = sp.get("brandPreferences", [])
        retailer_prefs = sp.get("retailerPreferences", [])
        price_sens = sp.get("priceSensitivity", "N/A")
        fashion = sp.get("fashionStyle", "N/A")
        print(f"  Gender pref: {gender_pref}")
        print(f"  Brand prefs: {brand_prefs}")
        print(f"  Retailer prefs: {retailer_prefs}")
        print(f"  Price sensitivity: {price_sens}")
        print(f"  Fashion style: {fashion}")

        # Print journey title/description
        print(f"\n  --- Journey ---")
        print(f"  Type: {journey.get('JourneyType', 'N/A')}")
        print(f"  Title: {journey.get('Title', 'N/A')}")
        print(f"  Description: {journey.get('Description', 'N/A')[:200]}")

        # Build OfferId -> product info map
        oid_map = build_offerid_map(journey)

        # Print all input products grouped by query
        print(f"\n  --- Input Products (total: {detail['n_input_products']}) ---")
        for qi, q in enumerate(journey.get("Queries", [])):
            prods = q.get("Products", [])
            print(f"    Query {qi}: \"{q.get('Query', '')}\" ({len(prods)} products)")
            for p in prods:
                oid = p.get("OfferId", "")
                title = p.get("Title", "N/A")[:80]
                seller = p.get("Seller", "N/A")
                brand = p.get("Brand", "N/A")
                price = p.get("Price", "N/A")
                gender = p.get("Gender", "")
                gender_str = f" [Gender:{gender}]" if gender else ""
                print(f"      {oid}: {title}")
                print(f"        Seller={seller} | Brand={brand} | Price={price}{gender_str}")

        # Print GT products with enriched info
        gt_ids = detail.get("gt_product_ids", [])
        print(f"\n  --- GT Products (LLM ground truth, {len(gt_ids)} products) ---")
        gt_sellers = []
        gt_brands = []
        for rank, oid in enumerate(gt_ids, 1):
            info = oid_map.get(str(oid), {})
            title = info.get("Title", "???")[:80]
            seller = info.get("Seller", "???")
            brand = info.get("Brand", "???")
            price = info.get("Price", "???")
            query = info.get("OriginalQuery", "???")[:60]
            gt_sellers.append(seller)
            gt_brands.append(brand)
            print(f"    Rank {rank:2d}: {oid}")
            print(f"      Title:  {title}")
            print(f"      Seller: {seller} | Brand: {brand} | Price: {price}")
            print(f"      Query:  {query}")

        # Print SLM products with enriched info
        slm_ids = detail.get("slm_product_ids", [])
        print(f"\n  --- SLM Products (model output, {len(slm_ids)} products) ---")
        slm_sellers = []
        slm_brands = []
        for rank, oid in enumerate(slm_ids, 1):
            info = oid_map.get(str(oid), {})
            title = info.get("Title", "??? [HALLUCINATED]")[:80]
            seller = info.get("Seller", "??? [HALLUCINATED]")
            brand = info.get("Brand", "??? [HALLUCINATED]")
            price = info.get("Price", "???")
            query = info.get("OriginalQuery", "???")[:60]
            in_gt = "✓" if oid in set(gt_ids) else "✗"
            in_input = "✓" if str(oid) in oid_map else "✗"
            slm_sellers.append(seller)
            slm_brands.append(brand)
            print(f"    Rank {rank:2d}: {oid}  [in_GT:{in_gt}] [in_input:{in_input}]")
            print(f"      Title:  {title}")
            print(f"      Seller: {seller} | Brand: {brand} | Price: {price}")
            print(f"      Query:  {query}")

        # Quick summary comparison for this case
        gt_set = set(gt_ids)
        slm_set = set(slm_ids)
        overlap = gt_set & slm_set
        gt_unique_sellers = set(s for s in gt_sellers if s != "???")
        slm_unique_sellers = set(s for s in slm_sellers if s != "???")
        gt_unique_brands = set(b for b in gt_brands if b != "???")
        slm_unique_brands = set(b for b in slm_brands if b != "???")

        print(f"\n  --- Case Summary ---")
        print(f"    Overlap:        {len(overlap)} / GT:{len(gt_ids)} / SLM:{len(slm_ids)}")
        print(f"    Precision:      {len(overlap)/max(len(slm_ids),1)*100:.1f}%")
        print(f"    Recall:         {len(overlap)/max(len(gt_ids),1)*100:.1f}%")
        print(f"    GT sellers:     {len(gt_unique_sellers)} unique: {sorted(gt_unique_sellers)}")
        print(f"    SLM sellers:    {len(slm_unique_sellers)} unique: {sorted(slm_unique_sellers)}")
        print(f"    GT brands:      {len(gt_unique_brands)} unique: {sorted(gt_unique_brands)}")
        print(f"    SLM brands:     {len(slm_unique_brands)} unique: {sorted(slm_unique_brands)}")

        # Check if SLM has all same seller
        if len(slm_unique_sellers) == 1 and slm_ids:
            print(f"    ⚠️  SLM ALL SAME SELLER: {list(slm_unique_sellers)[0]}")
        if len(slm_unique_brands) == 1 and slm_ids:
            print(f"    ⚠️  SLM ALL SAME BRAND: {list(slm_unique_brands)[0]}")

        # Hallucinated IDs
        input_ids = set(str(x) for x in detail.get("input_product_ids", []))
        slm_hallucinated = [oid for oid in slm_ids if str(oid) not in input_ids]
        if slm_hallucinated:
            print(f"    ⚠️  SLM hallucinated OfferIds: {slm_hallucinated}")

        # Duplicate check
        if len(slm_ids) != len(set(slm_ids)):
            from collections import Counter
            dups = [oid for oid, cnt in Counter(slm_ids).items() if cnt > 1]
            print(f"    ⚠️  SLM duplicate OfferIds: {dups}")

    # =========================================================================
    # Aggregate stats across all 10 cases
    # =========================================================================
    print("\n" + "=" * 100)
    print("AGGREGATE STATS ACROSS ALL SAMPLED CASES")
    print("=" * 100)

    all_same_seller = 0
    all_same_brand = 0
    total_hallucinated = 0
    total_duplicated = 0
    precisions = []
    recalls = []

    for detail in sampled:
        gt_ids = detail.get("gt_product_ids", [])
        slm_ids = detail.get("slm_product_ids", [])
        input_ids = set(str(x) for x in detail.get("input_product_ids", []))

        key = (detail["UserId"], str(detail["JourneyIndex"]))
        test_row = matched.get(key)
        if test_row is None:
            continue

        journey = parse_journey_from_input(test_row.get("input", ""))
        if journey is None:
            continue

        oid_map = build_offerid_map(journey)

        # Sellers/brands
        slm_sellers = set()
        slm_brands = set()
        for oid in slm_ids:
            info = oid_map.get(str(oid), {})
            s = info.get("Seller", "")
            b = info.get("Brand", "")
            if s:
                slm_sellers.add(s)
            if b:
                slm_brands.add(b)

        if len(slm_sellers) == 1 and slm_ids:
            all_same_seller += 1
        if len(slm_brands) == 1 and slm_ids:
            all_same_brand += 1

        # Hallucination
        hall = sum(1 for oid in slm_ids if str(oid) not in input_ids)
        total_hallucinated += hall

        # Duplicates
        total_duplicated += len(slm_ids) - len(set(slm_ids))

        # Overlap
        gt_set = set(gt_ids)
        slm_set = set(slm_ids)
        overlap = gt_set & slm_set
        if slm_ids:
            precisions.append(len(overlap) / len(slm_set))
        if gt_ids:
            recalls.append(len(overlap) / len(gt_set))

    print(f"  Cases with ALL same seller: {all_same_seller}/{len(sampled)}")
    print(f"  Cases with ALL same brand:  {all_same_brand}/{len(sampled)}")
    print(f"  Total hallucinated IDs:     {total_hallucinated}")
    print(f"  Total duplicate IDs:        {total_duplicated}")
    print(f"  Mean precision:             {sum(precisions)/len(precisions)*100:.1f}%" if precisions else "  No precisions")
    print(f"  Mean recall:                {sum(recalls)/len(recalls)*100:.1f}%" if recalls else "  No recalls")


if __name__ == "__main__":
    main()
