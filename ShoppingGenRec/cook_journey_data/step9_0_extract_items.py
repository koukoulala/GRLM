#!/usr/bin/env python3
"""Step 9.0: Extract OfferIds from step6 ranked TSV and/or step8 vis JSONL.

Collects all unique product OfferIds and writes them to a text file
(one ID per line) for use with s1_generate_tid.py --filter_items_file.

Supports two input formats (auto-detected by extension):
  - .tsv: step6 ranked TSV (RankedJourneys JSON column with OfferId)
  - .jsonl: step8 vis JSONL (global_offer_id field)

Usage:
    python step9_0_extract_items.py \
        --input .../ranker_output/vip_users_..._Ranked.tsv \
        --output .../vip_case_study_IDB_new/filter_offer_ids.txt
"""

import argparse
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)


def extract_from_tsv(filepath):
    """Extract OfferIds from step6 ranked TSV (RankedJourneys column)."""
    ids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rj_raw = row.get("RankedJourneys", "")
            try:
                rj = json.loads(rj_raw) if rj_raw else {}
            except (json.JSONDecodeError, TypeError):
                continue
            for j in rj.get("ContinuedJourneys", []):
                for p in j.get("Products", []):
                    oid = str(p.get("OfferId", "")).strip()
                    if oid:
                        ids.add(oid)
    return ids


def extract_from_jsonl(filepath):
    """Extract OfferIds from step8 vis JSONL (global_offer_id field)."""
    ids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            user = json.loads(line)
            for j in user.get("journeys", []):
                for p in j.get("products", []):
                    gid = p.get("global_offer_id", "")
                    if gid:
                        ids.add(gid)
    return ids


def main():
    ap = argparse.ArgumentParser(
        description="Extract unique OfferIds from ranked TSV or vis JSONL"
    )
    ap.add_argument(
        "--input", type=str, nargs="+",
        default=["/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                 "Data/LLMTrainingData/20260528/vip_case_study_IDB_new/"
                 "ranker_output/vip_users_journey_with_products_Ranked.tsv"],
        help="Input file(s): step6 ranked TSV (.tsv) or step8 vis JSONL (.jsonl)",
    )
    ap.add_argument(
        "--output", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260528/vip_case_study_IDB_new/"
                "filter_offer_ids.txt",
        help="Output file (one OfferId per line)",
    )
    args = ap.parse_args()

    offer_ids = set()
    for filepath in args.input:
        print(f"Reading: {filepath}")
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".tsv":
            ids = extract_from_tsv(filepath)
        else:
            ids = extract_from_jsonl(filepath)
        print(f"  Found {len(ids):,} unique OfferIds")
        offer_ids.update(ids)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for gid in sorted(offer_ids):
            f.write(gid + "\n")

    print(f"Total unique OfferIds: {len(offer_ids):,}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
