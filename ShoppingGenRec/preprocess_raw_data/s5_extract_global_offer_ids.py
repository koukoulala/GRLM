"""
Extract all GlobalOfferIds from merged_clean_item.json and write them
to a simple TSV file (one ID per line) for downstream joining.

Reads:
  1. merged_clean_item.json - unified item metadata from s4.

Produces:
  1. global_offer_ids.tsv   - one GlobalOfferId per line (no header).
     Only non-P-prefixed keys are included.

Usage:
    python s5_extract_global_offer_ids.py \
        --merged_item_file ../raw_data/merged_clean_item.json \
        --output_file      ../raw_data/global_offer_ids.tsv
"""

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract GlobalOfferIds from merged_clean_item.json "
                    "into a one-ID-per-line TSV for joining."
    )
    parser.add_argument(
        "--merged_item_file",
        type=str,
        default="./raw_data/merged_clean_item.json",
        help="Path to merged_clean_item.json from s4 "
             "(default: ./raw_data/merged_clean_item.json)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/CombinedRawData/merged_global_offer_ids.tsv",
        help="Output TSV path, one GlobalOfferId per line "
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading: {args.merged_item_file}")
    with open(args.merged_item_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    gids = sorted(k for k in data if not k.startswith("P"))
    ptids = total - len(gids)

    print(f"  Total entries:      {total:>10,}")
    print(f"  GlobalOfferIds:     {len(gids):>10,}")
    print(f"  PageTitle entries:  {ptids:>10,}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        for gid in gids:
            f.write(gid + "\n")

    print(f"Written {len(gids):,} IDs to: {args.output_file}")


if __name__ == "__main__":
    main()
