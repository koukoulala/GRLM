"""Step 3: Build Meta-to-TID SFT Data

Creates SFT training data for teaching the model to generate text IDs (TID)
from product metadata (title + description + categories
+ structured attributes -> 7-word summary).

Reads merged_clean_item_with_attr.json (from preprocess s6) via id2meta.json
which inherits the enriched attributes.

Supports PageTitle item down-sampling via --pagetitle_sample_prob.

Usage:
    python s2_build_meta2tid_sft_data.py \
        --id2meta_file ./processed/id2meta.json \
        --output_file ./sft_data/meta2tid_sft.json \
        --pagetitle_sample_prob 0.5
"""

import os
import json
import random
import argparse

# Attributes to include in the product information.
# Only Brand, Seller, Model, and non-default Gender/AgeGroup.
# Exclude Color, Size, Material (too granular), Price, Market.


def prepare_data(item: dict) -> dict:
    """Prepare a single SFT training sample."""
    info_lines = ["Product Information:"]

    title = item.get("title", "")
    if title:
        if len(title) > 150:
            title = title[:150] + "..."
        info_lines.append(f"Title: {title}")

    description = item.get("description", "")
    if description:
        if len(description) > 150:
            description = description[:150] + "..."
        info_lines.append(f"Description: {description}")

    categories = item.get("categories", "")
    if categories:
        if len(categories) > 150:
            categories = categories[:150] + "..."
        info_lines.append(f"Categories: {categories}")

    # Append structured attributes (from s6 enrichment)
    attributes = item.get("attributes", {})
    brand = attributes.get("Brand", "")
    if isinstance(brand, str):
        brand = " ".join(brand.split())
    seller = attributes.get("Seller", "")
    if isinstance(seller, str):
        seller = " ".join(seller.split())
    if brand and seller and brand.lower() == seller.lower():
        info_lines.append(f"Brand/Seller: {brand}")
    else:
        if brand:
            info_lines.append(f"Brand: {brand}")
        if seller:
            info_lines.append(f"Seller: {seller}")
    for attr_name in ["Color", "Size"]:
        attr_val = attributes.get(attr_name, "")
        if isinstance(attr_val, str):
            attr_val = attr_val.strip()
        if attr_val:
            info_lines.append(f"{attr_name}: {attr_val}")
    gender = attributes.get("Gender", "").strip()
    if gender and gender.lower() != "unisex":
        info_lines.append(f"Gender: {gender}")
    age_group = attributes.get("AgeGroup", "").strip()
    if age_group and age_group.lower() != "adult":
        info_lines.append(f"AgeGroup: {age_group}")

    input_str = "\n".join(info_lines) + "\n"

    return {
        "instruction": (
            "Summarize the product into a text ID of exactly 7 distinct slots. "
            "Each slot is one base-form word; use a multi-word phrase only for "
            "brand/seller names or fixed proper nouns. "
            "Priority: category, function, feature, attribute, brand, seller, audience/style. "
            "Output strictly in the format: Item text ID: [s1, s2, s3, s4, s5, s6, s7]."
        ),
        "input": input_str,
        "output": "Item text ID: [" + ", ".join(item.get("summary_words", [])) + "]",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build meta-to-TID SFT training data"
    )
    parser.add_argument(
        "--id2meta_file",
        type=str,
        #default="./processed/id2meta.json",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed/id2meta.json",
        help="Path to id2meta JSON from step 1 (default: ./processed/id2meta.json)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        #default="./sft_data/meta2tid_sft.json",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/sft_data/meta2tid_sft.json",
        help="Output path for SFT training data JSON",
    )
    parser.add_argument(
        "--pagetitle_sample_prob",
        type=float,
        default=0.5,
        help="Sampling probability for PageTitle items (P-prefixed). "
             "Set to 1.0 to keep all, 0.0 to exclude all (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PageTitle sampling (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading metadata: {args.id2meta_file}")
    with open(args.id2meta_file, "r", encoding="utf-8") as f:
        parent_asin2meta = json.load(f)
    print(f"Loaded {len(parent_asin2meta)} product mappings")

    sft_data = []
    sft_data_full = []
    item_id2tid = {}       # item_id -> [7 summary words]
    num_gid = 0
    num_ptid = 0
    num_ptid_total = 0
    num_ptid_sampled = 0
    num_no_tid = 0

    for key, value in parent_asin2meta.items():
        # Build item_id2tid for ALL items (regardless of PageTitle sampling)
        summary_words = value.get("summary_words", [])
        valid_words = [
            word.replace("[", "").replace("]", "")
            for word in summary_words
            if word and word.strip()
        ]
        if len(valid_words) >= 7 and "" not in valid_words:
            item_id2tid[key] = valid_words[:7]
        else:
            num_no_tid += 1

        is_pagetitle = key.startswith("P")
        if is_pagetitle:
            num_ptid_total += 1
            # Down-sample PageTitle items
            if random.random() >= args.pagetitle_sample_prob:
                continue
            num_ptid_sampled += 1
            num_ptid += 1
        else:
            num_gid += 1

        sample = prepare_data(value)
        sft_data.append(sample)
        # Full version preserves GlobalOfferId for downstream test-set building
        full_sample = dict(sample)
        full_sample["metadata"] = {"GlobalOfferId": key}
        sft_data_full.append(full_sample)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Save training-only version (instruction/input/output)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    # Save full version with metadata
    full_file = args.output_file.replace(".json", "_full.json")
    with open(full_file, "w", encoding="utf-8") as f:
        json.dump(sft_data_full, f, ensure_ascii=False, indent=2)
    full_mb = os.path.getsize(full_file) / (1024 * 1024)
    print(f"Full data saved: {full_file} ({full_mb:.1f} MB)")

    # =========================================================================
    # Save item_id2tid and tid2item_id mappings
    # =========================================================================
    tid_dir = os.path.join(os.path.dirname(args.output_file), "item_id2tid")
    os.makedirs(tid_dir, exist_ok=True)

    # item_id2tid: item_id -> [7 summary words]
    id2tid_file = os.path.join(tid_dir, "item_id2tid.json")
    with open(id2tid_file, "w", encoding="utf-8") as f:
        json.dump(item_id2tid, f, ensure_ascii=False, indent=2)

    # tid2item_id: comma-joined TID string -> [list of item_ids]
    tid2item_id = {}
    for item_id, tid_words in item_id2tid.items():
        tid_key = ",".join(tid_words)
        if tid_key not in tid2item_id:
            tid2item_id[tid_key] = []
        tid2item_id[tid_key].append(item_id)

    tid2id_file = os.path.join(tid_dir, "tid2item_id.json")
    with open(tid2id_file, "w", encoding="utf-8") as f:
        json.dump(tid2item_id, f, ensure_ascii=False, indent=2)

    print(f"\n  item_id2tid mappings saved to: {tid_dir}")
    print(f"    item_id2tid: {len(item_id2tid):,} items")
    print(f"    tid2item_id: {len(tid2item_id):,} unique TIDs")
    print(f"    Items without valid TID: {num_no_tid:,}")

    # TID multiplicity statistics
    tid_counts = [len(ids) for ids in tid2item_id.values()]
    multi_tids = [c for c in tid_counts if c > 1]
    print(f"\n  --- TID Multiplicity (1 TID -> N items) ---")
    print(f"    TIDs mapping to 1 item:   {sum(1 for c in tid_counts if c == 1):>10,}")
    print(f"    TIDs mapping to 2+ items: {len(multi_tids):>10,}")
    if multi_tids:
        import numpy as np
        arr = np.array(multi_tids)
        print(f"    Among multi-mapped TIDs:")
        print(f"      Mean items/TID: {arr.mean():.1f}")
        print(f"      Max items/TID:  {arr.max()}")
        # Distribution buckets
        for threshold in [2, 3, 5, 10, 50]:
            count = sum(1 for c in multi_tids if c >= threshold)
            print(f"      TIDs with >= {threshold:>2} items: {count:>10,}")

    # Print summary statistics
    train_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"\nSummary:")
    print(f"  Total items in id2meta:   {len(parent_asin2meta):>10,}")
    print(f"  GlobalOfferId items:      {num_gid:>10,}")
    print(f"  PageTitle items (total):  {num_ptid_total:>10,}")
    print(f"  PageTitle items (sampled):{num_ptid_sampled:>10,}")
    print(f"  SFT training samples:     {len(sft_data):>10,}")
    print(f"  Training data: {args.output_file} ({train_mb:.1f} MB)")

    # Show examples
    print(f"\nExample cases (first 2):")
    for idx, sample in enumerate(sft_data[:2]):
        print(f"\n--- Example {idx + 1} ---")
        print(f"  Instruction: {sample['instruction'][:100]}...")
        print(f"  Input: {sample['input'][:200]}...")
        print(f"  Output: {sample['output']}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()