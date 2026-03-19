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
        default="./processed/id2meta.json",
        help="Path to id2meta JSON from step 1 (default: ./processed/id2meta.json)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./sft_data/meta2tid_sft.json",
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
    num_gid = 0
    num_ptid = 0
    num_ptid_total = 0
    num_ptid_sampled = 0

    for key, value in parent_asin2meta.items():
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

        sft_data.append(prepare_data(value))

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"\nSFT data saved to: {args.output_file}")
    print(f"Generated {len(sft_data)} training samples")
    print(f"  GlobalOfferId items:  {num_gid:>10,}")
    print(f"  PageTitle items:      {num_ptid:>10,} "
          f"(sampled from {num_ptid_total:,}, prob={args.pagetitle_sample_prob})")


if __name__ == "__main__":
    main()
