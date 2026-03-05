"""Step 3: Build Meta-to-TID SFT Data

Creates SFT training data for teaching the model to generate text IDs (TID)
from product metadata (title + description + categories + related_queries
-> 7-word summary).

Usage:
    python s3_build_meta2tid_sft_data.py \
        --id2meta_file ./processed/id2meta.json \
        --output_file ./processed/sft_data/meta2tid_sft.json
"""

import os
import json
import argparse


def prepare_data(item: dict) -> dict:
    """Prepare a single SFT training sample."""
    info_lines = ["Product Information:"]
    
    title = item.get("title", "")
    if title:
        info_lines.append(f"Title: {title}")

    description = item.get("description", "")
    if description:
        if len(description) > 150:
            description = description[:150] + "..."
        info_lines.append(f"Description: {description}")

    categories = item.get("categories", "")
    if categories:
        info_lines.append(f"Categories: {categories}")

    related_queries = item.get("related_queries", "")
    if related_queries:
        if len(related_queries) > 150:
            related_queries = related_queries[:150] + "..."
        info_lines.append(f"Related Queries: {related_queries}")

    input_str = "\n".join(info_lines) + "\n"

    return {
        "instruction": (
            "Summarize the product into a text ID consisting of exactly 7 distinct, base-form words (nouns/adjectives). "
            "Order by importance (Category, Key Attribute, Brand/Ecosystem, Seller/Retailer, Gender/Audience, Style/Occasion, Unique Point) to highlight its uniqueness. "
            "Output strictly in the format: Item text ID: [word1, word2, word3, word4, word5, word6, word7]."
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
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading metadata: {args.id2meta_file}")
    with open(args.id2meta_file, "r", encoding="utf-8") as f:
        parent_asin2meta = json.load(f)
    print(f"Loaded {len(parent_asin2meta)} product mappings")

    sft_data = [prepare_data(value) for value in parent_asin2meta.values()]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"SFT data saved to: {args.output_file}")
    print(f"Generated {len(sft_data)} training samples")


if __name__ == "__main__":
    main()
