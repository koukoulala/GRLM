"""Step 3: Build Meta-to-TID SFT Data

Creates SFT training data for teaching the model to generate text IDs (TID)
from product metadata (title + description + categories + related_queries
-> 5-word summary).

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
    title = item.get("title", "")
    title_str = f"Title: {title}" if title else ""

    description = item.get("description", "")
    if description and len(description) > 150:
        description = description[:150] + "..."
    desc_str = f"Description: {description}" if description else ""

    categories = item.get("categories", "")
    categories_str = f"Categories: {categories}" if categories else ""

    related_queries = item.get("related_queries", "")
    if related_queries and len(related_queries) > 150:
        related_queries = related_queries[:150] + "..."
    related_queries_str = f"Related Queries: {related_queries}" if related_queries else ""

    return {
        "instruction": (
            "Please generate exactly five words to summarize this product. "
            "Follow these guidelines carefully:\n\n"
            "1. Words must be in their base form (noun or adjective, no -ed, -ing, -s endings)\n"
            "2. Order words by importance (most important aspect first)\n"
            "3. Focus on product category, function, key features, and target users\n"
            "4. Each word should represent a distinct aspect\n"
            "5. The word should be able to express the uniqueness of the product to "
            "ensure that it is distinguishable from other similar products\n"
            "6. Provide ONLY the five words in the specified format, with no additional "
            "text, explanations, or content\n"
            "7. Output format: [word1, word2, word3, word4, word5]"
        ),
        "input": (
            f"\n\nProduct Information:\n{title_str}\n{desc_str}"
            f"\n{categories_str}\n{related_queries_str}\n\n"
            "Please provide exactly five words separated by commas:"
        ),
        "output": "[" + ", ".join(item.get("summary_words", [])) + "]",
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
