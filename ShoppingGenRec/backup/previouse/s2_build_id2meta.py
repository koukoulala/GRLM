"""Step 2: Build Item ID to Metadata Mapping

Reads summaries from step 1 and builds a JSON mapping from item ID to
full metadata (including summary words).

Usage:
    python s2_build_id2meta.py \
        --input_file ./processed/sum_data/summaries_with_similarity.jsonl \
        --output_file ./processed/sum_data/id2meta.json
"""

import json
import argparse
from tqdm import tqdm


def create_mapping(input_file: str, output_file: str):
    """Create item_id -> metadata mapping from summary JSONL."""
    parent_asin2meta = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing data"):
            if not line.strip():
                continue
            item = json.loads(line)
            if len(item.get("summary_words", [])) != 5:
                print(f"Warning: item {item.get('id', '?')} has != 5 summary words")
            # Normalize multi-word summaries (join with hyphen)
            item["summary_words"] = [
                "-".join(word.split()) for word in item.get("summary_words", [])
            ]
            parent_asin = item.get("id")
            if parent_asin:
                parent_asin2meta[parent_asin] = item

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parent_asin2meta, f, ensure_ascii=False, indent=2)

    print(f"Completed! Mapped {len(parent_asin2meta)} products -> {output_file}")
    return parent_asin2meta


def parse_args():
    parser = argparse.ArgumentParser(description="Build item ID to metadata mapping")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to summaries JSONL file from step 1",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output path for id2meta JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mapping = create_mapping(args.input_file, args.output_file)

    # Show example
    sample_key = next(iter(mapping), None)
    if sample_key:
        info = mapping[sample_key]
        print(f"Example - ID: {sample_key}")
        print(f"  Title: {info.get('title', 'N/A')}")
        print(f"  Summary: {info.get('summary_words', [])}")


if __name__ == "__main__":
    main()
