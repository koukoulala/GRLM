"""Step 4b: Build Collaborative Filtering SFT Data

Creates SFT training data from SASRec similarity outputs. Given a target
product, the model learns to predict collaborative-filtering-based similar
products.

Usage:
    python s4_build_collaborative_sft_data.py \
        --id2meta_file ./processed/sum_data/id2meta.json \
        --sasrec_file ./processed/sum_data/similar_item_sasrec_num.txt \
        --output_file ./processed/sft_data/collaborative_sft.json \
        --top_k 5
"""

import os
import json
import argparse
from tqdm import tqdm


def load_id2meta(path: str) -> dict:
    """Load item ID to metadata mapping."""
    print(f"Loading metadata from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sasrec_sims(path: str, top_k: int = 5) -> dict:
    """Load SASRec similar items (numeric IDs)."""
    print(f"Loading SASRec similarities from {path}...")
    sims = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            anchor = parts[0]
            similar_items = parts[1 : top_k + 1]
            sims[anchor] = similar_items
    return sims


def format_item_input(item_meta: dict) -> str:
    """Format item information for input."""
    title = item_meta.get("title", "")
    description = item_meta.get("description", "")
    summary_words = item_meta.get("summary_words", [])
    text_id = "[" + ", ".join(summary_words) + "]"
    return f"Item text ID: {text_id} Title: {title}. Description: {description}.\n"


def format_sim_item_output(item_meta: dict) -> str:
    """Format similar item information for output."""
    title = item_meta.get("title", "")
    summary_words = item_meta.get("summary_words", [])
    text_id = "[" + ", ".join(summary_words) + "]"
    return f"Item text ID: {text_id} Title: {title}.\n"


def create_sft_sample(anchor_id: str, sim_ids: list, id2meta: dict) -> dict:
    """Create a single SFT training sample."""
    if anchor_id not in id2meta:
        return None

    anchor_meta = id2meta[anchor_id]
    valid_sims = [id2meta[sid] for sid in sim_ids if sid in id2meta]
    if not valid_sims:
        return None

    instruction = (
        "Analyze the input product's information and identifiers. Based on "
        "collaborative filtering patterns (co-purchase or co-view signals), "
        "recommend similar products.\n"
        "For each recommendation, provide its Title and Identifiers (5-word summary)."
    )

    input_text = (
        f"\nTarget Product:\n{format_item_input(anchor_meta)}\n"
        f"Please recommend {len(valid_sims)} similar products:"
    )

    output_parts = [
        f"{i}. {format_sim_item_output(sim_meta)}"
        for i, sim_meta in enumerate(valid_sims, 1)
    ]
    output_text = "\n\n".join(output_parts)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build collaborative filtering SFT data"
    )
    parser.add_argument(
        "--id2meta_file",
        type=str,
        required=True,
        help="Path to id2meta JSON from step 2",
    )
    parser.add_argument(
        "--sasrec_file",
        type=str,
        required=True,
        help="Path to SASRec similarity file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output path for collaborative SFT data JSON",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of similar items per sample (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    id2meta = load_id2meta(args.id2meta_file)
    sims = load_sasrec_sims(args.sasrec_file, top_k=args.top_k)

    sft_data = []
    for anchor, similar_list in tqdm(sims.items(), desc="Generating SFT data"):
        sample = create_sft_sample(anchor, similar_list, id2meta)
        if sample:
            sft_data.append(sample)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(sft_data)} samples to {args.output_file}")


if __name__ == "__main__":
    main()
