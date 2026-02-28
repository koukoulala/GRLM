"""Step 7: Build RL Training Data for GRPO

Converts SFT recommendation data to RL training format.

Data split strategy (NO DATA LEAKAGE):
  - Original sequence: [item1, ..., itemN-2, itemN-1(valid), itemN(test)]
  - RL Prompt:       instruction + input + output (ends at itemN-2)
  - RL Ground Truth: valid_ground_truth (itemN-1) — used for reward computation
  - Test Evaluation: test_ground_truth (itemN)   — held out for s5_eval.py

Usage:
    python s7_build_rl_data.py \
        --input_file ./processed/sft_data/rec_sft.json \
        --output_dir ./processed/rl_data \
        --test_size 1000
"""

import json
import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


def format_tid_as_text(tid_list):
    """Format TID list as text: [w1, w2, w3, w4, w5]."""
    if isinstance(tid_list, list):
        return "[" + ", ".join(tid_list) + "]"
    return str(tid_list)


def build_rl_prompt(item):
    """Build RL prompt: instruction + input + output (ends at itemN-2)."""
    return item["instruction"] + item["input"] + item["output"]


def build_ground_truth(item):
    """Build ground truth from valid_ground_truth_tid (itemN-1)."""
    return "Item text ID: " + format_tid_as_text(item["valid_ground_truth_tid"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build RL data for GRLM GRPO training"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="SFT rec data JSON (from step 4a)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save RL training data",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="shopping",
        help="Data source label (default: shopping)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of samples for test set (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="both",
        choices=["parquet", "json", "both"],
        help="Output format (default: both)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("GRLM RL Data Builder")
    print("=" * 60)
    print(f"  Input:       {args.input_file}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Data source: {args.data_source}")
    print(f"  Test size:   {args.test_size}")
    print(f"  Seed:        {args.seed}")
    print("=" * 60)

    # Load SFT data
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    # Convert to EasyR1 format
    required_fields = [
        "instruction", "input", "output",
        "valid_ground_truth_tid", "test_ground_truth_tid",
        "test_ground_truth_id",
    ]
    formatted_data = []
    skipped = 0

    for item in tqdm(data, desc="Converting"):
        if not all(key in item for key in required_fields):
            skipped += 1
            continue

        formatted_data.append(
            {
                "prompt": build_rl_prompt(item),
                "answer": build_ground_truth(item),
                "data_source": args.data_source,
                "user_id": item.get("metadata", {}).get("user_id"),
                "valid_ground_truth_id": item.get("valid_ground_truth_id"),
                "valid_ground_truth_tid": item["valid_ground_truth_tid"],
                "test_ground_truth_id": item["test_ground_truth_id"],
                "test_ground_truth_tid": item["test_ground_truth_tid"],
            }
        )

    print(f"Converted {len(formatted_data)} samples, skipped {skipped}")

    # Train / test split
    if args.test_size >= len(formatted_data):
        print(
            f"Warning: test_size ({args.test_size}) >= total ({len(formatted_data)}), "
            "using all data as test"
        )
        test_data = formatted_data
        train_data = []
    else:
        indices = list(range(len(formatted_data)))
        random.shuffle(indices)
        test_indices = set(indices[: args.test_size])
        train_data = [
            formatted_data[i]
            for i in range(len(formatted_data))
            if i not in test_indices
        ]
        test_data = [formatted_data[i] for i in indices[: args.test_size]]
        random.shuffle(train_data)
        random.shuffle(test_data)

    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    # Save
    if args.output_format in ("parquet", "both"):
        train_path = os.path.join(args.output_dir, "train.parquet")
        test_path = os.path.join(args.output_dir, "test.parquet")
        pd.DataFrame(train_data).to_parquet(
            train_path, engine="pyarrow", index=False, compression="snappy"
        )
        pd.DataFrame(test_data).to_parquet(
            test_path, engine="pyarrow", index=False, compression="snappy"
        )
        print(f"Saved: {train_path}, {test_path}")

    if args.output_format in ("json", "both"):
        train_path = os.path.join(args.output_dir, "train.jsonl")
        test_path = os.path.join(args.output_dir, "test.jsonl")
        for path, items in [(train_path, train_data), (test_path, test_data)]:
            with open(path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved: {train_path}, {test_path}")

    # Print sample
    if formatted_data:
        sample = formatted_data[0]
        print(f"\nSample prompt (first 300 chars):\n{sample['prompt'][:300]}...")
        print(f"Sample answer: {sample['answer']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
