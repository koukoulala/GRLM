"""Step 7: Build RL Training Data for Journey GRPO

Converts journey SFT data (event2journey / profile2journey) from the
individual *_full.json files (output of s3) to RL training format.

The RL task: given user events/profile, generate shopping journeys.
Reward signals (computed during training by grlm_journey_recipe.py):
  1. Instruction following — journey count + min products compliance
  2. Product diversity within journeys

Data format (output parquet columns):
  - prompt    : str  — instruction + "\\n" + input (wrapped as user message by dataset)
  - answer    : str  — ground-truth journey JSON (reference for logging / future rewards)
  - data_source         : str  — "shopping_journey"
  - task_type           : str  — "event2journey" | "profile2journey"
  - user_id             : str  — original user ID from metadata
  - required_journey_count    : int  — N from instruction (-1 = not specified)
  - min_products_per_journey  : int  — M from instruction (default 8)

Usage:
    python s7_build_journey_rl_data.py \\
        --event2journey_file ./sft_data/event2journey_sft_full.json \\
        --profile2journey_file ./sft_data/profile2journey_sft_full.json \\
        --output_dir ./rl_data/journey \\
        --test_size 1000
"""

import json
import os
import re
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# Instruction Parsing
# ============================================================================

def parse_instruction_requirements(instruction, input_text):
    """Extract required journey count (N) and min products (M) from instruction/input.

    Parses patterns produced by s3_build_journey_sft_data.py:
      instruction: "... predict N shopping journey(s) ..."
      input:       "... exactly N journeys, at least M products in each journey ..."

    Returns:
        (required_journey_count, min_products_per_journey)
        required_journey_count = -1 when instruction says "appropriate number"
    """
    full_text = instruction + "\n" + input_text

    # --- Journey count ---
    journey_count = -1
    # "predict 3 shopping journey(s)"
    m = re.search(r"predict\s+(\d+)\s+shopping\s+journey", full_text, re.IGNORECASE)
    if m:
        journey_count = int(m.group(1))
    else:
        # "exactly 3 journeys"
        m = re.search(r"exactly\s+(\d+)\s+journey", full_text, re.IGNORECASE)
        if m:
            journey_count = int(m.group(1))

    # --- Min products per journey ---
    min_products = 8  # fallback default
    m = re.search(r"at\s+least\s+(\d+)\s+(?:recommended\s+)?products", full_text, re.IGNORECASE)
    if m:
        min_products = int(m.group(1))

    return journey_count, min_products


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build RL data for journey GRPO training"
    )
    parser.add_argument(
        "--event2journey_file", type=str, default=None,
        help="Path to event2journey_sft_full.json (from s3)",
    )
    parser.add_argument(
        "--profile2journey_file", type=str, default=None,
        help="Path to profile2journey_sft_full.json (from s3)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save RL training data",
    )
    parser.add_argument(
        "--data_source", type=str, default="shopping_journey",
        help="Data source label (default: shopping_journey)",
    )
    parser.add_argument(
        "--test_size", type=int, default=1000,
        help="Number of samples for test set (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output_format", type=str, default="both",
        choices=["parquet", "json", "both"],
        help="Output format (default: both)",
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Journey RL Data Builder")
    print("=" * 60)
    print(f"  Event2Journey:  {args.event2journey_file}")
    print(f"  Profile2Journey: {args.profile2journey_file}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Data source:    {args.data_source}")
    print(f"  Test size:      {args.test_size}")
    print(f"  Seed:           {args.seed}")
    print("=" * 60)

    all_data = []

    for file_path, task_type in [
        (args.event2journey_file, "event2journey"),
        (args.profile2journey_file, "profile2journey"),
    ]:
        if file_path is None:
            print(f"  Skipping {task_type}: not provided")
            continue
        if not os.path.exists(file_path):
            print(f"  Skipping {task_type}: file not found ({file_path})")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} samples from {task_type}")

        skipped = 0
        for item in tqdm(data, desc=f"Converting {task_type}"):
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            if not instruction or not input_text or not output:
                skipped += 1
                continue

            req_count, min_products = parse_instruction_requirements(
                instruction, input_text
            )

            # RL prompt = instruction + "\n" + input
            # (the dataset class wraps this as a user chat message)
            prompt = instruction + "\n" + input_text

            user_id = item.get("metadata", {}).get("user_id", "")

            all_data.append({
                "prompt": prompt,
                "answer": output,
                "data_source": args.data_source,
                "task_type": task_type,
                "user_id": str(user_id),
                "required_journey_count": int(req_count),
                "min_products_per_journey": int(min_products),
            })

        if skipped:
            print(f"    Skipped {skipped} samples with missing fields")

    print(f"\nTotal valid samples: {len(all_data)}")
    if not all_data:
        print("No data to process. Exiting.")
        return

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    random.shuffle(all_data)
    if args.test_size >= len(all_data):
        print(
            f"Warning: test_size ({args.test_size}) >= total ({len(all_data)}), "
            "using all data as test"
        )
        test_data = all_data
        train_data = []
    else:
        test_data = all_data[: args.test_size]
        train_data = all_data[args.test_size :]
        random.shuffle(train_data)

    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.output_format in ("parquet", "both"):
        for name, items in [("train", train_data), ("test", test_data)]:
            if not items:
                continue
            path = os.path.join(args.output_dir, f"{name}.parquet")
            pd.DataFrame(items).to_parquet(
                path, engine="pyarrow", index=False, compression="snappy"
            )
            print(f"  Saved: {path} ({len(items)} samples)")

    if args.output_format in ("json", "both"):
        for name, items in [("train", train_data), ("test", test_data)]:
            if not items:
                continue
            path = os.path.join(args.output_dir, f"{name}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  Saved: {path} ({len(items)} samples)")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    task_counts = {}
    has_count_cnt = 0
    min_prod_vals = []
    for item in all_data:
        task_counts[item["task_type"]] = task_counts.get(item["task_type"], 0) + 1
        if item["required_journey_count"] > 0:
            has_count_cnt += 1
        min_prod_vals.append(item["min_products_per_journey"])

    print(f"\nStatistics:")
    print(f"  Task distribution: {task_counts}")
    print(
        f"  With explicit journey count: "
        f"{has_count_cnt}/{len(all_data)} "
        f"({100 * has_count_cnt / len(all_data):.1f}%)"
    )
    print(
        f"  Min products per journey: "
        f"mean={np.mean(min_prod_vals):.1f}, "
        f"min={np.min(min_prod_vals)}, max={np.max(min_prod_vals)}"
    )

    # Print sample
    if all_data:
        sample = all_data[0]
        print(f"\n--- Sample ---")
        print(f"Prompt (first 300 chars):\n{sample['prompt'][:300]}...")
        print(f"Answer (first 200 chars):\n{sample['answer'][:200]}...")
        print(f"Required journey count: {sample['required_journey_count']}")
        print(f"Min products/journey:   {sample['min_products_per_journey']}")
        print(f"Task type: {sample['task_type']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
