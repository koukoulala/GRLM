"""Step 8: Build RL Training Data for Journey GRPO

Converts journey SFT data (event2journey / profile2journey) from the
individual *_full.json files (output of s3) to RL training format.
Both tasks are merged, independently sampled, and jointly shuffled.

The RL task: given user events/profile, generate shopping journeys.
Reward signals (computed during training by shopping_grlm_journey_recipe.py):
  score = format * (0.2*IF + 0.3*diversity + 0.5*relevance) * volume_factor

Data format (output parquet columns):
  - prompt    : str  — JSON-serialized chat messages [{"role":"user","content":"..."}]
  - answer    : str  — ground-truth journey JSON (for relevance reward)
  - data_source         : str  — "shopping_journey"
  - task_type           : str  — "event2journey" | "profile2journey"
  - user_id             : str  — original user ID from metadata
  - required_journey_count    : int  — N from instruction (-1 = not specified)
  - min_products_per_journey  : int  — M from instruction (default 8)
  - gt_journey_count          : int  — number of journeys in GT answer
  - gt_total_products         : int  — total number of products in GT answer
  - num_events                : int  — number of user events in the input

Outputs:
  - train.parquet / train.jsonl          — full training set
  - test.parquet  / test.jsonl           — test set (per-task sampled, then merged)
  - train_easy.parquet / train_easy.jsonl — curriculum phase-1 data (easy half)

Usage:
    python s8_build_journey_rl_data.py \\
        --event2journey_file ./sft_data/event2journey_sft_full.json \\
        --profile2journey_file ./sft_data/profile2journey_sft_full.json \\
        --output_dir ./rl_data/journey \\
        --max_event2journey_samples 100000 \\
        --max_profile2journey_samples 100000 \\
        --test_event2journey 1000 --test_profile2journey 1000
"""

import json
import os
import re
import argparse
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


# ============================================================================
# Helpers
# ============================================================================

def parse_instruction_requirements(instruction, input_text):
    """Extract required journey count (N) and min products (M) from instruction/input.

    Returns:
        (required_journey_count, min_products_per_journey)
        required_journey_count = -1 when instruction says "appropriate number"
    """
    full_text = instruction + "\n" + input_text

    journey_count = -1
    m = re.search(r"predict\s+(\d+)\s+shopping\s+journey", full_text, re.IGNORECASE)
    if m:
        journey_count = int(m.group(1))
    else:
        m = re.search(r"exactly\s+(\d+)\s+journey", full_text, re.IGNORECASE)
        if m:
            journey_count = int(m.group(1))

    min_products = 8
    m = re.search(r"at\s+least\s+(\d+)\s+(?:recommended\s+)?products", full_text, re.IGNORECASE)
    if m:
        min_products = int(m.group(1))

    return journey_count, min_products


def count_gt_stats(output_text):
    """Parse GT answer to count journeys and total products."""
    try:
        obj = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return 0, 0
    journeys = obj.get("ContinuedJourneys", [])
    if not isinstance(journeys, list):
        return 0, 0
    total_products = 0
    for j in journeys:
        if isinstance(j, dict):
            tids = j.get("ProductTIDs", [])
            if isinstance(tids, list):
                total_products += len(tids)
    return len(journeys), total_products


def count_events(input_text):
    """Count user shopping events in the input text.

    Events are formatted as numbered items like '1. ...' or as bullet points.
    """
    lines = input_text.strip().split("\n")
    count = 0
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d+\.\s", stripped):
            count += 1
    return count if count > 0 else len([l for l in lines if l.strip()])


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build RL data for journey GRPO training"
    )
    parser.add_argument(
        "--event2journey_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260415/sft_data/event2journey_sft_full.json",
        help="Path to event2journey_sft_full.json (from s3)",
    )
    parser.add_argument(
        "--profile2journey_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260415/sft_data/profile2journey_sft_full.json",
        help="Path to profile2journey_sft_full.json (from s3)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260415/rl_data",
        help="Directory to save RL training data",
    )
    parser.add_argument(
        "--max_event2journey_samples", type=int, default=100000,
        help="Max samples to keep from event2journey (default: 100000)",
    )
    parser.add_argument(
        "--max_profile2journey_samples", type=int, default=100000,
        help="Max samples to keep from profile2journey (default: 100000)",
    )
    parser.add_argument(
        "--data_source", type=str, default="shopping_journey",
        help="Data source label (default: shopping_journey)",
    )
    parser.add_argument(
        "--test_event2journey", type=int, default=500,
        help="Test samples for event2journey (default: 500)",
    )
    parser.add_argument(
        "--test_profile2journey", type=int, default=500,
        help="Test samples for profile2journey (default: 500)",
    )
    parser.add_argument(
        "--easy_ratio", type=float, default=0.5,
        help="Fraction of training data for curriculum easy split (default: 0.5)",
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
# Per-task statistics
# ============================================================================

def print_task_stats(items, task_label):
    """Print distribution statistics for a single task's samples."""
    if not items:
        return
    n_events = [it["num_events"] for it in items]
    n_journeys = [it["gt_journey_count"] for it in items]
    n_products = [it["gt_total_products"] for it in items]
    req_counts = [it["required_journey_count"] for it in items]
    min_prods = [it["min_products_per_journey"] for it in items]
    has_explicit = sum(1 for r in req_counts if r > 0)

    def _stats(vals):
        a = np.array(vals)
        return f"mean={a.mean():.1f}, median={np.median(a):.0f}, min={a.min()}, max={a.max()}"

    print(f"\n  [{task_label}] — {len(items)} samples")
    print(f"    Events/sample:           {_stats(n_events)}")
    print(f"    GT journeys/sample:      {_stats(n_journeys)}")
    print(f"    GT products/sample:      {_stats(n_products)}")
    print(f"    Min products/journey:    {_stats(min_prods)}")
    print(f"    Explicit journey count:  {has_explicit}/{len(items)} ({100*has_explicit/len(items):.1f}%)")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    max_samples_map = {
        "event2journey": args.max_event2journey_samples,
        "profile2journey": args.max_profile2journey_samples,
    }

    test_size_map = {
        "event2journey": args.test_event2journey,
        "profile2journey": args.test_profile2journey,
    }

    print("=" * 60)
    print("Journey RL Data Builder")
    print("=" * 60)
    print(f"  Event2Journey:    {args.event2journey_file}")
    print(f"    max samples:    {args.max_event2journey_samples}")
    print(f"    test samples:   {args.test_event2journey}")
    print(f"  Profile2Journey:  {args.profile2journey_file}")
    print(f"    max samples:    {args.max_profile2journey_samples}")
    print(f"    test samples:   {args.test_profile2journey}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Easy ratio:       {args.easy_ratio}")
    print(f"  Seed:             {args.seed}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load and convert both tasks, then merge
    # ------------------------------------------------------------------
    task_pools: dict[str, list] = {}

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

        converted = []
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
            gt_jcount, gt_tprods = count_gt_stats(output)
            n_events = count_events(input_text)

            # Store prompt as chat-message JSON string for verl compatibility
            prompt_content = instruction + "\n" + input_text
            prompt = json.dumps(
                [{"role": "user", "content": prompt_content}],
                ensure_ascii=False,
            )

            user_id = item.get("metadata", {}).get("user_id", "")

            converted.append({
                "prompt": prompt,
                "answer": output,
                "data_source": args.data_source,
                "task_type": task_type,
                "user_id": str(user_id),
                "required_journey_count": int(req_count),
                "min_products_per_journey": int(min_products),
                "gt_journey_count": int(gt_jcount),
                "gt_total_products": int(gt_tprods),
                "num_events": int(n_events),
            })

        if skipped:
            print(f"    Skipped {skipped} samples with missing fields")
        print(f"    Valid: {len(converted)}")

        # Random sample if exceeding max
        max_n = max_samples_map[task_type]
        if max_n > 0 and len(converted) > max_n:
            random.shuffle(converted)
            converted = converted[:max_n]
            print(f"    Sampled down to {max_n}")

        task_pools[task_type] = converted

    if not task_pools:
        print("No data to process. Exiting.")
        return

    total_samples = sum(len(v) for v in task_pools.values())
    print(f"\nTotal samples across tasks: {total_samples}")

    # ------------------------------------------------------------------
    # Per-task test sampling, then merge
    # ------------------------------------------------------------------
    train_data = []
    test_data = []

    for task_type, items in task_pools.items():
        random.shuffle(items)
        t_size = test_size_map.get(task_type, 1000)
        if t_size >= len(items):
            print(f"  Warning: test_size ({t_size}) >= {task_type} pool ({len(items)}), using all as test")
            test_data.extend(items)
        else:
            test_data.extend(items[:t_size])
            train_data.extend(items[t_size:])

    random.shuffle(train_data)
    random.shuffle(test_data)

    print(f"\nTrain: {len(train_data)} | Test: {len(test_data)}")
    for task_type in sorted(task_pools.keys()):
        n_train = sum(1 for x in train_data if x["task_type"] == task_type)
        n_test = sum(1 for x in test_data if x["task_type"] == task_type)
        print(f"  {task_type}: train={n_train}, test={n_test}")

    # ------------------------------------------------------------------
    # Curriculum split — easy half based on difficulty
    # ------------------------------------------------------------------
    # Difficulty = num_events + gt_journey_count + gt_total_products
    # Lower difficulty = easier sample
    if train_data and args.easy_ratio > 0:
        for item in train_data:
            item["_difficulty"] = (
                item["num_events"] + item["gt_journey_count"] + item["gt_total_products"]
            )
        sorted_by_diff = sorted(train_data, key=lambda x: x["_difficulty"])
        easy_n = int(len(sorted_by_diff) * args.easy_ratio)
        easy_data = sorted_by_diff[:easy_n]
        random.shuffle(easy_data)

        # Clean up temp field
        for item in train_data:
            item.pop("_difficulty", None)
        for item in easy_data:
            item.pop("_difficulty", None)

        print(f"\nCurriculum easy split: {len(easy_data)} samples "
              f"(difficulty threshold at {args.easy_ratio*100:.0f}th percentile)")
    else:
        easy_data = []

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    all_data = train_data + test_data

    def _save(name, items):
        if not items:
            return
        if args.output_format in ("parquet", "both"):
            path = os.path.join(args.output_dir, f"{name}.parquet")
            pd.DataFrame(items).to_parquet(
                path, engine="pyarrow", index=False, compression="snappy"
            )
            print(f"  Saved: {path} ({len(items)} samples)")
        if args.output_format in ("json", "both"):
            path = os.path.join(args.output_dir, f"{name}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  Saved: {path} ({len(items)} samples)")

    _save("train", train_data)
    _save("test", test_data)
    if easy_data:
        _save("train_easy", easy_data)

    # ------------------------------------------------------------------
    # Statistics — per-task distributions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    by_task = defaultdict(list)
    for item in all_data:
        by_task[item["task_type"]].append(item)

    for task_type in sorted(by_task.keys()):
        print_task_stats(by_task[task_type], task_type)

    print_task_stats(all_data, "ALL (merged)")

    if easy_data:
        print_task_stats(easy_data, "EASY (curriculum)")

    # Print sample
    if all_data:
        sample = all_data[0]
        prompt_display = json.loads(sample["prompt"])[0]["content"][:300]
        print(f"\n--- Sample ---")
        print(f"Prompt (first 300 chars):\n{prompt_display}...")
        print(f"Answer (first 200 chars):\n{sample['answer'][:200]}...")
        print(f"Required journey count:  {sample['required_journey_count']}")
        print(f"Min products/journey:    {sample['min_products_per_journey']}")
        print(f"GT journeys:             {sample['gt_journey_count']}")
        print(f"GT total products:       {sample['gt_total_products']}")
        print(f"Num events:              {sample['num_events']}")
        print(f"Task type:               {sample['task_type']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
