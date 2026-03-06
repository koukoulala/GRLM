"""Step 3: Merge and Shuffle All SFT Datasets

Reads SFT JSON files from multiple tasks (meta2tid, rec, journey2product,
journey_prediction), applies per-task sampling, merges them into a single
dataset, and shuffles randomly.

Each input file should be a JSON list of dicts with at least
{instruction, input, output} keys.

Usage:
    python s3_merge_sft_data.py \
        --meta2tid_file ./sft_data/meta2tid_sft.json \
        --rec_file ./sft_data/rec_sft.json \
        --journey2product_file ./sft_data/journey_sft.json \
        --journey_prediction_file ./sft_data/journey_prediction_sft.json \
        --output_file ./sft_data/combined_sft.json \
        --meta2tid_prob 1.0 \
        --rec_prob 1.0 \
        --journey2product_prob 1.0 \
        --journey_prediction_prob 1.0
"""

import argparse
import json
import os
import random


def load_and_sample(file_path, sample_prob, task_name):
    """Load a JSON list file and apply random sampling.

    Args:
        file_path: Path to the JSON file.
        sample_prob: Probability of keeping each sample (0.0-1.0).
        task_name: Name of the task (for logging).

    Returns:
        Tuple of (sampled_data, original_count).
    """
    if not file_path or not os.path.exists(file_path):
        print(f"  [{task_name}] File not found or not specified: {file_path}")
        return [], 0

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"  [{task_name}] WARNING: expected list, got {type(data).__name__}")
        return [], 0

    original_count = len(data)

    if sample_prob >= 1.0:
        sampled = data
    elif sample_prob <= 0.0:
        sampled = []
    else:
        sampled = [d for d in data if random.random() < sample_prob]

    print(f"  [{task_name}] Loaded {original_count:>10,} -> "
          f"Sampled {len(sampled):>10,} (prob={sample_prob})")

    return sampled, original_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge and shuffle multiple SFT datasets into one"
    )
    # Input files
    parser.add_argument(
        "--meta2tid_file", type=str,
        default="./sft_data/meta2tid_sft.json",
        help="Path to meta2tid SFT data (default: ./sft_data/meta2tid_sft.json)",
    )
    parser.add_argument(
        "--rec_file", type=str,
        default="./sft_data/rec_sft.json",
        help="Path to rec SFT data (default: ./sft_data/rec_sft.json)",
    )
    parser.add_argument(
        "--journey2product_file", type=str,
        default="./sft_data/journey_sft.json",
        help="Path to journey2product SFT data "
    )
    parser.add_argument(
        "--journey_prediction_file", type=str,
        default="./sft_data/journey_prediction_sft.json",
        help="Path to journey prediction SFT data "
    )
    # Sampling probabilities
    parser.add_argument(
        "--meta2tid_prob", type=float, default=0.5,
        help="Sampling probability for meta2tid data",
    )
    parser.add_argument(
        "--rec_prob", type=float, default=1.0,
        help="Sampling probability for rec data",
    )
    parser.add_argument(
        "--journey2product_prob", type=float, default=1.0,
        help="Sampling probability for journey2product data",
    )
    parser.add_argument(
        "--journey_prediction_prob", type=float, default=1.0,
        help="Sampling probability for journey prediction data",
    )
    # Output
    parser.add_argument(
        "--output_file", type=str,
        default="./sft_data/combined_sft.json",
        help="Output path for merged SFT data",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling and shuffling",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Task configurations: (file_path, sample_prob, task_name)
    tasks = [
        (args.meta2tid_file, args.meta2tid_prob, "meta2tid"),
        (args.rec_file, args.rec_prob, "rec"),
        (args.journey2product_file, args.journey2product_prob, "journey2product"),
        (args.journey_prediction_file, args.journey_prediction_prob, "journey_prediction"),
    ]

    # =========================================================================
    # Step 1: Load and sample each task
    # =========================================================================
    print("=" * 70)
    print("Step 1: Loading and sampling SFT datasets")
    print("=" * 70)

    all_data = []
    stats = {}  # task_name -> (original, sampled)

    for file_path, prob, name in tasks:
        sampled, original = load_and_sample(file_path, prob, name)
        stats[name] = (original, len(sampled))
        all_data.extend(sampled)

    total_before_merge = sum(v[1] for v in stats.values())
    print(f"\n  Total samples after sampling: {total_before_merge:,}")

    # =========================================================================
    # Step 2: Shuffle
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Shuffling merged dataset")
    print("=" * 70)

    random.shuffle(all_data)
    print(f"  Shuffled {len(all_data):,} samples (seed={args.seed})")

    # =========================================================================
    # Step 3: Save
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Saving merged dataset")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"  Output: {args.output_file}")
    print(f"  Size:   {file_size_mb:.2f} MB")
    print(f"  Total:  {len(all_data):,} samples")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  {'Task':<25s} {'Original':>10s} {'Sampled':>10s} {'Ratio':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for name in ["meta2tid", "rec", "journey2product", "journey_prediction"]:
        orig, samp = stats.get(name, (0, 0))
        ratio = f"{samp/orig*100:.1f}%" if orig > 0 else "N/A"
        print(f"  {name:<25s} {orig:>10,} {samp:>10,} {ratio:>8s}")
    total_orig = sum(v[0] for v in stats.values())
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'TOTAL':<25s} {total_orig:>10,} {len(all_data):>10,}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
