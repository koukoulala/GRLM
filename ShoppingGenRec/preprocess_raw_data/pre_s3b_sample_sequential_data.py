"""
Sample a subset of users from sequential data files.

Reads the output of pre_s3 (item_sequential_data.txt and
full_sequential_data.json) and randomly samples a fraction of users,
producing sampled versions of both files.

This is an optional step between pre_s3 and pre_s4, useful for
reducing dataset size for faster experimentation.

Usage:
    python preprocess_raw_data/pre_s3b_sample_sequential_data.py \
        --sample_prob 0.4 \
        --input_dir ./raw_data \
        --output_dir ./raw_data
"""

import argparse
import json
import os
import random
from collections import defaultdict

from tqdm import tqdm


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a subset of users from sequential data files."
    )
    parser.add_argument(
        "--sample_prob",
        type=float,
        default=0.4,
        help="Probability of keeping each user (default: 0.4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./raw_data",
        help="Directory containing pre_s3 outputs (default: ./raw_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw_data",
        help="Directory for sampled outputs (default: ./raw_data)",
    )
    parser.add_argument(
        "--item_seq_file",
        type=str,
        default="item_sequential_data.txt",
        help="Filename of item sequential data (default: item_sequential_data.txt)",
    )
    parser.add_argument(
        "--full_seq_file",
        type=str,
        default="full_sequential_data.json",
        help="Filename of full sequential data (default: full_sequential_data.json)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)

    item_seq_path = os.path.join(args.input_dir, args.item_seq_file)
    full_seq_path = os.path.join(args.input_dir, args.full_seq_file)

    print("=" * 70)
    print("Sample Sequential Data")
    print(f"  sample_prob = {args.sample_prob}")
    print(f"  seed        = {args.seed}")
    print("=" * 70)

    # =========================================================================
    # Step 1: Read item_sequential_data.txt and decide which users to keep
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 1: Sampling users from item_sequential_data.txt")
    print("=" * 70)

    sampled_user_ids = set()
    original_users = 0
    original_gids = set()
    original_items_total = 0

    # Sequence length stats
    original_seq_lens = []

    with open(item_seq_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            original_users += 1
            user_id = parts[0]
            item_ids = parts[1:]
            original_items_total += len(item_ids)
            original_seq_lens.append(len(item_ids))
            for gid in item_ids:
                original_gids.add(gid)

            if random.random() < args.sample_prob:
                sampled_user_ids.add(user_id)

    print(f"  Original users:       {original_users:>12,}")
    print(f"  Original distinct GIDs: {len(original_gids):>12,}")
    print(f"  Original total items: {original_items_total:>12,}")
    print(f"  Sampled users:        {len(sampled_user_ids):>12,}")
    print(f"  Sample rate (actual): "
          f"{len(sampled_user_ids) / original_users * 100:.2f}%"
          if original_users > 0 else "  Sample rate: N/A")

    # =========================================================================
    # Step 2: Write sampled item_sequential_data.txt
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Writing sampled item_sequential_data.txt")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    base_name, ext = os.path.splitext(args.item_seq_file)
    out_item_seq_path = os.path.join(args.output_dir, f"{base_name}_sample{ext}")

    sampled_gids = set()
    sampled_items_total = 0
    sampled_users_written = 0
    sampled_seq_lens = []

    with open(item_seq_path, "r", encoding="utf-8") as fin, \
         open(out_item_seq_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = parts[0]
            if user_id in sampled_user_ids:
                fout.write(line)
                sampled_users_written += 1
                item_ids = parts[1:]
                sampled_items_total += len(item_ids)
                sampled_seq_lens.append(len(item_ids))
                for gid in item_ids:
                    sampled_gids.add(gid)

    out_size = os.path.getsize(out_item_seq_path) / (1024 * 1024)
    print(f"  Written: {out_item_seq_path}")
    print(f"    Size:  {out_size:.2f} MB")
    print(f"    Users: {sampled_users_written:,}")
    print(f"    Total items: {sampled_items_total:,}")

    # =========================================================================
    # Step 3: Write sampled full_sequential_data.json (streaming)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Writing sampled full_sequential_data.json")
    print("=" * 70)

    base_name_full, ext_full = os.path.splitext(args.full_seq_file)
    out_full_seq_path = os.path.join(
        args.output_dir, f"{base_name_full}_sample{ext_full}"
    )

    # Stream-read the JSON: read full file, filter, stream-write
    print(f"  Loading: {full_seq_path}")
    with open(full_seq_path, "r", encoding="utf-8") as f:
        full_seq_data = json.load(f)
    print(f"    Original users in JSON: {len(full_seq_data):,}")

    sampled_full_users = 0
    sampled_full_actions = 0

    with open(out_full_seq_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        first = True
        for user_id in tqdm(sampled_user_ids, desc="  Writing sampled JSON",
                            dynamic_ncols=True):
            if user_id not in full_seq_data:
                continue
            actions = full_seq_data[user_id]
            sampled_full_users += 1
            sampled_full_actions += len(actions)

            if not first:
                f.write(",\n")
            first = False

            key_str = json.dumps(user_id, ensure_ascii=False)
            val_str = json.dumps(actions, indent=2, ensure_ascii=False)
            val_str = val_str.replace("\n", "\n  ")
            f.write(f"  {key_str}: {val_str}")
        f.write("\n}\n")

    out_full_size = os.path.getsize(out_full_seq_path) / (1024 * 1024)
    print(f"\n  Written: {out_full_seq_path}")
    print(f"    Size:    {out_full_size:.2f} MB")
    print(f"    Users:   {sampled_full_users:,}")
    print(f"    Actions: {sampled_full_actions:,}")

    # =========================================================================
    # Step 4: Statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Summary Statistics")
    print("=" * 70)

    print(f"\n  Users:")
    print(f"    Original:  {original_users:>12,}")
    print(f"    Sampled:   {sampled_users_written:>12,}")
    print(f"    Reduction: {(1 - sampled_users_written / original_users) * 100:>11.2f}%"
          if original_users > 0 else "")

    print(f"\n  Distinct GlobalOfferIds:")
    print(f"    Original:  {len(original_gids):>12,}")
    print(f"    Sampled:   {len(sampled_gids):>12,}")
    lost_gids = original_gids - sampled_gids
    print(f"    Lost:      {len(lost_gids):>12,}")
    print(f"    Retention: {len(sampled_gids) / len(original_gids) * 100:>11.2f}%"
          if original_gids else "")

    print(f"\n  Total item entries:")
    print(f"    Original:  {original_items_total:>12,}")
    print(f"    Sampled:   {sampled_items_total:>12,}")
    print(f"    Reduction: {(1 - sampled_items_total / original_items_total) * 100:>11.2f}%"
          if original_items_total > 0 else "")

    print(f"\n  Full sequence actions:")
    original_full_actions = sum(len(v) for v in full_seq_data.values())
    print(f"    Original:  {original_full_actions:>12,}")
    print(f"    Sampled:   {sampled_full_actions:>12,}")
    print(f"    Reduction: {(1 - sampled_full_actions / original_full_actions) * 100:>11.2f}%"
          if original_full_actions > 0 else "")

    # Sequence length comparison
    if sampled_seq_lens and original_seq_lens:
        def percentile(arr, p):
            s = sorted(arr)
            idx = int(len(s) * p)
            return s[min(idx, len(s) - 1)]

        print(f"\n  Item sequence length (original → sampled):")
        print(f"    Min:  {min(original_seq_lens):>6} → {min(sampled_seq_lens):>6}")
        print(f"    P25:  {percentile(original_seq_lens, 0.25):>6} → "
              f"{percentile(sampled_seq_lens, 0.25):>6}")
        print(f"    P50:  {percentile(original_seq_lens, 0.5):>6} → "
              f"{percentile(sampled_seq_lens, 0.5):>6}")
        print(f"    Mean: {sum(original_seq_lens)/len(original_seq_lens):>6.1f} → "
              f"{sum(sampled_seq_lens)/len(sampled_seq_lens):>6.1f}")
        print(f"    P75:  {percentile(original_seq_lens, 0.75):>6} → "
              f"{percentile(sampled_seq_lens, 0.75):>6}")
        print(f"    P90:  {percentile(original_seq_lens, 0.9):>6} → "
              f"{percentile(sampled_seq_lens, 0.9):>6}")
        print(f"    Max:  {max(original_seq_lens):>6} → {max(sampled_seq_lens):>6}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
