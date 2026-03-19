"""
Continue sampling from pre_s3b's sampled sequential data, guided by
which GlobalOfferIds have already been processed by s1_generate_tid.

Reads pre_s3b outputs (item_sequential_data_sample.txt and
full_sequential_data_sample.json) plus the summaries_with_similarity.jsonl
from s1. For each user sequence, counts how many GlobalOfferIds appear in
the summaries file ("covered" items). Users whose sequences contain more
than --min_covered_items covered GlobalOfferIds are considered "rich" and
are always kept. "Poor" users (<=min_covered_items covered) are randomly
sampled so the total user count stays within --max_users.

This avoids the need for s1 to process additional GlobalOfferIds.

Usage:
    python preprocess_raw_data/pre_s3c_continue_sample.py \
        --input_dir ./raw_data \
        --summaries_file ./processed/summaries_with_similarity.jsonl \
        --max_users 400000 \
        --min_covered_items 1 \
        --output_dir ./raw_data
"""

import argparse
import json
import os
import random

from tqdm import tqdm


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue sampling from pre_s3b output, keeping users "
                    "whose sequences have sufficient coverage in s1 summaries."
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=400000,
        help="Maximum total number of users to keep (default: 400000)",
    )
    parser.add_argument(
        "--min_covered_items",
        type=int,
        default=1,
        help="Minimum number of covered GlobalOfferIds in a sequence for "
             "a user to be considered 'rich' (users with > this value are "
             "always kept) (default: 1)",
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
        help="Directory containing pre_s3b outputs (default: ./raw_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw_data",
        help="Directory for output files (default: ./raw_data)",
    )
    parser.add_argument(
        "--item_seq_file",
        type=str,
        default="item_sequential_data_sample.txt",
        help="Filename of sampled item sequential data from pre_s3b "
             "(default: item_sequential_data_sample.txt)",
    )
    parser.add_argument(
        "--full_seq_file",
        type=str,
        default="full_sequential_data_sample.json",
        help="Filename of sampled full sequential data from pre_s3b "
             "(default: full_sequential_data_sample.json)",
    )
    parser.add_argument(
        "--summaries_file",
        type=str,
        default="./processed/summaries_with_similarity.jsonl",
        help="Path to summaries_with_similarity.jsonl from s1 "
             "(default: ./processed/summaries_with_similarity.jsonl)",
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
    print("Continue Sample Sequential Data (pre_s3c)")
    print(f"  max_users          = {args.max_users:,}")
    print(f"  min_covered_items  = {args.min_covered_items}")
    print(f"  seed               = {args.seed}")
    print(f"  item_seq_file      = {item_seq_path}")
    print(f"  full_seq_file      = {full_seq_path}")
    print(f"  summaries_file     = {args.summaries_file}")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load covered GlobalOfferIds from summaries_with_similarity.jsonl
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 1: Loading covered GlobalOfferIds from summaries file")
    print("=" * 70)

    covered_gids = set()
    with open(args.summaries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                item_id = record.get("id", "")
                if item_id:
                    covered_gids.add(item_id)

    print(f"  Covered GlobalOfferIds: {len(covered_gids):>12,}")

    # =========================================================================
    # Step 2: Read item_sequential_data and classify users as rich/poor
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Classifying users (rich vs poor)")
    print("=" * 70)

    rich_user_ids = []
    poor_user_ids = []
    user_sequences = {}  # user_id -> list of item_ids (for stats)
    user_coverage = {}   # user_id -> number of covered items

    with open(item_seq_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = parts[0]
            item_ids = parts[1:]
            user_sequences[user_id] = item_ids

            num_covered = sum(1 for gid in item_ids if gid in covered_gids)
            user_coverage[user_id] = num_covered

            if num_covered > args.min_covered_items:
                rich_user_ids.append(user_id)
            else:
                poor_user_ids.append(user_id)

    total_users = len(rich_user_ids) + len(poor_user_ids)
    print(f"  Total users:  {total_users:>12,}")
    print(f"  Rich users (>{args.min_covered_items} covered):  "
          f"{len(rich_user_ids):>12,}  "
          f"({len(rich_user_ids) / total_users * 100:.2f}%)"
          if total_users > 0 else "")
    print(f"  Poor users (<={args.min_covered_items} covered): "
          f"{len(poor_user_ids):>12,}  "
          f"({len(poor_user_ids) / total_users * 100:.2f}%)"
          if total_users > 0 else "")

    # Coverage distribution
    coverage_counts = {}
    for uid, cov in user_coverage.items():
        coverage_counts[cov] = coverage_counts.get(cov, 0) + 1
    print(f"\n  Coverage distribution (top 15):")
    for cov_val, count in sorted(coverage_counts.items(),
                                  key=lambda x: -x[1])[:15]:
        print(f"    {cov_val:>4} covered items: {count:>10,} users")

    # =========================================================================
    # Step 3: Decide which poor users to keep (sample to fit max_users)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Sampling poor users to fit max_users budget")
    print("=" * 70)

    budget_for_poor = args.max_users - len(rich_user_ids)

    if budget_for_poor <= 0:
        print(f"  Rich users ({len(rich_user_ids):,}) already >= max_users "
              f"({args.max_users:,})")
        print(f"  Keeping all rich users, dropping all poor users")
        sampled_poor = []
    elif budget_for_poor >= len(poor_user_ids):
        print(f"  Budget for poor users ({budget_for_poor:,}) >= all poor "
              f"users ({len(poor_user_ids):,})")
        print(f"  Keeping all poor users")
        sampled_poor = poor_user_ids
    else:
        print(f"  Budget for poor users: {budget_for_poor:,} out of "
              f"{len(poor_user_ids):,}")
        sampled_poor = random.sample(poor_user_ids, budget_for_poor)
        print(f"  Sampled {len(sampled_poor):,} poor users")

    # Final set of users to keep
    keep_user_ids = set(rich_user_ids) | set(sampled_poor)
    print(f"\n  Final users to keep: {len(keep_user_ids):>12,}")
    print(f"    Rich:  {len(rich_user_ids):>12,}")
    print(f"    Poor:  {len(sampled_poor):>12,}")

    # =========================================================================
    # Step 4: Write sampled item_sequential_data
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Writing sampled item_sequential_data")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    base_name, ext = os.path.splitext(args.item_seq_file)
    out_item_seq_path = os.path.join(args.output_dir, f"{base_name}2{ext}")

    sampled_gids = set()
    sampled_items_total = 0
    sampled_users_written = 0
    sampled_seq_lens = []
    original_gids = set()
    original_items_total = 0
    original_seq_lens = []

    with open(item_seq_path, "r", encoding="utf-8") as fin, \
         open(out_item_seq_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = parts[0]
            item_ids = parts[1:]
            original_items_total += len(item_ids)
            original_seq_lens.append(len(item_ids))
            for gid in item_ids:
                original_gids.add(gid)

            if user_id in keep_user_ids:
                fout.write(line)
                sampled_users_written += 1
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
    # Step 5: Write sampled full_sequential_data.json (streaming)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 5: Writing sampled full_sequential_data.json")
    print("=" * 70)

    base_name_full, ext_full = os.path.splitext(args.full_seq_file)
    out_full_seq_path = os.path.join(
        args.output_dir, f"{base_name_full}2{ext_full}"
    )

    print(f"  Loading: {full_seq_path}")
    with open(full_seq_path, "r", encoding="utf-8") as f:
        full_seq_data = json.load(f)
    print(f"    Original users in JSON: {len(full_seq_data):,}")

    sampled_full_users = 0
    sampled_full_actions = 0

    with open(out_full_seq_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        first = True
        for user_id in tqdm(keep_user_ids, desc="  Writing sampled JSON",
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
    # Step 6: Statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 6: Summary Statistics")
    print("=" * 70)

    print(f"\n  Users:")
    print(f"    Original (pre_s3b): {total_users:>12,}")
    print(f"    Sampled:            {sampled_users_written:>12,}")
    print(f"    Reduction:          "
          f"{(1 - sampled_users_written / total_users) * 100:>11.2f}%"
          if total_users > 0 else "")

    print(f"\n  Distinct GlobalOfferIds:")
    print(f"    Original: {len(original_gids):>12,}")
    print(f"    Sampled:  {len(sampled_gids):>12,}")
    lost_gids = original_gids - sampled_gids
    print(f"    Lost:     {len(lost_gids):>12,}")
    print(f"    Retention: {len(sampled_gids) / len(original_gids) * 100:>11.2f}%"
          if original_gids else "")

    # Coverage stats for kept users
    kept_rich = len(rich_user_ids)
    kept_poor = len(sampled_poor)
    print(f"\n  Coverage breakdown (kept users):")
    print(f"    Rich (>{args.min_covered_items} covered):  {kept_rich:>12,}")
    print(f"    Poor (<={args.min_covered_items} covered): {kept_poor:>12,}")

    # How many distinct GIDs in kept sequences are already covered?
    covered_in_sampled = sampled_gids & covered_gids
    uncovered_in_sampled = sampled_gids - covered_gids
    print(f"\n  GlobalOfferId coverage in sampled data:")
    print(f"    Already covered by s1: {len(covered_in_sampled):>12,}")
    print(f"    Not yet covered:       {len(uncovered_in_sampled):>12,}")
    print(f"    Coverage rate:         "
          f"{len(covered_in_sampled) / len(sampled_gids) * 100:>11.2f}%"
          if sampled_gids else "")

    print(f"\n  Total item entries:")
    print(f"    Original: {original_items_total:>12,}")
    print(f"    Sampled:  {sampled_items_total:>12,}")
    print(f"    Reduction: "
          f"{(1 - sampled_items_total / original_items_total) * 100:>11.2f}%"
          if original_items_total > 0 else "")

    print(f"\n  Full sequence actions:")
    original_full_actions = sum(len(v) for v in full_seq_data.values())
    print(f"    Original: {original_full_actions:>12,}")
    print(f"    Sampled:  {sampled_full_actions:>12,}")
    print(f"    Reduction: "
          f"{(1 - sampled_full_actions / original_full_actions) * 100:>11.2f}%"
          if original_full_actions > 0 else "")

    # Sequence length comparison
    if sampled_seq_lens and original_seq_lens:
        def percentile(arr, p):
            s = sorted(arr)
            idx = int(len(s) * p)
            return s[min(idx, len(s) - 1)]

        print(f"\n  Item sequence length (original → sampled):")
        print(f"    Min:  {min(original_seq_lens):>6} → "
              f"{min(sampled_seq_lens):>6}")
        print(f"    P25:  {percentile(original_seq_lens, 0.25):>6} → "
              f"{percentile(sampled_seq_lens, 0.25):>6}")
        print(f"    P50:  {percentile(original_seq_lens, 0.5):>6} → "
              f"{percentile(sampled_seq_lens, 0.5):>6}")
        print(f"    Mean: "
              f"{sum(original_seq_lens)/len(original_seq_lens):>6.1f} → "
              f"{sum(sampled_seq_lens)/len(sampled_seq_lens):>6.1f}")
        print(f"    P75:  {percentile(original_seq_lens, 0.75):>6} → "
              f"{percentile(sampled_seq_lens, 0.75):>6}")
        print(f"    P90:  {percentile(original_seq_lens, 0.9):>6} → "
              f"{percentile(sampled_seq_lens, 0.9):>6}")
        print(f"    Max:  {max(original_seq_lens):>6} → "
              f"{max(sampled_seq_lens):>6}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
