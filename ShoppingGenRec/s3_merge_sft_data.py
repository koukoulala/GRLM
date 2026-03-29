"""Step 3: Merge and Shuffle All SFT Datasets + Build Test Sets

Three tasks: meta2tid, event2journey, profile2journey.

For event2journey and profile2journey:
  - Shared users between the two tasks are identified.
  - test_sample_n test users are sampled from shared users and excluded
    from BOTH tasks' training data.
  - Remaining shared users are split or shared between tasks to reach
    the --journey_target_total.  If the deduped total is already above
    the target, shared users are simply split 50/50 with no duplication.
    If below, some shared users appear in both tasks to fill the gap.

For meta2tid:
  - Loads meta2tid_sft_full.json (with metadata.GlobalOfferId).
  - test_sample_n items sampled as test set (with GlobalOfferId).
  - Remaining items sampled by --meta2tid_prob.

Outputs (in --output_dir):
  1. combined_sft.jsonl          - Merged training data (shuffled).
  2. meta2tid_test.jsonl         - Test set (instruction/input/output + GlobalOfferId).
  3. event2journey_test.jsonl    - Test set (instruction/input/output + UserId).
  4. profile2journey_test.jsonl  - Test set (instruction/input/output + UserId).

Usage:
    python s3_merge_sft_data.py \\
        --meta2tid_full_file ./sft_data/meta2tid_sft_full.json \\
        --event2journey_full_file ./sft_data/event2journey_sft_full.json \\
        --profile2journey_full_file ./sft_data/profile2journey_sft_full.json \\
        --journey_target_total 1000000 \\
        --test_sample_n 5000
"""

import argparse
import json
import os
import random





# =============================================================================
# Common helpers
# =============================================================================

def extract_sft_fields(sample):
    """Extract only instruction/input/output from a full sample dict."""
    return {
        "instruction": sample["instruction"],
        "input": sample["input"],
        "output": sample["output"],
    }


def percentile(sorted_list, p):
    """Return p-th percentile from a pre-sorted list."""
    idx = int(len(sorted_list) * p / 100)
    return sorted_list[min(idx, len(sorted_list) - 1)]


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge and shuffle SFT datasets (meta2tid + two journey "
                    "tasks) into one training file, plus three test files."
    )

    # --- Input files ---
    parser.add_argument(
        "--meta2tid_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260324/sft_data/"
                "meta2tid_sft_full.json",
        help="Path to meta2tid *_full.json (with metadata.GlobalOfferId)",
    )
    parser.add_argument(
        "--event2journey_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260324/sft_data/"
                "event2journey_sft_full.json",
        help="Path to event2journey *_full.json (with metadata.user_id)",
    )
    parser.add_argument(
        "--profile2journey_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260324/sft_data/"
                "profile2journey_sft_full.json",
        help="Path to profile2journey *_full.json (with metadata.user_id)",
    )

    # --- Sampling ---
    parser.add_argument(
        "--meta2tid_prob", type=float, default=0.1,
        help="Sampling probability for meta2tid training data",
    )
    parser.add_argument(
        "--meta2tid_max_train", type=int, default=500_000,
        help="Maximum number of meta2tid training samples (default: 500000)",
    )
    parser.add_argument(
        "--journey_target_total", type=int, default=1_000_000,
        help="Target total for event2journey + profile2journey combined. "
             "If deduped total is below this, shared users are duplicated "
             "across both tasks to fill the gap (default: 1000000)",
    )

    # --- Test set ---
    parser.add_argument(
        "--test_sample_n", type=int, default=5000,
        help="Number of test samples per task (default: 5000)",
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260324/sft_data",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    all_training = []
    stats = {}

    # =========================================================================
    # 1. meta2tid: load _full.json, split test, sample training
    # =========================================================================
    print("=" * 70)
    print("1. meta2tid - loading meta2tid_sft_full.json")
    print("=" * 70)

    meta_by_gid = {}  # GlobalOfferId -> full sample
    if args.meta2tid_full_file and os.path.exists(args.meta2tid_full_file):
        with open(args.meta2tid_full_file, "r", encoding="utf-8") as f:
            meta_full_data = json.load(f)
        for sample in meta_full_data:
            gid = sample.get("metadata", {}).get("GlobalOfferId", "")
            if gid:
                meta_by_gid[gid] = sample
        print(f"  Loaded {len(meta_full_data):,} samples, "
              f"{len(meta_by_gid):,} unique GlobalOfferIds")
    else:
        print(f"  File not found: {args.meta2tid_full_file}")

    # Shuffle GlobalOfferIds, then split test / train
    meta_gids = list(meta_by_gid.keys())
    rng.shuffle(meta_gids)
    test_n_meta = min(args.test_sample_n, len(meta_gids))
    meta_test_gids = meta_gids[:test_n_meta]
    meta_train_gids = meta_gids[test_n_meta:]

    # Test set: sft fields + GlobalOfferId
    meta_test = []
    for gid in meta_test_gids:
        entry = extract_sft_fields(meta_by_gid[gid])
        entry["GlobalOfferId"] = gid
        meta_test.append(entry)

    # Training: apply sampling probability, then cap at max_train
    meta_train = []
    for gid in meta_train_gids:
        if rng.random() < args.meta2tid_prob:
            meta_train.append(extract_sft_fields(meta_by_gid[gid]))

    if args.meta2tid_max_train and len(meta_train) > args.meta2tid_max_train:
        rng.shuffle(meta_train)
        meta_train = meta_train[:args.meta2tid_max_train]

    stats["meta2tid"] = (len(meta_train_gids), len(meta_train))
    all_training.extend(meta_train)
    print(f"  Test:  {len(meta_test):,}")
    print(f"  Train: {len(meta_train_gids):,} -> sampled {len(meta_train):,} "
          f"(prob={args.meta2tid_prob})")

    # =========================================================================
    # 2. Journey tasks: load, test split, dedup/share, build training
    # =========================================================================
    print()
    print("=" * 70)
    print("2. Journey tasks - event2journey + profile2journey")
    print("=" * 70)

    # --- Load both full files, group by user_id ---
    e2j_by_user = {}
    if args.event2journey_full_file and os.path.exists(args.event2journey_full_file):
        with open(args.event2journey_full_file, "r", encoding="utf-8") as f:
            e2j_data = json.load(f)
        for sample in e2j_data:
            uid = sample.get("metadata", {}).get("user_id", "")
            if uid:
                e2j_by_user[uid] = sample
        print(f"  [event2journey]    Loaded {len(e2j_data):,} samples, "
              f"{len(e2j_by_user):,} unique users")
    else:
        print(f"  [event2journey]    File not found: "
              f"{args.event2journey_full_file}")

    p2j_by_user = {}
    if args.profile2journey_full_file and os.path.exists(args.profile2journey_full_file):
        with open(args.profile2journey_full_file, "r", encoding="utf-8") as f:
            p2j_data = json.load(f)
        for sample in p2j_data:
            uid = sample.get("metadata", {}).get("user_id", "")
            if uid:
                p2j_by_user[uid] = sample
        print(f"  [profile2journey]  Loaded {len(p2j_data):,} samples, "
              f"{len(p2j_by_user):,} unique users")
    else:
        print(f"  [profile2journey]  File not found: "
              f"{args.profile2journey_full_file}")

    # --- Identify user groups ---
    shared_uids = set(e2j_by_user.keys()) & set(p2j_by_user.keys())
    e2j_only_uids = set(e2j_by_user.keys()) - shared_uids
    p2j_only_uids = set(p2j_by_user.keys()) - shared_uids

    print(f"\n  User breakdown:")
    print(f"    Shared:               {len(shared_uids):>10,}")
    print(f"    event2journey only:   {len(e2j_only_uids):>10,}")
    print(f"    profile2journey only: {len(p2j_only_uids):>10,}")

    # --- Sample test users from shared users ---
    shared_sorted = sorted(shared_uids)
    rng.shuffle(shared_sorted)
    test_n_journey = min(args.test_sample_n, len(shared_uids))
    test_uids = set(shared_sorted[:test_n_journey])

    # Build test sets: sft fields + UserId
    e2j_test = []
    p2j_test = []
    for uid in sorted(test_uids):
        if uid in e2j_by_user:
            entry = extract_sft_fields(e2j_by_user[uid])
            entry["UserId"] = uid
            e2j_test.append(entry)
        if uid in p2j_by_user:
            entry = extract_sft_fields(p2j_by_user[uid])
            entry["UserId"] = uid
            p2j_test.append(entry)

    print(f"\n  Test set ({test_n_journey:,} shared users):")
    print(f"    event2journey test:    {len(e2j_test):,}")
    print(f"    profile2journey test:  {len(p2j_test):,}")

    # --- Remove test users from all pools ---
    shared_train_uids = shared_uids - test_uids
    e2j_only_uids -= test_uids
    p2j_only_uids -= test_uids

    # --- Compute sharing strategy ---
    #   min_total: split shared 50/50, each shared user in one task only
    #   max_total: all shared users in BOTH tasks
    S = len(shared_train_uids)
    min_total = len(e2j_only_uids) + len(p2j_only_uids) + S
    max_total = len(e2j_only_uids) + len(p2j_only_uids) + 2 * S
    target = args.journey_target_total

    if target <= min_total:
        num_to_share = 0
    elif target <= max_total:
        num_to_share = target - min_total
    else:
        num_to_share = S

    shared_train_list = sorted(shared_train_uids)
    rng.shuffle(shared_train_list)

    # First num_to_share users go to BOTH tasks
    shared_both = set(shared_train_list[:num_to_share])
    # Remaining split 50/50
    remaining = shared_train_list[num_to_share:]
    mid = len(remaining) // 2
    e2j_exclusive_shared = set(remaining[:mid])
    p2j_exclusive_shared = set(remaining[mid:])

    e2j_final_uids = e2j_only_uids | shared_both | e2j_exclusive_shared
    p2j_final_uids = p2j_only_uids | shared_both | p2j_exclusive_shared

    total_journey = len(e2j_final_uids) + len(p2j_final_uids)

    print(f"\n  Sharing strategy (target={target:,}):")
    print(f"    Shared train users:     {S:,}")
    print(f"    Min total (no share):   {min_total:,}")
    print(f"    Max total (full share): {max_total:,}")
    print(f"    Users shared in both:   {num_to_share:,}")
    print(f"    Exclusive to e2j:       {len(e2j_exclusive_shared):,}")
    print(f"    Exclusive to p2j:       {len(p2j_exclusive_shared):,}")
    print(f"    event2journey train:    {len(e2j_final_uids):,}")
    print(f"    profile2journey train:  {len(p2j_final_uids):,}")
    print(f"    Journey total:          {total_journey:,}")

    # --- Build SFT training samples ---
    e2j_train = [
        extract_sft_fields(e2j_by_user[uid])
        for uid in e2j_final_uids if uid in e2j_by_user
    ]
    p2j_train = [
        extract_sft_fields(p2j_by_user[uid])
        for uid in p2j_final_uids if uid in p2j_by_user
    ]

    stats["event2journey"] = (len(e2j_final_uids), len(e2j_train))
    stats["profile2journey"] = (len(p2j_final_uids), len(p2j_train))
    all_training.extend(e2j_train)
    all_training.extend(p2j_train)

    # =========================================================================
    # 3. Shuffle
    # =========================================================================
    print()
    print("=" * 70)
    print("3. Shuffling merged dataset")
    print("=" * 70)

    random.shuffle(all_training)
    print(f"  Total training samples: {len(all_training):,} (seed={args.seed})")

    # =========================================================================
    # 4. Save
    # =========================================================================
    print()
    print("=" * 70)
    print("4. Saving training data and test sets")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Training data ---
    train_file = os.path.join(args.output_dir, "combined_sft.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in all_training:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    train_size = os.path.getsize(train_file)
    size_str = (f"{train_size / (1024**3):.2f} GB" if train_size > 1024**3
                else f"{train_size / (1024**2):.2f} MB")
    print(f"  Training: {train_file}")
    print(f"    Samples: {len(all_training):,}  Size: {size_str}")

    # --- meta2tid test ---
    meta_test_file = os.path.join(args.output_dir, "meta2tid_test.jsonl")
    with open(meta_test_file, "w", encoding="utf-8") as f:
        for item in meta_test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  meta2tid test: {meta_test_file}  ({len(meta_test):,})")

    # --- event2journey test ---
    e2j_test_file = os.path.join(args.output_dir, "event2journey_test.jsonl")
    with open(e2j_test_file, "w", encoding="utf-8") as f:
        for item in e2j_test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  event2journey test: {e2j_test_file}  ({len(e2j_test):,})")

    # --- profile2journey test ---
    p2j_test_file = os.path.join(args.output_dir, "profile2journey_test.jsonl")
    with open(p2j_test_file, "w", encoding="utf-8") as f:
        for item in p2j_test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  profile2journey test: {p2j_test_file}  ({len(p2j_test):,})")

    # =========================================================================
    # 5. Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\n  {'Task':<25s} {'Pool':>10s} {'Train':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for name in ["meta2tid", "event2journey", "profile2journey"]:
        pool, train = stats.get(name, (0, 0))
        print(f"  {name:<25s} {pool:>10,} {train:>10,}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    total_pool = sum(v[0] for v in stats.values())
    print(f"  {'TOTAL':<25s} {total_pool:>10,} {len(all_training):>10,}")

    print(f"\n  Test Sets:")
    print(f"    meta2tid:          {len(meta_test):>10,}")
    print(f"    event2journey:     {len(e2j_test):>10,}")
    print(f"    profile2journey:   {len(p2j_test):>10,}")

    # =========================================================================
    # 6. Length Statistics
    # =========================================================================
    if all_training:
        print()
        print("=" * 70)
        print("Length Statistics (character count)")
        print("=" * 70)

        input_lens = sorted(
            len(d.get("instruction", "") + d.get("input", ""))
            for d in all_training
        )
        output_lens = sorted(len(d.get("output", "")) for d in all_training)
        n = len(input_lens)

        for label, lens in [("instruction + input", input_lens),
                            ("output", output_lens)]:
            print(f"\n  {label}:")
            print(f"    Count:  {n:>10,}")
            print(f"    Min:    {lens[0]:>10,}")
            print(f"    Max:    {lens[-1]:>10,}")
            print(f"    Mean:   {sum(lens) / n:>10,.1f}")
            print(f"    Median: {percentile(lens, 50):>10,}")
            print(f"    P25:    {percentile(lens, 25):>10,}")
            print(f"    P75:    {percentile(lens, 75):>10,}")
            print(f"    P95:    {percentile(lens, 95):>10,}")
            print(f"    P99:    {percentile(lens, 99):>10,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
