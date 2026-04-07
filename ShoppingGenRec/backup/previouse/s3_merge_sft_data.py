"""Step 3: Merge and Shuffle All SFT Datasets + Build Test Sets

Reads SFT JSON files from multiple tasks (meta2tid, rec, event2product,
event2journey, profile2journey), applies per-task sampling, merges them
into a single dataset, and shuffles randomly.

For task pairs that share the same underlying users (rec & event2product,
event2journey & profile2journey), this script loads the *_full.json
versions (which contain metadata.user_id), randomly splits shared user
IDs into two disjoint halves, assigns one half to each task, then
extracts only the SFT fields before sampling.

Additionally, samples N user_ids from the intersection of event2journey
and profile2journey to build two test JSONL files (one per journey task).
Test users are excluded from the training data.

Outputs (in --output_dir):
  1. combined_sft.jsonl          - Merged training data (no test users).
  2. event2journey_test.jsonl    - Test set for event2journey task.
  3. profile2journey_test.jsonl  - Test set for profile2journey task.

Usage:
    python s3_merge_sft_data.py \\
        --meta2tid_file ./sft_data/meta2tid_sft.json \\
        --rec_full_file ./sft_data/rec_sft_full.json \\
        --event2product_full_file ./sft_data/event2product_sft_full.json \\
        --event2journey_full_file ./sft_data/event2journey_sft_full.json \\
        --profile2journey_full_file ./sft_data/profile2journey_sft_full.json \\
        --output_dir ./sft_data \\
        --test_sample_n 5000
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


def extract_sft_fields(sample):
    """Extract only instruction/input/output from a full sample dict."""
    return {
        "instruction": sample["instruction"],
        "input": sample["input"],
        "output": sample["output"],
    }


def load_and_split_paired_tasks(
    task_a_full_file,
    task_b_full_file,
    task_a_prob,
    task_b_prob,
    task_a_name,
    task_b_name,
    rng,
):
    """Load two paired full-data files, deduplicate shared users by
    splitting them randomly, then sample and return SFT-ready data.

    Both tasks may share the same underlying users. To avoid duplicating
    users across both tasks, shared user IDs are randomly split into two
    disjoint halves — one half assigned to each task. Users that appear
    in only one task are kept as-is.

    Args:
        task_a_full_file: Path to task A's *_full.json.
        task_b_full_file: Path to task B's *_full.json.
        task_a_prob: Sampling probability for task A.
        task_b_prob: Sampling probability for task B.
        task_a_name: Display name for task A (for logging).
        task_b_name: Display name for task B (for logging).
        rng: random.Random instance for reproducibility.

    Returns:
        Tuple of (task_a_samples, task_b_samples,
                  task_a_stats, task_b_stats)
        where each stats is (original_count, sampled_count).
    """
    # Load both full files
    a_by_user = {}
    if task_a_full_file and os.path.exists(task_a_full_file):
        with open(task_a_full_file, "r", encoding="utf-8") as f:
            a_data = json.load(f)
        for sample in a_data:
            uid = sample.get("metadata", {}).get("user_id", "")
            if uid:
                a_by_user[uid] = sample
        print(f"  [{task_a_name}] Loaded full data: {len(a_data):,} samples, "
              f"{len(a_by_user):,} unique users")
    else:
        print(f"  [{task_a_name}] File not found: {task_a_full_file}")

    b_by_user = {}
    if task_b_full_file and os.path.exists(task_b_full_file):
        with open(task_b_full_file, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        for sample in b_data:
            uid = sample.get("metadata", {}).get("user_id", "")
            if uid:
                b_by_user[uid] = sample
        print(f"  [{task_b_name}] Loaded full data: {len(b_data):,} samples, "
              f"{len(b_by_user):,} unique users")
    else:
        print(f"  [{task_b_name}] File not found: {task_b_full_file}")

    # Find shared users (present in both tasks)
    shared_uids = set(a_by_user.keys()) & set(b_by_user.keys())
    a_only_uids = set(a_by_user.keys()) - shared_uids
    b_only_uids = set(b_by_user.keys()) - shared_uids

    pair_label = f"{task_a_name}/{task_b_name}"
    print(f"  [{pair_label} dedup] Shared users: {len(shared_uids):,}, "
          f"{task_a_name}-only: {len(a_only_uids):,}, "
          f"{task_b_name}-only: {len(b_only_uids):,}")

    # Randomly split shared users into two halves
    shared_list = sorted(shared_uids)  # sort for reproducibility
    rng.shuffle(shared_list)
    mid = len(shared_list) // 2
    a_shared = set(shared_list[:mid])
    b_shared = set(shared_list[mid:])

    print(f"  [{pair_label} dedup] Split shared users: "
          f"{len(a_shared):,} -> {task_a_name}, "
          f"{len(b_shared):,} -> {task_b_name}")

    # Build final user sets
    a_final_uids = a_only_uids | a_shared
    b_final_uids = b_only_uids | b_shared

    # Extract SFT samples for assigned users
    a_assigned = [
        extract_sft_fields(a_by_user[uid])
        for uid in a_final_uids if uid in a_by_user
    ]
    b_assigned = [
        extract_sft_fields(b_by_user[uid])
        for uid in b_final_uids if uid in b_by_user
    ]

    # Apply sampling probability
    a_original = len(a_assigned)
    if task_a_prob < 1.0:
        a_assigned = [d for d in a_assigned if rng.random() < task_a_prob]
    b_original = len(b_assigned)
    if task_b_prob < 1.0:
        b_assigned = [d for d in b_assigned if rng.random() < task_b_prob]

    print(f"  [{task_a_name}] After dedup: {a_original:>10,} -> "
          f"Sampled {len(a_assigned):>10,} (prob={task_a_prob})")
    print(f"  [{task_b_name}] After dedup: {b_original:>10,} -> "
          f"Sampled {len(b_assigned):>10,} (prob={task_b_prob})")

    return (
        a_assigned,
        b_assigned,
        (a_original, len(a_assigned)),
        (b_original, len(b_assigned)),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge and shuffle multiple SFT datasets into one"
    )
    # Input files
    parser.add_argument(
        "--meta2tid_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/sft_data/meta2tid_sft.json",
        help="Path to meta2tid SFT data",
    )
    parser.add_argument(
        "--rec_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/sft_data/rec_sft_full.json",
        help="Path to rec full SFT data (with metadata)",
    )
    parser.add_argument(
        "--event2product_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/sft_data/event2product_sft_full.json",
        help="Path to event2product full SFT data (with metadata)",
    )
    parser.add_argument(
        "--event2journey_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/sft_data/event2journey_sft_full.json",
        help="Path to event2journey full SFT data (with metadata)",
    )
    parser.add_argument(
        "--profile2journey_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/sft_data/profile2journey_sft_full.json",
        help="Path to profile2journey full SFT data (with metadata)",
    )
    # Sampling probabilities
    parser.add_argument(
        "--meta2tid_prob", type=float, default=0.05,
        help="Sampling probability for meta2tid data",
    )
    parser.add_argument(
        "--rec_prob", type=float, default=1.0,
        help="Sampling probability for rec data",
    )
    parser.add_argument(
        "--event2product_prob", type=float, default=1.0,
        help="Sampling probability for event2product data",
    )
    parser.add_argument(
        "--event2journey_prob", type=float, default=1.0,
        help="Sampling probability for event2journey data",
    )
    parser.add_argument(
        "--profile2journey_prob", type=float, default=1.0,
        help="Sampling probability for profile2journey data",
    )
    # Output
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/sft_data",
        help="Output directory for merged training data and test sets",
    )
    # Test set
    parser.add_argument(
        "--test_sample_n", type=int, default=5000,
        help="Number of user_ids to sample as test set from the intersection "
             "of event2journey and profile2journey (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling and shuffling",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    # =========================================================================
    # Step 1: Load and sample independent tasks
    # =========================================================================
    print("=" * 70)
    print("Step 1: Loading and sampling SFT datasets")
    print("=" * 70)

    all_data = []
    stats = {}

    # Independent tasks (no dedup needed)
    independent_tasks = [
        (args.meta2tid_file, args.meta2tid_prob, "meta2tid"),
    ]

    for file_path, prob, name in independent_tasks:
        sampled, original = load_and_sample(file_path, prob, name)
        stats[name] = (original, len(sampled))
        all_data.extend(sampled)

    # =========================================================================
    # Step 1b: Load, deduplicate, and sample rec & event2product
    # =========================================================================
    print()
    print("  --- rec / event2product deduplication ---")
    rec_samples, e2p_samples, rec_stats, e2p_stats = load_and_split_paired_tasks(
        args.rec_full_file,
        args.event2product_full_file,
        args.rec_prob,
        args.event2product_prob,
        "rec",
        "event2product",
        rng,
    )
    stats["rec"] = rec_stats
    stats["event2product"] = e2p_stats
    all_data.extend(rec_samples)
    all_data.extend(e2p_samples)

    # =========================================================================
    # Step 1c: Load journey task data, extract test set, then dedup & sample
    # =========================================================================
    print()
    print("  --- event2journey / profile2journey: test set + deduplication ---")

    # Load both full files first to find shared users for test set
    e2j_by_user = {}
    if args.event2journey_full_file and os.path.exists(args.event2journey_full_file):
        with open(args.event2journey_full_file, "r", encoding="utf-8") as f:
            e2j_data = json.load(f)
        for sample in e2j_data:
            uid = sample.get("metadata", {}).get("user_id", "")
            if uid:
                e2j_by_user[uid] = sample
        print(f"  [event2journey] Loaded full data: {len(e2j_data):,} samples, "
              f"{len(e2j_by_user):,} unique users")
    else:
        print(f"  [event2journey] File not found: {args.event2journey_full_file}")

    p2j_by_user = {}
    if args.profile2journey_full_file and os.path.exists(args.profile2journey_full_file):
        with open(args.profile2journey_full_file, "r", encoding="utf-8") as f:
            p2j_data = json.load(f)
        for sample in p2j_data:
            uid = sample.get("metadata", {}).get("user_id", "")
            if uid:
                p2j_by_user[uid] = sample
        print(f"  [profile2journey] Loaded full data: {len(p2j_data):,} samples, "
              f"{len(p2j_by_user):,} unique users")
    else:
        print(f"  [profile2journey] File not found: {args.profile2journey_full_file}")

    # Find users present in BOTH journey tasks for test set sampling
    both_journey_uids = set(e2j_by_user.keys()) & set(p2j_by_user.keys())
    print(f"  Users in both journey tasks: {len(both_journey_uids):,}")

    # Sample test user_ids
    test_n = min(args.test_sample_n, len(both_journey_uids))
    both_sorted = sorted(both_journey_uids)
    rng.shuffle(both_sorted)
    test_uids = set(both_sorted[:test_n])
    print(f"  Sampled {len(test_uids):,} test user_ids from {len(both_journey_uids):,} shared users")

    # Build test sets (full metadata preserved)
    e2j_test = [e2j_by_user[uid] for uid in sorted(test_uids) if uid in e2j_by_user]
    p2j_test = [p2j_by_user[uid] for uid in sorted(test_uids) if uid in p2j_by_user]
    print(f"  Test set: {len(e2j_test):,} event2journey, {len(p2j_test):,} profile2journey")

    # Remove test users from journey data before dedup/training split
    e2j_by_user_train = {uid: s for uid, s in e2j_by_user.items() if uid not in test_uids}
    p2j_by_user_train = {uid: s for uid, s in p2j_by_user.items() if uid not in test_uids}
    print(f"  After removing test users: {len(e2j_by_user_train):,} event2journey, "
          f"{len(p2j_by_user_train):,} profile2journey")

    # Dedup shared training users between the two journey tasks
    shared_train = set(e2j_by_user_train.keys()) & set(p2j_by_user_train.keys())
    e2j_only = set(e2j_by_user_train.keys()) - shared_train
    p2j_only = set(p2j_by_user_train.keys()) - shared_train

    shared_train_list = sorted(shared_train)
    rng.shuffle(shared_train_list)
    mid = len(shared_train_list) // 2
    e2j_shared = set(shared_train_list[:mid])
    p2j_shared = set(shared_train_list[mid:])

    print(f"  [dedup] Shared train users: {len(shared_train):,}, "
          f"split {len(e2j_shared):,} -> event2journey, "
          f"{len(p2j_shared):,} -> profile2journey")

    e2j_final_uids = e2j_only | e2j_shared
    p2j_final_uids = p2j_only | p2j_shared

    e2j_assigned = [
        extract_sft_fields(e2j_by_user_train[uid])
        for uid in e2j_final_uids if uid in e2j_by_user_train
    ]
    p2j_assigned = [
        extract_sft_fields(p2j_by_user_train[uid])
        for uid in p2j_final_uids if uid in p2j_by_user_train
    ]

    e2j_orig = len(e2j_assigned)
    if args.event2journey_prob < 1.0:
        e2j_assigned = [d for d in e2j_assigned if rng.random() < args.event2journey_prob]
    p2j_orig = len(p2j_assigned)
    if args.profile2journey_prob < 1.0:
        p2j_assigned = [d for d in p2j_assigned if rng.random() < args.profile2journey_prob]

    print(f"  [event2journey] After dedup: {e2j_orig:>10,} -> "
          f"Sampled {len(e2j_assigned):>10,} (prob={args.event2journey_prob})")
    print(f"  [profile2journey] After dedup: {p2j_orig:>10,} -> "
          f"Sampled {len(p2j_assigned):>10,} (prob={args.profile2journey_prob})")

    stats["event2journey"] = (e2j_orig, len(e2j_assigned))
    stats["profile2journey"] = (p2j_orig, len(p2j_assigned))
    all_data.extend(e2j_assigned)
    all_data.extend(p2j_assigned)

    total_before_merge = sum(v[1] for v in stats.values())
    print(f"\n  Total training samples after sampling: {total_before_merge:,}")

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
    # Step 3: Save training data + test sets
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Saving training data and test sets")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Training data ---
    train_file = os.path.join(args.output_dir, "combined_sft.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    train_size_mb = os.path.getsize(train_file) / (1024 * 1024)
    print(f"  Training data: {train_file}")
    print(f"    Size:   {train_size_mb:.2f} MB")
    print(f"    Total:  {len(all_data):,} samples")

    # --- Test sets (full metadata preserved) ---
    e2j_test_file = os.path.join(args.output_dir, "event2journey_test.jsonl")
    with open(e2j_test_file, "w", encoding="utf-8") as f:
        for item in e2j_test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  event2journey test: {e2j_test_file}")
    print(f"    Samples: {len(e2j_test):,}")

    p2j_test_file = os.path.join(args.output_dir, "profile2journey_test.jsonl")
    with open(p2j_test_file, "w", encoding="utf-8") as f:
        for item in p2j_test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  profile2journey test: {p2j_test_file}")
    print(f"    Samples: {len(p2j_test):,}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  {'Task':<25s} {'Original':>10s} {'Sampled':>10s} {'Ratio':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for name in ["meta2tid", "rec", "event2product",
                  "event2journey", "profile2journey"]:
        orig, samp = stats.get(name, (0, 0))
        ratio = f"{samp/orig*100:.1f}%" if orig > 0 else "N/A"
        print(f"  {name:<25s} {orig:>10,} {samp:>10,} {ratio:>8s}")
    total_orig = sum(v[0] for v in stats.values())
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'TOTAL':<25s} {total_orig:>10,} {len(all_data):>10,}")

    # Test set summary
    print(f"\n  Test Set:")
    print(f"    Test user_ids sampled:    {len(test_uids):>10,}")
    print(f"    event2journey test:       {len(e2j_test):>10,}")
    print(f"    profile2journey test:     {len(p2j_test):>10,}")

    # =========================================================================
    # Length Statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Length Statistics (character count)")
    print("=" * 70)

    if all_data:
        input_lens = [len(d.get("instruction", "") + d.get("input", "")) for d in all_data]
        output_lens = [len(d.get("output", "")) for d in all_data]

        input_lens_sorted = sorted(input_lens)
        output_lens_sorted = sorted(output_lens)
        n = len(input_lens)

        def percentile(sorted_list, p):
            idx = int(len(sorted_list) * p / 100)
            return sorted_list[min(idx, len(sorted_list) - 1)]

        print(f"\n  instruction + input:")
        print(f"    Count:  {n:>10,}")
        print(f"    Min:    {input_lens_sorted[0]:>10,}")
        print(f"    Max:    {input_lens_sorted[-1]:>10,}")
        print(f"    Mean:   {sum(input_lens) / n:>10,.1f}")
        print(f"    Median: {percentile(input_lens_sorted, 50):>10,}")
        print(f"    P25:    {percentile(input_lens_sorted, 25):>10,}")
        print(f"    P75:    {percentile(input_lens_sorted, 75):>10,}")
        print(f"    P95:    {percentile(input_lens_sorted, 95):>10,}")
        print(f"    P99:    {percentile(input_lens_sorted, 99):>10,}")

        print(f"\n  output:")
        print(f"    Count:  {n:>10,}")
        print(f"    Min:    {output_lens_sorted[0]:>10,}")
        print(f"    Max:    {output_lens_sorted[-1]:>10,}")
        print(f"    Mean:   {sum(output_lens) / n:>10,.1f}")
        print(f"    Median: {percentile(output_lens_sorted, 50):>10,}")
        print(f"    P25:    {percentile(output_lens_sorted, 25):>10,}")
        print(f"    P75:    {percentile(output_lens_sorted, 75):>10,}")
        print(f"    P95:    {percentile(output_lens_sorted, 95):>10,}")
        print(f"    P99:    {percentile(output_lens_sorted, 99):>10,}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
