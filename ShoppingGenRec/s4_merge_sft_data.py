"""Step 4: Merge and Shuffle All SFT Datasets + Build Test Sets

Three tasks: meta2tid, event2journey, profile2journey.
Any task can be skipped by leaving its file path empty or pointing to
a non-existent file.

Pipeline per journey task:
  1. Load full JSON; group by user_id.
  2. Random-sample test_sample_n users as test set.
  3. Apply per-bucket balance cap on remaining users.
  4. Sample from capped pool to reach journey_target_total.

For meta2tid:
  - Loads meta2tid_sft_full.json (with metadata.GlobalOfferId).
  - test_sample_n items sampled as test set.
  - Remaining items sampled by --meta2tid_prob, capped at --meta2tid_max_train.

Outputs (in --output_dir):
  1. combined_sft.jsonl          - Merged training data (shuffled).
  2. meta2tid_test.jsonl         - Test set (instruction/input/output + GlobalOfferId).
  3. event2journey_test.jsonl    - Test set (instruction/input/output + UserId).
  4. profile2journey_test.jsonl  - Test set (instruction/input/output + UserId).
  5. *_full_cleaned_test.tsv     - Test TSV files (if source TSV exists).

Usage:
    python s4_merge_sft_data.py \\
        --meta2tid_full_file ./sft_data/meta2tid_sft_full.json \\
        --event2journey_full_file ./sft_data/event2journey_sft_full.json \\
        --profile2journey_full_file ./sft_data/profile2journey_sft_full.json \\
        --journey_target_total 500000 \\
        --test_sample_n 1000
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict


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
    if not sorted_list:
        return 0
    idx = int(len(sorted_list) * p / 100)
    return sorted_list[min(idx, len(sorted_list) - 1)]


def _load_full_json(path, label):
    """Load a *_full.json file. Returns list or empty list if missing."""
    if not path or not path.strip() or not os.path.exists(path):
        print(f"  [{label}] File not found or empty path: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  [{label}] Loaded {len(data):,} samples from {path}")
    return data


def balance_by_journey_count(user_samples, rng, balance_cap):
    """Balance users by capping each journey-count bucket.

    Groups users by num_journeys, randomly samples at most balance_cap
    from each bucket.

    Args:
        user_samples: Dict mapping user_id -> full sample dict.
        rng: Random instance.
        balance_cap: Max samples per journey-count bucket.

    Returns:
        Tuple of (kept_user_ids_set, stats_str).
    """
    buckets = defaultdict(list)
    for uid, sample in user_samples.items():
        n_j = sample.get("metadata", {}).get("num_journeys", 0)
        buckets[n_j].append(uid)

    kept = set()
    print(f"    Balance cap = {balance_cap:,} per bucket:")
    for cnt in sorted(buckets.keys()):
        uids = buckets[cnt]
        if len(uids) > balance_cap:
            sampled = rng.sample(uids, balance_cap)
            print(f"      {cnt:>2} journeys: {len(uids):>8,} -> {balance_cap:>8,} (capped)")
        else:
            sampled = uids
            print(f"      {cnt:>2} journeys: {len(uids):>8,} -> {len(uids):>8,}")
        kept.update(sampled)

    print(f"    Total after balance: {len(kept):,}")
    return kept


# =============================================================================
# Test TSV builder
# =============================================================================

def _read_test_uids(jsonl_path):
    """Read UserId values from a test JSONL file."""
    uids = set()
    if not os.path.exists(jsonl_path):
        return uids
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            uid = d.get("UserId", "").strip()
            if uid:
                uids.add(uid)
    return uids


def _filter_tsv_by_uids(tsv_path, uids, out_path):
    """Read a TSV, keep only rows whose UserId is in uids, write to out_path.

    Returns (total_rows, matched_rows).
    """
    total = 0
    matched = 0
    remaining = set(uids)

    with open(tsv_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        header_line = fin.readline()
        if not header_line:
            return 0, 0
        fout.write(header_line)

        for line in fin:
            total += 1
            tab_pos = line.find("\t")
            uid = line[:tab_pos].strip() if tab_pos > 0 else line.strip()
            if uid in remaining:
                fout.write(line)
                matched += 1
                remaining.discard(uid)
                if not remaining:
                    break
            if total % 100_000 == 0:
                print(f"      Scanned {total:>10,} rows, "
                      f"matched {matched:,}/{len(uids):,} ...")

    return total, matched


def build_test_tsv(output_dir, test_output_dir, tasks_tsv_map):
    """Build *_full_cleaned_test.tsv for evaluation.

    Args:
        output_dir: Directory containing *_test.jsonl files.
        test_output_dir: Directory to save test TSV files.
        tasks_tsv_map: Dict of task_name -> source_tsv_path.
    """
    os.makedirs(test_output_dir, exist_ok=True)

    for task_name, full_tsv in tasks_tsv_map.items():
        test_jsonl = os.path.join(output_dir, f"{task_name}_test.jsonl")
        print(f"\n  [{task_name}]")
        if not os.path.exists(test_jsonl):
            print(f"    SKIP: test JSONL not found: {test_jsonl}")
            continue
        if not full_tsv or not full_tsv.strip() or not os.path.exists(full_tsv):
            print(f"    SKIP: source TSV not found: {full_tsv}")
            continue

        uids = _read_test_uids(test_jsonl)
        print(f"    Test users from JSONL: {len(uids):,}")

        out_path = os.path.join(
            test_output_dir, f"{task_name}_full_cleaned_test.tsv")
        total, matched = _filter_tsv_by_uids(full_tsv, uids, out_path)

        file_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"    TSV rows scanned: {total:,}")
        print(f"    Matched:          {matched:,}")
        print(f"    Missing:          {len(uids) - matched:,}")
        print(f"    Output: {out_path} ({file_mb:.1f} MB)")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge and shuffle SFT datasets (meta2tid + journey "
                    "tasks) into one training file, plus test files."
    )

    # --- Input files (empty string = skip this task) ---
    parser.add_argument(
        "--meta2tid_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260424/sft_data/"
                "meta2tid_sft_full.json",
        help="Path to meta2tid *_full.json. Empty or missing = skip.",
    )
    parser.add_argument(
        "--event2journey_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260424/sft_data/"
                "event2journey_sft_full.json",
        help="Path to event2journey *_full.json. Empty or missing = skip.",
    )
    parser.add_argument(
        "--profile2journey_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260424/sft_data/"
                "profile2journey_sft_full.json",
        help="Path to profile2journey *_full.json. Empty or missing = skip.",
    )

    # --- Sampling ---
    parser.add_argument(
        "--meta2tid_prob", type=float, default=0.5,
        help="Sampling probability for meta2tid training data (default: 0.5)",
    )
    parser.add_argument(
        "--meta2tid_max_train", type=int, default=500000,
        help="Maximum number of meta2tid training samples (default: 500000)",
    )
    parser.add_argument(
        "--journey_target_total", type=int, default=500000,
        help="Target total for event2journey + profile2journey combined "
             "(default: 500000). Split proportionally across available tasks.",
    )
    parser.add_argument(
        "--journey_balance_cap", type=int, default=20000,
        help="Max samples per journey-count bucket for balancing "
             "(default: 20000)",
    )
    parser.add_argument(
        "--test_sample_n", type=int, default=1000,
        help="Number of test samples per task (default: 1000)",
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260424/sft_data",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=43,
        help="Random seed (default: 43)",
    )

    # --- Test TSV sources ---
    parser.add_argument(
        "--test_output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260424/test_data",
        help="Directory to save test TSV files.",
    )
    parser.add_argument(
        "--event2journey_tsv", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260424/raw_data/"
                "event2journey_full_cleaned.tsv",
        help="Path to event2journey source TSV. Empty or missing = skip.",
    )
    parser.add_argument(
        "--profile2journey_tsv", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260424/raw_data/"
                "profile2journey_full_cleaned.tsv",
        help="Path to profile2journey source TSV. Empty or missing = skip.",
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
    # 1. meta2tid: load, test split, sample training
    # =========================================================================
    print("=" * 70)
    print("1. meta2tid")
    print("=" * 70)

    meta_full_data = _load_full_json(args.meta2tid_full_file, "meta2tid")
    meta_by_gid = {}
    for sample in meta_full_data:
        gid = sample.get("metadata", {}).get("GlobalOfferId", "")
        if gid:
            meta_by_gid[gid] = sample

    if meta_by_gid:
        meta_gids = list(meta_by_gid.keys())
        rng.shuffle(meta_gids)
        test_n = min(args.test_sample_n, len(meta_gids))
        meta_test_gids = meta_gids[:test_n]
        meta_train_gids = meta_gids[test_n:]

        # Test set
        meta_test = []
        for gid in meta_test_gids:
            entry = extract_sft_fields(meta_by_gid[gid])
            entry["GlobalOfferId"] = gid
            meta_test.append(entry)

        # Training: sample + cap
        meta_train = []
        for gid in meta_train_gids:
            if rng.random() < args.meta2tid_prob:
                meta_train.append(extract_sft_fields(meta_by_gid[gid]))
        if args.meta2tid_max_train and len(meta_train) > args.meta2tid_max_train:
            rng.shuffle(meta_train)
            meta_train = meta_train[:args.meta2tid_max_train]

        stats["meta2tid"] = (len(meta_by_gid), len(meta_train))
        all_training.extend(meta_train)
        print(f"  Unique items:  {len(meta_by_gid):,}")
        print(f"  Test:          {len(meta_test):,}")
        print(f"  Train:         {len(meta_train_gids):,} -> sampled {len(meta_train):,} "
              f"(prob={args.meta2tid_prob}, cap={args.meta2tid_max_train:,})")
    else:
        meta_test = []
        print(f"  SKIP: no data loaded")

    # =========================================================================
    # 2. Journey tasks: load, test split, balance cap, target sampling
    # =========================================================================
    print()
    print("=" * 70)
    print("2. Journey tasks")
    print("=" * 70)

    # --- Load both full files, group by user_id ---
    journey_tasks = {}  # task_name -> {uid: sample}

    for task_name, full_file in [
        ("event2journey", args.event2journey_full_file),
        ("profile2journey", args.profile2journey_full_file),
    ]:
        data = _load_full_json(full_file, task_name)
        if not data:
            continue
        by_user = {}
        for sample in data:
            uid = sample.get("metadata", {}).get("user_id", "")
            if uid:
                by_user[uid] = sample
        print(f"    {task_name}: {len(by_user):,} unique users")
        journey_tasks[task_name] = by_user

    if not journey_tasks:
        print(f"  No journey tasks loaded. Skipping journey section.")
    else:
        # --- Per-task: test split -> balance cap -> collect training pool ---
        journey_test_sets = {}  # task_name -> test list
        journey_train_pools = {}  # task_name -> list of sft dicts

        for task_name, by_user in journey_tasks.items():
            print(f"\n  [{task_name}]")

            # Step 1: Random sample test users
            all_uids = list(by_user.keys())
            rng.shuffle(all_uids)
            test_n = min(args.test_sample_n, len(all_uids))
            test_uids = set(all_uids[:test_n])

            test_samples = []
            for uid in sorted(test_uids):
                entry = extract_sft_fields(by_user[uid])
                entry["UserId"] = uid
                test_samples.append(entry)
            journey_test_sets[task_name] = test_samples
            print(f"    Test users: {len(test_samples):,}")

            # Step 2: Remove test users, then balance cap
            train_candidates = {
                uid: sample for uid, sample in by_user.items()
                if uid not in test_uids
            }
            print(f"    Train candidates (before balance): {len(train_candidates):,}")

            kept_uids = balance_by_journey_count(
                train_candidates, rng, args.journey_balance_cap,
            )

            # Build training pool from kept uids
            pool = []
            for uid in kept_uids:
                pool.append(extract_sft_fields(train_candidates[uid]))
            journey_train_pools[task_name] = pool
            print(f"    Train pool (after balance): {len(pool):,}")

        # --- Distribute journey_target_total across available tasks ---
        target = args.journey_target_total
        total_pool = sum(len(p) for p in journey_train_pools.values())
        print(f"\n  Journey target total: {target:,}")
        print(f"  Total pool available: {total_pool:,}")

        journey_train = {}
        if total_pool <= target:
            # Use everything
            for task_name, pool in journey_train_pools.items():
                journey_train[task_name] = pool
                print(f"    [{task_name}] Using all {len(pool):,} samples")
        else:
            # Proportional sampling to reach target
            for task_name, pool in journey_train_pools.items():
                task_target = int(target * len(pool) / total_pool)
                task_target = min(task_target, len(pool))
                rng.shuffle(pool)
                sampled = pool[:task_target]
                journey_train[task_name] = sampled
                print(f"    [{task_name}] {len(pool):,} -> sampled {len(sampled):,}")

        combined_journey = sum(len(v) for v in journey_train.values())
        print(f"    Combined journey train: {combined_journey:,} "
              f"(target: {target:,}, diff: {combined_journey - target:+,})")

        # Add to all_training and stats
        for task_name, train_list in journey_train.items():
            stats[task_name] = (
                len(journey_train_pools[task_name]),
                len(train_list),
            )
            all_training.extend(train_list)

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
    if meta_test:
        with open(meta_test_file, "w", encoding="utf-8") as f:
            for item in meta_test:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  meta2tid test: {meta_test_file}  ({len(meta_test):,})")

    # --- Journey test sets ---
    if journey_tasks:
        for task_name in journey_tasks:
            test_list = journey_test_sets.get(task_name, [])
            if not test_list:
                continue
            test_file = os.path.join(args.output_dir, f"{task_name}_test.jsonl")
            with open(test_file, "w", encoding="utf-8") as f:
                for item in test_list:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  {task_name} test: {test_file}  ({len(test_list):,})")

    # =========================================================================
    # 5. Build test TSV (automatically, skip if source TSV missing)
    # =========================================================================
    print()
    print("=" * 70)
    print("5. Building test TSV files")
    print("=" * 70)

    tasks_tsv_map = {
        "event2journey": args.event2journey_tsv,
        "profile2journey": args.profile2journey_tsv,
    }
    build_test_tsv(args.output_dir, args.test_output_dir, tasks_tsv_map)

    # =========================================================================
    # 6. Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\n  {'Task':<25s} {'Pool':>10s} {'Train':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for name in ["meta2tid", "event2journey", "profile2journey"]:
        pool, train = stats.get(name, (0, 0))
        if pool == 0 and train == 0:
            print(f"  {name:<25s} {'(skipped)':>10s} {'':>10s}")
        else:
            print(f"  {name:<25s} {pool:>10,} {train:>10,}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    total_pool = sum(v[0] for v in stats.values())
    print(f"  {'TOTAL':<25s} {total_pool:>10,} {len(all_training):>10,}")

    print(f"\n  Test Sets:")
    print(f"    meta2tid:          {len(meta_test):>10,}")
    if journey_tasks:
        for task_name in journey_tasks:
            test_list = journey_test_sets.get(task_name, [])
            print(f"    {task_name}:     {len(test_list):>10,}")

    # =========================================================================
    # 7. Length Statistics
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
