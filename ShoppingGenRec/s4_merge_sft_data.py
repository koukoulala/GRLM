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
    idx = int(len(sorted_list) * p / 100)
    return sorted_list[min(idx, len(sorted_list) - 1)]


def count_journey_buckets(user_samples_dict, task_uids, keep_threshold):
    """Count users by journey-count bucket for a single task.

    Returns:
        Tuple of (n_high, low_bucket_counts) where
        n_high is the number of users with >= keep_threshold journeys,
        low_bucket_counts is a dict mapping journey_count -> user_count
        for users with < keep_threshold journeys.
    """
    n_high = 0
    low_buckets = defaultdict(int)
    for uid in task_uids:
        if uid not in user_samples_dict:
            continue
        num_j = user_samples_dict[uid].get("metadata", {}).get("num_journeys", 0)
        if num_j >= keep_threshold:
            n_high += 1
        else:
            low_buckets[num_j] += 1
    return n_high, dict(low_buckets)


def compute_uniform_bucket_probs(n_high_total, low_buckets_list,
                                 keep_threshold, target_total):
    """Compute per-bucket sampling probs for uniform kept-count distribution.

    Goal: each low-journey bucket contributes the same number of kept samples.
    For bucket j with count_j users: prob_j = C / count_j, capped at 1.0.
    C is chosen so that n_high_total + sum(kept) ≈ target_total.

    Uses iterative saturation: buckets too small to downsample (count_j <= C)
    are saturated (kept 100%), and remaining quota is redistributed.

    Returns:
        Dict mapping journey_count -> sampling probability.
    """
    merged = defaultdict(int)
    for low_buckets in low_buckets_list:
        for j, n_j in low_buckets.items():
            merged[j] += n_j

    if not merged or target_total <= n_high_total:
        return {j: 1.0 for j in merged}

    needed = target_total - n_high_total
    saturated_count = 0
    unsaturated = dict(merged)
    saturated_probs = {}  # j -> 1.0 for saturated buckets

    for _ in range(keep_threshold + 2):
        n_buckets = len(unsaturated)
        if n_buckets == 0:
            break

        remaining_needed = needed - saturated_count
        if remaining_needed <= 0:
            result = {j: 0.0 for j in unsaturated}
            result.update(saturated_probs)
            return result

        # Each unsaturated bucket should keep C samples
        C = remaining_needed / n_buckets

        # Saturate buckets where count_j <= C (can't downsample, keep all)
        newly_saturated = [j for j, count_j in unsaturated.items()
                           if count_j <= C]

        if not newly_saturated:
            # Done — compute final probs
            result = {}
            for j, count_j in unsaturated.items():
                result[j] = min(C / count_j, 1.0)
            result.update(saturated_probs)
            return result

        for j in newly_saturated:
            saturated_count += unsaturated.pop(j)
            saturated_probs[j] = 1.0

    # Fallback: all saturated
    result = {j: 1.0 for j in merged}
    return result


def sample_by_journey_count(user_samples_dict, task_uids, rng,
                            keep_threshold=5, bucket_probs=None):
    """Sample users based on per-bucket probabilities for uniform distribution.

    Users with num_journeys >= keep_threshold are always kept.
    Users with fewer journeys are kept with the probability from bucket_probs.

    Args:
        user_samples_dict: Dict mapping user_id -> full sample dict.
        task_uids: Set/list of user IDs to consider.
        rng: Random instance.
        keep_threshold: Journey count threshold for guaranteed inclusion.
        bucket_probs: Dict mapping journey_count -> sampling probability.

    Returns:
        Tuple of (training_samples_list, bucket_stats_dict) where
        bucket_stats_dict maps journey_count -> [total, kept, effective_prob].
    """
    if bucket_probs is None:
        bucket_probs = {}

    kept = []
    bucket_stats = defaultdict(lambda: [0, 0, 0.0])

    for uid in task_uids:
        if uid not in user_samples_dict:
            continue
        sample = user_samples_dict[uid]
        num_j = sample.get("metadata", {}).get("num_journeys", 0)
        bucket_stats[num_j][0] += 1

        if num_j >= keep_threshold:
            kept.append(extract_sft_fields(sample))
            bucket_stats[num_j][1] += 1
            bucket_stats[num_j][2] = 1.0
        else:
            keep_prob = bucket_probs.get(num_j, 1.0)
            bucket_stats[num_j][2] = keep_prob
            if rng.random() < keep_prob:
                kept.append(extract_sft_fields(sample))
                bucket_stats[num_j][1] += 1

    return kept, dict(bucket_stats)


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
                "OneRec/Data/LLMTrainingData/20260324/sft_data_v4/"
                "all_meta2tid_sft_full.json",
        help="Path to meta2tid *_full.json (with metadata.GlobalOfferId)",
    )
    parser.add_argument(
        "--event2journey_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260407/sft_data/"
                "event2journey_sft_full.json",
        help="Path to event2journey *_full.json (with metadata.user_id)",
    )
    parser.add_argument(
        "--profile2journey_full_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260406/sft_data/"
                "profile2journey_sft_full.json",
        help="Path to profile2journey *_full.json (with metadata.user_id)",
    )

    # --- Sampling ---
    parser.add_argument(
        "--meta2tid_prob", type=float, default=0.5,
        help="Sampling probability for meta2tid training data",
    )
    parser.add_argument(
        "--meta2tid_max_train", type=int, default=1000000,
        help="Maximum number of meta2tid training samples (default: 500000)",
    )
    parser.add_argument(
        "--journey_target_total", type=int, default=1000000,
        help="Target total for event2journey + profile2journey combined. "
             "If deduped total is below this, shared users are duplicated "
             "across both tasks to fill the gap (default: 1000000)",
    )

    # --- Test set ---
    parser.add_argument(
        "--journey_keep_threshold", type=int, default=5,
        help="Users with >= this many journeys are always kept; "
             "users below are sampled with prob = num_journeys / threshold "
             "(default: 5)",
    )
    parser.add_argument(
        "--test_sample_n", type=int, default=2000,
        help="Number of test samples per task (default: 5000)",
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260407/sft_data",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    # --- Build test TSV mode ---
    parser.add_argument(
        "--build_test_tsv", action="store_true", default=False,
        help="Build *_full_cleaned_test.tsv files from test JSONL + merged TSV. "
             "Skips the normal merge/train pipeline.",
    )
    parser.add_argument(
        "--event2journey_tsv", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260407/raw_data/"
                "event2journey_full_cleaned.tsv",
        help="Path to merged event2journey TSV (from pre_s2)",
    )
    parser.add_argument(
        "--profile2journey_tsv", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/LLMTrainingData/20260407/raw_data/"
                "profile2journey_full_cleaned.tsv",
        help="Path to merged profile2journey TSV (from pre_s2)",
    )

    return parser.parse_args()


# =============================================================================
# Build test TSV from test JSONL + merged full_cleaned TSV
# =============================================================================

def _read_test_uids(jsonl_path):
    """Read UserId values from a test JSONL file."""
    uids = set()
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
    csv.field_size_limit(sys.maxsize)
    total = 0
    matched = 0
    with open(tsv_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return 0, 0
        writer = csv.writer(fout, delimiter="\t", quoting=csv.QUOTE_NONE,
                            escapechar="\\")
        writer.writerow(header)

        uid_idx = header.index("UserId") if "UserId" in header else 0
        for row in reader:
            total += 1
            if len(row) > uid_idx and row[uid_idx].strip() in uids:
                writer.writerow(row)
                matched += 1
    return total, matched


def build_test_tsv(args):
    """Build *_full_cleaned_test.tsv for evaluation."""
    print("=" * 70)
    print("Build Test TSV Mode")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = [
        (
            "event2journey",
            os.path.join(args.output_dir, "event2journey_test.jsonl"),
            args.event2journey_tsv,
        ),
        (
            "profile2journey",
            os.path.join(args.output_dir, "profile2journey_test.jsonl"),
            args.profile2journey_tsv,
        ),
    ]

    for task_name, test_jsonl, full_tsv in tasks:
        print(f"\n  [{task_name}]")
        if not os.path.exists(test_jsonl):
            print(f"    SKIP: test JSONL not found: {test_jsonl}")
            continue
        if not os.path.exists(full_tsv):
            print(f"    SKIP: full TSV not found: {full_tsv}")
            continue

        uids = _read_test_uids(test_jsonl)
        print(f"    Test users from JSONL: {len(uids):,}")

        out_path = os.path.join(
            args.output_dir, f"{task_name}_full_cleaned_test.tsv")
        total, matched = _filter_tsv_by_uids(full_tsv, uids, out_path)

        file_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"    TSV rows scanned: {total:,}")
        print(f"    Matched:          {matched:,}")
        print(f"    Missing:          {len(uids) - matched:,}")
        print(f"    Output: {out_path} ({file_mb:.1f} MB)")

    print("\nDone!")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    # --build_test_tsv: fast path
    if args.build_test_tsv:
        build_test_tsv(args)
        return

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

    # --- Sample test users independently from each task ---
    # event2journey test
    e2j_all_uids = sorted(e2j_by_user.keys())
    rng.shuffle(e2j_all_uids)
    test_n_e2j = min(args.test_sample_n, len(e2j_all_uids))
    e2j_test_uids = set(e2j_all_uids[:test_n_e2j])

    # profile2journey test
    p2j_all_uids = sorted(p2j_by_user.keys())
    rng.shuffle(p2j_all_uids)
    test_n_p2j = min(args.test_sample_n, len(p2j_all_uids))
    p2j_test_uids = set(p2j_all_uids[:test_n_p2j])

    test_uids = e2j_test_uids | p2j_test_uids

    # Build test sets: sft fields + UserId
    e2j_test = []
    p2j_test = []
    for uid in sorted(e2j_test_uids):
        entry = extract_sft_fields(e2j_by_user[uid])
        entry["UserId"] = uid
        e2j_test.append(entry)
    for uid in sorted(p2j_test_uids):
        entry = extract_sft_fields(p2j_by_user[uid])
        entry["UserId"] = uid
        p2j_test.append(entry)

    print(f"\n  Test sets (independently sampled):")
    print(f"    event2journey test:    {len(e2j_test):,} (from {len(e2j_by_user):,} users)")
    print(f"    profile2journey test:  {len(p2j_test):,} (from {len(p2j_by_user):,} users)")
    print(f"    Total test users:      {len(test_uids):,} (union)")

    # --- Remove test users from all pools ---
    shared_train_uids = shared_uids - test_uids
    e2j_only_uids -= test_uids
    p2j_only_uids -= test_uids

    # --- Classify users by journey count (high vs low) ---
    threshold = args.journey_keep_threshold
    target = args.journey_target_total

    def _get_num_j(uid):
        """Get max num_journeys across both tasks for a user."""
        nj_e = e2j_by_user.get(uid, {}).get("metadata", {}).get("num_journeys", 0)
        nj_p = p2j_by_user.get(uid, {}).get("metadata", {}).get("num_journeys", 0)
        return max(nj_e, nj_p)

    # Shared users: high → BOTH tasks unconditionally; low → sharing strategy
    shared_high = set()  # >= threshold
    shared_low = set()   # < threshold
    for uid in shared_train_uids:
        if _get_num_j(uid) >= threshold:
            shared_high.add(uid)
        else:
            shared_low.add(uid)

    print(f"\n  Journey-count classification (threshold={threshold}):")
    print(f"    Shared high (>= {threshold}j, both tasks): {len(shared_high):>10,}")
    print(f"    Shared low  (<  {threshold}j, to split):   {len(shared_low):>10,}")

    # --- Sharing strategy for LOW-journey shared users only ---
    # High-journey users contribute 2 * len(shared_high) to both tasks
    n_guaranteed = len(shared_high) * 2 + len(e2j_only_uids) + len(p2j_only_uids)
    # (e2j_only and p2j_only high users are already in their task's pool)

    S_low = len(shared_low)
    # min: split shared_low 50/50
    min_low_total = n_guaranteed + S_low
    # max: all shared_low in both
    max_low_total = n_guaranteed + 2 * S_low

    if target <= min_low_total:
        num_to_share_low = 0
    elif target <= max_low_total:
        num_to_share_low = target - min_low_total
    else:
        num_to_share_low = S_low

    shared_low_list = sorted(shared_low)
    rng.shuffle(shared_low_list)

    # First num_to_share_low go to BOTH tasks
    shared_low_both = set(shared_low_list[:num_to_share_low])
    # Remaining split 50/50
    remaining_low = shared_low_list[num_to_share_low:]
    mid = len(remaining_low) // 2
    e2j_exclusive_low = set(remaining_low[:mid])
    p2j_exclusive_low = set(remaining_low[mid:])

    # --- Final uid pools ---
    # High-journey shared users go to BOTH tasks
    e2j_final_uids = e2j_only_uids | shared_high | shared_low_both | e2j_exclusive_low
    p2j_final_uids = p2j_only_uids | shared_high | shared_low_both | p2j_exclusive_low

    total_journey = len(e2j_final_uids) + len(p2j_final_uids)

    print(f"\n  Sharing strategy (target={target:,}):")
    print(f"    Guaranteed (high both + exclusive): {n_guaranteed:,}")
    print(f"    Low shared users:       {S_low:,}")
    print(f"    Min total (no share):   {min_low_total:,}")
    print(f"    Max total (full share): {max_low_total:,}")
    print(f"    Low users shared both:  {num_to_share_low:,}")
    print(f"    Low exclusive to e2j:   {len(e2j_exclusive_low):,}")
    print(f"    Low exclusive to p2j:   {len(p2j_exclusive_low):,}")
    print(f"    event2journey pool:     {len(e2j_final_uids):,}")
    print(f"    profile2journey pool:   {len(p2j_final_uids):,}")
    print(f"    Journey pool total:     {total_journey:,}")

    # --- Journey-count sampling on final pools ---
    # Step 1: Count buckets to compute uniform per-bucket probs
    e2j_n_high, e2j_low_buckets = count_journey_buckets(
        e2j_by_user, e2j_final_uids, threshold,
    )
    p2j_n_high, p2j_low_buckets = count_journey_buckets(
        p2j_by_user, p2j_final_uids, threshold,
    )
    n_high_total = e2j_n_high + p2j_n_high
    bucket_probs = compute_uniform_bucket_probs(
        n_high_total, [e2j_low_buckets, p2j_low_buckets],
        threshold, target,
    )

    # Merge bucket counts for display
    merged_buckets = defaultdict(int)
    for lb in [e2j_low_buckets, p2j_low_buckets]:
        for j, n in lb.items():
            merged_buckets[j] += n

    print(f"\n  Journey-count sampling (uniform per-bucket):")
    print(f"    High-journey users (always kept): {n_high_total:,} (across both tasks)")
    print(f"    Low-bucket target per bucket: "
          f"{(target - n_high_total) / max(len(merged_buckets), 1):,.0f}")
    print(f"    Per-bucket probabilities:")
    for j in sorted(merged_buckets.keys()):
        prob = bucket_probs.get(j, 1.0)
        cnt = merged_buckets[j]
        expected = int(cnt * prob)
        print(f"      {j} journeys: {cnt:>10,} users, "
              f"prob={prob:.4f}, expected_kept~{expected:,}")

    # Step 2: Sample with per-bucket probs
    e2j_train, e2j_bucket = sample_by_journey_count(
        e2j_by_user, e2j_final_uids, rng,
        keep_threshold=threshold, bucket_probs=bucket_probs,
    )
    p2j_train, p2j_bucket = sample_by_journey_count(
        p2j_by_user, p2j_final_uids, rng,
        keep_threshold=threshold, bucket_probs=bucket_probs,
    )

    # Print per-bucket sampling stats
    for label, bucket in [("event2journey", e2j_bucket),
                          ("profile2journey", p2j_bucket)]:
        total_all = sum(v[0] for v in bucket.values())
        kept_all = sum(v[1] for v in bucket.values())
        print(f"\n    [{label}] per-bucket stats:")
        print(f"      {'Journeys':>8s}  {'Total':>8s}  {'Kept':>8s}  {'Pct':>6s}")
        for jc in sorted(bucket.keys()):
            total, kept, prob = bucket[jc]
            pct = total / max(total_all, 1) * 100
            print(f"      {jc:>8d}  {total:>8,}  {kept:>8,}  {pct:>5.1f}%")
        print(f"      {'ALL':>8s}  {total_all:>8,}  {kept_all:>8,}  100.0%")

    combined_journey = len(e2j_train) + len(p2j_train)
    print(f"\n    Combined journey train: {combined_journey:,} "
          f"(target: {target:,}, diff: {combined_journey - target:+,})")

    stats["event2journey"] = (len(e2j_final_uids), len(e2j_train))
    stats["profile2journey"] = (len(p2j_final_uids), len(p2j_train))
    all_training.extend(e2j_train)
    all_training.extend(p2j_train)

    # --- Explicit vs Related journey distribution ---
    print(f"\n  Explicit vs Related journey distribution:")
    for label, by_user, final_uids in [
        ("event2journey", e2j_by_user, e2j_final_uids),
        ("profile2journey", p2j_by_user, p2j_final_uids),
    ]:
        all_explicit = []        # per-user explicit counts
        all_related = []         # per-user related counts
        all_explicit_pct = []    # per-user explicit percentage
        all_related_pct = []     # per-user related percentage
        total_explicit = 0
        total_related = 0

        for uid in final_uids:
            if uid not in by_user:
                continue
            sample = by_user[uid]
            try:
                out = json.loads(sample["output"])
                types = [j.get("JourneyType", "explicit")
                         for j in out.get("ContinuedJourneys", [])]
            except (json.JSONDecodeError, KeyError):
                continue
            n_exp = types.count("explicit")
            n_rel = types.count("related")
            n_total = n_exp + n_rel
            if n_total == 0:
                continue
            total_explicit += n_exp
            total_related += n_rel
            all_explicit.append(n_exp)
            all_related.append(n_rel)
            all_explicit_pct.append(n_exp / n_total * 100)
            all_related_pct.append(n_rel / n_total * 100)

        n_users = len(all_explicit)
        if n_users == 0:
            print(f"\n    [{label}] No data")
            continue

        grand_total = total_explicit + total_related
        sorted_exp = sorted(all_explicit)
        sorted_rel = sorted(all_related)
        sorted_exp_pct = sorted(all_explicit_pct)
        sorted_rel_pct = sorted(all_related_pct)

        print(f"\n    [{label}] ({n_users:,} users, "
              f"{grand_total:,} total journeys)")
        print(f"      Overall: explicit {total_explicit:,} "
              f"({total_explicit / grand_total * 100:.1f}%), "
              f"related {total_related:,} "
              f"({total_related / grand_total * 100:.1f}%)")
        print(f"      Per-user explicit:  "
              f"count  min={sorted_exp[0]}, max={sorted_exp[-1]}, "
              f"mean={sum(sorted_exp)/n_users:.1f}, "
              f"median={percentile(sorted_exp, 50)}  |  "
              f"pct  min={sorted_exp_pct[0]:.1f}%, "
              f"max={sorted_exp_pct[-1]:.1f}%, "
              f"mean={sum(sorted_exp_pct)/n_users:.1f}%, "
              f"median={percentile(sorted_exp_pct, 50):.1f}%")
        print(f"      Per-user related:   "
              f"count  min={sorted_rel[0]}, max={sorted_rel[-1]}, "
              f"mean={sum(sorted_rel)/n_users:.1f}, "
              f"median={percentile(sorted_rel, 50)}  |  "
              f"pct  min={sorted_rel_pct[0]:.1f}%, "
              f"max={sorted_rel_pct[-1]:.1f}%, "
              f"mean={sum(sorted_rel_pct)/n_users:.1f}%, "
              f"median={percentile(sorted_rel_pct, 50):.1f}%")

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
