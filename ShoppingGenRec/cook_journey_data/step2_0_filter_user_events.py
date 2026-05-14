"""Step 2.0: Filter Raw User Events for Journey Generation.

Reads the merged user-events TSV (from step0_1), applies two filters:
  1. Remove users whose HisCount >= max_his_count (default 500).
  2. Remove users whose UserId already appears in a previously-generated
     journey TSV file (to avoid duplicate work).

Writes the remaining rows as a TSV with the same schema:
    UserId | ReadableUserEvents | RequestTime | UserHistory | HisCount

Usage:
    python cook_full_journey_data/step2_0_filter_user_events.py

    python cook_full_journey_data/step2_0_filter_user_events.py \\
        --input_file /path/to/Step1_Merged_600K.tsv \\
        --exclude_tsv /path/to/existing_journey.tsv \\
        --max_his_count 500 \\
        --output_dir /path/to/output/
"""

import argparse
import os
import random
import sys
import time
from collections import Counter

# 8 MB I/O buffer for faster file reads/writes
_IO_BUFFER = 8 * 1024 * 1024


def load_existing_user_ids(tsv_path):
    """Load user IDs from an existing TSV file with a UserId column.

    Args:
        tsv_path: Path to the TSV file.

    Returns:
        Set of user ID strings.
    """
    user_ids = set()
    if not tsv_path or not os.path.isfile(tsv_path):
        return user_ids

    with open(tsv_path, "r", encoding="utf-8", buffering=_IO_BUFFER) as f:
        header_line = f.readline()
        if not header_line:
            return user_ids

        cols = header_line.rstrip("\n\r").split("\t")
        try:
            uid_idx = [c.strip() for c in cols].index("UserId")
        except ValueError:
            uid_idx = 0  # fallback: first column

        for line in f:
            fields = line.split("\t", uid_idx + 1)
            if len(fields) > uid_idx:
                uid = fields[uid_idx].strip()
                if uid:
                    user_ids.add(uid)

    return user_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2.0: Filter raw user events — remove high-HisCount "
                    "users and already-processed users"
    )
    parser.add_argument(
        "--input_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
                "/Data/1225_0325/Step1_Merged_600K.tsv",
        help="Path to merged user-events TSV from step0_1",
    )
    parser.add_argument(
        "--exclude_tsv", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
                "/Data/1225_0325/JourneyWithProfile"
                "/JourneyWithConversationStarterAndDesc/v3"
                "/Step0_UserProfile_500KEnUsHisRandom0408_500K_Journey.tsv",
        help="Path to existing journey TSV file with UserId column. "
             "Users in this file will be excluded from output.",
    )
    parser.add_argument(
        "--max_his_count", type=int, default=500,
        help="Maximum HisCount (exclusive). Users with HisCount >= this "
             "value are removed (default: 500)",
    )
    parser.add_argument(
        "--max_low_his_users", type=int, default=10000,
        help="Maximum number of users with HisCount < 10 to keep. "
             "Excess users are randomly downsampled (default: 10000). "
             "Set to 0 to disable downsampling.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for downsampling (default: 42)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
                "/Data/LLMTrainingData/20260513/raw_data",
        help="Directory to save the filtered output TSV",
    )
    parser.add_argument(
        "--output_name", type=str, default="UserEvents_clean.tsv",
        help="Output file name (default: auto-generated)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Step 2.0: Filter Raw User Events")
    print("=" * 70)
    print(f"  Input file:      {args.input_file}")
    print(f"  Exclude TSV:     {args.exclude_tsv}")
    print(f"  Max HisCount:    < {args.max_his_count}")
    print(f"  Max low-his users (HisCount<10): {args.max_low_his_users}")
    print(f"  Seed:            {args.seed}")
    print(f"  Output dir:      {args.output_dir}")
    print()

    start_time = time.time()

    # =========================================================================
    # Step 1: Load existing user IDs to exclude
    # =========================================================================
    print("=" * 70)
    print("Step 1: Loading existing user IDs to exclude")
    print("=" * 70)

    exclude_ids = set()
    if args.exclude_tsv and os.path.isfile(args.exclude_tsv):
        print(f"  Reading: {args.exclude_tsv}")
        exclude_ids = load_existing_user_ids(args.exclude_tsv)
        print(f"  Loaded {len(exclude_ids):,} user IDs to exclude")
    else:
        print(f"  No exclude TSV file found, skipping exclusion")
    print()

    # =========================================================================
    # Step 2: Stream-filter input TSV and write output directly
    # =========================================================================
    print("=" * 70)
    print("Step 2: Reading, filtering, and writing (streaming)")
    print("=" * 70)
    print(f"  Reading: {args.input_file}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name:
        out_name = args.output_name
    else:
        out_name = None  # will be determined after counting

    # Use a temp file first, rename after we know the count
    tmp_output = os.path.join(args.output_dir, "_tmp_filtered.tsv")

    total_read = 0
    kept_count = 0
    stats = {
        "his_count_filtered": 0,
        "his_count_parse_error": 0,
        "exclude_filtered": 0,
        "no_userid": 0,
        "empty_events": 0,
        "low_his_downsampled": 0,
    }
    his_buckets = Counter()
    _HIS_BOUNDARIES = [10, 50, 100, 200, 300, 500]
    _HIS_LABELS = ["[1, 10)", "[10, 50)", "[50, 100)",
                   "[100, 200)", "[200, 300)", "[300, 500)", "[500+)"]

    max_hc = args.max_his_count
    # Buffer for low-HisCount (<10) lines to downsample later
    low_his_buffer = []

    with open(args.input_file, "r", encoding="utf-8",
             buffering=_IO_BUFFER) as fin, \
         open(tmp_output, "w", encoding="utf-8",
              buffering=_IO_BUFFER) as fout:

        # Read and write header
        header_line = fin.readline()
        if not header_line:
            print("  ERROR: Empty file!")
            return

        header_clean = [h.strip() for h in
                        header_line.rstrip("\n\r").split("\t")]
        print(f"  Header: {header_clean}")

        col_map = {name: idx for idx, name in enumerate(header_clean)}
        uid_idx = col_map.get("UserId")
        his_count_idx = col_map.get("HisCount")
        events_idx = col_map.get("ReadableUserEvents")
        # We need to split up to the max index we care about
        max_split = max(uid_idx or 0, his_count_idx or 0,
                        events_idx or 0) + 1

        if uid_idx is None:
            print("  ERROR: 'UserId' column not found!")
            return

        fout.write(header_line if header_line.endswith("\n")
                   else header_line + "\n")

        for line in fin:
            total_read += 1

            # Split up to the columns we need
            fields = line.split("\t", max_split)

            # Check UserId
            if len(fields) <= uid_idx:
                stats["no_userid"] += 1
                continue
            uid = fields[uid_idx].strip()
            if not uid:
                stats["no_userid"] += 1
                continue

            # Filter empty ReadableUserEvents
            if events_idx is not None and len(fields) > events_idx:
                events = fields[events_idx].strip()
                if not events:
                    stats["empty_events"] += 1
                    continue
            elif events_idx is not None:
                stats["empty_events"] += 1
                continue

            # Filter by HisCount
            his_count = -1
            if his_count_idx is not None and len(fields) > his_count_idx:
                hc_str = fields[his_count_idx].strip().rstrip("\n\r")
                if hc_str:
                    try:
                        his_count = int(hc_str)
                        if his_count >= max_hc:
                            stats["his_count_filtered"] += 1
                            continue
                    except ValueError:
                        stats["his_count_parse_error"] += 1

            # Filter by exclude list
            if uid in exclude_ids:
                stats["exclude_filtered"] += 1
                continue

            # Low-HisCount (<10): buffer for downsampling
            if args.max_low_his_users > 0 and his_count >= 0 and his_count < 10:
                low_his_buffer.append(line)
                continue

            # Write raw line directly (no re-serialization)
            fout.write(line if line.endswith("\n") else line + "\n")
            kept_count += 1

            # Track HisCount bucket
            if his_count >= 0:
                bucket_idx = 0
                for boundary in _HIS_BOUNDARIES:
                    if his_count < boundary:
                        break
                    bucket_idx += 1
                his_buckets[_HIS_LABELS[bucket_idx]] += 1

        # Downsample low-HisCount buffer
        low_his_total = len(low_his_buffer)
        max_low = args.max_low_his_users
        if max_low > 0 and low_his_total > max_low:
            rng = random.Random(args.seed)
            low_his_buffer = rng.sample(low_his_buffer, max_low)
            stats["low_his_downsampled"] = low_his_total - max_low
            print(f"\n  Low-HisCount (<10): {low_his_total:,} -> "
                  f"sampled {max_low:,} (removed {stats['low_his_downsampled']:,})")
        elif max_low > 0:
            print(f"\n  Low-HisCount (<10): {low_his_total:,} "
                  f"(under limit {max_low:,}, kept all)")

        # Write buffered low-his lines
        for buf_line in low_his_buffer:
            fout.write(buf_line if buf_line.endswith("\n")
                       else buf_line + "\n")
            kept_count += 1
            his_buckets["[1, 10)"] += 1

    print(f"\n  Total rows read:                     {total_read:>10,}")
    print(f"  Unique UserIds in input:             {total_read - stats['no_userid']:>10,}")
    print(f"  Exclude list UserIds:                {len(exclude_ids):>10,}")
    print(f"  Removed (empty events):              {stats['empty_events']:>10,}")
    print(f"  Removed (HisCount >= {max_hc}):         {stats['his_count_filtered']:>10,}")
    print(f"  Removed (already in exclude TSV):    {stats['exclude_filtered']:>10,}")
    if stats["low_his_downsampled"] > 0:
        print(f"  Removed (low-his downsampled):       {stats['low_his_downsampled']:>10,}")
    if stats["no_userid"] > 0:
        print(f"  Removed (missing UserId):            {stats['no_userid']:>10,}")
    if stats["his_count_parse_error"] > 0:
        print(f"  HisCount parse errors (kept):        {stats['his_count_parse_error']:>10,}")
    print(f"  Rows kept after filtering:           {kept_count:>10,}")
    print()

    # =========================================================================
    # Step 3: HisCount distribution (kept rows)
    # =========================================================================
    print("=" * 70)
    print("Step 3: HisCount distribution (kept rows)")
    print("=" * 70)

    bucket_order = ["[1, 10)", "[10, 50)", "[50, 100)", "[100, 200)",
                    "[200, 300)", "[300, 500)", "[500+)", "(parse error)"]
    print(f"  {'Bucket':<20s} {'Count':>10s}")
    print(f"  {'-'*20} {'-'*10}")
    for bucket in bucket_order:
        count = his_buckets.get(bucket, 0)
        if count > 0:
            print(f"  {bucket:<20s} {count:>10,}")
    print()

    # =========================================================================
    # Step 4: Rename output file
    # =========================================================================
    print("=" * 70)
    print("Step 4: Finalizing output")
    print("=" * 70)

    if out_name is None:
        total_k = kept_count // 1000
        out_name = f"UserEvents_Filtered_{total_k}K.tsv"

    output_file = os.path.join(args.output_dir, out_name)
    os.replace(tmp_output, output_file)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    elapsed = time.time() - start_time

    print(f"  Output: {output_file}")
    print(f"  Rows:   {kept_count:,}")
    print(f"  Size:   {file_size_mb:.1f} MB")
    print(f"  Time:   {elapsed:.1f}s")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Input rows:                          {total_read:>10,}")
    print(f"  Removed (HisCount >= {max_hc}):         {stats['his_count_filtered']:>10,}")
    print(f"  Removed (already in exclude TSV):    {stats['exclude_filtered']:>10,}")
    if stats["low_his_downsampled"] > 0:
        print(f"  Removed (low-his downsampled):       {stats['low_his_downsampled']:>10,}")
    print(f"  Output rows:                         {kept_count:>10,}")
    print(f"  Output: {output_file}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
