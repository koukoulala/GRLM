"""Step 0.1: Merge and Sample Multiple ShoppingJourney Input Files.

Reads multiple TSV files (same format as step0_0 output / step0_2 input),
samples up to --sample_per_file rows from each, merges them, shuffles,
and writes a single output TSV.

Input files must have header:
  UserId | ReadableUserEvents | RequestTime | UserHistory | HisCount

Usage:
    python cook_data/step0_1_merge_inputs.py

    python cook_data/step0_1_merge_inputs.py \\
        --input_files /path/to/file1.tsv /path/to/file2.tsv \\
        --sample_per_file 200000 \\
        --output_name merged_700K.tsv
"""

import argparse
import csv
import os
import random
import sys
import time

csv.field_size_limit(sys.maxsize)

DEFAULT_DIR = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325"

DEFAULT_INPUT_FILES = [
    os.path.join(DEFAULT_DIR, "Step1_200K_EnUs_UserReadableHis_HisLarge50.tsv"),
    os.path.join(DEFAULT_DIR, "Step1_300K_EnUs_UserReadableHis_HisLess50.tsv"),
    os.path.join(DEFAULT_DIR, "Step1_ShoppingJourney_HisLarge_100_200K.tsv"),
]

EXPECTED_HEADER = ["UserId", "ReadableUserEvents", "RequestTime",
                   "UserHistory", "HisCount"]


def read_and_sample(filepath, sample_count, seed):
    """Read a TSV file and return up to sample_count rows (as raw lists).

    Args:
        filepath: Path to TSV file.
        sample_count: Max rows to sample. 0 = keep all.
        seed: Random seed for sampling.

    Returns:
        Tuple of (rows, header) where rows is a list of raw field lists.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        header_clean = [h.strip() for h in header]
        if header_clean != EXPECTED_HEADER:
            raise ValueError(
                f"Header mismatch in {filepath}:\n"
                f"  Expected: {EXPECTED_HEADER}\n"
                f"  Got:      {header_clean}")

        for row in reader:
            if len(row) >= len(EXPECTED_HEADER):
                rows.append(row)

    total = len(rows)
    if sample_count > 0 and total > sample_count:
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_count)
        print(f"  {os.path.basename(filepath)}: {total:,} rows -> "
              f"sampled {len(rows):,}")
    else:
        print(f"  {os.path.basename(filepath)}: {total:,} rows (kept all)")

    return rows, header_clean


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 0.1: Merge and sample multiple ShoppingJourney "
                    "input TSV files"
    )
    parser.add_argument(
        "--input_files", type=str, nargs="+",
        default=DEFAULT_INPUT_FILES,
        help="List of input TSV files to merge",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_DIR,
        help="Output directory (default: same as input files)",
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Output file name (default: auto-generated from total count)",
    )
    parser.add_argument(
        "--sample_per_file", type=int, default=200000,
        help="Max rows to sample from each file. 0 = keep all (default: 200000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 70)
    print("Step 0.1: Merge and Sample Input Files")
    print("=" * 70)
    print(f"  Input files:      {len(args.input_files)}")
    print(f"  Sample per file:  {args.sample_per_file or 'all'}")
    print(f"  Seed:             {args.seed}")
    print()

    start_time = time.time()

    # Read and sample each file
    all_rows = []
    seen_uids = set()
    duplicates = 0

    for fpath in args.input_files:
        if not os.path.exists(fpath):
            print(f"  [WARNING] File not found, skipping: {fpath}")
            continue
        rows, _ = read_and_sample(fpath, args.sample_per_file, args.seed)

        # Deduplicate by UserId across files
        for row in rows:
            uid = row[0].strip()
            if uid in seen_uids:
                duplicates += 1
                continue
            seen_uids.add(uid)
            all_rows.append(row)

    print(f"\n  Total rows after merge: {len(all_rows):,}")
    if duplicates > 0:
        print(f"  Duplicate UserIds removed: {duplicates:,}")

    # Shuffle
    random.shuffle(all_rows)
    print(f"  Shuffled {len(all_rows):,} rows")

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_name:
        out_name = args.output_name
    else:
        total_k = len(all_rows) // 1000
        out_name = f"Step1_Merged_{total_k}K.tsv"

    out_file = os.path.join(args.output_dir, out_name)

    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(EXPECTED_HEADER)
        for row in all_rows:
            writer.writerow(row)

    out_mb = os.path.getsize(out_file) / (1024 * 1024)
    elapsed = time.time() - start_time

    print(f"\n  Output: {out_file}")
    print(f"    Users: {len(all_rows):,}  Size: {out_mb:.1f} MB")
    print(f"    Time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
