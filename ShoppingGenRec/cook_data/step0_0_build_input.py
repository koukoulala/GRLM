"""Step 0.0: Build ShoppingJourney Input from Raw Sequence & Product Data.

Reads Step0_Sequence.tsv (user event sequences) and Step0_Product.tsv
(product catalog), then constructs the ShoppingJourney input TSV format
expected by step0_generate_journey.py.

Input files:
  Step0_Sequence.tsv - columns: UserId, PageTitle, GlobalOfferId,
                       Timestamp, Vertical, Query, RowNumber
  Step0_Product.tsv  - columns: OfferId, GlobalOfferId, Title, ... (25 cols)

Vertical → Action mapping:
  - *PageTitle* (Flyout_PageTitle, UET_PageTitle, SAN_PageTitle, etc.)
    → "Browsed", content = PageTitle
  - *Click* (msnClick, etc.)
    → "Clicked", content = Product title from Step0_Product.tsv via GlobalOfferId
  - *Query* (bingPAQuery, etc.)
    → "Searched", content = Query column

Output TSV columns:
  UserId | ReadableUserEvents | RequestTime | UserHistory | HisCount

Usage:
    python cook_data/step0_0_build_input.py \\
        --sequence_file /path/to/Step0_Sequence.tsv \\
        --product_file /path/to/Step0_Product.tsv \\
        --output_dir /path/to/output/ \\
        --request_time "03/25/2026" \\
        --min_events 10 \\
        --sample_count 200000

    # With user exclusion:
    python cook_data/step0_0_build_input.py \\
        --exclude_files /path/to/existing1.tsv /path/to/existing2.tsv \\
        --min_events 50 \\
        --sample_count 200000
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict

from tqdm import tqdm

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)

# Reference timestamp for "time ago" calculation
# Default request time: 03/25/2026 00:00:00 UTC
DEFAULT_REQUEST_TIME = "03/25/2026"
DEFAULT_REQUEST_TIMESTAMP = 1774396800  # 2026-03-25 00:00:00 UTC

# Event tuple field indices: (page_title, global_offer_id, timestamp, vertical, query)
_EV_PT    = 0
_EV_GID   = 1
_EV_TS    = 2
_EV_VERT  = 3
_EV_QUERY = 4


# =============================================================================
# Vertical → Action Classification
# =============================================================================

def classify_vertical(vertical):
    """Classify a Vertical string into an action type.

    Args:
        vertical: Raw Vertical string from Step0_Sequence.tsv.

    Returns:
        Tuple of (action, source) where:
          action: "Browsed", "Clicked", or "Searched"
          source: "pagetitle", "offerid", or "query"
        Returns (None, None) for unrecognized verticals.
    """
    v = vertical.strip().lower()
    if "query" in v:
        return "Searched", "query"
    elif "click" in v:
        return "Clicked", "offerid"
    elif "pagetitle" in v:
        return "Browsed", "pagetitle"
    else:
        # Fallback: treat as Browsed if PageTitle-like
        return "Browsed", "pagetitle"


# =============================================================================
# Time Formatting
# =============================================================================

def format_time_ago(event_ts, reference_ts):
    """Format a timestamp as a human-readable 'X time ago' string.

    Args:
        event_ts: Event timestamp (Unix epoch seconds).
        reference_ts: Reference timestamp (request time).

    Returns:
        str like "2 hours ago", "5 days ago", etc.
    """
    diff_seconds = max(reference_ts - event_ts, 0)
    diff_minutes = diff_seconds // 60
    diff_hours = diff_seconds // 3600
    diff_days = diff_seconds // 86400

    if diff_days > 0:
        return f"{diff_days} days ago"
    elif diff_hours > 0:
        return f"{diff_hours} hours ago"
    elif diff_minutes > 0:
        return f"{diff_minutes} minutes ago"
    else:
        return "0 minutes ago"


# =============================================================================
# Data Loading
# =============================================================================

def load_product_titles(product_file):
    """Load GlobalOfferId → Title mapping from Step0_Product.tsv.

    Uses pandas with usecols to read only the two needed columns.

    Args:
        product_file: Path to Step0_Product.tsv.

    Returns:
        Dict mapping GlobalOfferId (str) → Title (str).
    """
    import pandas as pd
    print(f"  Loading product titles from: {product_file}")
    df = pd.read_csv(
        product_file, sep="\t",
        usecols=["GlobalOfferId", "Title"],
        dtype=str, engine="c", on_bad_lines="skip",
    )
    df["GlobalOfferId"] = df["GlobalOfferId"].fillna("").str.strip()
    df["Title"] = df["Title"].fillna("").str.strip()
    df = df[(df["GlobalOfferId"] != "") & (df["Title"] != "")]
    titles = dict(zip(df["GlobalOfferId"], df["Title"]))
    print(f"  Loaded {len(titles):,} product titles")
    return titles


def load_exclude_user_ids(exclude_files):
    """Load user IDs to exclude from a list of TSV files.

    Each file should have a 'UserId' column in the header.

    Args:
        exclude_files: List of file paths.

    Returns:
        Set of user ID strings to exclude.
    """
    exclude_ids = set()
    for fpath in exclude_files:
        if not os.path.exists(fpath):
            print(f"  [WARNING] Exclude file not found: {fpath}")
            continue

        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header is None:
                continue
            col_map = {name.strip(): idx for idx, name in enumerate(header)}
            uid_idx = col_map.get("UserId", 0)

            for row in reader:
                if len(row) > uid_idx:
                    uid = row[uid_idx].strip()
                    if uid:
                        exclude_ids.add(uid)
                        count += 1

        print(f"  Loaded {count:,} user IDs from {os.path.basename(fpath)}")

    print(f"  Total unique user IDs to exclude: {len(exclude_ids):,}")
    return exclude_ids


def load_all_user_events(sequence_file, exclude_ids, scan_ratio=0.5):
    """Single-pass load: read all 6 columns, accumulate events per user.

    Reads the sequence file in one pass, storing events for all non-excluded
    users. If scan_ratio < 1.0, only scans that fraction of the file
    (estimated via reading 100K lines to measure bytes-per-row).

    Args:
        sequence_file: Path to Step0_Sequence.tsv.
        exclude_ids: Set of user IDs to skip.
        scan_ratio: Fraction of the file to scan (0.0-1.0). Default 0.5.

    Returns:
        Tuple of (user_events, total_rows_scanned) where
        user_events maps UserId → list of event tuples (sorted by ts desc).
    """
    import pandas as pd
    import numpy as np

    COLS = ["UserId", "PageTitle", "GlobalOfferId", "Timestamp",
            "Vertical", "Query"]
    exclude_set = frozenset(exclude_ids) if exclude_ids else frozenset()

    file_size = os.path.getsize(sequence_file)
    print(f"  Loading from: {sequence_file} ({file_size / 1e9:.1f} GB)")

    # Compute max rows to scan based on scan_ratio
    max_rows = None
    if scan_ratio < 1.0:
        # Read 100K raw lines to measure bytes-per-row
        n_probe = 100_000
        with open(sequence_file, "r", encoding="utf-8") as fh:
            for _ in range(n_probe + 1):  # +1 for header
                fh.readline()
            bytes_read = fh.tell()
        avg_bytes_per_row = bytes_read / n_probe
        est_total_rows = int(file_size / avg_bytes_per_row)
        max_rows = int(est_total_rows * scan_ratio)
        print(f"  Estimated ~{est_total_rows:,} total rows "
              f"({avg_bytes_per_row:.0f} bytes/row)")
        print(f"  Will scan ~{max_rows:,} rows ({scan_ratio:.0%})")

    CHUNK_SIZE = 5_000_000
    total_rows = 0
    excluded_rows = 0
    chunks = []
    stopped_early = False

    for chunk in tqdm(
        pd.read_csv(sequence_file, sep="\t", usecols=COLS, dtype=str,
                     engine="c", on_bad_lines="skip", chunksize=CHUNK_SIZE),
        desc="Loading",
    ):
        total_rows += len(chunk)
        chunk["UserId"] = chunk["UserId"].fillna("").str.strip()

        mask = (chunk["UserId"] != "")
        if exclude_set:
            exclude_mask = chunk["UserId"].isin(exclude_set)
            excluded_rows += exclude_mask.sum()
            mask = mask & (~exclude_mask)
        filtered = chunk[mask]

        if len(filtered) > 0:
            chunks.append(filtered)

        if max_rows is not None and total_rows >= max_rows:
            stopped_early = True
            break

    if stopped_early:
        print(f"  Stopped early at {total_rows:,} rows "
              f"(~{scan_ratio:.0%} of file)")
    else:
        print(f"  Scanned all {total_rows:,} rows")
    if exclude_set:
        print(f"  Excluded {excluded_rows:,} rows from {len(exclude_set):,} users")

    if not chunks:
        print(f"  No valid rows found")
        return {}, total_rows

    print(f"  Concatenating chunks ...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    kept_rows = len(df)

    # Parse timestamps; drop rows where Timestamp is invalid
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["Timestamp"] = df["Timestamp"].astype("int64")

    for col in ["PageTitle", "GlobalOfferId", "Vertical", "Query"]:
        df[col] = df[col].fillna("").str.replace(r'[\r\n\t]', ' ', regex=True).str.strip()

    # Sort by UserId asc, Timestamp desc so each user's events are contiguous
    print("  Sorting by user and timestamp ...")
    df = df.sort_values(["UserId", "Timestamp"], ascending=[True, False])
    df = df.reset_index(drop=True)

    # Extract numpy arrays; convert timestamps to Python ints for JSON safety
    uid_arr  = df["UserId"].to_numpy()
    pt_arr   = df["PageTitle"].to_numpy()
    gid_arr  = df["GlobalOfferId"].to_numpy()
    ts_list  = df["Timestamp"].to_numpy(dtype="int64").tolist()
    vert_arr = df["Vertical"].to_numpy()
    q_arr    = df["Query"].to_numpy()
    n = len(uid_arr)
    del df

    # Find positions where UserId changes (C-level, no Python loop needed)
    boundaries = np.concatenate(
        [[0], np.where(uid_arr[1:] != uid_arr[:-1])[0] + 1, [n]]
    )

    user_events = {}
    for k in range(len(boundaries) - 1):
        s, e = int(boundaries[k]), int(boundaries[k + 1])
        user_events[uid_arr[s]] = list(zip(
            pt_arr[s:e], gid_arr[s:e], ts_list[s:e],
            vert_arr[s:e], q_arr[s:e],
        ))

    print(f"  Kept {kept_rows:,} rows, {len(user_events):,} unique users")
    return user_events, total_rows


# =============================================================================
# Event Formatting
# =============================================================================

def build_readable_event(event, idx, reference_ts, product_titles):
    """Build a single readable event string.

    Args:
        event: Event tuple (page_title, global_offer_id, timestamp,
               vertical, query); see _EV_* constants for field indices.
        idx: 1-based event index.
        reference_ts: Reference timestamp for 'X time ago' calculation.
        product_titles: Dict of GlobalOfferId → Title.

    Returns:
        Tuple of (readable_str, history_dict) or (None, None) if event
        should be skipped (e.g., no content).
    """
    action, source = classify_vertical(event[_EV_VERT])
    if action is None:
        return None, None

    time_ago = format_time_ago(event[_EV_TS], reference_ts)

    # Determine content based on source type
    if source == "query":
        content = event[_EV_QUERY]
    elif source == "offerid":
        gid = event[_EV_GID]
        content = product_titles.get(gid, "")
        if not content:
            # Fallback to page title if product not found
            content = event[_EV_PT]
    else:  # pagetitle
        content = event[_EV_PT]

    if not content:
        return None, None

    # Clean content: remove tabs, newlines, carriage returns
    content = content.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()

    readable = f"{idx} | {time_ago} | {action} | {content}"

    # Build UserHistory JSON entry
    history = {
        "TimeStamp": event[_EV_TS],
        "GlobalOfferId": event[_EV_GID],
        "PageTitle": event[_EV_PT],
        "Query": event[_EV_QUERY],
        "Type": event[_EV_VERT],
    }

    return readable, history


def build_user_row(uid, events, reference_ts, request_time_str,
                   product_titles):
    """Build a complete output row for a single user.

    Args:
        uid: User ID string.
        events: List of event dicts (sorted by timestamp desc).
        reference_ts: Reference timestamp for time calculation.
        request_time_str: Request time string for output.
        product_titles: Dict of GlobalOfferId → Title.

    Returns:
        Dict with keys: UserId, ReadableUserEvents, RequestTime,
        UserHistory, HisCount. Or None if no valid events.
    """
    readable_parts = []
    history_list = []
    idx = 0

    for event in events:
        idx += 1
        readable, history = build_readable_event(
            event, idx, reference_ts, product_titles)
        if readable is None:
            idx -= 1  # Don't increment for skipped events
            continue
        readable_parts.append(readable)
        history_list.append(history)

    if not readable_parts:
        return None

    readable_events = "#N#".join(readable_parts)
    user_history = json.dumps(history_list, ensure_ascii=False,
                              separators=(',', ':'))

    return {
        "UserId": uid,
        "ReadableUserEvents": readable_events,
        "RequestTime": request_time_str,
        "UserHistory": user_history,
        "HisCount": str(len(readable_parts)),
    }


# =============================================================================
# Multiprocessing Worker
# =============================================================================

# Module-level globals used by worker processes.
# On Linux, mp.Pool uses fork by default: child processes inherit these via
# copy-on-write — no serialization/pickling of the large dicts needed.
_MP_USER_EVENTS: dict = {}
_MP_PRODUCT_TITLES: dict = {}
_MP_REFERENCE_TS: int = DEFAULT_REQUEST_TIMESTAMP
_MP_REQUEST_TIME: str = DEFAULT_REQUEST_TIME


def _mp_worker(uid):
    """Build output row for one user (called in Pool worker)."""
    return build_user_row(
        uid, _MP_USER_EVENTS[uid],
        _MP_REFERENCE_TS, _MP_REQUEST_TIME, _MP_PRODUCT_TITLES,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 0.0: Build ShoppingJourney input TSV from raw "
                    "sequence and product data"
    )

    parser.add_argument(
        "--sequence_file", type=str,
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/Step0_Sequence.tsv",
        default="/vc_data/users/xiaoyukou/GRLM/ShoppingGenRec/resources/Step0_Sequence.tsv",
        help="Path to Step0_Sequence.tsv",
    )
    parser.add_argument(
        "--product_file", type=str,
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/Step0_Product.tsv",
        default="/vc_data/users/xiaoyukou/GRLM/ShoppingGenRec/resources/Step0_Product.tsv",
        help="Path to Step0_Product.tsv",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/"
                "OneRec/Data/1225_0325/CookData_merged",
        help="Output directory",
    )
    parser.add_argument(
        "--request_time", type=str, default=DEFAULT_REQUEST_TIME,
        help=f"Request time string for output (default: {DEFAULT_REQUEST_TIME})",
    )
    parser.add_argument(
        "--reference_timestamp", type=int, default=DEFAULT_REQUEST_TIMESTAMP,
        help="Unix timestamp for 'X time ago' calculation "
             f"(default: {DEFAULT_REQUEST_TIMESTAMP} = 2026-03-25 00:00 UTC)",
    )
    parser.add_argument(
        "--exclude_files", type=str, nargs="*", 
        default=["/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/JourneyWithProfile/Step0_UserProfile_200KHisLarge50_300KHisLess50.tsv"],
        help="TSV files whose UserId values will be excluded from processing",
    )
    parser.add_argument(
        "--min_events", type=int, default=5,
        help="Minimum number of valid events per user. Users below this "
             "threshold are dropped (default: 5)",
    )
    parser.add_argument(
        "--sample_count", type=int, default=200000,
        help="Max users to sample per bucket (default: 200000)",
    )
    parser.add_argument(
        "--scan_ratio", type=float, default=0.8,
        help="Fraction of sequence file to scan (0.0-1.0). Default 1.0 "
             "scans the entire file (default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 70)
    print("Step 0.0: Build ShoppingJourney Input from Raw Data")
    print("=" * 70)
    print(f"  Sequence file:  {args.sequence_file}")
    print(f"  Product file:   {args.product_file}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Request time:   {args.request_time}")
    print(f"  Reference TS:   {args.reference_timestamp}")
    print(f"  Min events:     {args.min_events}")
    print(f"  Sample count:   {args.sample_count or 'all'}")
    print(f"  Scan ratio:     {args.scan_ratio}")
    if args.exclude_files:
        print(f"  Exclude files:  {len(args.exclude_files)} file(s)")
    print()

    start_time = time.time()

    # ---- Step 1: Load exclude user IDs ----
    exclude_ids = set()
    if args.exclude_files:
        print("Step 1: Loading exclude user IDs")
        exclude_ids = load_exclude_user_ids(args.exclude_files)
        print()

    # ---- Step 2: Load product titles ----
    print("Step 2: Loading product catalog")
    product_titles = load_product_titles(args.product_file)
    print()

    # ---- Step 3: Single-pass load of all user events ----
    print("Step 3: Loading all user events (single-pass)")
    user_events, total_seq_rows = load_all_user_events(
        args.sequence_file, exclude_ids, scan_ratio=args.scan_ratio)

    # Distribution before filtering
    user_counts = {uid: len(evts) for uid, evts in user_events.items()}
    all_counts = sorted(user_counts.values())
    n_all = len(all_counts)
    if n_all > 0:
        print(f"  All-user event count distribution:")
        print(f"    Min={all_counts[0]}, "
              f"P50={all_counts[n_all // 2]}, "
              f"P90={all_counts[int(n_all * 0.9)]}, "
              f"Max={all_counts[-1]}")
    del all_counts
    print()

    # ---- Step 4: Bucket users by event count ----
    print("Step 4: Bucketing users by event count")
    BUCKETS = [
        ("HisLess50",   args.min_events, 49),
        ("His50to100",  50, 99),
        ("HisLarge100", 100, None),  # None = no upper bound
    ]

    bucket_uids = {}
    for name, lo, hi in BUCKETS:
        uids = [uid for uid, cnt in user_counts.items()
                if cnt >= lo and (hi is None or cnt <= hi)]
        bucket_uids[name] = uids
        print(f"  {name} ({lo}-{hi or '∞'} events): {len(uids):,} users")
    del user_counts
    print()

    # Sample each bucket
    for name in bucket_uids:
        uids = bucket_uids[name]
        if args.sample_count > 0 and len(uids) > args.sample_count:
            bucket_uids[name] = random.sample(sorted(uids), args.sample_count)
            print(f"  {name}: sampled {args.sample_count:,} from {len(uids):,}")
        else:
            print(f"  {name}: using all {len(uids):,}")

    # Collect all target UIDs and prune memory
    all_target = set()
    for uids in bucket_uids.values():
        all_target.update(uids)
    user_events = {uid: evts for uid, evts in user_events.items()
                   if uid in all_target}
    print(f"  Total retained in memory: {len(user_events):,} users")
    print()

    # ---- Step 5: Build output rows (parallelized) ----
    print("Step 5: Building output rows")

    # Vertical + GID statistics (from loaded data)
    vertical_stats = defaultdict(int)
    gid_total = 0       # total Click events with a GID
    gid_found = 0       # Click events whose GID was found in product_titles
    gid_fallback_pt = 0 # Click events that fell back to PageTitle
    gid_no_content = 0  # Click events with no title and no PageTitle
    for events in user_events.values():
        for e in events:
            vertical_stats[e[_EV_VERT]] += 1
            action, source = classify_vertical(e[_EV_VERT])
            if source == "offerid":
                gid = e[_EV_GID]
                if gid:
                    gid_total += 1
                    if gid in product_titles:
                        gid_found += 1
                    elif e[_EV_PT]:
                        gid_fallback_pt += 1
                    else:
                        gid_no_content += 1

    print(f"\n  --- GID Mapping Statistics (Click events) ---")
    print(f"    Total Click events with GID:   {gid_total:>12,}")
    print(f"    GID found in product catalog:   {gid_found:>12,} "
          f"({gid_found / max(gid_total, 1) * 100:.1f}%)")
    print(f"    Fallback to PageTitle:          {gid_fallback_pt:>12,} "
          f"({gid_fallback_pt / max(gid_total, 1) * 100:.1f}%)")
    print(f"    No content (skipped):           {gid_no_content:>12,} "
          f"({gid_no_content / max(gid_total, 1) * 100:.1f}%)")

    # Set fork-safe globals before spawning Pool workers (Linux COW fork:
    # child processes see these as read-only without copying)
    global _MP_USER_EVENTS, _MP_PRODUCT_TITLES, _MP_REFERENCE_TS, _MP_REQUEST_TIME
    _MP_USER_EVENTS = user_events
    _MP_PRODUCT_TITLES = product_titles
    _MP_REFERENCE_TS = args.reference_timestamp
    _MP_REQUEST_TIME = args.request_time

    nproc = min(mp.cpu_count(), 16)
    print(f"  Using {nproc} worker processes")
    with mp.Pool(processes=nproc) as pool:
        results = list(tqdm(
            pool.imap_unordered(_mp_worker, user_events.keys(),
                                chunksize=5000),
            total=len(user_events), desc="Building rows",
        ))

    # Index results by UserId
    row_map = {}
    skipped_no_valid = 0
    for row in results:
        if row is None:
            skipped_no_valid += 1
            continue
        row_map[row["UserId"]] = row
    del results

    print(f"\n  Target users: {len(user_events):,}")
    print(f"  Users with valid output: {len(row_map):,}")
    print(f"  Skipped (no valid events after formatting): {skipped_no_valid:,}")

    # Event count statistics
    all_event_counts = sorted(int(r["HisCount"]) for r in row_map.values())
    if all_event_counts:
        n = len(all_event_counts)
        print(f"\n  Output event count distribution:")
        print(f"    Min: {all_event_counts[0]:>6}")
        print(f"    P25: {all_event_counts[n // 4]:>6}")
        print(f"    P50: {all_event_counts[n // 2]:>6}")
        print(f"    P75: {all_event_counts[3 * n // 4]:>6}")
        print(f"    P90: {all_event_counts[int(n * 0.9)]:>6}")
        print(f"    Max: {all_event_counts[-1]:>6}")
        print(f"    Avg: {sum(all_event_counts) / n:>6.1f}")

    # Vertical type statistics
    print(f"\n  Vertical type distribution:")
    for v, cnt in sorted(vertical_stats.items(), key=lambda x: -x[1]):
        action, _ = classify_vertical(v)
        print(f"    {v:30s} -> {action:10s}  {cnt:>12,}")
    print()

    # ---- Step 6: Save output (merge all buckets into one file) ----
    print("Step 6: Saving output (merged)")
    os.makedirs(args.output_dir, exist_ok=True)

    output_header = ["UserId", "ReadableUserEvents", "RequestTime",
                     "UserHistory", "HisCount"]

    # Collect rows from all buckets, then shuffle
    all_output_rows = []
    for name, lo, hi in BUCKETS:
        uids = bucket_uids[name]
        bucket_rows = [row_map[uid] for uid in uids if uid in row_map]
        print(f"  {name} ({lo}-{hi or '∞'}): {len(bucket_rows):,} users")
        all_output_rows.extend(bucket_rows)

    random.shuffle(all_output_rows)
    total_written = len(all_output_rows)

    total_k = total_written // 1000
    out_file = os.path.join(
        args.output_dir,
        f"Step1_ShoppingJourney_Mixed_{total_k}K.tsv")

    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(output_header)
        for row in all_output_rows:
            writer.writerow([row[col] for col in output_header])

    out_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n  Output: {out_file}")
    print(f"    Users: {total_written:,}  Size: {out_mb:.1f} MB")

    # ---- Summary ----
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Done! ({elapsed:.1f}s)")
    print(f"  Sequence rows scanned: {total_seq_rows:,}")
    print(f"  Total output users: {total_written:,}")
    for name, lo, hi in BUCKETS:
        uids = bucket_uids[name]
        written = sum(1 for uid in uids if uid in row_map)
        print(f"    {name}: {written:,}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
