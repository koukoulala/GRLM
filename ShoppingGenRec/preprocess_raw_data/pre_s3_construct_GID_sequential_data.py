"""
Construct user interaction sequence data from a SequenceData_Plat TSV file.

Combines raw browsing data with item data from s0 (item.json) to produce:
  1. item_sequential_data.txt    - compact sequential data where each line is
                                   UserId followed by valid GlobalOfferIds in
                                   chronological order
  2. full_sequential_data.json   - full sequence data with all fields per action

Pipeline:
  1. Read TSV (UserId, PageTitle, GlobalOfferId, Timestamp, Source, Query)
  2. Temporal deduplication: within each user, collapse actions within a
     configurable time window (default: 5s), keeping the highest-priority
     source (Click-types > others > Query-types)
  3. GlobalOfferId validation: load item.json, keep only valid GlobalOfferIds.
     Invalid GlobalOfferIds are cleared but the action is kept in full_seq.
  4. Minimum sequence length filter: users whose item sequence (valid
     GlobalOfferIds only) is shorter than the threshold are discarded
  5. Generate output files

Item identifiers in item_sequential_data.txt:
  - Only valid GlobalOfferIds (e.g., "88880989482")

full_sequential_data.json format per action:
  - Timestamp: original timestamp string
  - Source: original source string
  - GlobalOfferId: GlobalOfferId if found in item.json, otherwise ""
  - PageTitle: raw PageTitle string (original text, not mapped)
  - Query: raw query text
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

from tqdm import tqdm

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)


# =============================================================================
# Constants
# =============================================================================

# Source priority keywords for temporal deduplication.
# Sources containing "Click" get high priority (kept first);
# sources containing "Query" get low priority (dropped first);
# all others fall in between.
_PRIORITY_CLICK = 0    # highest: explicit click actions
_PRIORITY_DEFAULT = 50 # middle: other interactions
_PRIORITY_QUERY = 100  # lowest: search query results


# =============================================================================
# Utility Functions
# =============================================================================

def read_tsv(filepath, expected_columns=None):
    """Read a TSV file and return rows as list of dicts.

    Args:
        filepath: Path to the TSV file.
        expected_columns: Optional list of column names to use instead of
            the file header.

    Returns:
        A list of dicts, one per row.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        if expected_columns:
            # Skip the header row if present
            next(reader, None)
            columns = expected_columns
        else:
            header = next(reader, None)
            if header is None:
                return rows
            columns = header

        for row in reader:
            if len(row) < len(columns):
                row.extend([""] * (len(columns) - len(row)))
            elif len(row) > len(columns):
                row = row[: len(columns)]
            rows.append(dict(zip(columns, row)))

    return rows


def get_source_priority(source):
    """Get the deduplication priority for a source type.

    Lower value = higher priority (kept first during dedup).
    Sources containing "Click" -> highest priority (0).
    Sources containing "Query" -> lowest priority (100).
    All others -> middle priority (50).

    Args:
        source: Source string (e.g., "msnClick", "bingPAQuery").

    Returns:
        Integer priority value.
    """
    if "Click" in source:
        return _PRIORITY_CLICK
    if "Query" in source:
        return _PRIORITY_QUERY
    return _PRIORITY_DEFAULT


def parse_timestamp(ts_str):
    """Parse a timestamp string to float (seconds since epoch).

    Auto-detects millisecond timestamps (value > 10^12) and converts
    to seconds.

    Args:
        ts_str: Timestamp string.

    Returns:
        Float timestamp in seconds, or None if parsing fails.
    """
    try:
        ts = float(ts_str)
        # Auto-detect millisecond timestamps
        if ts > 1e12:
            ts = ts / 1000.0
        return ts
    except (ValueError, TypeError):
        return None


def deduplicate_within_time_window(actions, window_seconds=5):
    """Deduplicate user actions within a time window.

    Groups consecutive actions that fall within window_seconds of the
    group's first action. Within each group, keeps only the action with
    the highest source priority (lowest priority number). Ties are broken
    by preferring the earlier timestamp.

    Args:
        actions: List of action dicts, each must have "_timestamp" (float)
            and "Source" (str) keys.
        window_seconds: Maximum time difference (in seconds) for actions
            to be considered duplicates (default: 5).

    Returns:
        Deduplicated list of action dicts, sorted by timestamp ascending.
    """
    if not actions:
        return []

    sorted_actions = sorted(
        actions,
        key=lambda a: (a["_timestamp"], get_source_priority(a["Source"])),
    )
    result = []
    i = 0
    n = len(sorted_actions)

    while i < n:
        # Start a new window from the current action
        window_start_ts = sorted_actions[i]["_timestamp"]
        window_group = [sorted_actions[i]]
        j = i + 1

        # Collect all actions within the time window
        while j < n and (sorted_actions[j]["_timestamp"] - window_start_ts) <= window_seconds:
            window_group.append(sorted_actions[j])
            j += 1

        # Keep the action with highest priority (lowest number);
        # break ties by earlier timestamp
        best = min(
            window_group,
            key=lambda a: (get_source_priority(a["Source"]), a["_timestamp"]),
        )
        result.append(best)
        i = j

    return result


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct user interaction sequence data from "
                    "SequenceData_Plat TSV"
    )
    parser.add_argument(
        "--sequence_data_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/0128_0301/SequenceData_Plat.tsv",
        help="Path to the SequenceData_Plat TSV file "
             "(columns: UserId, PageTitle, GlobalOfferId, Timestamp, "
             "Source, Query)",
    )
    parser.add_argument(
        "--item_json_file",
        type=str,
        default="./raw_data/item.json",
        help="Path to item.json from s0 (default: ./raw_data/item.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw_data",
        help="Path to the output directory (default: ./raw_data)",
    )
    parser.add_argument(
        "--dedup_window",
        type=float,
        default=5.0,
        help="Time window in seconds for temporal deduplication (default: 5)",
    )
    parser.add_argument(
        "--min_seq_length",
        type=int,
        default=3,
        help="Minimum item sequence length per user to be kept (default: 3)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # =========================================================================
    # Step 1: Read item.json
    # =========================================================================
    print("=" * 70)
    print("Step 1: Reading item.json")
    print("=" * 70)

    # Read item.json (from s0)
    with open(args.item_json_file, "r", encoding="utf-8") as f:
        item_data = json.load(f)
    print(f"  Items in item.json: {len(item_data):>12,}")

    # =========================================================================
    # Step 2: Streaming TSV -> group actions by user (with progress bar)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Streaming TSV and grouping actions by user")
    print("=" * 70)

    columns = ["UserId", "PageTitle", "GlobalOfferId", "Timestamp", "Source", "Query"]
    user_actions = defaultdict(list)
    skipped_no_user = 0
    skipped_no_timestamp = 0
    source_counts = defaultdict(int)
    total_rows = 0

    file_size = os.path.getsize(args.sequence_data_file)

    with open(args.sequence_data_file, "r", encoding="utf-8") as f:
        pbar = tqdm(total=file_size, unit="B", unit_scale=True,
                    desc="  Reading TSV", dynamic_ncols=True)
        reader = csv.reader(f, delimiter="\t")
        # Skip the header row
        next(reader, None)

        for row in reader:
            # Update progress bar by bytes consumed
            pbar.update(f.tell() - pbar.n)
            total_rows += 1

            if len(row) < len(columns):
                row.extend([""] * (len(columns) - len(row)))
            elif len(row) > len(columns):
                row = row[:len(columns)]

            row_dict = dict(zip(columns, row))

            user_id = row_dict["UserId"].strip()
            if not user_id:
                skipped_no_user += 1
                continue

            ts_str = row_dict["Timestamp"].strip()
            ts = parse_timestamp(ts_str)
            if ts is None:
                skipped_no_timestamp += 1
                continue

            source = row_dict["Source"].strip()
            source_counts[source] += 1

            action = {
                "UserId": user_id,
                "PageTitle": row_dict["PageTitle"].strip(),
                "GlobalOfferId": row_dict["GlobalOfferId"].strip(),
                "_timestamp": ts,
                "Timestamp": ts_str,
                "Source": source,
                "Query": row_dict["Query"].strip(),
            }
            user_actions[user_id].append(action)

        pbar.update(file_size - pbar.n)  # Ensure bar reaches 100%
        pbar.close()

    total_actions = sum(len(v) for v in user_actions.values())
    print(f"  Total rows in TSV: {total_rows:>12,}")
    print(f"  Skipped rows (no UserId): {skipped_no_user:>12,}")
    print(f"  Skipped rows (bad Timestamp): {skipped_no_timestamp:>12,}")
    print(f"  Total users: {len(user_actions):>12,}")
    print(f"  Total valid actions: {total_actions:>12,}")

    # Source distribution
    print(f"\n  Source distribution:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:40s} {cnt:>10,}")

    # User sequence length statistics
    seq_lengths = [len(v) for v in user_actions.values()]
    if seq_lengths:
        seq_sorted = sorted(seq_lengths)
        print(f"\n  User sequence length stats (before cleaning):")
        print(f"    Min:  {min(seq_lengths):>6}")
        print(f"    P50:  {seq_sorted[len(seq_sorted) // 2]:>6}")
        print(f"    P90:  {seq_sorted[int(len(seq_sorted) * 0.9)]:>6}")
        print(f"    Max:  {max(seq_lengths):>6}")
        print(f"    Avg:  {sum(seq_lengths) / len(seq_lengths):>6.1f}")

    # =========================================================================
    # Step 3: Temporal deduplication
    # =========================================================================
    print()
    print("=" * 70)
    print(f"Step 3: Temporal deduplication (window = {args.dedup_window:.1f}s)")
    print("=" * 70)

    total_before_dedup = 0
    total_after_dedup = 0
    source_removed = defaultdict(int)

    for user_id in tqdm(user_actions, desc="  Dedup users", dynamic_ncols=True):
        actions = user_actions[user_id]
        total_before_dedup += len(actions)

        # Count source types before dedup for this user
        before_sources = defaultdict(int)
        for a in actions:
            before_sources[a["Source"]] += 1

        deduped = deduplicate_within_time_window(actions, args.dedup_window)
        user_actions[user_id] = deduped
        total_after_dedup += len(deduped)

        # Count source types after dedup for this user
        after_sources = defaultdict(int)
        for a in deduped:
            after_sources[a["Source"]] += 1

        for src, cnt in before_sources.items():
            removed = cnt - after_sources.get(src, 0)
            if removed > 0:
                source_removed[src] += removed

    removed_dedup = total_before_dedup - total_after_dedup
    print(f"  Actions before dedup: {total_before_dedup:>12,}")
    print(f"  Actions after dedup: {total_after_dedup:>12,}")
    print(f"  Actions removed: {removed_dedup:>12,}")
    if total_before_dedup > 0:
        print(f"  Dedup rate: {removed_dedup / total_before_dedup * 100:>11.2f}%")

    if source_removed:
        print(f"\n  Actions removed by source:")
        for src, cnt in sorted(source_removed.items(), key=lambda x: -x[1]):
            print(f"    {src:40s} {cnt:>10,}")

    # =========================================================================
    # Step 4: Validate GlobalOfferIds against item.json
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Validating GlobalOfferIds against item.json")
    print("=" * 70)

    item_data_keys = set(item_data.keys())
    total_gids_checked = 0
    valid_gids_count = 0
    invalid_gid_count = 0
    invalid_gid_set = set()

    for user_id, actions in tqdm(user_actions.items(),
                                   desc="  Validating GIDs",
                                   total=len(user_actions),
                                   dynamic_ncols=True):
        for action in actions:
            gid = action["GlobalOfferId"]
            if gid:
                total_gids_checked += 1
                if gid in item_data_keys:
                    valid_gids_count += 1
                else:
                    # Clear invalid GIDs (keep the action for full_seq)
                    invalid_gid_count += 1
                    invalid_gid_set.add(gid)
                    action["GlobalOfferId"] = ""

    print(f"  Non-empty GlobalOfferIds checked: {total_gids_checked:>12,}")
    print(f"  Valid GlobalOfferIds: {valid_gids_count:>12,}")
    print(f"  Invalid GlobalOfferIds cleared: {invalid_gid_count:>12,}")
    print(f"  Invalid GlobalOfferIds (distinct): {len(invalid_gid_set):>12,}")
    print(f"  Remaining users: {len(user_actions):>12,}")

    if invalid_gid_set:
        sample_invalid = sorted(invalid_gid_set)[:5]
        print(f"\n  Sample invalid GlobalOfferIds:")
        for gid in sample_invalid:
            print(f"    -> {gid}")

    # =========================================================================
    # Step 5: Build item sequences and filter short sequences
    # =========================================================================
    print()
    print("=" * 70)
    print(f"Step 5: Building sequences and filtering "
          f"(min_seq_length = {args.min_seq_length})")
    print("=" * 70)

    users_too_short = 0
    actions_lost_short = 0
    final_users = {}

    for user_id, actions in tqdm(user_actions.items(),
                                   desc="  Building seqs",
                                   total=len(user_actions),
                                   dynamic_ncols=True):
        # Sort actions by timestamp (ascending = chronological order)
        actions.sort(key=lambda a: a["_timestamp"])

        # Build item sequence: only valid GlobalOfferIds
        item_sequence = []
        for action in actions:
            gid = action["GlobalOfferId"]
            if gid and gid in item_data_keys:
                item_sequence.append(gid)

        # Apply minimum sequence length filter on item sequence
        if len(item_sequence) < args.min_seq_length:
            users_too_short += 1
            actions_lost_short += len(actions)
            continue

        final_users[user_id] = {
            "actions": actions,
            "item_sequence": item_sequence,
        }

    total_items = sum(len(u["item_sequence"]) for u in final_users.values())
    total_actions_final = sum(len(u["actions"]) for u in final_users.values())

    print(f"  Users with short item sequence "
          f"(< {args.min_seq_length}): {users_too_short:>12,}")
    print(f"  Actions lost from short sequences: {actions_lost_short:>12,}")
    print(f"  Final users: {len(final_users):>12,}")
    print(f"  Total items in item sequences: {total_items:>12,}")
    print(f"  Total actions in full sequences: {total_actions_final:>12,}")

    # Sequence length statistics
    if final_users:
        item_seq_lengths = [
            len(u["item_sequence"]) for u in final_users.values()
        ]
        full_seq_lengths = [
            len(u["actions"]) for u in final_users.values()
        ]

        item_sorted = sorted(item_seq_lengths)
        full_sorted = sorted(full_seq_lengths)

        def percentile(arr, p):
            idx = int(len(arr) * p)
            return arr[min(idx, len(arr) - 1)]

        print(f"\n  Item sequence length stats (GlobalOfferId only):")
        print(f"    Min:  {min(item_seq_lengths):>6}")
        print(f"    P25:  {percentile(item_sorted, 0.25):>6}")
        print(f"    P50:  {percentile(item_sorted, 0.5):>6}")
        print(f"    Mean: {sum(item_seq_lengths) / len(item_seq_lengths):>6.1f}")
        print(f"    P75:  {percentile(item_sorted, 0.75):>6}")
        print(f"    P90:  {percentile(item_sorted, 0.9):>6}")
        print(f"    P95:  {percentile(item_sorted, 0.95):>6}")
        print(f"    P99:  {percentile(item_sorted, 0.99):>6}")
        print(f"    Max:  {max(item_seq_lengths):>6}")

        print(f"\n  Full sequence length stats (all actions):")
        print(f"    Min:  {min(full_seq_lengths):>6}")
        print(f"    P25:  {percentile(full_sorted, 0.25):>6}")
        print(f"    P50:  {percentile(full_sorted, 0.5):>6}")
        print(f"    Mean: {sum(full_seq_lengths) / len(full_seq_lengths):>6.1f}")
        print(f"    P75:  {percentile(full_sorted, 0.75):>6}")
        print(f"    P90:  {percentile(full_sorted, 0.9):>6}")
        print(f"    P95:  {percentile(full_sorted, 0.95):>6}")
        print(f"    P99:  {percentile(full_sorted, 0.99):>6}")
        print(f"    Max:  {max(full_seq_lengths):>6}")

    # =========================================================================
    # Step 6: Final source distribution statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 6: Final source distribution (after all filtering)")
    print("=" * 70)

    final_source_counts = defaultdict(int)
    for user_id, user_data in final_users.items():
        for action in user_data["actions"]:
            final_source_counts[action["Source"]] += 1

    total_final_actions = sum(final_source_counts.values())
    print(f"  Total final actions: {total_final_actions:>12,}")
    print(f"  Source distribution:")
    for src, cnt in sorted(final_source_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total_final_actions * 100 if total_final_actions > 0 else 0
        print(f"    {src:40s} {cnt:>10,} ({pct:5.1f}%)")

    # =========================================================================
    # Step 7: Write output files
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 7: Writing output files")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- item_sequential_data.txt ---
    item_seq_path = os.path.join(args.output_dir, "item_sequential_data.txt")

    # Sort users: numeric IDs sorted numerically, others alphabetically
    def user_sort_key(uid):
        try:
            return (0, int(uid), "")
        except ValueError:
            return (1, 0, uid)

    sorted_user_ids = sorted(final_users.keys(), key=user_sort_key)

    with open(item_seq_path, "w", encoding="utf-8") as f:
        for user_id in sorted_user_ids:
            seq = final_users[user_id]["item_sequence"]
            f.write(user_id + " " + " ".join(seq) + "\n")

    item_seq_size = os.path.getsize(item_seq_path) / (1024 * 1024)
    print(f"  Written: {item_seq_path}")
    print(f"    Size: {item_seq_size:.2f} MB")
    print(f"    Users: {len(final_users):,}")
    print(f"    Total item entries: {total_items:,}")

    # --- full_sequential_data.json ---
    full_seq_data = {}

    for user_id in sorted_user_ids:
        user_data = final_users[user_id]
        seq = []

        for action in user_data["actions"]:
            entry = {
                "Timestamp": action["Timestamp"],
                "Source": action["Source"],
                "GlobalOfferId": action["GlobalOfferId"],
                "PageTitle": action["PageTitle"],
                "Query": action["Query"],
            }
            seq.append(entry)

        full_seq_data[user_id] = seq

    full_seq_path = os.path.join(args.output_dir, "full_sequential_data.json")
    with open(full_seq_path, "w", encoding="utf-8") as f:
        json.dump(full_seq_data, f, indent=2, ensure_ascii=False)

    full_seq_size = os.path.getsize(full_seq_path) / (1024 * 1024)
    print(f"\n  Written: {full_seq_path}")
    print(f"    Size: {full_seq_size:.2f} MB")
    print(f"    Users: {len(full_seq_data):,}")
    print(f"    Total action entries: {total_actions_final:,}")

    # =========================================================================
    # Step 8: Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"  Input:")
    print(f"    TSV rows:                         {total_rows:>12,}")
    print(f"    Distinct users (before filter):   "
          f"{len(user_actions) + users_too_short:>12,}")
    print(f"  Filtering:")
    print(f"    Skipped (no UserId/Timestamp):    "
          f"{skipped_no_user + skipped_no_timestamp:>12,}")
    print(f"    Removed by temporal dedup:        {removed_dedup:>12,}")
    print(f"    Invalid GlobalOfferIds cleared:   {invalid_gid_count:>12,}")
    print(f"    Users discarded (short sequence): {users_too_short:>12,}")
    print(f"  Output:")
    print(f"    Final users:                      {len(final_users):>12,}")
    print(f"    Item sequence entries (GID only): {total_items:>12,}")
    print(f"    Full sequence entries:            {total_actions_final:>12,}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
