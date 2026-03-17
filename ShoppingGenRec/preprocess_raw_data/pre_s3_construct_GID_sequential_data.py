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
        default=4,
        help="Minimum item sequence length per user to be kept",
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
    # Step 2: Streaming TSV -> group actions by user (with GID validation)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Streaming TSV, grouping by user & validating GIDs")
    print("=" * 70)

    columns = ["UserId", "PageTitle", "GlobalOfferId", "Timestamp", "Source", "Query"]
    item_data_keys = set(item_data.keys())
    user_actions = defaultdict(list)
    skipped_no_user = 0
    skipped_no_timestamp = 0
    source_counts = defaultdict(int)
    total_rows = 0
    # GID validation counters (inline during read)
    total_gids_checked = 0
    valid_gids_count = 0
    invalid_gid_count = 0
    invalid_gid_set = set()

    file_size = os.path.getsize(args.sequence_data_file)

    # Wrapper to track bytes read (f.tell() is disabled after csv.reader calls next())
    class _ByteCounter:
        __slots__ = ("_f", "bytes_read")
        def __init__(self, f):
            self._f = f
            self.bytes_read = 0
        def __iter__(self):
            return self
        def __next__(self):
            line = next(self._f)
            self.bytes_read += len(line.encode("utf-8"))
            return line

    with open(args.sequence_data_file, "r", encoding="utf-8") as f:
        counter = _ByteCounter(f)
        pbar = tqdm(total=file_size, unit="B", unit_scale=True,
                    desc="  Reading TSV", dynamic_ncols=True)
        reader = csv.reader(counter, delimiter="\t")
        # Skip the header row
        next(reader, None)

        for row in reader:
            # Update progress bar by bytes consumed
            pbar.update(counter.bytes_read - pbar.n)
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

            # Validate GlobalOfferId inline
            gid = row_dict["GlobalOfferId"].strip()
            if gid:
                total_gids_checked += 1
                if gid in item_data_keys:
                    valid_gids_count += 1
                else:
                    invalid_gid_count += 1
                    invalid_gid_set.add(gid)
                    gid = ""  # Clear invalid GID

            action = {
                "PageTitle": row_dict["PageTitle"].strip(),
                "GlobalOfferId": gid,
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

    # GID validation summary
    print(f"\n  GID validation (inline):")
    print(f"    Non-empty GlobalOfferIds checked: {total_gids_checked:>12,}")
    print(f"    Valid GlobalOfferIds: {valid_gids_count:>12,}")
    print(f"    Invalid GlobalOfferIds cleared: {invalid_gid_count:>12,}")
    print(f"    Invalid GlobalOfferIds (distinct): {len(invalid_gid_set):>12,}")
    if invalid_gid_set:
        sample_invalid = sorted(invalid_gid_set)[:5]
        print(f"    Sample invalid GIDs:")
        for gid in sample_invalid:
            print(f"      -> {gid}")

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
        print(f"    P90:  {seq_sorted[int(len(seq_lengths) * 0.9)]:>6}")
        print(f"    Max:  {max(seq_lengths):>6}")
        print(f"    Avg:  {sum(seq_lengths) / len(seq_lengths):>6.1f}")

    # =========================================================================
    # Step 3: Dedup + build sequences + filter (single pass per user)
    # =========================================================================
    print()
    print("=" * 70)
    print(f"Step 3: Dedup + build sequences + filter "
          f"(window={args.dedup_window:.1f}s, "
          f"min_seq={args.min_seq_length})")
    print("=" * 70)

    total_before_dedup = 0
    total_after_dedup = 0
    source_removed = defaultdict(int)
    users_too_short = 0
    actions_lost_short = 0
    final_users = {}
    total_users = len(user_actions)

    # Iterate over a snapshot of keys so we can pop from user_actions
    user_ids = list(user_actions.keys())
    for user_id in tqdm(user_ids, desc="  Processing users",
                        dynamic_ncols=True):
        actions = user_actions.pop(user_id)  # pop to free memory immediately
        total_before_dedup += len(actions)

        # --- Temporal deduplication ---
        before_sources = defaultdict(int)
        for a in actions:
            before_sources[a["Source"]] += 1

        deduped = deduplicate_within_time_window(actions, args.dedup_window)
        del actions  # free original list
        total_after_dedup += len(deduped)

        after_sources = defaultdict(int)
        for a in deduped:
            after_sources[a["Source"]] += 1
        for src, cnt in before_sources.items():
            removed = cnt - after_sources.get(src, 0)
            if removed > 0:
                source_removed[src] += removed

        # --- Sort by timestamp ---
        deduped.sort(key=lambda a: a["_timestamp"])

        # --- Build item sequence (GIDs already validated in Step 2) ---
        item_sequence = []
        for action in deduped:
            gid = action["GlobalOfferId"]
            if gid:
                item_sequence.append(gid)

        # --- Min sequence length filter ---
        if len(item_sequence) < args.min_seq_length:
            users_too_short += 1
            actions_lost_short += len(deduped)
            continue

        final_users[user_id] = {
            "actions": deduped,
            "item_sequence": item_sequence,
        }

    del user_ids  # free the keys list

    removed_dedup = total_before_dedup - total_after_dedup
    total_items = sum(len(u["item_sequence"]) for u in final_users.values())
    total_actions_final = sum(len(u["actions"]) for u in final_users.values())

    print(f"  Actions before dedup: {total_before_dedup:>12,}")
    print(f"  Actions after dedup: {total_after_dedup:>12,}")
    print(f"  Actions removed by dedup: {removed_dedup:>12,}")
    if total_before_dedup > 0:
        print(f"  Dedup rate: {removed_dedup / total_before_dedup * 100:>11.2f}%")
    if source_removed:
        print(f"\n  Actions removed by source:")
        for src, cnt in sorted(source_removed.items(), key=lambda x: -x[1]):
            print(f"    {src:40s} {cnt:>10,}")

    print(f"\n  Users with short item sequence "
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
    # Step 4: Final source distribution statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Final source distribution (after all filtering)")
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
    # Step 5: Write output files (streaming JSON)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 5: Writing output files")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Sort users: numeric IDs sorted numerically, others alphabetically
    def user_sort_key(uid):
        try:
            return (0, int(uid), "")
        except ValueError:
            return (1, 0, uid)

    sorted_user_ids = sorted(final_users.keys(), key=user_sort_key)

    # --- item_sequential_data.txt ---
    item_seq_path = os.path.join(args.output_dir, "item_sequential_data.txt")

    with open(item_seq_path, "w", encoding="utf-8") as f:
        for user_id in sorted_user_ids:
            seq = final_users[user_id]["item_sequence"]
            f.write(user_id + " " + " ".join(seq) + "\n")

    item_seq_size = os.path.getsize(item_seq_path) / (1024 * 1024)
    print(f"  Written: {item_seq_path}")
    print(f"    Size: {item_seq_size:.2f} MB")
    print(f"    Users: {len(final_users):,}")
    print(f"    Total item entries: {total_items:,}")

    # --- full_sequential_data.json (streaming write) ---
    full_seq_path = os.path.join(args.output_dir, "full_sequential_data.json")

    with open(full_seq_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        for idx, user_id in enumerate(tqdm(sorted_user_ids,
                                           desc="  Writing JSON",
                                           dynamic_ncols=True)):
            user_data = final_users[user_id]
            seq = []
            for action in user_data["actions"]:
                seq.append({
                    "Timestamp": action["Timestamp"],
                    "Source": action["Source"],
                    "GlobalOfferId": action["GlobalOfferId"],
                    "PageTitle": action["PageTitle"],
                    "Query": action["Query"],
                })

            # Write this user's entry
            key_str = json.dumps(user_id, ensure_ascii=False)
            val_str = json.dumps(seq, indent=2, ensure_ascii=False)
            # Indent the value block by 2 spaces for readability
            val_str = val_str.replace("\n", "\n  ")
            f.write(f"  {key_str}: {val_str}")
            if idx < len(sorted_user_ids) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("}\n")

    full_seq_size = os.path.getsize(full_seq_path) / (1024 * 1024)
    print(f"\n  Written: {full_seq_path}")
    print(f"    Size: {full_seq_size:.2f} MB")
    print(f"    Users: {len(final_users):,}")
    print(f"    Total action entries: {total_actions_final:,}")

    # =========================================================================
    # Step 6: Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"  Input:")
    print(f"    TSV rows:                         {total_rows:>12,}")
    print(f"    Distinct users (before filter):   "
          f"{total_users:>12,}")
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
