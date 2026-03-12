"""
Clean item.json and page_title_item.json by removing unused entries,
then merge them into a single merged_clean_item.json.

Reads:
  1. item_sequential_data.txt    - compact sequential data (from s2).
     Each line: UserId id1 id2 ... where ids are either GlobalOfferIds
     (numeric strings) or PageTitle indices (P-prefixed, e.g., "P0", "P123").
  2. shopping_journey.json       - shopping journey data (from s3). Keyed by
     uuid; each entry has a "journeys" list, each journey has "product_ids"
     (list of GlobalOfferId strings, no P-prefixed ids).
  3. item.json                   - item data keyed by GlobalOfferId (from s0).
  4. page_title_item.json        - page title item data keyed by P-prefixed
     indices (from s1).

Produces:
  1. merged_clean_item.json      - merged file containing cleaned items from
     both item.json and page_title_item.json, all in a unified format with
     fields: title, description, categories, attributes.  Page-title entries
     have empty strings for description and categories, and an empty dict
     for attributes.

Pipeline:
  1. Read item_sequential_data.txt and collect all referenced item IDs.
     Separate into GlobalOfferIds (non-P-prefixed) and PageTitle indices
     (P-prefixed).
  2. Read shopping_journey.json and collect all product_ids (GlobalOfferIds)
     from every journey.
  3. Merge the GlobalOfferId sets from both sources.
  4. Read item.json and page_title_item.json. Remove entries not in the
     referenced ID sets.
  5. Merge cleaned entries into unified format and write merged_clean_item.json.
"""

import argparse
import json
import os
import sys


# =============================================================================
# Utility Functions
# =============================================================================

def load_json(filepath):
    """Load a JSON file and return the parsed object.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON object (usually a dict).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    """Save a Python object as a formatted JSON file.

    Args:
        data: Object to serialize.
        filepath: Output file path.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_ids_from_sequential_data(filepath):
    """Read item_sequential_data.txt and collect all referenced IDs.

    Each line: UserId id1 id2 id3 ...
    IDs starting with 'P' are PageTitle indices; others are GlobalOfferIds.

    Args:
        filepath: Path to item_sequential_data.txt.

    Returns:
        Tuple of (global_offer_ids, page_title_ids) where each is a set of
        strings.
    """
    global_offer_ids = set()
    page_title_ids = set()
    user_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_count += 1
            # First token is UserId, rest are item IDs
            for item_id in parts[1:]:
                if item_id.startswith("P"):
                    page_title_ids.add(item_id)
                else:
                    global_offer_ids.add(item_id)

    return global_offer_ids, page_title_ids, user_count


def collect_ids_from_shopping_journey(filepath):
    """Read shopping_journey.json and collect all product IDs.

    Args:
        filepath: Path to shopping_journey.json.

    Returns:
        Tuple of (global_offer_ids, journey_count, entry_count) where
        global_offer_ids is a set of GlobalOfferId strings.
    """
    data = load_json(filepath)
    global_offer_ids = set()
    journey_count = 0
    entry_count = len(data)

    for uuid, entry in data.items():
        for journey in entry.get("journeys", []):
            journey_count += 1
            for pid in journey.get("product_ids", []):
                global_offer_ids.add(pid)

    return global_offer_ids, journey_count, entry_count


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean item.json and page_title_item.json by removing "
                    "entries not referenced in sequential data or shopping "
                    "journey data."
    )
    parser.add_argument(
        "--sequential_data_file",
        type=str,
        default="./raw_data/item_sequential_data.txt",
        help="Path to item_sequential_data.txt (from s2). "
             "(default: ./raw_data/item_sequential_data.txt)",
    )
    parser.add_argument(
        "--shopping_journey_file",
        type=str,
        default="./raw_data/shopping_journeys.json",
        help="Path to shopping_journeys.json (from s3). "
             "(default: ./raw_data/shopping_journeys.json)",
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="./raw_data/item.json",
        help="Path to item.json (from s0). "
             "(default: ./raw_data/item.json)",
    )
    parser.add_argument(
        "--page_title_item_file",
        type=str,
        default="./raw_data/page_title_item.json",
        help="Path to page_title_item.json (from s1). "
             "(default: ./raw_data/page_title_item.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw_data",
        help="Path to the output directory (default: ./raw_data)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # =========================================================================
    # Step 1: Collect IDs from item_sequential_data.txt
    # =========================================================================
    print("=" * 70)
    print("Step 1: Collecting IDs from item_sequential_data.txt")
    print("=" * 70)

    seq_gids, seq_ptids, seq_user_count = collect_ids_from_sequential_data(
        args.sequential_data_file,
    )

    print(f"  File: {args.sequential_data_file}")
    print(f"  Users:                              {seq_user_count:>12,}")
    print(f"  Distinct GlobalOfferIds:            {len(seq_gids):>12,}")
    print(f"  Distinct PageTitle indices:         {len(seq_ptids):>12,}")

    # =========================================================================
    # Step 2: Collect IDs from shopping_journey.json
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Collecting IDs from shopping_journey.json")
    print("=" * 70)

    journey_gids, journey_count, journey_entry_count = (
        collect_ids_from_shopping_journey(args.shopping_journey_file)
    )

    print(f"  File: {args.shopping_journey_file}")
    print(f"  Entries (uuids):                    {journey_entry_count:>12,}")
    print(f"  Total journeys:                     {journey_count:>12,}")
    print(f"  Distinct GlobalOfferIds:            {len(journey_gids):>12,}")

    # =========================================================================
    # Step 3: Merge referenced IDs
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Merging referenced IDs")
    print("=" * 70)

    # GlobalOfferIds: union of sequential data and shopping journey
    merged_gids = seq_gids | journey_gids
    # PageTitle indices: only from sequential data (shopping journeys don't
    # contain P-prefixed ids)
    merged_ptids = seq_ptids

    # Overlap analysis
    gid_overlap = seq_gids & journey_gids
    gid_only_seq = seq_gids - journey_gids
    gid_only_journey = journey_gids - seq_gids

    print(f"  GlobalOfferIds:")
    print(f"    From sequential data only:        {len(gid_only_seq):>12,}")
    print(f"    From shopping journey only:       {len(gid_only_journey):>12,}")
    print(f"    In both sources:                  {len(gid_overlap):>12,}")
    print(f"    Merged total:                     {len(merged_gids):>12,}")
    print(f"  PageTitle indices:")
    print(f"    From sequential data:             {len(merged_ptids):>12,}")

    # =========================================================================
    # Step 4: Clean item.json
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Cleaning item.json")
    print("=" * 70)

    item_data = load_json(args.item_file)
    original_item_count = len(item_data)

    # Keep only entries whose key is in the merged GlobalOfferId set
    cleaned_item_data = {
        gid: info for gid, info in item_data.items() if gid in merged_gids
    }
    cleaned_item_count = len(cleaned_item_data)
    removed_item_count = original_item_count - cleaned_item_count

    # Check for referenced IDs missing from item.json
    item_keys = set(item_data.keys())
    missing_gids = merged_gids - item_keys

    print(f"  Original entries:                   {original_item_count:>12,}")
    print(f"  Referenced GlobalOfferIds:           {len(merged_gids):>12,}")
    print(f"  Entries kept:                       {cleaned_item_count:>12,}")
    print(f"  Entries removed (unused):           {removed_item_count:>12,}")
    if missing_gids:
        print(f"  Referenced but missing in item.json:{len(missing_gids):>12,}")

    # =========================================================================
    # Step 5: Clean page_title_item.json
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 5: Cleaning page_title_item.json")
    print("=" * 70)

    pt_data = load_json(args.page_title_item_file)
    original_pt_count = len(pt_data)

    # Keep only entries whose key is in the referenced PageTitle index set
    cleaned_pt_data = {
        ptid: info for ptid, info in pt_data.items() if ptid in merged_ptids
    }
    cleaned_pt_count = len(cleaned_pt_data)
    removed_pt_count = original_pt_count - cleaned_pt_count

    # Check for referenced IDs missing from page_title_item.json
    pt_keys = set(pt_data.keys())
    missing_ptids = merged_ptids - pt_keys

    print(f"  Original entries:                   {original_pt_count:>12,}")
    print(f"  Referenced PageTitle indices:        {len(merged_ptids):>12,}")
    print(f"  Entries kept:                       {cleaned_pt_count:>12,}")
    print(f"  Entries removed (unused):           {removed_pt_count:>12,}")
    if missing_ptids:
        print(f"  Referenced but missing in file:     {len(missing_ptids):>12,}")

    # =========================================================================
    # Step 6: Write cleaned files
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 6: Writing cleaned files")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- merged_clean_item.json ---
    # Merge cleaned item.json and page_title_item.json into a unified format.
    # All entries use the item.json schema: title, description, categories,
    # attributes. Page-title entries only have title; the rest are "" / {}.
    ITEM_FIELDS = ["title", "description", "categories"]

    merged_data = {}
    for gid, info in cleaned_item_data.items():
        merged_data[gid] = {field: info.get(field, "") for field in ITEM_FIELDS}
        merged_data[gid]["attributes"] = info.get("attributes", {})

    for ptid, info in cleaned_pt_data.items():
        merged_data[ptid] = {
            "title": info.get("title", ""),
            "description": "",
            "categories": "",
            "attributes": {},
        }

    merged_output_path = os.path.join(args.output_dir, "merged_clean_item.json")
    save_json(merged_data, merged_output_path)
    merged_size_mb = os.path.getsize(merged_output_path) / (1024 * 1024)
    merged_count = len(merged_data)
    print(f"  Written: {merged_output_path}")
    print(f"    Size:    {merged_size_mb:.2f} MB")
    print(f"    Entries: {merged_count:,} "
          f"({cleaned_item_count:,} items + {cleaned_pt_count:,} page titles)")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"  Sources:")
    print(f"    Sequential data users:            {seq_user_count:>12,}")
    print(f"    Sequential data GlobalOfferIds:   {len(seq_gids):>12,}")
    print(f"    Sequential data PageTitle indices: {len(seq_ptids):>12,}")
    print(f"    Shopping journey entries:          {journey_entry_count:>12,}")
    print(f"    Shopping journey GlobalOfferIds:   {len(journey_gids):>12,}")
    print(f"  Cleaning:")
    print(f"    item.json: {original_item_count:,} -> {cleaned_item_count:,} "
          f"(removed {removed_item_count:,})")
    print(f"    page_title_item.json: {original_pt_count:,} -> "
          f"{cleaned_pt_count:,} (removed {removed_pt_count:,})")
    print(f"  Merged output:")
    print(f"    merged_clean_item.json:           {merged_count:>12,} entries")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
