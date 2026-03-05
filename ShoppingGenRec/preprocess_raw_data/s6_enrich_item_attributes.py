"""
Enrich merged_clean_item.json with structured attributes from
RawOfferAttributes.tsv.

Reads:
  1. merged_clean_item.json      - unified item metadata from s4.
     Keys are either GlobalOfferIds (numeric strings) or PageTitle indices
     (P-prefixed, e.g., "P0", "P123").
  2. RawOfferAttributes.tsv      - tab-separated file with per-offer
     attributes including Brand, Seller, Gender, AgeGroup, Model, Color,
     Size, Material, Price, Market, etc.  Keyed by GlobalOfferId column.

Produces:
  1. merged_clean_item_with_attr.json - same as merged_clean_item.json but
     with two additions per entry:
       a) If a GlobalOfferId entry had an empty description, it is back-filled
          from the TSV's Description column.
       b) An "attributes" dict is added to every entry.  For PageTitle entries
          it is {}.  For GlobalOfferId entries it contains whichever of the
          following fields are non-empty in the TSV (in this order):
            Brand, Seller, Gender, AgeGroup, Model, Color, Size, Material,
            Price, Market.

Pipeline:
  1. Load merged_clean_item.json.
  2. Classify keys into GlobalOfferId vs PageTitle sets; print counts.
  3. Stream-read RawOfferAttributes.tsv, indexing rows by GlobalOfferId.
  4. Compute join statistics (how many GlobalOfferIds are found in the TSV).
  5. Back-fill missing descriptions.
  6. Build per-item attributes dict.
  7. Print per-attribute coverage statistics.
  8. Write merged_clean_item_with_attr.json.
  9. Print 10 sample entries.

Usage:
    python s5_enrich_item_attributes.py \
        --merged_item_file ../raw_data/merged_clean_item.json \
        --raw_attr_file    ../raw_data/RawOfferAttributes.tsv \
        --output_file      ../raw_data/merged_clean_item_with_attr.json
"""

import argparse
import csv
import json
import os
import sys
from tqdm import tqdm


# =============================================================================
# Constants
# =============================================================================

# Attributes to extract from the TSV, in the required order.
ATTRIBUTE_FIELDS = [
    "Brand",
    "Seller",
    "Gender",
    "AgeGroup",
    "Model",
    "Color",
    "Size",
    "Material",
    "Price",
    "Market",
]


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


def load_raw_attributes(tsv_path, needed_gids):
    """Stream-read RawOfferAttributes.tsv and index rows by GlobalOfferId.

    Optimized for very large files (TB-scale):
      - Reads raw lines instead of csv.DictReader to avoid building a
        Python dict for every row.
      - Parses the header once to find column indices.
      - For each data line, splits by tab and checks only the
        GlobalOfferId column.  Full dict is built only for matching rows.
      - Shows a tqdm progress bar based on bytes read.

    Only rows whose GlobalOfferId appears in *needed_gids* are kept in
    memory, so we don't need to hold the entire TSV.

    Args:
        tsv_path: Path to the TSV file.
        needed_gids: Set of GlobalOfferId strings we care about.

    Returns:
        Dict mapping GlobalOfferId -> dict of column values.
    """
    gid_to_row = {}
    total_rows = 0
    matched_rows = 0

    file_size = os.path.getsize(tsv_path)

    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        # Parse header to get column indices
        header_line = f.readline()
        if not header_line:
            print("  WARNING: TSV file is empty!")
            return gid_to_row
        columns = header_line.rstrip("\n\r").split("\t")

        try:
            gid_col_idx = columns.index("GlobalOfferId")
        except ValueError:
            print("  ERROR: 'GlobalOfferId' column not found in TSV header!")
            print(f"  Available columns: {columns}")
            return gid_to_row

        bytes_read = len(header_line.encode("utf-8", errors="replace"))

        with tqdm(
            total=file_size,
            initial=bytes_read,
            unit="B",
            unit_scale=True,
            desc="  Reading TSV",
            ncols=90,
        ) as pbar:
            for line in f:
                line_bytes = len(line.encode("utf-8", errors="replace"))
                pbar.update(line_bytes)
                total_rows += 1

                fields = line.rstrip("\n\r").split("\t")

                # Quick check: does this row's GlobalOfferId match?
                if gid_col_idx >= len(fields):
                    continue
                gid = fields[gid_col_idx].strip()
                if not gid or gid not in needed_gids:
                    continue

                matched_rows += 1

                # Only build full dict for matching rows (first occurrence)
                if gid not in gid_to_row:
                    row_dict = {}
                    for i, col_name in enumerate(columns):
                        row_dict[col_name] = fields[i] if i < len(fields) else ""
                    gid_to_row[gid] = row_dict

                    # Early termination: if we've found all needed IDs, stop
                    if len(gid_to_row) == len(needed_gids):
                        pbar.update(file_size - pbar.n)  # fill progress bar
                        break

    print(f"  TSV total rows scanned:             {total_rows:>12,}")
    print(f"  Rows matching needed GlobalOfferIds: {matched_rows:>12,}")
    print(f"  Unique GlobalOfferIds matched:       {len(gid_to_row):>12,}")

    return gid_to_row


def build_attributes(tsv_row):
    """Extract structured attributes from a TSV row.

    Only non-empty fields are included, in the canonical order defined by
    ATTRIBUTE_FIELDS.

    Args:
        tsv_row: Dict of column name -> value from the TSV.

    Returns:
        OrderedDict-like dict with non-empty attribute values.
    """
    attrs = {}
    for field in ATTRIBUTE_FIELDS:
        value = tsv_row.get(field, "").strip()
        if value:
            # Keep Price as a number if possible
            if field == "Price":
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
            attrs[field] = value
    return attrs


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Enrich merged_clean_item.json with structured attributes "
                    "from RawOfferAttributes.tsv."
    )
    parser.add_argument(
        "--merged_item_file",
        type=str,
        default="./raw_data/merged_clean_item.json",
        help="Path to merged_clean_item.json from s4 "
             "(default: ./raw_data/merged_clean_item.json)",
    )
    parser.add_argument(
        "--raw_attr_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/Resource/Joint_OfferAttributes.tsv",
        help="Path to RawOfferAttributes.tsv "
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./raw_data/merged_clean_item_with_attr.json",
        help="Output path for the enriched item JSON "
             "(default: ./raw_data/merged_clean_item_with_attr.json)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # =========================================================================
    # Step 1: Load merged_clean_item.json
    # =========================================================================
    print("=" * 70)
    print("Step 1: Loading merged_clean_item.json")
    print("=" * 70)

    merged_data = load_json(args.merged_item_file)
    total_items = len(merged_data)
    print(f"  File: {args.merged_item_file}")
    print(f"  Total entries:                      {total_items:>12,}")

    # =========================================================================
    # Step 2: Classify keys into GlobalOfferId vs PageTitle
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Classifying item keys")
    print("=" * 70)

    global_offer_ids = set()
    page_title_ids = set()

    for key in merged_data:
        if key.startswith("P"):
            page_title_ids.add(key)
        else:
            global_offer_ids.add(key)

    num_gids = len(global_offer_ids)
    num_ptids = len(page_title_ids)
    gid_pct = num_gids / total_items * 100 if total_items > 0 else 0
    ptid_pct = num_ptids / total_items * 100 if total_items > 0 else 0

    print(f"  GlobalOfferId items:                {num_gids:>12,} ({gid_pct:.1f}%)")
    print(f"  PageTitle items:                    {num_ptids:>12,} ({ptid_pct:.1f}%)")

    # =========================================================================
    # Step 3: Load RawOfferAttributes.tsv (only needed GlobalOfferIds)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Loading RawOfferAttributes.tsv")
    print("=" * 70)

    print(f"  File: {args.raw_attr_file}")
    gid_to_row = load_raw_attributes(args.raw_attr_file, global_offer_ids)

    # =========================================================================
    # Step 4: Join statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Join statistics (GlobalOfferId coverage)")
    print("=" * 70)

    found_gids = global_offer_ids & set(gid_to_row.keys())
    missing_gids = global_offer_ids - set(gid_to_row.keys())
    found_pct = len(found_gids) / num_gids * 100 if num_gids > 0 else 0

    print(f"  GlobalOfferIds in merged_clean_item:{num_gids:>12,}")
    print(f"  Found in TSV:                       {len(found_gids):>12,} ({found_pct:.1f}%)")
    print(f"  Not found in TSV:                   {len(missing_gids):>12,} ({100 - found_pct:.1f}%)")

    # =========================================================================
    # Step 5: Back-fill descriptions & build attributes
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 5: Enriching items (back-fill descriptions + attributes)")
    print("=" * 70)

    desc_backfilled = 0
    attr_counts = {field: 0 for field in ATTRIBUTE_FIELDS}

    for key, info in merged_data.items():
        if key.startswith("P"):
            # PageTitle entry: empty attributes
            info["attributes"] = {}
            continue

        tsv_row = gid_to_row.get(key)
        if tsv_row is None:
            # GlobalOfferId not found in TSV: empty attributes
            info["attributes"] = {}
            continue

        # Back-fill empty description
        current_desc = info.get("description", "").strip()
        if not current_desc:
            tsv_desc = tsv_row.get("Description", "").strip()
            if tsv_desc:
                info["description"] = tsv_desc
                desc_backfilled += 1

        # Build structured attributes
        attrs = build_attributes(tsv_row)
        info["attributes"] = attrs

        # Track per-attribute coverage
        for field in ATTRIBUTE_FIELDS:
            if field in attrs:
                attr_counts[field] += 1

    print(f"  Descriptions back-filled:           {desc_backfilled:>12,}")

    # =========================================================================
    # Step 6: Per-attribute coverage statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 6: Per-attribute coverage (among GlobalOfferId items)")
    print("=" * 70)

    print(f"  {'Attribute':<20s} {'Count':>10s} {'Coverage':>10s}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10}")
    for field in ATTRIBUTE_FIELDS:
        count = attr_counts[field]
        pct = count / num_gids * 100 if num_gids > 0 else 0
        print(f"  {field:<20s} {count:>10,} {pct:>9.1f}%")

    # =========================================================================
    # Step 7: Write output
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 7: Writing enriched output")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    save_json(merged_data, args.output_file)
    output_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)

    print(f"  Output: {args.output_file}")
    print(f"  Size:   {output_size_mb:.2f} MB")
    print(f"  Total entries: {total_items:,}")

    # =========================================================================
    # Step 8: Sample 10 entries
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 8: Sample entries (up to 10)")
    print("=" * 70)

    # Show a mix: some GlobalOfferId with attributes, some PageTitle
    sample_keys = []

    # Pick up to 7 GlobalOfferId items that have attributes
    gid_with_attrs = [
        k for k in global_offer_ids if merged_data[k].get("attributes")
    ]
    sample_keys.extend(gid_with_attrs[:7])

    # Pick up to 2 GlobalOfferId items without attributes (if any)
    gid_without_attrs = [
        k for k in global_offer_ids if not merged_data[k].get("attributes")
    ]
    sample_keys.extend(gid_without_attrs[:2])

    # Pick up to 1 PageTitle item
    pt_sample = list(page_title_ids)[:1]
    sample_keys.extend(pt_sample)

    # Trim to 10
    sample_keys = sample_keys[:10]

    for idx, key in enumerate(sample_keys, 1):
        info = merged_data[key]
        print(f"\n--- Sample {idx} (key={key}) ---")
        print(f"  title:        {info.get('title', '')[:120]}")
        desc = info.get("description", "")
        print(f"  description:  {desc[:100]}{'...' if len(desc) > 100 else ''}")
        print(f"  categories:   {info.get('categories', '')[:100]}")
        print(f"  related_q:    {info.get('related_queries', '')[:100]}")
        attrs = info.get("attributes", {})
        if attrs:
            print(f"  attributes:")
            for af, av in attrs.items():
                print(f"    {af}: {av}")
        else:
            print(f"  attributes:   {{}}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total items:                        {total_items:>12,}")
    print(f"  GlobalOfferId items:                {num_gids:>12,}")
    print(f"  PageTitle items:                    {num_ptids:>12,}")
    print(f"  GlobalOfferIds found in TSV:        {len(found_gids):>12,} ({found_pct:.1f}%)")
    print(f"  Descriptions back-filled:           {desc_backfilled:>12,}")
    print(f"  Output: {args.output_file}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
