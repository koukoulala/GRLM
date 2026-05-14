"""
Build enriched item JSON from one or more JourneyProduct TSV files.

Input TSV files:
  JourneyProduct: columns include Date, GlobalOfferId, Title, Seller, Gender,
     OriginalPrice, LLMCatId, CategoryName, AgeGroup, Brand, Description,
     OfferUrl, ImageUrl, etc.

  When multiple files are provided, rows are merged. If the same
  GlobalOfferId appears in multiple rows, the row with the latest Date is
  kept.

Output:
  A JSON file keyed by GlobalOfferId, each item containing:
    - title (str)
    - description (str)
    - categories (str) - from CategoryName
    - attributes (dict): Brand, Seller, Gender, AgeGroup, Price, Model,
      Color, Size, Material, Market. Only non-empty fields are included.

Rules:
  - Items without a title are removed.
  - Fields exceeding max_field_length are truncated.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)

# Attribute fields to extract, in canonical order.
# Category is stored as a top-level "categories" field, not here.
# Keep in sync with s6_enrich_item_attributes.py ATTRIBUTE_FIELDS.
ATTRIBUTE_FIELDS = [
    "Brand", "Seller", "Gender", "AgeGroup",
    "Model", "Color", "Size", "Material",
    "Price", "Market",
]

# Common date formats to try when parsing the Date column
DATE_FORMATS = [
    "%m/%d/%Y %I:%M:%S %p",  # e.g. 7/21/2025 12:00:00 AM
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%m/%d/%Y",
]


def parse_date(date_str):
    """Parse a date string, trying several common formats.

    Returns a datetime object, or datetime.min if parsing fails.
    """
    date_str = date_str.strip()
    if not date_str:
        return datetime.min
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime.min


def read_tsv(filepath, expected_columns=None):
    """Read a TSV file and return rows as list of dicts.

    Args:
        filepath: Path to the TSV file.
        expected_columns: Optional list of column names. If provided, will be
            used as header instead of first row.

    Returns:
        A list of dicts, one per row.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        if expected_columns:
            next(reader, None)  # skip header
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
                row = row[:len(columns)]
            rows.append(dict(zip(columns, row)))

    return rows


def dedup_rows_by_date(rows):
    """Deduplicate rows by GlobalOfferId, keeping the row with the latest Date.

    Args:
        rows: List of row dicts, each containing 'GlobalOfferId' and 'Date'.
            Rows should have a '_source_file' key for source tracking.

    Returns:
        List of deduplicated row dicts.
    """
    best = {}  # gid -> (parsed_date, row)
    for row in rows:
        gid = row.get("GlobalOfferId", "").strip()
        if not gid:
            continue
        date = parse_date(row.get("Date", ""))
        if gid not in best or date > best[gid][0]:
            best[gid] = (date, row)
    return [entry[1] for entry in best.values()]


def build_item(row, category_key="CategoryName"):
    """Build a single item dict from a row.

    Args:
        row: Dict of column name -> value.
        category_key: Column name for category.

    Returns:
        Dict with title, description, categories, attributes.
        Returns None if the row has no title.
    """
    title = row.get("Title", "").strip()
    if not title:
        return None

    description = row.get("Description", "").strip()
    categories = row.get(category_key, "").strip()

    attrs = {}
    for field in ATTRIBUTE_FIELDS:
        value = row.get(field, "").strip()
        if not value:
            continue
        if field == "Price":
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
        attrs[field] = value

    # Also check OriginalPrice as fallback for Price
    if "Price" not in attrs:
        orig_price = row.get("OriginalPrice", "").strip()
        if orig_price:
            try:
                attrs["Price"] = float(orig_price)
            except (ValueError, TypeError):
                attrs["Price"] = orig_price

    return {
        "title": title,
        "description": description,
        "categories": categories,
        "attributes": attrs,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build enriched item.json from JourneyProduct TSV files"
    )
    parser.add_argument(
        "--journey_product_files",
        type=str,
        nargs="+",
        default=["/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/20250401_20260331/EnUs_Product.tsv",
                 "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/20250401_20260331/EnUs_Product_UpdatedBlocklist.tsv"],
        help="Path(s) to JourneyProduct TSV files. Multiple files will be "
             "merged; duplicate GlobalOfferIds are resolved by keeping the "
             "row with the latest Date.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260509/raw_data",
        help="Directory to save output item.json (default: ./raw_data)",
    )
    parser.add_argument(
        "--max_field_length",
        type=int,
        default=1000,
        help="Maximum allowed character length for title/description. "
             "Items exceeding this are removed (default: 1000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # =========================================================================
    # Step 1: Read input TSV files
    # =========================================================================
    print("=" * 70)
    print("Step 1: Reading input TSV files")
    print("=" * 70)

    all_rows = []
    for filepath in args.journey_product_files:
        print(f"\n  Reading: {filepath}")
        # Peek at first line to check if it's a header
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        first_fields = first_line.split("\t")
        if "GlobalOfferId" in first_fields:
            print(f"    Detected header row: {first_fields[:5]}...")
            rows = read_tsv(filepath)  # uses its own header
        else:
            print(f"    No header detected, using predefined columns")
            journey_columns_fallback = [
                "GlobalOfferId", "Title", "Embedding", "Seller", "Gender",
                "OriginalPrice", "LLMCatId", "CategoryName", "AgeGroup",
                "Brand", "Description", "OfferUrl", "ImageUrl",
            ]
            rows = read_tsv(filepath, expected_columns=journey_columns_fallback)
        print(f"    Rows: {len(rows):,}")
        if rows:
            print(f"    Columns: {list(rows[0].keys())}")
        # Tag each row with its source file for later statistics
        for r in rows:
            r["_source_file"] = filepath
        all_rows.extend(rows)

    print(f"\n  Total rows across all files: {len(all_rows):,}")

    # =========================================================================
    # Step 2: Deduplicate by GlobalOfferId (keep latest Date)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Deduplicating by GlobalOfferId (keeping latest Date)")
    print("=" * 70)

    before_dedup = len(all_rows)
    deduped_rows = dedup_rows_by_date(all_rows)
    after_dedup = len(deduped_rows)
    print(f"  Rows before dedup:  {before_dedup:>10,}")
    print(f"  Rows after dedup:   {after_dedup:>10,}")
    print(f"  Duplicates removed: {before_dedup - after_dedup:>10,}")

    # Per-file GID contribution after dedup
    from collections import Counter
    file_gid_counts = Counter(row.get("_source_file", "unknown") for row in deduped_rows)
    print(f"\n  Per-file GID contribution (after dedup):")
    for fpath in args.journey_product_files:
        count = file_gid_counts.get(fpath, 0)
        print(f"    {os.path.basename(fpath):<50s} {count:>10,} GIDs")

    # Per-date distribution after dedup
    date_counts = Counter()
    for row in deduped_rows:
        raw_date = row.get("Date", "").strip()
        parsed = parse_date(raw_date)
        if parsed != datetime.min:
            date_key = parsed.strftime("%Y-%m-%d")
        else:
            date_key = "(no date)"
        date_counts[date_key] += 1
    print(f"\n  GID count by Date (top 20):")
    for date_key, count in date_counts.most_common(20):
        print(f"    {date_key:<20s} {count:>10,}")
    if len(date_counts) > 20:
        print(f"    ... and {len(date_counts) - 20} more dates")

    # =========================================================================
    # Step 3: Build item data with attributes
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Building item data with attributes")
    print("=" * 70)

    items = {}
    stats = {
        "no_title": 0,
        "truncated_title": 0,
        "truncated_description": 0,
        "truncated_categories": 0,
    }
    max_field_len = args.max_field_length

    # Attribute coverage counters
    attr_counts = {field: 0 for field in ATTRIBUTE_FIELDS}

    for row in deduped_rows:
        gid = row.get("GlobalOfferId", "").strip()
        if not gid:
            continue

        item = build_item(row, category_key="CategoryName")
        if item is None:
            stats["no_title"] += 1
            continue

        # Truncate overly long fields
        for key in ("title", "description", "categories"):
            if len(item[key]) > max_field_len:
                stats[f"truncated_{key}"] += 1
                item[key] = item[key][:max_field_len]

        # Track attribute coverage
        for field in attr_counts:
            if field in item["attributes"]:
                attr_counts[field] += 1

        items[gid] = item

    print(f"  Items removed (missing title):             {stats['no_title']:>10,}")
    print(f"  Titles truncated (over {max_field_len} chars):       {stats['truncated_title']:>10,}")
    print(f"  Descriptions truncated (over {max_field_len} chars): {stats['truncated_description']:>10,}")
    print(f"  Categories truncated (over {max_field_len} chars):   {stats['truncated_categories']:>10,}")
    print(f"  Total items in final output:               {len(items):>10,}")

    # =========================================================================
    # Step 4: Statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Field & attribute coverage statistics")
    print("=" * 70)

    has_title = sum(1 for v in items.values() if v["title"])
    has_desc = sum(1 for v in items.values() if v["description"])
    has_cat = sum(1 for v in items.values() if v["categories"])
    has_attrs = sum(1 for v in items.values() if v["attributes"])

    print(f"  Items with title:        {has_title:>10,}")
    print(f"  Items with description:  {has_desc:>10,}")
    print(f"  Items with categories:   {has_cat:>10,}")
    print(f"  Items with attributes:   {has_attrs:>10,}")

    print()
    total = len(items)
    print(f"  {'Attribute':<20s} {'Count':>10s} {'Coverage':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    for field in ATTRIBUTE_FIELDS:
        count = attr_counts[field]
        pct = count / total * 100 if total > 0 else 0
        print(f"  {field:<20s} {count:>10,} {pct:>9.1f}%")

    print(f"\n  Total unique GIDs: {after_dedup:>10,} -> {len(items):>10,} kept")

    # =========================================================================
    # Step 5: Write output
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 5: Writing output")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "item.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output: {output_path}")
    print(f"  Size:   {file_size_mb:.2f} MB")
    print(f"  Total items: {len(items):,}")

    # =========================================================================
    # Step 6: Sample entries
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 6: Sample entries")
    print("=" * 70)

    # Pick samples: some with attributes, some without
    all_keys = list(items.keys())
    with_attrs = [k for k in all_keys if items[k]["attributes"]]
    without_attrs = [k for k in all_keys if not items[k]["attributes"]]

    sample_keys = []
    if with_attrs:
        sample_keys.extend(random.sample(with_attrs, min(3, len(with_attrs))))
    if without_attrs:
        sample_keys.extend(random.sample(without_attrs, min(2, len(without_attrs))))
    sample_keys = sample_keys[:5]

    for idx, key in enumerate(sample_keys, 1):
        info = items[key]
        print(f"\n--- Sample {idx} (GlobalOfferId={key}) ---")
        print(f"  title:        {info['title'][:120]}")
        desc = info["description"]
        print(f"  description:  {desc[:100]}{'...' if len(desc) > 100 else ''}")
        print(f"  categories:   {info['categories'][:100]}")
        attrs = info["attributes"]
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
    print(f"  Total items:               {len(items):>10,}")
    print(f"  With description:          {has_desc:>10,}")
    print(f"  With categories:           {has_cat:>10,}")
    print(f"  With attributes:           {has_attrs:>10,}")
    print(f"  Output: {output_path}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
