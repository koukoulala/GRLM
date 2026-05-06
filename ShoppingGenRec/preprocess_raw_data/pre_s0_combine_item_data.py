"""
Build enriched item JSON from JourneyProduct TSV file, optionally enriched
with a secondary Step0_Product TSV file.

Input TSV files:
  JourneyProduct: columns are:
     GlobalOfferId, Title, Embedding, Seller, Gender, OriginalPrice,
     LLMCatId, CategoryName, AgeGroup, Brand, Description, OfferUrl, ImageUrl

  Step0_Product (optional, has its own header row): columns are:
     OfferId, GlobalOfferId, Title, Price, Brand, Model, Color, Size,
     Material, Condition, Gender, AgeGroup, Description, LLMCatId,
     BingCategory, GoogleCategory, MerchantCategory, Seller, Market,
     MerchantId, MerchantProductId, CdmProperties, OfferURL,
     OriginalImageURL, RowNumber

  When both files are provided, Step0_Product has higher priority:
  for overlapping GlobalOfferIds, non-empty fields from Step0_Product
  overwrite JourneyProduct fields, and empty fields are filled in.

Output:
  A JSON file keyed by GlobalOfferId, each item containing:
    - title (str)
    - description (str)
    - categories (str) - from CategoryName / BingCategory
    - attributes (dict): Brand, Seller, Gender, AgeGroup, Price, Model,
      Color, Size, Material, Market. Only non-empty fields are included.

Rules:
  - Items without a title are removed.
  - Items with conflicting titles (same GlobalOfferId, different titles) are removed.
  - Fields exceeding max_field_length are truncated.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

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


def collect_from_rows(rows, title_key="Title", desc_key="Description",
                      category_key=None, category_fallback_keys=None):
    """Collect item data grouped by GlobalOfferId from a set of rows.

    Args:
        rows: List of row dicts.
        title_key: Column name for title.
        desc_key: Column name for description.
        category_key: Column name for category (e.g., "Category" or
                      "CategoryName"). If None, category is not extracted.
        category_fallback_keys: Optional list of fallback column names for
            category. Tried in order when category_key value is empty.

    Returns:
        Dict: GlobalOfferId -> {
            "titles": set, "description": str, "category": str, "row": dict
        }
    """
    data = defaultdict(lambda: {
        "titles": set(), "description": "", "category": "", "row": {}
    })
    for row in rows:
        gid = row.get("GlobalOfferId", "").strip()
        if not gid:
            continue
        title = row.get(title_key, "").strip()
        desc = row.get(desc_key, "").strip()

        if title:
            data[gid]["titles"].add(title)
        if desc and not data[gid]["description"]:
            data[gid]["description"] = desc

        # Category (with fallback keys)
        if category_key and not data[gid]["category"]:
            cat = row.get(category_key, "").strip()
            if not cat and category_fallback_keys:
                for fb_key in category_fallback_keys:
                    cat = row.get(fb_key, "").strip()
                    if cat:
                        break
            if cat:
                data[gid]["category"] = cat

        # Store full row for attribute extraction (first occurrence wins)
        if not data[gid]["row"]:
            data[gid]["row"] = row

    return data


def build_attributes(row):
    """Extract structured attributes from a row dict.

    Only non-empty fields in ATTRIBUTE_FIELDS are included.
    Price fields (OriginalPrice or Price) are converted to float if possible.

    Args:
        row: Dict of column name -> value.

    Returns:
        Dict with non-empty attribute values.
    """
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

    return attrs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build enriched item.json from JourneyProduct TSV file"
    )
    parser.add_argument(
        "--journey_product_file",
        type=str,
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/0307_EnUs_Product.tsv",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/20250401_20260331/EnUs_Product.tsv",
        help="Path to JourneyProduct TSV file",
    )
    parser.add_argument(
        "--more_product_file",
        type=str,
        default="",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/Step0_Product.tsv",
        help="Optional path to Step0_Product TSV file (has its own header). "
             "Higher priority than JourneyProduct for overlapping GIDs. "
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260430/raw_data",
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

    # JourneyProduct: auto-detect header or use predefined columns
    # Old files have no header (first row is data); new files have header with "GlobalOfferId" in it
    journey_columns_fallback = [
        "GlobalOfferId", "Title", "Embedding", "Seller", "Gender",
        "OriginalPrice", "LLMCatId", "CategoryName", "AgeGroup",
        "Brand", "Description", "OfferUrl", "ImageUrl",
    ]
    print(f"\n  Reading JourneyProduct: {args.journey_product_file}")
    # Peek at first line to check if it's a header
    with open(args.journey_product_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    first_fields = first_line.split("\t")
    if "GlobalOfferId" in first_fields:
        print(f"    Detected header row: {first_fields[:5]}...")
        journey_rows = read_tsv(args.journey_product_file)  # uses its own header
    else:
        print(f"    No header detected, using predefined columns")
        journey_rows = read_tsv(args.journey_product_file,
                                expected_columns=journey_columns_fallback)
    print(f"    Rows: {len(journey_rows):,}")
    if journey_rows:
        print(f"    Columns: {list(journey_rows[0].keys())}")

    # Step0_Product (optional): self-describing header
    product_rows = []
    if args.more_product_file:
        print(f"\n  Reading Step0_Product: {args.more_product_file}")
        product_rows = read_tsv(args.more_product_file)  # uses its own header
        print(f"    Rows: {len(product_rows):,}")

    # =========================================================================
    # Step 2: Group data by GlobalOfferId
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Grouping data by GlobalOfferId")
    print("=" * 70)

    journey_data = collect_from_rows(journey_rows, category_key="CategoryName")
    journey_gids = set(journey_data.keys())
    print(f"  Unique GIDs in JourneyProduct:             {len(journey_gids):>10,}")

    product_data = {}
    product_gids = set()
    if product_rows:
        product_data = collect_from_rows(
            product_rows,
            category_key="BingCategory",
            category_fallback_keys=["GoogleCategory", "MerchantCategory"],
        )
        product_gids = set(product_data.keys())
        print(f"  Unique GIDs in Step0_Product:              {len(product_gids):>10,}")

    # Overlap statistics
    all_gids = journey_gids | product_gids
    overlap_gids = journey_gids & product_gids
    journey_only_gids = journey_gids - product_gids
    product_only_gids = product_gids - journey_gids

    print(f"  Overlapping GIDs (in both files):          {len(overlap_gids):>10,}")
    print(f"  GIDs only in JourneyProduct:               {len(journey_only_gids):>10,}")
    print(f"  GIDs only in Step0_Product:                {len(product_only_gids):>10,}")
    print(f"  Total unique GIDs (union):                 {len(all_gids):>10,}")

    # =========================================================================
    # Step 3: Identify conflicting titles
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Identifying items with conflicting titles")
    print("=" * 70)

    # Merge title sets for conflict detection
    merged_titles = defaultdict(set)
    for gid in all_gids:
        if gid in journey_data:
            merged_titles[gid] |= journey_data[gid]["titles"]
        if gid in product_data:
            merged_titles[gid] |= product_data[gid]["titles"]

    conflicting_gids = set()
    for gid in all_gids:
        if len(merged_titles[gid]) > 1:
            conflicting_gids.add(gid)

    print(f"  GlobalOfferIds with conflicting titles:     {len(conflicting_gids):>10,}")

    # =========================================================================
    # Step 4: Build combined item data with attributes
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Building combined item data with attributes")
    print("=" * 70)

    items = {}
    stats = {
        "conflict_removed": 0,
        "no_title": 0,
        "field_truncated": 0,
        "product_enriched": 0,
        "product_new": 0,
        "fields_filled": 0,
    }
    max_field_len = args.max_field_length

    # Attribute coverage counters
    attr_counts = {field: 0 for field in ATTRIBUTE_FIELDS}

    for gid in sorted(all_gids):
        # Skip conflicting titles
        if gid in conflicting_gids:
            stats["conflict_removed"] += 1
            continue

        j_data = journey_data.get(gid)
        p_data = product_data.get(gid)

        # --- Title: product_data has higher priority ---
        title = ""
        if p_data and p_data["titles"]:
            title = next(iter(p_data["titles"]))
        if not title and j_data and j_data["titles"]:
            title = next(iter(j_data["titles"]))

        if not title:
            stats["no_title"] += 1
            continue

        # --- Description: product_data preferred, fallback to journey ---
        description = ""
        if p_data and p_data["description"]:
            description = p_data["description"]
        if not description and j_data and j_data["description"]:
            description = j_data["description"]

        # --- Category: product_data preferred, fallback to journey ---
        categories = ""
        if p_data and p_data["category"]:
            categories = p_data["category"]
        if not categories and j_data and j_data["category"]:
            categories = j_data["category"]

        # Truncate overly long fields
        if len(title) > max_field_len:
            stats["field_truncated"] += 1
            title = title[:max_field_len]
        if len(description) > max_field_len:
            stats["field_truncated"] += 1
            description = description[:max_field_len]
        if len(categories) > max_field_len:
            stats["field_truncated"] += 1
            categories = categories[:max_field_len]

        # --- Attributes: product_data preferred, journey fills gaps ---
        attrs = {}
        if p_data and p_data["row"]:
            attrs = build_attributes(p_data["row"])
        if j_data and j_data["row"]:
            j_attrs = build_attributes(j_data["row"])
            filled_count = 0
            for k, v in j_attrs.items():
                if k not in attrs or not attrs[k]:
                    attrs[k] = v  # journey fills empty fields only
                    filled_count += 1
            if filled_count > 0:
                stats["fields_filled"] += filled_count

        # Track source contribution
        if p_data and j_data:
            stats["product_enriched"] += 1
        elif p_data and not j_data:
            stats["product_new"] += 1

        # Track attribute coverage
        for field in attr_counts:
            if field in attrs:
                attr_counts[field] += 1

        items[gid] = {
            "title": title,
            "description": description,
            "categories": categories,
            "attributes": attrs,
        }

    print(f"  Items removed (conflicting titles):        {stats['conflict_removed']:>10,}")
    print(f"  Items removed (missing title):             {stats['no_title']:>10,}")
    print(f"  Fields truncated (over max length):        {stats['field_truncated']:>10,}")
    print(f"  Items enriched by Step0_Product:           {stats['product_enriched']:>10,}")
    print(f"  Items new from Step0_Product only:         {stats['product_new']:>10,}")
    print(f"  Attribute fields filled from Step0_Product:{stats['fields_filled']:>10,}")
    print(f"  Total items in final output:               {len(items):>10,}")

    # =========================================================================
    # Step 5: Statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 5: Field & attribute coverage statistics")
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

    # Source stats
    print()
    print(f"  JourneyProduct:  {len(journey_gids):>10,} GIDs")
    if product_gids:
        print(f"  Step0_Product:   {len(product_gids):>10,} GIDs")
        print(f"  Overlap:         {len(overlap_gids):>10,} GIDs")
        print(f"  Journey only:    {len(journey_only_gids):>10,} GIDs")
        print(f"  Product only:    {len(product_only_gids):>10,} GIDs")
    print(f"  Union total:     {len(all_gids):>10,} GIDs -> {len(items):>10,} kept")

    # =========================================================================
    # Step 6: Write output
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 6: Writing output")
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
    # Step 7: Sample entries
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 7: Sample entries")
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
