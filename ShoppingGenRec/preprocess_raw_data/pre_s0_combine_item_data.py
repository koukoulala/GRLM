"""
Combine item data from two TSV files into a single enriched JSON file.

Input TSV files:
  1. ProductsData (high priority): columns are:
     OfferId, GlobalOfferId, Title, Price, Brand, Model, Color, Size,
     Material, Condition, Gender, AgeGroup, Description, LLMCatId,
     BingCategory, GoogleCategory, MerchantCategory, Seller, Market,
     MerchantId, MerchantProductId, CdmProperties, OfferURL, OriginalImageURL
  2. JourneyProduct (lower priority): columns are:
     GlobalOfferId, Title, Embedding, Seller, Gender, OriginalPrice,
     LLMCatId, CategoryName, AgeGroup, Brand, Description, OfferUrl, ImageUrl

Output:
  A JSON file keyed by GlobalOfferId, each item containing:
    - title (str)
    - description (str)
    - categories (str) - from BingCategory (ProductsData) or CategoryName (JourneyProduct)
    - attributes (dict): Brand, Seller, Gender, AgeGroup, Price, Model,
      Color, Size, Material. Only non-empty fields are included.

Priority rules:
  - ProductsData is high priority for title, description, categories, attributes.
  - JourneyProduct fills in missing fields.
  - If a GlobalOfferId has conflicting titles across sources, it is removed.
  - Items without a title are removed.
  - Fields exceeding max_field_length are removed.
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
                      category_key=None):
    """Collect item data grouped by GlobalOfferId from a set of rows.

    Args:
        rows: List of row dicts.
        title_key: Column name for title.
        desc_key: Column name for description.
        category_key: Column name for category (e.g., "Category" or
                      "CategoryName"). If None, category is not extracted.

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

        # Category
        if category_key:
            cat = row.get(category_key, "").strip()
            if cat and not data[gid]["category"]:
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
        description="Combine item data from two TSV files into enriched item.json"
    )
    parser.add_argument(
        "--products_data_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/ProductsData.tsv",
        help="Path to ProductsData TSV file (auto-detected columns from header)",
    )
    parser.add_argument(
        "--journey_product_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/0307_EnUs_Product.tsv",
        help="Path to JourneyProduct TSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw_data",
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

    # ProductsData: explicit columns
    prod_columns = [
        "OfferId", "GlobalOfferId", "Title", "Price", "Brand", "Model",
        "Color", "Size", "Material", "Condition", "Gender", "AgeGroup",
        "Description", "LLMCatId", "BingCategory", "GoogleCategory",
        "MerchantCategory", "Seller", "Market", "MerchantId",
        "MerchantProductId", "CdmProperties", "OfferURL", "OriginalImageURL",
    ]
    print(f"\n  Reading ProductsData: {args.products_data_file}")
    prod_rows = read_tsv(args.products_data_file, expected_columns=prod_columns)
    print(f"    Rows: {len(prod_rows):,}")

    # JourneyProduct: specific columns
    journey_columns = [
        "GlobalOfferId", "Title", "Embedding", "Seller", "Gender",
        "OriginalPrice", "LLMCatId", "CategoryName", "AgeGroup",
        "Brand", "Description", "OfferUrl", "ImageUrl",
    ]
    print(f"\n  Reading JourneyProduct: {args.journey_product_file}")
    journey_rows = read_tsv(args.journey_product_file, expected_columns=journey_columns)
    print(f"    Rows: {len(journey_rows):,}")

    # =========================================================================
    # Step 2: Group data by GlobalOfferId
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Grouping data by GlobalOfferId")
    print("=" * 70)

    # ProductsData uses BingCategory for categories
    prod_data = collect_from_rows(prod_rows, category_key="BingCategory")
    journey_data = collect_from_rows(journey_rows, category_key="CategoryName")

    prod_gids = set(prod_data.keys())
    journey_gids = set(journey_data.keys())
    all_gids = prod_gids | journey_gids
    overlap_gids = prod_gids & journey_gids

    print(f"  Unique GlobalOfferIds in ProductsData:     {len(prod_gids):>10,}")
    print(f"  Unique GlobalOfferIds in JourneyProduct:   {len(journey_gids):>10,}")
    print(f"  Total unique GlobalOfferIds (union):       {len(all_gids):>10,}")
    print(f"  Overlap (in both files):                   {len(overlap_gids):>10,}")

    # =========================================================================
    # Step 3: Identify conflicting titles
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Identifying items with conflicting titles")
    print("=" * 70)

    # Merge title sets across sources
    conflicting_gids = set()
    for gid in all_gids:
        merged_titles = set()
        if gid in prod_data:
            merged_titles.update(prod_data[gid]["titles"])
        if gid in journey_data:
            merged_titles.update(journey_data[gid]["titles"])
        if len(merged_titles) > 1:
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
    }
    max_field_len = args.max_field_length

    # Attribute coverage counters
    attr_counts = {field: 0 for field in ATTRIBUTE_FIELDS}

    for gid in sorted(all_gids):
        # Skip conflicting titles
        if gid in conflicting_gids:
            stats["conflict_removed"] += 1
            continue

        # Determine title (prefer ProductsData, then JourneyProduct)
        title = ""
        for source in [prod_data, journey_data]:
            if gid in source and source[gid]["titles"]:
                title = next(iter(source[gid]["titles"]))
                break

        if not title:
            stats["no_title"] += 1
            continue

        # Get description (prefer ProductsData, then JourneyProduct)
        description = ""
        for source in [prod_data, journey_data]:
            if gid in source and source[gid]["description"]:
                description = source[gid]["description"]
                break

        # Get category (prefer ProductsData BingCategory, then JourneyProduct CategoryName)
        categories = ""
        for source in [prod_data, journey_data]:
            if gid in source and source[gid]["category"]:
                categories = source[gid]["category"]
                break

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

        # Build attributes (ProductsData high priority, JourneyProduct as base)
        attrs = {}
        # Start with JourneyProduct attributes as base
        if gid in journey_data and journey_data[gid]["row"]:
            attrs = build_attributes(journey_data[gid]["row"])
        # Overlay with ProductsData attributes (higher priority)
        if gid in prod_data and prod_data[gid]["row"]:
            prod_attrs = build_attributes(prod_data[gid]["row"])
            attrs.update(prod_attrs)

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

    # Per-source stats
    print()
    prod_kept = sum(1 for gid in prod_gids if gid in items)
    journey_kept = sum(1 for gid in journey_gids if gid in items)
    print(f"  ProductsData:    {len(prod_gids):>10,} total -> {prod_kept:>10,} kept")
    print(f"  JourneyProduct:  {len(journey_gids):>10,} total -> {journey_kept:>10,} kept")

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
