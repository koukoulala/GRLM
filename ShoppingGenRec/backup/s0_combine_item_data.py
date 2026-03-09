"""
Combine item data from three TSV files into a single JSON file (item.json).

Input TSV files:
  1. TitleAndCategory: SID, GlobalOfferId, Title, Category
  2. Item_Description: SID, GlobalOfferId, Title, Description
  3. ShoppingJourney_Query_Products: query_id, SID, GlobalOfferId, Title, SimilarityScore

Output:
  A JSON file keyed by GlobalOfferId, each item containing:
    - title (str)
    - description (str)
    - categories (str)
    - related_queries (str, pipe-separated, max 3 distinct queries)

Rules:
  - If a GlobalOfferId maps to multiple distinct titles, it is considered invalid and removed.
  - Items without a title are removed.
  - Missing fields default to empty string.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

# Increase CSV field size limit to handle very large fields (e.g. long descriptions)
csv.field_size_limit(sys.maxsize)


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
            # Skip the header row if present
            header = next(reader, None)
            # Use the expected column names
            columns = expected_columns
        else:
            header = next(reader, None)
            if header is None:
                return rows
            columns = header

        for line_num, row in enumerate(reader, start=2):
            if len(row) < len(columns):
                # Pad short rows with empty strings
                row.extend([""] * (len(columns) - len(row)))
            elif len(row) > len(columns):
                # Truncate extra columns
                row = row[: len(columns)]
            rows.append(dict(zip(columns, row)))

    return rows


def collect_titles_and_categories(rows):
    """Collect title and category data grouped by GlobalOfferId.

    Returns:
        A dict: GlobalOfferId -> {"titles": set, "category": str}
    """
    data = defaultdict(lambda: {"titles": set(), "category": ""})
    for row in rows:
        gid = row.get("GlobalOfferId", "").strip()
        title = row.get("Title", "").strip()
        category = row.get("Category", "").strip()
        if not gid:
            continue
        if title:
            data[gid]["titles"].add(title)
        if category:
            data[gid]["category"] = category
    return data


def collect_descriptions(rows):
    """Collect description data grouped by GlobalOfferId.

    Returns:
        A dict: GlobalOfferId -> {"titles": set, "description": str}
    """
    data = defaultdict(lambda: {"titles": set(), "description": ""})
    for row in rows:
        gid = row.get("GlobalOfferId", "").strip()
        title = row.get("Title", "").strip()
        description = row.get("Description", "").strip()
        if not gid:
            continue
        if title:
            data[gid]["titles"].add(title)
        if description:
            data[gid]["description"] = description
    return data


def collect_queries(rows):
    """Collect query data grouped by GlobalOfferId.

    Returns:
        A dict: GlobalOfferId -> {"titles": set, "queries": list of (query_id, score)}
    """
    data = defaultdict(lambda: {"titles": set(), "queries": []})
    for row in rows:
        gid = row.get("GlobalOfferId", "").strip()
        title = row.get("Title", "").strip()
        query_id = row.get("query_id", "").strip()
        score_str = row.get("SimilarityScore", "").strip()
        if not gid:
            continue
        if title:
            data[gid]["titles"].add(title)
        if query_id:
            try:
                score = float(score_str) if score_str else 0.0
            except ValueError:
                score = 0.0
            data[gid]["queries"].append((query_id, score))
    return data


def find_conflicting_title_gids(title_sets_list):
    """Find GlobalOfferIds that have conflicting (multiple distinct) titles
    across all data sources.

    Args:
        title_sets_list: A list of dicts, each mapping GlobalOfferId -> set of titles.

    Returns:
        A set of GlobalOfferIds with conflicting titles.
    """
    # Merge all title sets per GlobalOfferId
    merged_titles = defaultdict(set)
    for title_sets in title_sets_list:
        for gid, titles in title_sets.items():
            merged_titles[gid].update(titles)

    conflicting = set()
    for gid, titles in merged_titles.items():
        if len(titles) > 1:
            conflicting.add(gid)

    return conflicting


def parse_args():
    parser = argparse.ArgumentParser(description="Combine item data from three TSV files into item.json")
    parser.add_argument(
        "--title_category_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/ItemDenseCaptioning/TitleAndCategory_All.tsv",
        help="Path to the TitleAndCategory TSV file (columns: SID, GlobalOfferId, Title, Category)",
    )
    parser.add_argument(
        "--description_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/ItemDenseCaptioning/Item_Description_1M.tsv",
        help="Path to the Item_Description TSV file (columns: SID, GlobalOfferId, Title, Description)",
    )
    parser.add_argument(
        "--query_products_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/ShoppingJourney_Query_Products_Resolved.tsv",
        help="Path to the ShoppingJourney_Query_Products TSV file "
        "(columns: query_id, SID, GlobalOfferId, Title, SimilarityScore)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw_data",
        help="Path to the output directory where item.json will be saved",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=3,
        help="Maximum number of distinct related queries to keep per item (default: 3)",
    )
    parser.add_argument(
        "--max_field_length",
        type=int,
        default=2000,
        help="Maximum allowed character length for any single field. "
        "Items with any field exceeding this limit are removed (default: 2000)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # Step 1: Read all three TSV files
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 1: Reading input TSV files")
    print("=" * 70)

    tc_columns = ["SID", "GlobalOfferId", "Title", "Category"]
    desc_columns = ["SID", "GlobalOfferId", "Title", "Description"]
    query_columns = ["query_id", "SID", "GlobalOfferId", "Title", "SimilarityScore"]

    tc_rows = read_tsv(args.title_category_file, expected_columns=tc_columns)
    desc_rows = read_tsv(args.description_file, expected_columns=desc_columns)
    query_rows = read_tsv(args.query_products_file, expected_columns=query_columns)

    print(f"  TitleAndCategory file:           {len(tc_rows):>10,} rows")
    print(f"  Item_Description file:           {len(desc_rows):>10,} rows")
    print(f"  ShoppingJourney_Query_Products:  {len(query_rows):>10,} rows")

    # -------------------------------------------------------------------------
    # Step 2: Group data by GlobalOfferId
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Step 2: Grouping data by GlobalOfferId")
    print("=" * 70)

    tc_data = collect_titles_and_categories(tc_rows)
    desc_data = collect_descriptions(desc_rows)
    query_data = collect_queries(query_rows)

    tc_gids = set(tc_data.keys())
    desc_gids = set(desc_data.keys())
    query_gids = set(query_data.keys())
    all_gids = tc_gids | desc_gids | query_gids

    print(f"  Unique GlobalOfferIds in TitleAndCategory:          {len(tc_gids):>10,}")
    print(f"  Unique GlobalOfferIds in Item_Description:          {len(desc_gids):>10,}")
    print(f"  Unique GlobalOfferIds in ShoppingJourney_Query:     {len(query_gids):>10,}")
    print(f"  Total unique GlobalOfferIds (union):                {len(all_gids):>10,}")

    # Query stats
    max_queries_per_item = 0
    total_queries = 0
    for gid, qdata in query_data.items():
        distinct_queries = len(set(q for q, _ in qdata["queries"]))
        max_queries_per_item = max(max_queries_per_item, distinct_queries)
        total_queries += distinct_queries
    avg_queries = total_queries / len(query_gids) if query_gids else 0
    print(f"  Max distinct queries for a single GlobalOfferId:    {max_queries_per_item:>10,}")
    print(f"  Avg distinct queries per GlobalOfferId (query file):{avg_queries:>10.2f}")

    # GlobalOfferId overlap statistics
    desc_tc_overlap = desc_gids & tc_gids
    query_tc_overlap = query_gids & tc_gids
    query_desc_overlap = query_gids & desc_gids
    all_overlap = tc_gids & desc_gids & query_gids
    print()
    print("  --- GlobalOfferId overlap between files ---")
    print(f"  Item_Description  ∩ TitleAndCategory:               {len(desc_tc_overlap):>10,}  ({len(desc_tc_overlap)/len(desc_gids)*100:.2f}% of Desc, {len(desc_tc_overlap)/len(tc_gids)*100:.2f}% of TC)")
    print(f"  Query_Products    ∩ TitleAndCategory:               {len(query_tc_overlap):>10,}  ({len(query_tc_overlap)/len(query_gids)*100:.2f}% of Query, {len(query_tc_overlap)/len(tc_gids)*100:.2f}% of TC)")
    print(f"  Query_Products    ∩ Item_Description:               {len(query_desc_overlap):>10,}  ({len(query_desc_overlap)/len(query_gids)*100:.2f}% of Query, {len(query_desc_overlap)/len(desc_gids)*100:.2f}% of Desc)")
    print(f"  All three files   ∩:                                {len(all_overlap):>10,}")

    # -------------------------------------------------------------------------
    # Step 3: Identify and remove items with conflicting titles
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Step 3: Identifying items with conflicting titles")
    print("=" * 70)

    title_sets_list = [
        {gid: d["titles"] for gid, d in tc_data.items()},
        {gid: d["titles"] for gid, d in desc_data.items()},
        {gid: d["titles"] for gid, d in query_data.items()},
    ]
    conflicting_gids = find_conflicting_title_gids(title_sets_list)
    print(f"  GlobalOfferIds with conflicting titles:             {len(conflicting_gids):>10,}")

    # -------------------------------------------------------------------------
    # Step 4: Build the combined item data
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Step 4: Building combined item data")
    print("=" * 70)

    items = {}
    no_title_count = 0
    conflict_removed_count = 0
    field_too_long_count = 0
    max_field_len = args.max_field_length

    # Track removal reason per GlobalOfferId for per-source diagnostics
    removed_conflict = set()
    removed_no_title = set()
    removed_field_long = set()

    for gid in sorted(all_gids):
        # Skip items with conflicting titles
        if gid in conflicting_gids:
            conflict_removed_count += 1
            removed_conflict.add(gid)
            continue

        # Determine the title (from any source)
        title = ""
        for source in [tc_data, desc_data, query_data]:
            if gid in source and source[gid]["titles"]:
                title = next(iter(source[gid]["titles"]))
                break

        # Skip items without any title
        if not title:
            no_title_count += 1
            removed_no_title.add(gid)
            continue

        # Get category
        category = tc_data[gid]["category"] if gid in tc_data else ""

        # Get description
        description = desc_data[gid]["description"] if gid in desc_data else ""

        # Get related queries (top N distinct queries by similarity score)
        related_queries_str = ""
        if gid in query_data:
            query_list = query_data[gid]["queries"]
            # Deduplicate and sort by score descending
            seen_queries = {}
            for q, score in query_list:
                if q not in seen_queries or score > seen_queries[q]:
                    seen_queries[q] = score
            sorted_queries = sorted(seen_queries.items(), key=lambda x: x[1], reverse=True)
            top_queries = [q for q, _ in sorted_queries[: args.max_queries]]
            related_queries_str = " | ".join(top_queries)

        # Skip items where any field exceeds the max length
        if (len(title) > max_field_len
                or len(description) > max_field_len
                or len(category) > max_field_len
                or len(related_queries_str) > max_field_len):
            field_too_long_count += 1
            removed_field_long.add(gid)
            continue

        items[gid] = {
            "title": title,
            "description": description,
            "categories": category,
            "related_queries": related_queries_str,
        }

    print(f"  Items removed due to conflicting titles:            {conflict_removed_count:>10,}")
    print(f"  Items removed due to missing title:                 {no_title_count:>10,}")
    print(f"  Items removed due to field length > {max_field_len}:  {field_too_long_count:>10,}")
    print(f"  Total items in final output:                        {len(items):>10,}")

    # Per-source-file breakdown of removal reasons
    source_labels = [
        ("TitleAndCategory", tc_gids),
        ("Item_Description", desc_gids),
        ("Query_Products", query_gids),
    ]
    print()
    print("  --- Per-source breakdown of removed GlobalOfferIds ---")
    for label, gid_set in source_labels:
        s_kept = len(gid_set & set(items.keys()))
        s_conflict = len(gid_set & removed_conflict)
        s_no_title = len(gid_set & removed_no_title)
        s_field_long = len(gid_set & removed_field_long)
        print(f"  {label}:")
        print(f"    Kept:               {s_kept:>10,}")
        print(f"    Conflicting title:  {s_conflict:>10,}")
        print(f"    Missing title:      {s_no_title:>10,}")
        print(f"    Field too long:     {s_field_long:>10,}")

    # -------------------------------------------------------------------------
    # Step 5: Compute and print detailed statistics
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Step 5: Final statistics")
    print("=" * 70)

    has_title = sum(1 for v in items.values() if v["title"])
    has_desc = sum(1 for v in items.values() if v["description"])
    has_cat = sum(1 for v in items.values() if v["categories"])
    has_queries = sum(1 for v in items.values() if v["related_queries"])
    has_all = sum(
        1
        for v in items.values()
        if v["title"] and v["description"] and v["categories"] and v["related_queries"]
    )

    print(f"  Items with title:                                   {has_title:>10,}")
    print(f"  Items with description:                             {has_desc:>10,}")
    print(f"  Items with categories:                              {has_cat:>10,}")
    print(f"  Items with related_queries:                         {has_queries:>10,}")
    print(f"  Items with ALL four fields:                         {has_all:>10,}")

    # Per-file filtering stats
    tc_kept = sum(1 for gid in tc_gids if gid in items)
    desc_kept = sum(1 for gid in desc_gids if gid in items)
    query_kept = sum(1 for gid in query_gids if gid in items)

    print()
    print(f"  TitleAndCategory:  {len(tc_gids):>10,} total -> {tc_kept:>10,} kept, {len(tc_gids) - tc_kept:>10,} filtered")
    print(f"  Item_Description:  {len(desc_gids):>10,} total -> {desc_kept:>10,} kept, {len(desc_gids) - desc_kept:>10,} filtered")
    print(f"  Query_Products:    {len(query_gids):>10,} total -> {query_kept:>10,} kept, {len(query_gids) - query_kept:>10,} filtered")

    # -------------------------------------------------------------------------
    # Step 6: Write output
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Step 6: Writing output")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "item.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output written to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Total items: {len(items):,}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
