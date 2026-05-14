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
  - Items whose Seller is in the seller blocklist are removed.
  - Fields exceeding max_field_length are truncated.
"""

import argparse
import csv
import json
import multiprocessing
import os
import random
import re
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


def load_seller_blocklist(filepath):
    """Load seller blocklist from a file (one seller per line).

    Returns a set of lowercased seller names for case-insensitive matching.
    """
    sellers = set()
    if not filepath or not os.path.isfile(filepath):
        return sellers
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            seller = line.strip()
            if seller:
                sellers.add(seller.lower())
    return sellers


def load_title_blocklist(filepath, language="en"):
    """Load title blocklist keywords for a given language from a TSV file.

    The TSV file has columns: Market\tkeywords
    Filters rows where Market matches the given language.

    Returns a list of normalized keyword strings.
    """
    keywords = []
    if not filepath or not os.path.isfile(filepath):
        return keywords
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            market = row[0].strip().lower()
            keyword = row[1].strip()
            if market == language and keyword:
                keywords.append(normalize_title(keyword))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            deduped.append(kw)
    return deduped


# Pre-compiled regex for normalize_title — matches C# NormalizeV3 exactly:
#   string patternStr = "[-!+/_\\s,.;:?\"']+";
#   return $" {Regex.Replace(s, patternStr, replaceStr)} ".ToLowerInvariant();
_RE_NORM_V3 = re.compile(r"[-!+/_\s,.;:?\"']+")


def normalize_title(text):
    """Normalize title text for blocklist matching (equivalent to C# NormalizeV3).

    Replaces specific punctuation [-!+/_\s,.;:?"'] with spaces,
    lowercases, and wraps with leading/trailing space for word-boundary
    matching via Contains/substring check.
    """
    text = _RE_NORM_V3.sub(" ", text)
    return f" {text} ".lower()


def build_title_blocklist_regex(blocklist_tokens):
    """Build a single compiled regex from all blocklist tokens.

    This is much faster than checking each token individually with `in`,
    because the regex engine uses an optimized automaton for alternation.

    Args:
        blocklist_tokens: List of pre-normalized keyword strings.

    Returns:
        Compiled regex pattern, or None if no tokens.
    """
    if not blocklist_tokens:
        return None
    # Sort by length descending so longer matches take priority
    sorted_tokens = sorted(blocklist_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in sorted_tokens)
    return re.compile(pattern)


# Global variable for worker processes (set via initializer)
_worker_regex = None


def _init_worker(pattern_str):
    """Initializer for multiprocessing workers: compile regex once per process."""
    global _worker_regex
    _worker_regex = re.compile(pattern_str)


def _check_title_batch(batch):
    """Check a batch of (gid, title, seller, categories) tuples.

    Returns list of (gid, title, seller, categories, matched_kw) for blocked items.
    """
    results = []
    for gid, title, seller, categories in batch:
        if not title:
            continue
        norm_title = _RE_NORM_V3.sub(" ", title)
        norm_title = f" {norm_title} ".lower()
        m = _worker_regex.search(norm_title)
        if m:
            results.append((gid, title, seller, categories, m.group(0)))
    return results


def is_title_blocked(title, blocklist_regex):
    """Check if a title contains any blocked ngram using compiled regex.

    Args:
        title: Raw title string.
        blocklist_regex: Compiled regex from build_title_blocklist_regex().

    Returns:
        The matched keyword string if blocked, or None.
    """
    if not title or blocklist_regex is None:
        return None
    norm_title = normalize_title(title)
    m = blocklist_regex.search(norm_title)
    if m:
        return m.group(0)
    return None


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
                 "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/20250401_20260331/EnUs_Product_UpdatedBlocklist.tsv",
                 "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/20250401_20260331/EnUs_Product_0509.tsv"],
        help="Path(s) to JourneyProduct TSV files. Multiple files will be "
             "merged; duplicate GlobalOfferIds are resolved by keeping the "
             "row with the latest Date.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data",
        help="Directory to save output item.json (default: ./raw_data)",
    )
    parser.add_argument(
        "--seller_blocklist",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "resources", "Seller.Blocklist.Clean.tsv"),
        help="Path to seller blocklist file (one seller per line). "
             "Items whose Seller matches a blocklisted seller will be removed. "
             "Set to empty string to disable.",
    )
    parser.add_argument(
        "--title_blocklist",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "resources", "GlobalMarketTitleBlocklist.Clean.tsv"),
        help="Path to title blocklist TSV file (Market\tkeywords). "
             "Items whose title contains a blocked keyword will be removed.",
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

    # ----- Seller blocklist filtering -----
    seller_blocklist = load_seller_blocklist(args.seller_blocklist)
    seller_blocked_count = 0
    if seller_blocklist:
        print(f"\n  Applying seller blocklist ({len(seller_blocklist):,} sellers loaded)")
        gids_to_remove = []
        for gid, item in items.items():
            seller = item["attributes"].get("Seller", "").lower()
            if seller and seller in seller_blocklist:
                gids_to_remove.append(gid)
        for gid in gids_to_remove:
            del items[gid]
        seller_blocked_count = len(gids_to_remove)
        print(f"  Items removed by seller blocklist:          {seller_blocked_count:>10,}")
    else:
        print(f"\n  Seller blocklist: not applied (no valid file)")

    # ----- Title blocklist filtering (parallel) -----
    title_blocklist_tokens = load_title_blocklist(args.title_blocklist, language="en")
    title_blocklist_regex = build_title_blocklist_regex(title_blocklist_tokens)
    title_blocked_count = 0
    blocked_items = []  # only title-blocked items for report
    if title_blocklist_regex:
        print(f"\n  Applying title blocklist ({len(title_blocklist_tokens):,} keywords loaded)")

        # Prepare data for parallel processing
        all_pairs = [
            (gid, item["title"], item["attributes"].get("Seller", ""),
             item["categories"])
            for gid, item in items.items()
        ]

        # Split into chunks for multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 16)
        chunk_size = max(1, len(all_pairs) // num_workers)
        batches = [all_pairs[i:i + chunk_size]
                   for i in range(0, len(all_pairs), chunk_size)]

        print(f"  Using {num_workers} workers, {len(batches)} batches "
              f"({chunk_size:,} items/batch)")

        # Run in parallel
        pattern_str = title_blocklist_regex.pattern
        with multiprocessing.Pool(
            num_workers, initializer=_init_worker,
            initargs=(pattern_str,)
        ) as pool:
            batch_results = pool.map(_check_title_batch, batches)

        # Collect results
        gids_to_remove = []
        for batch_blocked in batch_results:
            for gid, title, seller, categories, matched_kw in batch_blocked:
                gids_to_remove.append(gid)
                blocked_items.append((gid, title, seller, categories, matched_kw))

        for gid in gids_to_remove:
            del items[gid]
        title_blocked_count = len(gids_to_remove)
        print(f"  Items removed by title blocklist:           {title_blocked_count:>10,}")
    else:
        print(f"\n  Title blocklist: not applied (no valid file)")

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

    # Write blocked items report
    if blocked_items:
        blocked_path = os.path.join(args.output_dir, "blocked_items.tsv")
        with open(blocked_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(["GlobalOfferId", "Title", "Seller",
                             "CategoryName", "MatchedKeyword"])
            for row in blocked_items:
                writer.writerow(row)
        blocked_mb = os.path.getsize(blocked_path) / (1024 * 1024)
        print(f"  Blocked items report: {blocked_path} ({blocked_mb:.2f} MB)")
        print(f"    Seller-blocked: {seller_blocked_count:,}")
        print(f"    Title-blocked:  {title_blocked_count:,}")
        print(f"    Total blocked:  {seller_blocked_count + title_blocked_count:,}")

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
    print(f"  Removed (missing title):   {stats['no_title']:>10,}")
    print(f"  Removed (seller blocklist):{seller_blocked_count:>10,}")
    print(f"  Removed (title blocklist): {title_blocked_count:>10,}")
    print(f"  With description:          {has_desc:>10,}")
    print(f"  With categories:           {has_cat:>10,}")
    print(f"  With attributes:           {has_attrs:>10,}")
    print(f"  Output: {output_path}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
