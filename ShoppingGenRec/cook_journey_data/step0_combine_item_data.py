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
import gc
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from tqdm import tqdm

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)


def _fast_json_save(obj, file_path):
    """Save JSON using orjson (fast, compact) with fallback to stdlib json."""
    t0 = time.time()
    try:
        import orjson
        raw = orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2)
    except ImportError:
        raw = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    with open(file_path, "wb") as f:
        f.write(raw)
    elapsed = time.time() - t0
    size_mb = len(raw) / (1024 * 1024)
    print(f"  Saved {file_path} ({size_mb:.2f} MB) in {elapsed:.1f}s")


def _streaming_json_save(obj, file_path, chunk_size=50_000):
    """Save a large dict as JSON by serializing in chunks.

    Unlike _fast_json_save, this never holds the entire serialized JSON in
    memory.  Entries are batched into mini-dicts of *chunk_size*, each
    serialized with orjson in one call (fast C-level bulk serialization),
    then the outer braces are stripped and the inner content is written.
    Peak extra memory ≈ one chunk (~50 K items, ~50-100 MB).
    """
    t0 = time.time()
    try:
        import orjson
        use_orjson = True
    except ImportError:
        use_orjson = False

    count = 0
    total = len(obj)
    chunk = {}
    first_chunk = True

    with open(file_path, "wb", buffering=16 * 1024 * 1024) as f:
        f.write(b"{")
        for key, value in obj.items():
            chunk[str(key)] = value
            count += 1
            if len(chunk) >= chunk_size or count == total:
                if use_orjson:
                    raw = orjson.dumps(chunk)
                else:
                    raw = json.dumps(chunk, ensure_ascii=False).encode("utf-8")
                # Strip outer { and } to get inner key:value pairs
                inner = raw[1:-1]
                if not first_chunk:
                    f.write(b",")
                f.write(inner)
                first_chunk = False
                chunk = {}
                if count % 5_000_000 == 0:
                    elapsed = time.time() - t0
                    print(f"    ... written {count:,}/{total:,} entries ({elapsed:.0f}s)")
        f.write(b"}")

    elapsed = time.time() - t0
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  Saved {file_path} ({size_mb:,.1f} MB, {count:,} entries) in {elapsed:.1f}s")


def load_llm_cat_mapping(path):
    """Load LLMCatId -> CategoryName from a 2-col TSV (`<name>\t<id>`)."""
    mapping = {}
    if not path or not os.path.isfile(path):
        print(f"[cat] WARNING: mapping file not found: {path}")
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            name, cat_id = parts[0].strip(), parts[1].strip()
            if cat_id and name:
                mapping[cat_id] = name
    print(f"[cat] loaded {len(mapping):,} category id->name entries")
    return mapping


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


def load_category_blocklist(filepath):
    """Load category blocklist from a TSV file.

    The TSV has columns: CategoryId, CategorHierarchy, CategoryName.
    First column (CategoryId) is loaded into a blocked set.
    Skips the header row.

    Returns:
        Set of blocked CategoryId strings.
    """
    blocked = set()
    if not filepath or not os.path.isfile(filepath):
        return blocked
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            cat_id = row[0].strip()
            if cat_id:
                blocked.add(cat_id)
    return blocked


def is_category_blocked(llm_cat_id, blocked_categories):
    """Check if a category is blocked.

    If LLMCatId is pipe-separated (hierarchy), checks each component.
    Empty LLMCatId is NOT blocked (not all items have a category).

    Args:
        llm_cat_id: LLMCatId string from the data.
        blocked_categories: Set of blocked CategoryId strings.

    Returns:
        The matched CategoryId string if blocked, or None.
    """
    if not llm_cat_id or not llm_cat_id.strip():
        return None
    for cat_id in llm_cat_id.split("|"):
        cat_id = cat_id.strip()
        if cat_id and cat_id in blocked_categories:
            return cat_id
    return None


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


# Pre-compiled regex for normalize_title — based on C# NormalizeV3,
# extended with () to handle parenthesized tokens like "(Renewed)":
#   Original C# pattern: "[-!+/_\\s,.;:?\"']+";
#   return $" {Regex.Replace(s, patternStr, replaceStr)} ".ToLowerInvariant();
_RE_NORM_V3 = re.compile(r"[-!+/_\s,.;:?\"'()]+") 


def normalize_title(text):
    r"""Normalize title text for blocklist matching (equivalent to C# NormalizeV3).

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


def read_tsv(filepath, expected_columns=None, keep_columns=None):
    """Read a TSV file and return rows as list of dicts.

    Uses streaming line-by-line reading with tqdm progress bar based on
    file size.

    Args:
        filepath: Path to the TSV file.
        expected_columns: Optional list of column names. If provided, will be
            used as header instead of first row.
        keep_columns: Optional set/list of column names to keep. If provided,
            only these columns are stored in the row dicts (saves memory
            when the TSV has many unused columns). All columns are still
            parsed per line, but only the requested ones are kept.

    Returns:
        A list of dicts, one per row.
    """
    file_size = os.path.getsize(filepath)
    t0 = time.time()
    rows = []
    bytes_read = 0
    total_lines = 0

    if keep_columns and not isinstance(keep_columns, set):
        keep_columns = set(keep_columns)

    with open(filepath, "r", encoding="utf-8", buffering=128 * 1024 * 1024) as f:
        pbar = tqdm(total=file_size, unit="B", unit_scale=True,
                    desc=f"    Reading {os.path.basename(filepath)}",
                    mininterval=2)

        # Parse header
        first_line = f.readline()
        bytes_read += len(first_line.encode("utf-8"))
        pbar.update(bytes_read)

        if expected_columns:
            columns = expected_columns
        else:
            header_fields = first_line.rstrip("\r\n").split("\t")
            if not header_fields or not header_fields[0]:
                pbar.close()
                return rows
            columns = header_fields

        num_cols = len(columns)

        # Pre-compute which column indices to keep for fast filtering
        if keep_columns:
            keep_indices = [i for i, c in enumerate(columns) if c in keep_columns]
            keep_col_names = [columns[i] for i in keep_indices]
        else:
            keep_indices = None

        # Stream lines
        for line in f:
            bytes_read += len(line.encode("utf-8"))
            if bytes_read % (64 * 1024 * 1024) < len(line.encode("utf-8")):
                pbar.update(bytes_read - pbar.n)

            line = line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            nf = len(fields)
            if nf < num_cols:
                fields.extend([""] * (num_cols - nf))
            elif nf > num_cols:
                fields = fields[:num_cols]

            if keep_indices is not None:
                row = {keep_col_names[j]: fields[keep_indices[j]]
                       for j in range(len(keep_indices))}
            else:
                row = dict(zip(columns, fields))
            total_lines += 1
            rows.append(row)

        pbar.update(file_size - pbar.n)  # ensure 100%
        pbar.close()

    elapsed = time.time() - t0
    print(f"    Parsed {len(rows):,} rows in {elapsed:.1f}s")
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


def build_item(row, category_key="CategoryName", cat_mapping=None):
    """Build a single item dict from a row.

    Args:
        row: Dict of column name -> value.
        category_key: Column name for category.
        cat_mapping: Optional dict of LLMCatId -> CategoryName.
            Used when category_key column is empty but LLMCatId exists.

    Returns:
        Dict with title, description, categories, attributes.
        Returns None if the row has no title.
    """
    title = row.get("Title", "").strip()
    if not title:
        return None

    description = row.get("Description", "").strip()
    categories = row.get(category_key, "").strip()
    # Fallback: derive category from LLMCatId via mapping
    if not categories and cat_mapping:
        llm_cat_id = row.get("LLMCatId", "").strip()
        if llm_cat_id:
            # LLMCatId can be pipe-separated; map each and join
            cat_names = []
            for cid in llm_cat_id.split("|"):
                cid = cid.strip()
                if cid and cid in cat_mapping:
                    cat_names.append(cat_mapping[cid])
            categories = " > ".join(cat_names)

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

    # URLs for product display
    offer_url = row.get("OfferURL", row.get("OfferUrl", "")).strip()
    image_url = row.get("ImageUrl", "").strip()

    item = {
        "title": title,
        "description": description,
        "categories": categories,
        "attributes": attrs,
    }
    if offer_url:
        item["offer_url"] = offer_url
    if image_url:
        item["image_url"] = image_url

    return item


def deduplicate_similar_items(items, target_count, seed=42):
    """Remove near-duplicate items based on title/seller/brand/description similarity.

    Strategy:
    1. Build a fingerprint from normalized (title[:50], seller, brand, description[:30])
    2. Group items by fingerprint — identical fingerprints = near-duplicates
    3. Keep the best item per group (most attributes filled)
    4. If still > target_count after dedup, randomly sample down

    Returns a *set* of GlobalOfferIds to keep (not a new dict, to save memory).
    """
    # Aggressive normalization: remove all non-alphanumeric, collapse spaces
    _re_nonalnum = re.compile(r'[^a-z0-9]')
    _re_spaces = re.compile(r'\s+')

    def _norm(text, max_len=100):
        """Normalize text for fingerprint: lowercase, strip all punctuation/
        special chars, collapse whitespace, truncate."""
        if not text:
            return ""
        text = str(text).lower().strip()
        text = _re_nonalnum.sub(' ', text)
        text = _re_spaces.sub(' ', text).strip()
        return text[:max_len]

    def _norm_desc(text, max_len=60):
        """Extra-aggressive normalization for description: also remove common
        boilerplate words that vary across near-duplicates."""
        if not text:
            return ""
        text = str(text).lower().strip()
        text = _re_nonalnum.sub(' ', text)
        text = _re_spaces.sub(' ', text).strip()
        # Take first N chars — descriptions often start the same for
        # near-dupes and diverge in boilerplate at the end
        return text[:max_len]

    def _fingerprint(item):
        t = _norm(item.get("title", ""), 60)
        attrs = item.get("attributes", {})
        s = _norm(attrs.get("Seller", ""), 30)
        b = _norm(attrs.get("Brand", ""), 30)
        d = _norm_desc(item.get("description", ""), 60)
        return f"{t}|{s}|{b}|{d}"

    def _quality(item):
        """Higher = better item to keep."""
        score = len(item.get("attributes", {}))
        if item.get("description"):
            score += 1
        if item.get("categories"):
            score += 1
        if item.get("offer_url"):
            score += 1
        return score

    print(f"[dedup] building fingerprints for {len(items):,} items...")
    t0 = time.time()

    # One pass: group by fingerprint, keep only the best per group
    best_per_fp = {}  # fingerprint -> (gid, quality_score)
    dup_count = 0
    for gid, item in tqdm(items.items(), desc="dedup-fingerprint", mininterval=5):
        fp = _fingerprint(item)
        q = _quality(item)
        if fp not in best_per_fp:
            best_per_fp[fp] = (gid, q)
        else:
            dup_count += 1
            if q > best_per_fp[fp][1]:
                best_per_fp[fp] = (gid, q)

    kept_gids = set(gid for gid, _ in best_per_fp.values())
    del best_per_fp  # free fingerprint memory
    gc.collect()
    print(f"[dedup] {len(items):,} -> {len(kept_gids):,} unique fingerprints "
          f"({dup_count:,} near-duplicates removed) in {time.time()-t0:.1f}s")

    # If still too many, random sample
    if 0 < target_count < len(kept_gids):
        rng = random.Random(seed)
        kept_gids = set(rng.sample(list(kept_gids), target_count))
        print(f"[dedup] further sampled to {target_count:,} items")

    return kept_gids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build enriched item.json from product TSV files"
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        default=["/cosmos/projects/Recommendations/Pipelines/QualityIndexAllMarketUnify/dev_Updated/2026/05/26/IndexData_en_us_all.tsv",
                 "/cosmos/projects/Recommendations/Pipelines/QualityIndexAllMarketUnify/dev_Updated/2026/05/25/IndexData_en_us_all.tsv",
                 "/cosmos/projects/Recommendations/Pipelines/QualityIndexAllMarketUnify/dev_Updated/2026/05/22/IndexData_en_us_all.tsv",
                 "/cosmos/projects/Recommendations/Pipelines/QualityIndexAllMarketUnify/dev_Updated/2026/05/12/IndexData_en_us_all.tsv",
                 "/cosmos/projects/Recommendations/Pipelines/QualityIndexAllMarketUnify/dev_Updated/2026/05/04/IndexData_en_us_all.tsv",
                 #"/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/ProductGroup/20260528_ProductBestOffer_Sampled.tsv",],
        help="Path(s) to product TSV files (JourneyProduct or IndexData). "
             "Multiple files will be merged; duplicate GlobalOfferIds are "
             "resolved by keeping the row with the latest Date.",
    )
    parser.add_argument(
        "--cat_mapping",
        type=str,
        default="/vc_data/users/wangying/OneRec/ShoppingJourney/Pipeline/res/LLMCatMapping.tsv",
        help="Path to LLMCatMapping.tsv ('<CategoryName>\\t<LLMCatId>'). "
             "Used to derive category names when CategoryName column is "
             "missing (e.g., IndexData TSV input).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_PG",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB",
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
        "--category_blocklist",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "resources", "CELA.Category.Blocklist.Clean.tsv"),
        help="Path to category blocklist TSV file (CategoryId as first column). "
             "Items whose LLMCatId matches a blocked CategoryId will be removed. "
             "Also used to look up CategoryName when not in data.",
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
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=15000000,
        help="If > 0, apply similarity-based dedup at output time to reduce "
             "item.json to approximately this many items. item_full.json is "
             "always written with all items. Near-duplicate items (same "
             "title/seller/brand/description) are merged first, then random "
             "sampling if still over the target. (default: 0 = no sampling)",
    )
    parser.add_argument(
        "--skip_filter",
        action="store_true",
        default=False,
        help="If set, skip all blocklist filtering (category/seller/title). "
             "Useful for quick sampling without waiting for filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args()


# Columns actually used by build_item() and dedup_rows_by_date().
# Only these are kept when reading TSV files to save memory (~4x reduction
# for IndexData files with 69 columns).
_NEEDED_COLUMNS = {
    "GlobalOfferId", "Title", "Description", "Seller", "Gender",
    "OriginalPrice", "LLMCatId", "CategoryName", "AgeGroup",
    "Brand", "Color", "Size", "Material", "Market", "Price",
    "Date", "OfferURL", "OfferUrl", "ImageUrl",
}


def main():
    args = parse_args()

    # =========================================================================
    # Step 1: Read input TSV files
    # =========================================================================
    print("=" * 70)
    print("Step 1: Reading input TSV files")
    print("=" * 70)

    all_rows = []
    for filepath in args.input_files:
        print(f"\n  Reading: {filepath}")
        # Peek at first line to check if it's a header
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        first_fields = first_line.split("\t")
        if "GlobalOfferId" in first_fields:
            print(f"    Detected header row: {first_fields[:5]}...")
            rows = read_tsv(filepath, keep_columns=_NEEDED_COLUMNS)
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
    del all_rows  # free memory (~hundreds of GB)
    after_dedup = len(deduped_rows)
    print(f"  Rows before dedup:  {before_dedup:>10,}")
    print(f"  Rows after dedup:   {after_dedup:>10,}")
    print(f"  Duplicates removed: {before_dedup - after_dedup:>10,}")

    # Per-file GID contribution after dedup
    from collections import Counter  # local import to avoid top-level
    file_gid_counts = Counter(row.get("_source_file", "unknown") for row in deduped_rows)
    print(f"\n  Per-file GID contribution (after dedup):")
    for fpath in args.input_files:
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
        "category_blocked": 0,
        "truncated_title": 0,
        "truncated_description": 0,
        "truncated_categories": 0,
    }
    max_field_len = args.max_field_length

    # Attribute coverage counters
    attr_counts = {field: 0 for field in ATTRIBUTE_FIELDS}

    # Load LLMCatId -> CategoryName mapping for TSV inputs without CategoryName
    cat_mapping = load_llm_cat_mapping(args.cat_mapping) if args.cat_mapping else {}

    # Load category blocklist
    cat_blocked_ids = set()
    has_cat_id_col = False
    if not args.skip_filter:
        cat_blocked_ids = load_category_blocklist(args.category_blocklist)
        # Check if data has LLMCatId column
        has_cat_id_col = len(deduped_rows) > 0 and "LLMCatId" in deduped_rows[0]
    else:
        print(f"\n  --skip_filter: skipping all blocklist filtering")

    for idx, row in enumerate(deduped_rows):
        gid = row.get("GlobalOfferId", "").strip()
        if not gid:
            continue

        # Category blocklist check (only if LLMCatId column exists)
        if cat_blocked_ids and has_cat_id_col:
            llm_cat_id = row.get("LLMCatId", "").strip()
            matched = is_category_blocked(llm_cat_id, cat_blocked_ids)
            if matched:
                stats["category_blocked"] += 1
                continue

        item = build_item(row, category_key="CategoryName", cat_mapping=cat_mapping)
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

        # Periodically clear processed rows to free memory
        deduped_rows[idx] = None

    del deduped_rows  # free remaining references
    gc.collect()
    print(f"  Built {len(items):,} items, freed raw row memory")

    # ----- Category blocklist result -----
    seller_blocked_count = 0
    title_blocked_count = 0
    n_title = 0
    n_desc = 0
    n_url = 0
    blocked_items = []

    if not args.skip_filter:
        if cat_blocked_ids and has_cat_id_col:
            print(f"\n  Applying category blocklist ({len(cat_blocked_ids):,} CategoryIds blocked)")
            print(f"  Items removed by category blocklist:       {stats['category_blocked']:>10,}")

        # ----- Seller blocklist filtering -----
        seller_blocklist = load_seller_blocklist(args.seller_blocklist)
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

        # ----- Title blocklist filtering (single-process, memory-efficient) -----
        title_blocklist_tokens = load_title_blocklist(args.title_blocklist, language="en")
        title_blocklist_regex = build_title_blocklist_regex(title_blocklist_tokens)
        if title_blocklist_regex:
            print(f"\n  Applying unified blocklist (title + description + URL) "
                  f"({len(title_blocklist_tokens):,} keywords)")

            gids_to_remove = []
            for gid, item in items.items():
                # Combine title + description + URL, normalize once
                parts = [item.get("title", "")]
                desc = item.get("description", "")
                url = item.get("offer_url", "")
                if desc:
                    parts.append(desc)
                if url:
                    # Normalize URL: replace path separators with spaces
                    # so "Restored-Apple-iPhone" becomes searchable words
                    parts.append(url.replace("/", " ").replace("-", " ")
                                    .replace("_", " ").replace("?", " "))
                combined = " ".join(parts)
                if not combined.strip():
                    continue
                norm = normalize_title(combined)
                m = title_blocklist_regex.search(norm)
                if m:
                    # Determine which field matched for the report
                    matched_in = "title"
                    norm_title = normalize_title(item.get("title", ""))
                    if not title_blocklist_regex.search(norm_title):
                        if desc and title_blocklist_regex.search(
                                normalize_title(desc)):
                            matched_in = "desc"
                        else:
                            matched_in = "url"
                    gids_to_remove.append(gid)
                    blocked_items.append((
                        gid, item.get("title", ""),
                        item["attributes"].get("Seller", ""),
                        item["categories"],
                        f"{matched_in}:{m.group(0)}",
                    ))

            for gid in gids_to_remove:
                del items[gid]
            title_blocked_count = len(gids_to_remove)
            # Count by source
            n_title = sum(1 for b in blocked_items if b[4].startswith("title:"))
            n_desc = sum(1 for b in blocked_items if b[4].startswith("desc:"))
            n_url = sum(1 for b in blocked_items if b[4].startswith("url:"))
            print(f"  Items removed by blocklist:                 {title_blocked_count:>10,}")
            print(f"    matched in title:                         {n_title:>10,}")
            print(f"    matched in description:                   {n_desc:>10,}")
            print(f"    matched in URL:                           {n_url:>10,}")
        else:
            print(f"\n  Title/description/URL blocklist: not applied (no valid file)")

    print(f"\n  Items removed (missing title):             {stats['no_title']:>10,}")
    print(f"  Titles truncated (over {max_field_len} chars):       {stats['truncated_title']:>10,}")
    print(f"  Descriptions truncated (over {max_field_len} chars): {stats['truncated_description']:>10,}")
    print(f"  Categories truncated (over {max_field_len} chars):   {stats['truncated_categories']:>10,}")
    print(f"\n  Total items in final output:               {len(items):>10,}")

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
    # Recount attribute coverage after filtering (skip if no filtering was done)
    if not args.skip_filter:
        attr_counts = {field: 0 for field in ATTRIBUTE_FIELDS}
        for item in items.values():
            for field in ATTRIBUTE_FIELDS:
                if field in item["attributes"]:
                    attr_counts[field] += 1
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
    total_item_count = len(items)

    # --- 5a. Always write item_full.json first (streaming to avoid OOM) ---
    full_output_path = os.path.join(args.output_dir, "item_full.json")
    print(f"  Writing item_full.json ({total_item_count:,} items, streaming)...")
    _streaming_json_save(items, full_output_path)
    print(f"  item_full.json: {total_item_count:,} items (full)")

    # --- 5b. Sample entries for display (before potential in-place deletion) ---
    sample_display = []
    if items:
        all_keys = list(items.keys())
        with_attrs = [k for k in all_keys[:200000] if items[k]["attributes"]]
        without_attrs = [k for k in all_keys[:200000] if not items[k]["attributes"]]
        sample_keys = []
        if with_attrs:
            sample_keys.extend(random.sample(with_attrs, min(3, len(with_attrs))))
        if without_attrs:
            sample_keys.extend(random.sample(without_attrs, min(2, len(without_attrs))))
        for key in sample_keys[:5]:
            sample_display.append((key, items[key].copy()))
        del all_keys, with_attrs, without_attrs, sample_keys

    # --- 5c. Apply similarity-based dedup if sample_rows > 0 ---
    if args.sample_rows > 0 and len(items) > args.sample_rows:
        kept_gids = deduplicate_similar_items(
            items, target_count=args.sample_rows, seed=args.seed)
        # Remove items not in kept_gids in-place (avoids copying the dict)
        gids_to_remove = [gid for gid in items if gid not in kept_gids]
        print(f"  Removing {len(gids_to_remove):,} items not in sample...")
        for gid in gids_to_remove:
            del items[gid]
        del gids_to_remove, kept_gids
        gc.collect()
        output_path = os.path.join(args.output_dir, "item.json")
        print(f"  Writing item.json ({len(items):,} items, streaming)...")
        _streaming_json_save(items, output_path)
        print(f"  item.json: {len(items):,} items (deduped/sampled from {total_item_count:,})")
    else:
        # No sampling — item.json == item_full.json, just symlink or copy
        output_path = os.path.join(args.output_dir, "item.json")
        print(f"  Writing item.json ({total_item_count:,} items, streaming)...")
        _streaming_json_save(items, output_path)
        print(f"  item.json: {total_item_count:,} items (same as full, no sampling)")

    # Free the large items dict
    del items
    gc.collect()
    print(f"  [mem] freed items dict")

    # Write blocked items report
    all_blocked = blocked_items
    if all_blocked:
        blocked_path = os.path.join(args.output_dir, "blocked_items.tsv")
        with open(blocked_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(["GlobalOfferId", "Title", "Seller",
                             "CategoryName", "MatchedKeyword"])
            for row in all_blocked:
                writer.writerow(row)
        blocked_mb = os.path.getsize(blocked_path) / (1024 * 1024)
        print(f"  Blocked items report: {blocked_path} ({blocked_mb:.2f} MB)")
        print(f"    Seller-blocked:  {seller_blocked_count:,}")
        print(f"    Content-blocked: {title_blocked_count:,}")
        print(f"      (title: {n_title:,}, desc: {n_desc:,}, url: {n_url:,})")
        total_blocked = seller_blocked_count + title_blocked_count
        print(f"    Total blocked:   {total_blocked:,}")

    # =========================================================================
    # Step 6: Sample entries
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 6: Sample entries")
    print("=" * 70)

    if sample_display:
        for idx, (key, info) in enumerate(sample_display, 1):
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
    del sample_display

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total items:               {total_item_count:>10,}")
    print(f"  Removed (missing title):   {stats['no_title']:>10,}")
    if cat_blocked_ids and has_cat_id_col:
        print(f"  Removed (blocked category):{stats['category_blocked']:>10,}")
    print(f"  Removed (seller blocklist):{seller_blocked_count:>10,}")
    print(f"  Removed (content blocklist): {title_blocked_count:>10,}")
    print(f"  With description:          {has_desc:>10,}")
    print(f"  With categories:           {has_cat:>10,}")
    print(f"  With attributes:           {has_attrs:>10,}")
    print(f"  Output dir: {args.output_dir}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
