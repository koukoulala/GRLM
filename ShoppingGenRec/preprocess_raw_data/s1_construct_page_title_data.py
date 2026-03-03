"""
Construct page title item data from a SequenceData_Plat TSV file.

Reads a TSV file containing user browsing sequence data and extracts distinct
page titles. Performs extensive cleaning and deduplication to produce:
  1. page_title_item.json    - indexed item data (keys: "P0", "P1", ...),
                               each containing the ORIGINAL page title text
  2. page_title_to_item.json - mapping from every original page title to its
                               assigned item index

Cleaning pipeline:
  1. Strip domain prefixes (e.g., "Amazon.com :") and suffixes (e.g., "| HSN")
     for deduplication purposes only
  2. Normalize whitespace and punctuation spacing for deduplication
  3. Filter non-product titles (e.g., "Sold Out", "Privacy Policy", "Clearance")
     via contains-matching on observed patterns
  4. Filter titles that are too short, too long (default: 500 chars), or empty
  5. Exact-match deduplication via canonical keys (case + punctuation insensitive)
  6. Near-duplicate merging using Jaccard word-overlap similarity (default >= 0.8)

Known issues addressed (observed from data samples):
  - Domain affixes: "Amazon.com :", "| HSN", "- QVC.com", ": Target", etc.
  - Whitespace variance: "Sheet Set , Plaid" vs "Sheet Set, Plaid"
  - Dash/paren spacing: "Temp - tations" vs "Temp-tations", "( 4 )" vs "(4)"
  - Non-product pages (contains-match): "Just Reduced", "Sold Out",
    "Clearance", "Shop All", "Weekly Ads", "Black Friday", "Cyber Monday",
    "Privacy Policy"
  - Near-duplicates differing by 1-2 extra words (e.g., "Watch" prefix)
  - Repeated/duplicate entries
  - Very short generic titles (e.g., "Women's Belts")
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)


# =============================================================================
# Constants
# =============================================================================

# Stopwords excluded when computing word-overlap similarity
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "its", "s", "t", "re", "ve", "ll", "d", "m",
    "this", "that", "these", "those", "not", "no", "so", "if", "then",
    "am", "has", "had", "have", "do", "does", "did", "will", "would",
    "can", "could", "may", "might", "shall", "should",
})

# Known domain / store names for suffix stripping (longest first for matching)
KNOWN_DOMAINS = sorted([
    # Major e-commerce
    "Amazon.com", "Amazon", "HSN", "QVC.com", "QVC", "Target", "Target.com",
    "Walmart.com", "Walmart", "eBay", "eBay.com", "Etsy", "Etsy.com",
    "Wayfair", "Wayfair.com", "Best Buy", "BestBuy.com",
    "Costco", "Costco.com", "Nordstrom", "Nordstrom.com",
    "Macy's", "Macys.com", "Kohl's", "Kohls.com",
    "Home Depot", "HomeDepot.com", "Lowe's", "Lowes.com",
    "Overstock", "Overstock.com", "Zappos", "Zappos.com",
    "Newegg", "Newegg.com", "Sephora", "Sephora.com",
    "SHEIN", "Temu", "Temu.com", "Chewy", "Chewy.com",
    # Streaming / media
    "Prime Video", "Netflix", "YouTube", "Hulu", "Disney+",
    # Grocery / retail
    "ALDI US", "ALDI", "Winn-Dixie", "Kroger", "Safeway", "Publix",
    "Whole Foods", "Trader Joe's",
    # Specialty / other
    "Lectric eBikes\u00ae", "Lectric eBikes",
    "Natural Life",
], key=len, reverse=True)

# Non-product page title patterns (from observed data samples).
# Matching is done on the FULL title after stripping special characters and
# lowercasing. Only exact whole-sentence matches are considered non-product.
NON_PRODUCT_PHRASES = {
    # Status / availability
    "just reduced",
    "sold out",
    "this item is sold out",
    # Promotional / sale pages
    "weekly ads",
    "weekly ad",
    "weekly ads discover deals on groceries and goods aldi us",
    "2025 woom black friday cyber monday sale",
    "clearance home",
    "clearance under 25",
    # Policy pages
    "privacy policy",
    "privacy policy winn dixie",
    # Non-product content pages
    "shop all electric bikes",
    "40 years of metallica night 1",
    "watch 40 years of metallica night 1"
}


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


def strip_domain_affixes(title):
    """Strip known domain prefixes and suffixes from a page title.

    Handles patterns like:
      - "Amazon.com : Product Name : Pet Supplies" -> "Product Name"
      - "Product Name | HSN" -> "Product Name"
      - "Product Name - QVC.com" -> "Product Name"
      - "Product Name : Target" -> "Product Name"

    Args:
        title: Raw page title string.

    Returns:
        Cleaned title with domain affixes removed.
    """
    # Strip leading domain prefix (e.g., "Amazon.com : ...", "Amazon.com | ...")
    title = re.sub(
        r"^(?:Amazon\.com|[A-Za-z0-9]+\.(?:com|org|net|co\.uk))\s*[:|]\s*",
        "", title, flags=re.IGNORECASE,
    )

    # Strip trailing domain/store suffixes iteratively (handles nested suffixes)
    for _ in range(3):
        changed = False

        # Try known domain names first (longest match first)
        for domain in KNOWN_DOMAINS:
            escaped = re.escape(domain)
            pattern = rf"\s*[|:\-\u2013\u2014]\s*{escaped}\s*$"
            new_title = re.sub(pattern, "", title, flags=re.IGNORECASE)
            if new_title != title:
                title = new_title
                changed = True
                break

        if not changed:
            # Generic pattern: trailing " | <word(s).com>" or similar
            new_title = re.sub(
                r"\s*[|:\-\u2013\u2014]\s*"
                r"[A-Za-z0-9][A-Za-z0-9\s&'.\u00ae]*"
                r"\.(?:com|org|net|co\.uk)\s*$",
                "", title, flags=re.IGNORECASE,
            )
            if new_title != title:
                title = new_title
                changed = True

        if not changed:
            break

    return title.strip()


def normalize_whitespace(title):
    """Normalize whitespace and punctuation spacing in a title.

    Fixes patterns like:
      - "Sheet Set , Plaid" -> "Sheet Set, Plaid"
      - "Temp - tations" -> "Temp-tations"
      - "( 4 )" -> "(4)"
      - Multiple spaces -> single space
      - "45 \" x 10 \"" spacing

    Args:
        title: Page title string.

    Returns:
        Whitespace-normalized title.
    """
    # Collapse all whitespace to single space
    title = re.sub(r"\s+", " ", title)
    # Remove space before punctuation: , ; : . ! ? ) ]
    title = re.sub(r"\s+([,;:.!?\)\]])", r"\1", title)
    # Remove space after opening brackets: ( [
    title = re.sub(r"([\(\[])\s+", r"\1", title)
    # Normalize isolated dashes: "word - word" -> "word-word" (compound words)
    title = re.sub(r"(\w)\s+\-\s+(\w)", r"\1-\2", title)
    # Normalize isolated slashes: "8 . 5" -> "8.5" (decimal numbers)
    title = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", title)
    # Normalize "word / word" -> "word/word" for size/option patterns
    title = re.sub(r"(\w)\s*/\s*(\w)", r"\1/\2", title)
    return title.strip()


def compute_canonical_key(title):
    """Compute a canonical key for exact-match deduplication.

    Lowercases the title and removes all non-alphanumeric characters (except
    spaces), then collapses whitespace. Two titles with the same canonical key
    are considered identical products.

    Args:
        title: Normalized page title string.

    Returns:
        Canonical key string.
    """
    key = title.lower()
    key = re.sub(r"[^a-z0-9\s]", " ", key)
    key = re.sub(r"\s+", " ", key)
    return key.strip()


def get_content_words(title):
    """Extract content words (excluding stopwords) from a title.

    Used for computing word-overlap similarity between titles.

    Args:
        title: Page title string.

    Returns:
        A frozenset of lowercase content words.
    """
    words = re.findall(r"[a-z0-9]+", title.lower())
    content = frozenset(w for w in words if w not in STOPWORDS and len(w) > 1)
    return content


def _normalize_for_non_product_check(title):
    """Normalize a title for non-product matching.

    Strips all non-alphanumeric characters (keeping spaces), lowercases,
    and collapses whitespace. This produces a clean string for exact
    whole-sentence comparison against NON_PRODUCT_PHRASES.

    Args:
        title: Raw or normalized page title string.

    Returns:
        Cleaned lowercase string with only alphanumeric chars and spaces.
    """
    text = title.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_non_product_title(title):
    """Check if a title matches known non-product page patterns.

    The title is first stripped of special characters and lowercased,
    then compared as a whole sentence against known non-product phrases.
    Only exact full-sentence matches are considered non-product.

    Args:
        title: Page title string (original or normalized).

    Returns:
        True if the title is a non-product page.
    """
    cleaned = _normalize_for_non_product_check(title)
    return cleaned in NON_PRODUCT_PHRASES


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets.

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Jaccard index (float in [0, 1]).
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class UnionFind:
    """Disjoint set data structure for grouping near-duplicate titles."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def deduplicate_by_word_overlap(titles, word_sets, threshold=0.8, min_words=3,
                                 max_posting_list_size=50000):
    """Deduplicate titles by word-overlap similarity using Union-Find.

    Uses an inverted index on content words to efficiently find candidate
    pairs, then checks Jaccard similarity. Titles with fewer than min_words
    content words are never merged by word overlap.

    For Jaccard >= 0.8, the word-count ratio must be <= 1.25, which
    significantly prunes the search space.

    Args:
        titles: List of normalized title strings.
        word_sets: Parallel list of frozensets of content words.
        threshold: Jaccard similarity threshold for merging (default: 0.8).
        min_words: Minimum content words required for word-overlap comparison.
        max_posting_list_size: Skip words whose posting list exceeds this size
            to avoid quadratic blowup on very common words.

    Returns:
        A list of sets, each containing indices of titles in the same group.
    """
    n = len(titles)
    uf = UnionFind(n)

    # Build inverted index: word -> list of title indices
    word_index = defaultdict(list)
    for i, ws in enumerate(word_sets):
        if len(ws) >= min_words:
            for w in ws:
                word_index[w].append(i)

    # Remove overly common words to avoid quadratic blowup
    oversized_words = [
        w for w, postings in word_index.items()
        if len(postings) > max_posting_list_size
    ]
    for w in oversized_words:
        del word_index[w]
    if oversized_words:
        print(f"    Skipped {len(oversized_words)} overly common words "
              f"(posting list > {max_posting_list_size:,})")

    # For Jaccard >= threshold, word count ratio must be <= 1/threshold
    max_ratio = 1.0 / threshold  # 1.25 for threshold=0.8

    merge_count = 0
    checked_pairs = 0

    for i in range(n):
        ws_i = word_sets[i]
        n_i = len(ws_i)
        if n_i < min_words:
            continue

        # Collect candidates from inverted index
        candidate_counts = defaultdict(int)
        for w in ws_i:
            if w in word_index:
                for j in word_index[w]:
                    if j > i:  # Only check each pair once
                        candidate_counts[j] += 1

        # Check promising candidates
        for j, shared_count in candidate_counts.items():
            n_j = len(word_sets[j])

            # Word count ratio check
            if max(n_i, n_j) > max_ratio * min(n_i, n_j):
                continue

            # Quick lower-bound: need |A∩B| >= threshold*(|A|+|B|)/(1+threshold)
            min_intersection = threshold * (n_i + n_j) / (1 + threshold)
            if shared_count < min_intersection * 0.9:  # Allow some slack
                continue

            # Compute exact Jaccard similarity
            checked_pairs += 1
            sim = jaccard_similarity(ws_i, word_sets[j])
            if sim >= threshold:
                uf.union(i, j)
                merge_count += 1

        # Progress reporting for large datasets
        if (i + 1) % 100000 == 0:
            print(f"    Processed {i + 1:,}/{n:,} titles, "
                  f"merges so far: {merge_count:,}")

    # Collect groups
    groups = defaultdict(set)
    for i in range(n):
        groups[uf.find(i)].add(i)

    print(f"    Candidate pairs checked: {checked_pairs:,}")
    print(f"    Word-overlap merges performed: {merge_count:,}")
    print(f"    Groups after word-overlap dedup: {len(groups):,}")

    return list(groups.values())


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct page title item data from SequenceData_Plat TSV"
    )
    parser.add_argument(
        "--sequence_data_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/1108_1208_SequenceData_Plat_Sampled_500000_User.tsv",
        help="Path to the SequenceData_Plat TSV file "
             "(columns: UserId, PageTitle, GlobalOfferId, Timestamp, Source, Query)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw_data",
        help="Path to the output directory (default: ./raw_data)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity threshold for near-duplicate merging (default: 0.8)",
    )
    parser.add_argument(
        "--min_title_chars",
        type=int,
        default=20,
        help="Minimum number of characters for a title to be kept (default: 20)",
    )
    parser.add_argument(
        "--max_title_chars",
        type=int,
        default=500,
        help="Maximum number of characters for a title (default: 500)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # =========================================================================
    # Step 1: Read TSV and extract distinct page titles
    # =========================================================================
    print("=" * 70)
    print("Step 1: Reading input TSV file")
    print("=" * 70)

    columns = ["UserId", "PageTitle", "GlobalOfferId", "Timestamp", "Source", "Query"]
    rows = read_tsv(args.sequence_data_file, expected_columns=columns)
    print(f"  Total rows in TSV: {len(rows):>12,}")

    # Collect all distinct non-empty page titles
    all_page_titles = set()
    empty_rows = 0
    for row in rows:
        pt = row.get("PageTitle", "").strip()
        if pt:
            all_page_titles.add(pt)
        else:
            empty_rows += 1

    print(f"  Rows with empty PageTitle: {empty_rows:>12,}")
    print(f"  Distinct non-empty page titles: {len(all_page_titles):>12,}")

    # =========================================================================
    # Step 2: Normalize titles and group by canonical key
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Normalizing titles and grouping by canonical key")
    print("=" * 70)

    # For each original title, compute a normalized version and canonical key
    original_to_normalized = {}  # original_title -> normalized_title
    canonical_groups = defaultdict(set)  # canonical_key -> set of original titles

    domain_stripped_count = 0
    ws_modified_count = 0

    for orig in all_page_titles:
        # Step 2a: Strip domain prefixes/suffixes
        after_domain = strip_domain_affixes(orig)
        if after_domain != orig:
            domain_stripped_count += 1

        # Step 2b: Normalize whitespace / punctuation spacing
        normalized = normalize_whitespace(after_domain)
        if normalized != after_domain:
            ws_modified_count += 1

        original_to_normalized[orig] = normalized

        # Step 2c: Canonical key for grouping
        key = compute_canonical_key(normalized)
        canonical_groups[key].add(orig)

    total_modified = sum(
        1 for o in all_page_titles if original_to_normalized[o] != o
    )
    print(f"  Titles with domain affixes stripped: {domain_stripped_count:>12,}")
    print(f"  Titles with whitespace normalized: {ws_modified_count:>12,}")
    print(f"  Total titles modified: {total_modified:>12,}")
    print(f"  Canonical groups (exact-match dedup): {len(canonical_groups):>12,}")
    print(f"  Dedup ratio: {len(all_page_titles):,} -> {len(canonical_groups):,} "
          f"({len(all_page_titles) - len(canonical_groups):,} duplicates removed)")

    # Show distribution of group sizes
    group_sizes = [len(v) for v in canonical_groups.values()]
    multi_groups = sum(1 for s in group_sizes if s > 1)
    max_group_size = max(group_sizes) if group_sizes else 0
    print(f"  Groups with multiple originals: {multi_groups:>12,}")
    print(f"  Largest group size: {max_group_size:>12,}")

    # =========================================================================
    # Step 3: Select representative title per canonical group
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Selecting representative titles")
    print("=" * 70)

    # For each group, pick the best ORIGINAL title as the representative.
    # The normalized form is still used for filtering and comparison only.
    # Selection criteria: prefer the shortest original title that is still
    # meaningful (avoids domain-appended versions), break ties alphabetically.
    group_representatives = {}  # canonical_key -> representative normalized title
    group_best_originals = {}  # canonical_key -> best original title
    group_originals = {}  # canonical_key -> set of original titles

    for key, orig_set in canonical_groups.items():
        # Pick normalized representative for filtering/comparison
        normalized_titles = [original_to_normalized[o] for o in orig_set]
        norm_counts = Counter(normalized_titles)
        best_norm = max(
            norm_counts.keys(),
            key=lambda t: (norm_counts[t], len(t)),
        )
        group_representatives[key] = best_norm
        group_originals[key] = orig_set

        # Pick the best original title for output:
        # Prefer originals that are within max_title_chars, then prefer
        # the shortest (cleanest) one; break ties alphabetically.
        valid_originals = [
            o for o in orig_set if len(o) <= args.max_title_chars
        ]
        if not valid_originals:
            valid_originals = list(orig_set)
        best_orig = min(valid_originals, key=lambda o: (len(o), o))
        group_best_originals[key] = best_orig

    print(f"  Representative titles selected: {len(group_representatives):>12,}")

    # =========================================================================
    # Step 4: Filter non-product and invalid titles
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Filtering non-product and invalid titles")
    print("=" * 70)

    filtered_keys = []
    filter_stats = {
        "empty_after_norm": 0,
        "too_short_chars": 0,
        "too_long": 0,
        "non_product": 0,
        "kept": 0,
    }
    removed_examples = defaultdict(list)  # reason -> [sample titles]

    for key, title in group_representatives.items():
        best_orig = group_best_originals[key]

        # Check empty after normalization
        if not title or not title.strip():
            filter_stats["empty_after_norm"] += 1
            if len(removed_examples["empty"]) < 3:
                removed_examples["empty"].append(repr(title))
            continue

        # Check character length (use original title length)
        if len(best_orig) < args.min_title_chars:
            filter_stats["too_short_chars"] += 1
            if len(removed_examples["short_chars"]) < 5:
                removed_examples["short_chars"].append(best_orig)
            continue

        if len(best_orig) > args.max_title_chars:
            filter_stats["too_long"] += 1
            if len(removed_examples["too_long"]) < 3:
                removed_examples["too_long"].append(best_orig[:80] + "...")
            continue

        # Check non-product patterns on both original and normalized titles.
        # If either contains a non-product pattern, remove the title.
        if is_non_product_title(title) or is_non_product_title(best_orig):
            filter_stats["non_product"] += 1
            if len(removed_examples["non_product"]) < 5:
                removed_examples["non_product"].append(best_orig)
            continue

        filtered_keys.append(key)
        filter_stats["kept"] += 1

    total_filtered = sum(v for k, v in filter_stats.items() if k != "kept")
    print(f"  Empty after normalization:          {filter_stats['empty_after_norm']:>10,}")
    print(f"  Too short (< {args.min_title_chars} chars):             "
          f"{filter_stats['too_short_chars']:>10,}")
    print(f"  Too long (> {args.max_title_chars} chars):             "
          f"{filter_stats['too_long']:>10,}")
    print(f"  Non-product patterns:               {filter_stats['non_product']:>10,}")
    print(f"  Total filtered out:                 {total_filtered:>10,}")
    print(f"  Kept:                               {filter_stats['kept']:>10,}")

    # Show sample removed titles
    for reason, samples in removed_examples.items():
        if samples:
            print(f"\n  Sample removed ({reason}):")
            for s in samples:
                print(f"    -> {s}")

    # =========================================================================
    # Step 5: Word-overlap deduplication
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 5: Near-duplicate detection (word-overlap similarity)")
    print("=" * 70)

    titles_list = [group_representatives[k] for k in filtered_keys]
    word_sets_list = [get_content_words(t) for t in titles_list]

    print(f"  Titles to deduplicate: {len(titles_list):>12,}")

    groups = deduplicate_by_word_overlap(
        titles_list, word_sets_list,
        threshold=args.similarity_threshold,
        min_words=3,
    )

    # For each group, pick the best representative:
    # prefer the canonical key with the most associated original titles,
    # then use the best original from that key.
    final_titles = []  # list of (best_original_title, set_of_canonical_keys)
    merged_group_count = 0

    for group_indices in groups:
        if len(group_indices) > 1:
            merged_group_count += 1

        # Pick the best canonical key from the group (most originals, then longest)
        best_idx = max(
            group_indices,
            key=lambda i: (len(group_originals[filtered_keys[i]]), len(titles_list[i])),
        )
        best_original = group_best_originals[filtered_keys[best_idx]]
        # Collect all canonical keys in this merged group
        all_keys = set()
        for idx in group_indices:
            all_keys.add(filtered_keys[idx])
        final_titles.append((best_original, all_keys))

    print(f"  Groups with merged near-duplicates: {merged_group_count:>12,}")
    print(f"  Final unique titles: {len(final_titles):>12,}")

    # Show sample merged groups
    sample_merges = 0
    for group_indices in groups:
        if len(group_indices) > 1 and sample_merges < 5:
            idxs = sorted(group_indices)
            print(f"\n  Sample merged group ({len(idxs)} titles):")
            for idx in idxs[:5]:
                print(f"    -> {titles_list[idx]}")
            if len(idxs) > 5:
                print(f"    ... and {len(idxs) - 5} more")
            sample_merges += 1

    # =========================================================================
    # Step 6: Assign indices and build output
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 6: Building output data")
    print("=" * 70)

    # Sort final titles alphabetically for deterministic output
    final_titles.sort(key=lambda x: x[0].lower())

    # Build page_title_item.json and page_title_to_item.json
    page_title_item = {}
    page_title_to_item = {}

    for idx, (title, key_set) in enumerate(final_titles):
        str_idx = f"P{idx}"
        page_title_item[str_idx] = {"title": title}

        # Map ALL original titles in this group to this index
        for key in key_set:
            for orig in group_originals[key]:
                page_title_to_item[orig] = str_idx

    print(f"  Total items in page_title_item.json:    {len(page_title_item):>10,}")
    print(f"  Total mappings in page_title_to_item.json: {len(page_title_to_item):>10,}")

    # Coverage: how many of the original distinct page titles got mapped
    mapped_originals = set(page_title_to_item.keys())
    unmapped = all_page_titles - mapped_originals
    print(f"  Original page titles mapped: {len(mapped_originals):>10,} "
          f"/ {len(all_page_titles):,} "
          f"({len(mapped_originals) / len(all_page_titles) * 100:.2f}%)")
    print(f"  Original page titles unmapped: {len(unmapped):>10,} "
          f"(filtered out)")

    # Title length statistics
    title_lengths = [len(v["title"]) for v in page_title_item.values()]
    if title_lengths:
        title_lengths_sorted = sorted(title_lengths)
        p50 = title_lengths_sorted[len(title_lengths_sorted) // 2]
        p90 = title_lengths_sorted[int(len(title_lengths_sorted) * 0.9)]
        p99 = title_lengths_sorted[int(len(title_lengths_sorted) * 0.99)]
        print(f"\n  Title length stats:")
        print(f"    Min:  {min(title_lengths):>6}")
        print(f"    P50:  {p50:>6}")
        print(f"    P90:  {p90:>6}")
        print(f"    P99:  {p99:>6}")
        print(f"    Max:  {max(title_lengths):>6}")
        print(f"    Avg:  {sum(title_lengths) / len(title_lengths):>6.1f}")

    # =========================================================================
    # Step 7: Write output files
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 7: Writing output files")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Write page_title_item.json
    item_path = os.path.join(args.output_dir, "page_title_item.json")
    with open(item_path, "w", encoding="utf-8") as f:
        json.dump(page_title_item, f, indent=2, ensure_ascii=False)
    item_size = os.path.getsize(item_path) / (1024 * 1024)
    print(f"  Written: {item_path}")
    print(f"    Size: {item_size:.2f} MB, Items: {len(page_title_item):,}")

    # Write page_title_to_item.json
    mapping_path = os.path.join(args.output_dir, "page_title_to_item.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(page_title_to_item, f, indent=2, ensure_ascii=False)
    mapping_size = os.path.getsize(mapping_path) / (1024 * 1024)
    print(f"  Written: {mapping_path}")
    print(f"    Size: {mapping_size:.2f} MB, Mappings: {len(page_title_to_item):,}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
