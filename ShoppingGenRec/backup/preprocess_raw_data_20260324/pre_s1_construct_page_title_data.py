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
import time
import hashlib
import struct
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
    # Domain / website fragments (should not count as content words)
    "amazon", "com", "co", "jp", "uk", "de", "fr", "ca", "au", "br",
    "www", "http", "https", "html", "htm", "org", "net",
    "walmart", "ebay", "etsy", "target", "hsn", "qvc",
})

# Known domain / store names for suffix stripping (longest first for matching)
KNOWN_DOMAINS = sorted([
    # Major e-commerce
    "Amazon.com", "Amazon.co.jp", "Amazon.co.uk", "Amazon.de",
    "Amazon.fr", "Amazon.ca", "Amazon.com.au", "Amazon",
    "HSN", "QVC.com", "QVC", "Target", "Target.com",
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


# Pre-compiled regex patterns for strip_domain_affixes
_RE_DOMAIN_PREFIX = re.compile(
    r"^(?:Amazon\.(?:com|co\.jp|co\.uk|de|fr|ca|com\.au)"
    r"|[A-Za-z0-9]+\.(?:com|org|net|co\.uk|co\.jp|co\.kr|com\.au|com\.br))\s*[:|]\s*",
    flags=re.IGNORECASE,
)

_RE_DOMAIN_SUFFIXES = []
for _domain in KNOWN_DOMAINS:
    _escaped = re.escape(_domain)
    _RE_DOMAIN_SUFFIXES.append(
        re.compile(rf"\s*[|:\-\u2013\u2014]\s*{_escaped}\s*$", flags=re.IGNORECASE)
    )

_RE_GENERIC_SUFFIX = re.compile(
    r"\s*[|:\-\u2013\u2014]\s*"
    r"[A-Za-z0-9][A-Za-z0-9\s&'.\u00ae]*"
    r"\.(?:com|org|net|co\.uk|co\.jp|co\.kr|com\.au|com\.br)\s*$",
    flags=re.IGNORECASE,
)

# Pre-compiled regex patterns for normalize_whitespace
_RE_WS_COLLAPSE = re.compile(r"\s+")
_RE_WS_BEFORE_PUNCT = re.compile(r"\s+([,;:.!?\)\]])")
_RE_WS_AFTER_OPEN = re.compile(r"([\(\[])\s+")
_RE_WS_DASH = re.compile(r"(\w)\s+\-\s+(\w)")
_RE_WS_DECIMAL = re.compile(r"(\d)\s*\.\s*(\d)")
_RE_WS_SLASH = re.compile(r"(\w)\s*/\s*(\w)")

# Pre-compiled regex for compute_canonical_key
_RE_CANON_NONALNUM = re.compile(r"[^a-z0-9\s]")
_RE_CANON_SPACES = re.compile(r"\s+")


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
    # Strip leading domain prefix (e.g., "Amazon.com : ...", "Amazon.co.jp: ...")
    title = _RE_DOMAIN_PREFIX.sub("", title)

    # Strip trailing domain/store suffixes iteratively (handles nested suffixes)
    for _ in range(3):
        changed = False

        # Try known domain names first (longest match first)
        for pat in _RE_DOMAIN_SUFFIXES:
            new_title = pat.sub("", title)
            if new_title != title:
                title = new_title
                changed = True
                break

        if not changed:
            # Generic pattern: trailing " | <word(s).com>" or similar
            new_title = _RE_GENERIC_SUFFIX.sub("", title)
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
    title = _RE_WS_COLLAPSE.sub(" ", title)
    # Remove space before punctuation: , ; : . ! ? ) ]
    title = _RE_WS_BEFORE_PUNCT.sub(r"\1", title)
    # Remove space after opening brackets: ( [
    title = _RE_WS_AFTER_OPEN.sub(r"\1", title)
    # Normalize isolated dashes: "word - word" -> "word-word" (compound words)
    title = _RE_WS_DASH.sub(r"\1-\2", title)
    # Normalize isolated slashes: "8 . 5" -> "8.5" (decimal numbers)
    title = _RE_WS_DECIMAL.sub(r"\1.\2", title)
    # Normalize "word / word" -> "word/word" for size/option patterns
    title = _RE_WS_SLASH.sub(r"\1/\2", title)
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
    key = _RE_CANON_NONALNUM.sub(" ", key)
    key = _RE_CANON_SPACES.sub(" ", key)
    return key.strip()


# Regex to match CJK Unified Ideographs, Hiragana, Katakana
_CJK_RANGE = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf]+"
)


def get_content_words(title):
    """Extract content words (excluding stopwords) from a title.

    Handles both ASCII alphanumeric words and CJK characters.
    For CJK text, extracts character bigrams as pseudo-words to enable
    meaningful similarity comparison between titles containing
    Japanese, Chinese, or Korean text.

    Args:
        title: Page title string.

    Returns:
        A frozenset of lowercase content words (and CJK bigrams).
    """
    # ASCII alphanumeric words
    words = re.findall(r"[a-z0-9]+", title.lower())
    content = set(w for w in words if w not in STOPWORDS and len(w) > 1)

    # CJK character bigrams: extract consecutive CJK runs, then split into
    # overlapping bigrams. E.g., "深蒸し茶" (CJK run "深蒸し茶") -> bigrams
    # "深蒸", "蒸し", "し茶". This provides meaningful similarity signals
    # for CJK text that would otherwise be entirely ignored.
    for match in _CJK_RANGE.finditer(title):
        cjk_run = match.group()
        if len(cjk_run) >= 2:
            for i in range(len(cjk_run) - 1):
                content.add(cjk_run[i:i+2])
        elif len(cjk_run) == 1:
            content.add(cjk_run)

    return frozenset(content)


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


# =============================================================================
# MinHash + LSH utilities for fast near-duplicate detection
# =============================================================================

# Global word-to-hash cache (built once, shared across processes)
_WORD_HASH_CACHE = {}

def _word_to_hashes(word, num_hashes):
    """Convert a word to a list of hash values for MinHash.

    Uses MD5-based double hashing: h_k(word) = (h1 + k * h2) mod p
    This is much faster than computing num_hashes independent hashes.
    """
    if word not in _WORD_HASH_CACHE:
        digest = hashlib.md5(word.encode("utf-8")).digest()
        h1, h2 = struct.unpack("<QQ", digest)
        _WORD_HASH_CACHE[word] = (h1, h2)
    h1, h2 = _WORD_HASH_CACHE[word]
    return [(h1 + k * h2) % ((1 << 61) - 1) for k in range(num_hashes)]


def compute_minhash_signature(word_set, num_hashes=128):
    """Compute a MinHash signature for a set of words.

    Args:
        word_set: frozenset of content words.
        num_hashes: Number of hash functions (signature length).

    Returns:
        Tuple of num_hashes minimum hash values.
    """
    if not word_set:
        return tuple([float("inf")] * num_hashes)

    sig = [float("inf")] * num_hashes
    for word in word_set:
        hashes = _word_to_hashes(word, num_hashes)
        for k in range(num_hashes):
            if hashes[k] < sig[k]:
                sig[k] = hashes[k]
    return tuple(sig)


def _compute_minhash_batch(args):
    """Worker function for parallel MinHash computation."""
    indices, word_sets_list, num_hashes = args
    results = []
    for i in indices:
        ws = word_sets_list[i]
        if len(ws) >= 3:  # min_words
            sig = compute_minhash_signature(ws, num_hashes)
        else:
            sig = None
        results.append((i, sig))
    return results


def deduplicate_by_word_overlap(titles, word_sets, threshold=0.8, min_words=3,
                                 max_posting_list_size=50000):
    """Deduplicate titles by MinHash LSH + exact Jaccard verification.

    Uses MinHash signatures with Locality-Sensitive Hashing (LSH) to
    efficiently find candidate near-duplicate pairs, then verifies with
    exact Jaccard similarity. Much faster than inverted-index approach
    for large datasets.

    The LSH parameters (num_hashes, num_bands) are automatically chosen
    to target the given threshold.

    Args:
        titles: List of normalized title strings.
        word_sets: Parallel list of frozensets of content words.
        threshold: Jaccard similarity threshold for merging (default: 0.8).
        min_words: Minimum content words required for word-overlap comparison.
        max_posting_list_size: Unused (kept for API compatibility).

    Returns:
        A list of sets, each containing indices of titles in the same group.
    """
    import math

    n = len(titles)
    uf = UnionFind(n)
    t0 = time.time()

    # ---- Choose LSH parameters ----
    # For threshold t, with b bands of r rows each:
    #   P(candidate) ≈ 1 - (1 - t^r)^b
    # We want high recall at threshold, so pick b,r to make P ≈ 0.95+ at t
    num_hashes = 128
    best_b, best_r = 16, 8  # default: 128 = 16 bands * 8 rows
    # For threshold=0.8: P(0.8) = 1-(1-0.8^8)^16 ≈ 0.9986 (very high recall)
    # For threshold=0.7: still ok. For threshold=0.9: even better.
    # Try to find better b,r if threshold is unusual
    best_prob = 0.0
    for b in range(2, num_hashes + 1):
        if num_hashes % b != 0:
            continue
        r = num_hashes // b
        prob_at_t = 1 - (1 - threshold ** r) ** b
        # Also check false positive rate at t/2
        prob_at_half = 1 - (1 - (threshold / 2) ** r) ** b
        score = prob_at_t - 0.5 * prob_at_half  # maximize recall, minimize FP
        if score > best_prob:
            best_prob = score
            best_b, best_r = b, r

    num_bands = best_b
    rows_per_band = best_r
    prob_at_t = 1 - (1 - threshold ** rows_per_band) ** num_bands
    print(f"    LSH parameters: {num_hashes} hashes = {num_bands} bands × "
          f"{rows_per_band} rows")
    print(f"    Estimated recall at Jaccard={threshold}: {prob_at_t:.4f}")

    # ---- Step A: Compute MinHash signatures (parallel) ----
    print(f"    Computing MinHash signatures for {n:,} titles...")
    t1 = time.time()

    eligible_count = sum(1 for ws in word_sets if len(ws) >= min_words)
    print(f"    Eligible titles (>= {min_words} content words): {eligible_count:,}")

    # Use multiprocessing for large datasets
    num_workers = min(cpu_count(), 50)
    signatures = [None] * n

    if n > 50000 and num_workers > 1:
        # Split indices into chunks for parallel processing
        chunk_size = max(1000, n // (num_workers * 4))
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunks.append((list(range(start, end)), word_sets, num_hashes))

        print(f"    Using {num_workers} workers, {len(chunks)} chunks...")
        with Pool(num_workers) as pool:
            for batch_results in pool.imap_unordered(_compute_minhash_batch, chunks):
                for i, sig in batch_results:
                    signatures[i] = sig
    else:
        for i in range(n):
            if len(word_sets[i]) >= min_words:
                signatures[i] = compute_minhash_signature(word_sets[i], num_hashes)

    t2 = time.time()
    print(f"    MinHash signatures computed in {t2 - t1:.1f}s")

    # ---- Step B: LSH banding — find candidate pairs ----
    print(f"    Running LSH banding to find candidate pairs...")

    # Precompute word set sizes
    ws_sizes = [len(ws) for ws in word_sets]
    max_ratio = 1.0 / threshold  # 1.25 for threshold=0.8

    candidate_pairs = set()
    for band_idx in range(num_bands):
        band_start = band_idx * rows_per_band
        band_end = band_start + rows_per_band

        # Hash each signature's band portion into buckets
        buckets = defaultdict(list)
        for i in range(n):
            sig = signatures[i]
            if sig is None:
                continue
            band_hash = hash(sig[band_start:band_end])
            buckets[band_hash].append(i)

        # Generate candidate pairs from each bucket
        for bucket_items in buckets.values():
            if len(bucket_items) < 2 or len(bucket_items) > 10000:
                # Skip huge buckets (likely noise)
                continue
            for idx_a in range(len(bucket_items)):
                i = bucket_items[idx_a]
                n_i = ws_sizes[i]
                for idx_b in range(idx_a + 1, len(bucket_items)):
                    j = bucket_items[idx_b]
                    n_j = ws_sizes[j]
                    # Quick size ratio filter
                    if max(n_i, n_j) > max_ratio * min(n_i, n_j):
                        continue
                    pair = (min(i, j), max(i, j))
                    candidate_pairs.add(pair)

    t3 = time.time()
    print(f"    LSH banding done in {t3 - t2:.1f}s")
    print(f"    Candidate pairs from LSH: {len(candidate_pairs):,}")

    # ---- Step C: Verify candidates with exact Jaccard ----
    print(f"    Verifying candidate pairs with exact Jaccard similarity...")

    merge_count = 0
    checked_pairs = 0
    for i, j in candidate_pairs:
        ws_i = word_sets[i]
        ws_j = word_sets[j]

        checked_pairs += 1
        intersection = len(ws_i & ws_j)
        union = len(ws_i) + len(ws_j) - intersection
        if union > 0 and intersection / union >= threshold:
            uf.union(i, j)
            merge_count += 1

    t4 = time.time()
    print(f"    Verification done in {t4 - t3:.1f}s")

    # Collect groups
    groups = defaultdict(set)
    for i in range(n):
        groups[uf.find(i)].add(i)

    total_time = time.time() - t0
    print(f"    Candidate pairs checked: {checked_pairs:,}")
    print(f"    Word-overlap merges performed: {merge_count:,}")
    print(f"    Groups after word-overlap dedup: {len(groups):,}")
    print(f"    Total Step 5 time: {total_time:.1f}s")

    return list(groups.values())


def _normalize_title_batch(titles_batch):
    """Worker function: normalize a batch of titles in a subprocess.

    Returns list of (original, normalized, canonical_key) tuples.
    """
    results = []
    for orig in titles_batch:
        after_domain = strip_domain_affixes(orig)
        normalized = normalize_whitespace(after_domain)
        key = compute_canonical_key(normalized)
        results.append((orig, normalized, key))
    return results


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
                "Data/0128_0301/SequenceData_Plat_Sampled_500000_User.tsv",
        help="Path to the SequenceData_Plat TSV file "
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
        default=0.75,
        help="Jaccard similarity threshold for near-duplicate merging",
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

    # Parallelize normalization for large datasets
    titles_list = list(all_page_titles)
    num_workers = min(cpu_count(), 50)
    chunk_size = max(5000, len(titles_list) // (num_workers * 4))

    if len(titles_list) > 100000 and num_workers > 1:
        print(f"  Using {num_workers} workers, chunk_size={chunk_size:,}...")
        batches = [
            titles_list[i:i + chunk_size]
            for i in range(0, len(titles_list), chunk_size)
        ]
        with Pool(num_workers) as pool:
            for batch_results in tqdm(
                pool.imap_unordered(_normalize_title_batch, batches),
                total=len(batches), desc="  Normalizing",
            ):
                for orig, normalized, key in batch_results:
                    if normalized != orig:
                        # Count domain stripping (approximation: any change counts)
                        domain_stripped_count += 1
                    original_to_normalized[orig] = normalized
                    canonical_groups[key].add(orig)
    else:
        for orig in tqdm(titles_list, desc="  Normalizing"):
            after_domain = strip_domain_affixes(orig)
            if after_domain != orig:
                domain_stripped_count += 1
            normalized = normalize_whitespace(after_domain)
            if normalized != after_domain:
                ws_modified_count += 1
            original_to_normalized[orig] = normalized
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
    # Step 4.5: Prefix-based deduplication (same-product SKU variants)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4.5: Prefix-based deduplication (SKU variants)")
    print("=" * 70)

    # Group titles by their canonical key prefix (first N chars).
    # Within each prefix group, merge titles with Jaccard >= threshold.
    # This catches "Product Name - Color A" vs "Product Name - Color B"
    # which MinHash may miss because they differ in multiple attribute words.
    PREFIX_LEN = 40
    prefix_groups = defaultdict(list)  # prefix -> list of indices into filtered_keys
    for i, key in enumerate(filtered_keys):
        title = group_representatives[key]
        canon = compute_canonical_key(title)
        prefix = canon[:PREFIX_LEN] if len(canon) >= PREFIX_LEN else canon
        prefix_groups[prefix].append(i)

    # Only process groups with 2+ members
    multi_prefix_groups = {
        p: idxs for p, idxs in prefix_groups.items() if len(idxs) >= 2
    }
    print(f"  Prefix length: {PREFIX_LEN} chars")
    print(f"  Total prefix groups: {len(prefix_groups):>12,}")
    print(f"  Groups with 2+ members: {len(multi_prefix_groups):>12,}")

    # Within each prefix group, compute Jaccard and merge
    prefix_uf = UnionFind(len(filtered_keys))
    prefix_merge_count = 0
    prefix_pairs_checked = 0

    for prefix, idxs in multi_prefix_groups.items():
        # Compute word sets for this group
        group_word_sets = []
        for i in idxs:
            title = group_representatives[filtered_keys[i]]
            group_word_sets.append(get_content_words(title))

        # Pairwise Jaccard within the group (small groups, so O(n^2) is fine)
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ws_a = group_word_sets[a]
                ws_b = group_word_sets[b]
                if len(ws_a) < 3 or len(ws_b) < 3:
                    continue
                prefix_pairs_checked += 1
                sim = jaccard_similarity(ws_a, ws_b)
                if sim >= args.similarity_threshold:
                    prefix_uf.union(idxs[a], idxs[b])
                    prefix_merge_count += 1

    # Rebuild filtered_keys by merging prefix groups
    if prefix_merge_count > 0:
        prefix_merged_groups = defaultdict(set)
        for i in range(len(filtered_keys)):
            prefix_merged_groups[prefix_uf.find(i)].add(i)

        new_filtered_keys = []
        for root, members in prefix_merged_groups.items():
            if len(members) == 1:
                idx = next(iter(members))
                new_filtered_keys.append(filtered_keys[idx])
            else:
                # Pick the key with most originals, then longest title
                best_idx = max(
                    members,
                    key=lambda i: (
                        len(group_originals[filtered_keys[i]]),
                        len(group_representatives[filtered_keys[i]]),
                    ),
                )
                best_key = filtered_keys[best_idx]
                # Merge all originals from other keys into the best key
                for idx in members:
                    if idx != best_idx:
                        other_key = filtered_keys[idx]
                        group_originals[best_key].update(
                            group_originals[other_key]
                        )
                new_filtered_keys.append(best_key)

        filtered_keys = new_filtered_keys

    print(f"  Pairs checked: {prefix_pairs_checked:>12,}")
    print(f"  Prefix merges: {prefix_merge_count:>12,}")
    print(f"  Keys after prefix dedup: {len(filtered_keys):>12,}")

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
