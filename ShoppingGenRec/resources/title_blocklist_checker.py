"""Title Blocklist Checker (equivalent to C# IsNgramBlockedV5 + NormalizeV3).

Loads a GlobalMarketTitleBlocklist TSV file, normalizes keywords and titles
using the same logic as the C# NormalizeV3 function, and checks if a title
contains any blocked keyword.

Usage as library:
    from title_blocklist_checker import load_title_blocklist, build_regex, is_title_blocked

    tokens = load_title_blocklist("GlobalMarketTitleBlocklist.Clean.tsv", "en")
    regex = build_regex(tokens)
    result = is_title_blocked("Cotton Bra Set", regex)  # returns " bra "

Usage as standalone script:
    python title_blocklist_checker.py
    python title_blocklist_checker.py --blocklist /path/to/blocklist.tsv --language en
"""

import argparse
import csv
import os
import re
import sys

# Pre-compiled regex — matches C# NormalizeV3 exactly:
#   string patternStr = "[-!+/_\\s,.;:?\"']+";
#   string replaceStr = " ";
#   return $" {Regex.Replace(s, patternStr, replaceStr)} ".ToLowerInvariant();
_RE_NORM_V3 = re.compile(r"[-!+/_\s,.;:?\"']+")


def normalize(text):
    """Normalize text for blocklist matching (equivalent to C# NormalizeV3).

    Replaces specific punctuation [-!+/_\\s,.;:?"'] with spaces,
    lowercases, and wraps with leading/trailing space for word-boundary
    matching via substring check.
    """
    text = _RE_NORM_V3.sub(" ", text)
    return f" {text} ".lower()


def load_title_blocklist(filepath, language="en"):
    """Load title blocklist keywords for a given language from a TSV file.

    TSV format: Market<TAB>keywords  (first row is header)

    Args:
        filepath: Path to the blocklist TSV file.
        language: Language/market code to filter (default: "en").

    Returns:
        List of normalized keyword strings (deduplicated).
    """
    keywords = []
    if not filepath or not os.path.isfile(filepath):
        return keywords
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            market = row[0].strip().lower()
            keyword = row[1].strip()
            if market == language and keyword:
                keywords.append(normalize(keyword))
    # Deduplicate while preserving order
    seen = set()
    return [kw for kw in keywords if not (kw in seen or seen.add(kw))]


def build_regex(blocklist_tokens):
    """Build a single compiled regex from all blocklist tokens.

    Much faster than checking each token individually with `in`.
    Sorts by length descending so longer matches take priority.

    Args:
        blocklist_tokens: List of pre-normalized keyword strings.

    Returns:
        Compiled regex pattern, or None if no tokens.
    """
    if not blocklist_tokens:
        return None
    sorted_tokens = sorted(blocklist_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in sorted_tokens)
    return re.compile(pattern)


def is_title_blocked(title, blocklist_regex):
    """Check if a title contains any blocked keyword.

    Args:
        title: Raw title string.
        blocklist_regex: Compiled regex from build_regex().

    Returns:
        The matched keyword string (with spaces) if blocked, or None.
    """
    if not title or blocklist_regex is None:
        return None
    norm_title = normalize(title)
    m = blocklist_regex.search(norm_title)
    return m.group(0) if m else None


def main():
    parser = argparse.ArgumentParser(
        description="Title Blocklist Checker (C# NormalizeV3 compatible)")
    parser.add_argument(
        "--blocklist", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "GlobalMarketTitleBlocklist.Clean.tsv"),
        help="Path to GlobalMarketTitleBlocklist TSV file",
    )
    parser.add_argument(
        "--language", type=str, default="en",
        help="Language/market code (default: en)",
    )
    parser.add_argument(
        "--titles", type=str, nargs="*", default=None,
        help="Titles to check (if not provided, runs built-in test cases)",
    )
    args = parser.parse_args()

    # Load blocklist
    print(f"Loading blocklist: {args.blocklist}")
    print(f"Language: {args.language}")
    tokens = load_title_blocklist(args.blocklist, args.language)
    regex = build_regex(tokens)
    print(f"Loaded {len(tokens)} keywords\n")

    if not tokens:
        print("No keywords loaded! Check file path and language.")
        return

    # Show a few sample keywords
    print("Sample keywords (first 10):")
    for kw in tokens[:10]:
        print(f"  {repr(kw)}")
    print()

    # Test titles
    if args.titles:
        test_titles = args.titles
    else:
        test_titles = [
            "Cotton Bra Set",
            "Bracket Mount",
            "Brand New Shoes",
            "Brass Fitting",
            "Library Shelf",
            "Vibrant Color",
            "Hunting Rifle Scope",
            "Nike Unisex Running Shoes",
            "Concealed Carry Holster",
            "ASICS Gunmetal Gray Shoes",
            "Porterhouse Steak Gift Set",
            "Leather Belt for Men",
            "Sex Toy Adult",
            "Burgundy Wine Glass",
            "Jewelry Box Organizer",
        ]

    print(f"{'Result':<30s}  Title")
    print(f"{'-'*30}  {'-'*50}")
    for title in test_titles:
        result = is_title_blocked(title, regex)
        if result:
            status = f"BLOCKED ({result.strip()})"
        else:
            status = "OK"
        print(f"{status:<30s}  {title}")


if __name__ == "__main__":
    main()
