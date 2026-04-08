"""
TermID Normalization

Three-stage normalization for shopping recommendation term IDs:
  Stage 1: Text Formatting (HTML entities, symbols, smart hyphen)
  Stage 2: Unit Normalization (full names to abbreviation, add number-unit join)
  Stage 3: Dictionary Normalization (singular/plural, spelling, synonyms)

Usage:
    from term_normalizer import normalize_term, normalize_file

    # Single term
    result = normalize_term("women's")  # -> "women"

    # Batch file
    normalize_file("path/to/termId_dict", "path/to/output")
"""

import html
import re
import time
from collections import Counter


# ===========================================================================
# Stage 1: Text Formatting
# ===========================================================================

_MOJIBAKE_TM = ["\u00e2\u0084\u00a2", "â\u0084¢"]
_MOJIBAKE_REG = ["\u00c2\u00ae"]
_MOJIBAKE_CR = ["\u00c2\u00a9"]

_SYMBOL_PATTERN = re.compile(
    r"[™®©]"
    r"|" + r"|".join(re.escape(s) for s in _MOJIBAKE_TM + _MOJIBAKE_REG + _MOJIBAKE_CR)
)


def _smart_hyphen_word(word: str) -> str:
    """Process a single word's hyphen. Only handles exactly 1 hyphen per word.
    Keep hyphen if either side is single char or number; else replace with space."""
    if word.count("-") != 1:
        return word
    left, right = word.split("-", 1)
    left_short = len(left) <= 1 or left.replace("/", "").replace(".", "").isdigit()
    right_short = len(right) <= 1 or right.replace("/", "").replace(".", "").isdigit()
    if left_short or right_short:
        return word
    return f"{left} {right}"


def _stage1(term: str) -> str:
    """HTML unescape, strip symbols, smart hyphen handling (per word)."""
    term = html.unescape(term)
    term = _SYMBOL_PATTERN.sub("", term)
    tokens = term.split()
    tokens = [_smart_hyphen_word(t) for t in tokens]
    term = " ".join(tokens)
    term = re.sub(r"\s+", " ", term).strip()
    return term


# ===========================================================================
# Stage 2: Unit Normalization
# ===========================================================================

_UNIT_TO_CANONICAL = {
    # Abbreviations (keep as-is)
    "oz": "oz", "lb": "lb", "kg": "kg",
    "ft": "ft", "mm": "mm", "cm": "cm", "inch": "inch",
    "ml": "ml", "gal": "gal", "lm": "lm",
    "gb": "gb", "tb": "tb", "mb": "mb",
    "hz": "hz", "mhz": "mhz", "ghz": "ghz", "mah": "mah",
    # Full names (singular) → abbreviation
    "ounce": "oz", "pound": "lb", "kilogram": "kg",
    "foot": "ft", "feet": "ft",
    "millimeter": "mm", "millimetre": "mm",
    "centimeter": "cm", "centimetre": "cm",
    "milliliter": "ml", "millilitre": "ml",
    "gallon": "gal", "lumen": "lm",
    "hertz": "hz", "megahertz": "mhz", "gigahertz": "ghz",
    "watt": "w", "volt": "v", "amp": "amp",
    "gram": "g", "milligram": "mg",
    "liter": "l", "litre": "l", "meter": "m", "metre": "m",
    "kilometer": "km", "kilometre": "km",
    "yard": "yd", "quart": "qt", "pint": "pt",
    "kilowatt": "kw", "milliamp": "ma", "megapixel": "mp",
    # Full names (plural) → abbreviation
    "ounces": "oz", "pounds": "lb", "kilograms": "kg",
    "millimeters": "mm", "millimetres": "mm",
    "centimeters": "cm", "centimetres": "cm",
    "milliliters": "ml", "millilitres": "ml",
    "gallons": "gal", "lumens": "lm", "inches": "inch",
    "watts": "w", "volts": "v", "amps": "amp",
    "grams": "g", "milligrams": "mg",
    "liters": "l", "litres": "l", "meters": "m", "metres": "m",
    "kilometers": "km", "kilometres": "km",
    "yards": "yd", "quarts": "qt", "pints": "pt",
    "kilowatts": "kw", "milliamps": "ma", "megapixels": "mp",
    # Abbreviation variant
    "lbs": "lb",
}

# Full names and variants that get converted to canonical abbreviation
_FULL_NAMES = {
    "ounce", "pound", "kilogram", "foot", "feet",
    "millimeter", "millimetre", "centimeter", "centimetre",
    "milliliter", "millilitre", "gallon", "lumen",
    "hertz", "megahertz", "gigahertz",
    "watt", "volt", "amp", "gram", "milligram",
    "liter", "litre", "meter", "metre",
    "kilometer", "kilometre", "yard", "quart", "pint",
    "kilowatt", "milliamp", "megapixel",
    "ounces", "pounds", "kilograms",
    "millimeters", "millimetres", "centimeters", "centimetres",
    "milliliters", "millilitres", "gallons", "lumens", "inches",
    "watts", "volts", "amps", "grams", "milligrams",
    "liters", "litres", "meters", "metres",
    "kilometers", "kilometres", "yards", "quarts", "pints",
    "kilowatts", "milliamps", "megapixels",
    "lbs",
}

_unit_tokens = sorted(_UNIT_TO_CANONICAL.keys(), key=len, reverse=True)
_UNIT_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9])(\d+(?:\.\d+)?)[\s-]*("
    + "|".join(re.escape(u) for u in _unit_tokens) + r")\b",
    re.IGNORECASE,
)


def _unit_replace(match: re.Match) -> str:
    """Full name → canonical abbreviation (lowercase); abbreviation → keep original case."""
    number, unit_raw = match.group(1), match.group(2)
    if unit_raw.lower() in _FULL_NAMES:
        return f"{number}{_UNIT_TO_CANONICAL[unit_raw.lower()]}"
    return f"{number}{unit_raw}"


def _stage2(term: str) -> str:
    """Normalize number+unit patterns. Full names become abbreviations."""
    return _UNIT_PATTERN.sub(_unit_replace, term)


# ===========================================================================
# Stage 3: Dictionary Normalization
# ===========================================================================

_NORMALIZE_DICT = {
    # Irregular forms (→ winner by count)
    "man": "men", "woman": "women", "child": "children",
    "foot": "feet", "tooth": "teeth", "goose": "geese", "mice": "mouse",
    # Possessive / variant forms
    "men's": "men", "women's": "women",
    "men\u2019s": "men", "women\u2019s": "women",
    "kid's": "kids", "kid\u2019s": "kids",
    "children's": "children", "children\u2019s": "children",
    "mens": "men", "womens": "women",
    # Product nouns (→ higher frequency form)
    "shoe": "shoes", "boot": "boots", "heel": "heels",
    "sock": "socks", "earring": "earrings", "girl": "girls",
    "boy": "boys", "kid": "kids", "cleat": "cleats",
    "glove": "gloves", "blind": "blinds", "headphone": "headphones",
    "bead": "beads", "goggle": "goggles", "lady": "ladies",
    "sweatpant": "sweatpants", "wipe": "wipes",
    "rail": "rails", "pant": "pants", "short": "shorts",
    "sandals": "sandal", "sneakers": "sneaker", "loafers": "loafer",
    "flats": "flat", "pumps": "pump", "slippers": "slipper",
    "slides": "slide", "rings": "ring", "necklaces": "necklace",
    "bracelets": "bracelet", "scarves": "scarf",
    "hats": "hat", "bags": "bag", "wallets": "wallet", "watches": "watch",
    "curtains": "curtain", "plates": "plate", "drawers": "drawer",
    "shelves": "shelf", "pillows": "pillow", "towels": "towel",
    "sheets": "sheet", "blankets": "blanket",
    "speakers": "speaker", "headlights": "headlight",
    "wheels": "wheel", "tires": "tire", "bulbs": "bulb",
    "seeds": "seed", "stickers": "sticker",
    "adults": "adult", "joggers": "jogger", "pockets": "pocket",
    "trousers": "trouser",
    "bottles": "bottle", "tablets": "tablet", "diapers": "diaper",
    "vitamins": "vitamin", "treats": "treat", "snacks": "snack",
    "mats": "mat", "rugs": "rug", "lamps": "lamp", "fans": "fan",
    "filters": "filter", "plugs": "plug", "hooks": "hook",
    "clips": "clip", "straps": "strap", "pads": "pad",
    "rods": "rod", "caps": "cap", "cups": "cup", "cans": "can",
    "bins": "bin", "bars": "bar", "ties": "tie", "pins": "pin",
    "tags": "tag", "knobs": "knob", "vents": "vent",
    "panels": "panel", "covers": "cover", "grips": "grip",
    "tubes": "tube", "cables": "cable", "cords": "cord",
    "markers": "marker", "rollers": "roller", "hangers": "hanger",
    "holders": "holder", "containers": "container",
    "organizers": "organizer", "baskets": "basket", "crates": "crate",
    "clogs": "clog", "booties": "bootie", "mules": "mule",
    "oxfords": "oxford", "wedges": "wedge", "studs": "stud",
    # British → American spelling
    "grey": "gray", "colour": "color", "colourful": "colorful",
    "colourblock": "colorblock", "colour-block": "color block",
    "multicolour": "multicolor", "multicoloured": "multicolored",
    "watercolour": "watercolor",
    "aluminium": "aluminum", "jewellery": "jewelry",
    "moulding": "molding", "mould": "mold",
    "fibre": "fiber", "litre": "liter", "centre": "center",
    "favour": "favor", "defence": "defense", "licence": "license",
    "organise": "organize", "organised": "organized",
    "customise": "customize", "personalised": "personalized",
    "mercerised": "mercerized", "modelling": "modeling",
    "travelling": "traveling",
    # Synonyms
    "wifi": "wireless", "wi fi": "wireless",
    "female": "women", "male": "men",
}

_NORMALIZE_LOOKUP = {k.lower(): v for k, v in _NORMALIZE_DICT.items()}
_SKIP_PATTERN = re.compile(r"\d|^.{0,2}$")
_WORD_PATTERN = re.compile(r"^[a-zA-Z''\u2019]+(?:\s+[a-zA-Z''\u2019]+)*$")


def _stage3(term: str) -> str:
    """Dictionary-based normalization. Single-word only. Lookup by lowercase."""
    if _SKIP_PATTERN.search(term):
        return term
    if not _WORD_PATTERN.match(term):
        return term
    if ' ' in term:
        return term
    lower = term.lower()
    if lower in _NORMALIZE_LOOKUP:
        return _NORMALIZE_LOOKUP[lower]
    return term


# ===========================================================================
# Public API
# ===========================================================================

def normalize_term(term: str) -> str:
    """Normalize a single TermID string through all 3 stages."""
    return _stage3(_stage2(_stage1(term)))


def normalize_file(input_path: str, output_path: str, mapping_path: str = None) -> dict:
    """
    Normalize a termId_dict file and write the result.

    Args:
        input_path:  TSV input (Word\\tCount, with header)
        output_path: TSV output (normalized Word\\tCount, merged counts)
        mapping_path: Optional TSV of changed terms (for debugging)

    Returns:
        dict with stats: input_count, output_count, merged_count, changed_count, elapsed_s
    """
    terms = []
    with open(input_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                try:
                    terms.append((parts[0], int(parts[1])))
                except ValueError:
                    continue

    start = time.time()
    normalized_counts = Counter()
    changes = []
    for word, count in terms:
        result = normalize_term(word)
        normalized_counts[result] += count
        if result != word:
            changes.append((word, result, count))
    elapsed = time.time() - start

    sorted_terms = sorted(normalized_counts.items(), key=lambda x: -x[1])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Word\tCount\n")
        for term, count in sorted_terms:
            f.write(f"{term}\t{count}\n")

    if mapping_path:
        changes.sort(key=lambda x: -x[2])
        with open(mapping_path, "w", encoding="utf-8") as f:
            f.write("original\tnormalized\tcount\n")
            for orig, norm, count in changes:
                f.write(f"{orig}\t{norm}\t{count}\n")

    return {
        "input_count": len(terms),
        "output_count": len(sorted_terms),
        "merged_count": len(terms) - len(sorted_terms),
        "changed_count": len(changes),
        "elapsed_s": round(elapsed, 1),
    }


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        stats = normalize_file(sys.argv[1], sys.argv[2],
                               sys.argv[3] if len(sys.argv) >= 4 else None)
        print(f"Done in {stats['elapsed_s']}s | "
              f"In: {stats['input_count']:,} | Out: {stats['output_count']:,} | "
              f"Merged: {stats['merged_count']:,} | Changed: {stats['changed_count']:,}")
    else:
        test_terms = [
            "Bed Bath &amp; Beyond", "H&amp;M", "Cricut\u00ae",
            "noise-canceling", "slip-on", "A-line", "T-shirt",
            "3-in-1", "set-of-2", "300-thread-count",
            "6-inch", "32-65-inch", "B&amp;H Photo-Video-Pro A",
            "1000LM", "1000lumen", "1000-lumen", "128GB",
            "16oz", "16-ounce", "5lb", "10ft", "2.4ghz",
            "1000-watt", "12-volt", "100-gram", "8400-lumens",
            "5W", "12V", "3M", "4K", "24in", "WMH32519HZ",
            "shoe", "sneakers", "men", "women", "Female", "Male",
            "women's", "mens", "grey", "aluminium", "wifi",
            "Zappos", "Nike", "Kohl's", "Macy's", "REI",
        ]
        print("TermID Normalizer - Quick Test")
        print("-" * 60)
        for t in test_terms:
            r = normalize_term(t)
            if r != t:
                print(f"  {t!r:40s} -> {r!r}")
            else:
                print(f"  {t!r:40s} -> (unchanged)")
