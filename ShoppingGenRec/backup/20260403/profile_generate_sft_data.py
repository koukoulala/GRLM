"""Step 0 (Profile): Build Shopping Profile SFT Data

Creates SFT training data for generating shopping profiles from user event
histories. Given a user's browsing, search, and click events, the model
learns to produce a structured JSON shopping profile.

Data sources:
  1. Input TSV file (same as pre_s3_construct_shopping_profile.py input):
     Columns: UserId, ReadableUserEvents (pipe-delimited event history,
     "#N#" as line separators).
  2. shopping_profiles.tsv (output of pre_s3_construct_shopping_profile.py):
     Columns: UserId, ShoppingProfile (JSON string).

Pipeline:
  1. Load the input TSV to get user event histories.
  2. Load shopping_profiles.tsv to get generated profiles.
  3. Match by UserId and filter invalid entries.
  4. Build SFT samples: instruction + input (events) -> output (profile JSON).
  5. Print statistics on event count distribution.
  6. Save full and training versions.

Usage:
    python profile_s0_generate_sft_data.py \
        --events_file ./raw_data/ShoppingJourney_Input_500K_His50.tsv \
        --profiles_file ./raw_data/shopping_profiles.tsv \
        --output_dir ./sft_data \
        --max_events 50
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)


# =============================================================================
# Constants
# =============================================================================

# Default maximum number of user events to include in the input.
DEFAULT_MAX_EVENTS = 100

# Shopping profile JSON fields (for validation and statistics).
PROFILE_FIELDS = [
    "shoppingGenderPreference",
    "categoryPreferences",
    "brandPreferences",
    "retailerPreferences",
    "priceSensitivity",
    "fashionStyle",
    "fashionFit",
    "shoppingValues",
    "negativePreferences",
    "contextualShoppingInterests",
    "suggestedRelatedBrands",
]


# =============================================================================
# Data Loading
# =============================================================================

def read_user_events_tsv(filepath):
    """Read TSV file and extract UserId and ReadableUserEvents columns.

    Reads column names from the header row so the script is resilient to
    column ordering and extra columns. Mirrors the loading logic in
    pre_s3_construct_shopping_profile.py.

    Args:
        filepath: Path to the input TSV file.

    Returns:
        Dict of UserId -> events_text (with "#N#" replaced by newlines).
    """
    users = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        col_map = {name.strip(): idx for idx, name in enumerate(header)}
        if "UserId" not in col_map:
            raise ValueError(f"Column 'UserId' not found in header: {header}")
        if "ReadableUserEvents" not in col_map:
            raise ValueError(
                f"Column 'ReadableUserEvents' not found in header: {header}"
            )

        uid_idx = col_map["UserId"]
        events_idx = col_map["ReadableUserEvents"]

        for row in reader:
            if len(row) <= max(uid_idx, events_idx):
                continue
            user_id = row[uid_idx].strip()
            events_raw = row[events_idx].strip()
            if not user_id or not events_raw:
                continue
            # Replace "#N#" separator with actual newlines
            events_text = events_raw.replace("#N#", "\n")
            users[user_id] = events_text

    return users


def read_shopping_profiles_tsv(filepath):
    """Read shopping_profiles.tsv and extract UserId -> profile JSON string.

    Args:
        filepath: Path to shopping_profiles.tsv.

    Returns:
        Dict of UserId -> profile JSON string (raw).
    """
    profiles = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        col_map = {name.strip(): idx for idx, name in enumerate(header)}
        if "UserId" not in col_map:
            raise ValueError(f"Column 'UserId' not found in header: {header}")
        if "ShoppingProfile" not in col_map:
            raise ValueError(
                f"Column 'ShoppingProfile' not found in header: {header}"
            )

        uid_idx = col_map["UserId"]
        profile_idx = col_map["ShoppingProfile"]

        for row in reader:
            if len(row) <= max(uid_idx, profile_idx):
                continue
            user_id = row[uid_idx].strip()
            profile_raw = row[profile_idx].strip()
            if not user_id or not profile_raw:
                continue
            profiles[user_id] = profile_raw

    return profiles


# =============================================================================
# Validation
# =============================================================================

# Expected types for each profile field.
# String fields: shoppingGenderPreference, priceSensitivity
# List fields: all others
_STRING_FIELDS = {"shoppingGenderPreference", "priceSensitivity"}

# Allowed enum values
_VALID_GENDERS = {"general", "men", "women"}
_VALID_PRICE = {"general", "low-tier shopper", "mid-tier shopper", "high-tier shopper"}


def validate_profile_json(profile_str):
    """Parse and validate a shopping profile JSON string.

    Checks:
      1. Valid JSON that parses to a dict.
      2. Contains all 11 required PROFILE_FIELDS.
      3. Each field has the correct type (str or list).
      4. Gender and price sensitivity have valid enum values.
      5. At least one field is non-empty (all-general is OK).

    Args:
        profile_str: JSON string from shopping_profiles.tsv.

    Returns:
        Tuple of (parsed_dict, skip_reason). skip_reason is None if valid,
        otherwise a string describing why validation failed.
    """
    try:
        obj = json.loads(profile_str)
    except json.JSONDecodeError:
        return None, "json_parse_error"

    if not obj or not isinstance(obj, dict):
        return None, "not_a_dict"

    # Locate the profile dict (may or may not be wrapped)
    if "userShoppingProfile" in obj:
        profile = obj["userShoppingProfile"]
    else:
        profile = obj

    # Check all required fields exist
    missing = [f for f in PROFILE_FIELDS if f not in profile]
    if missing:
        return None, "missing_fields"

    # Check field types
    for field in PROFILE_FIELDS:
        val = profile[field]
        if field in _STRING_FIELDS:
            if not isinstance(val, str):
                return None, f"bad_type_{field}"
        else:
            if not isinstance(val, list):
                return None, f"bad_type_{field}"

    # Validate enum values
    gender = profile["shoppingGenderPreference"].strip().lower()
    if gender not in _VALID_GENDERS:
        return None, "invalid_gender_value"
    # Normalize gender to lowercase
    profile["shoppingGenderPreference"] = gender

    price = profile["priceSensitivity"].strip().lower()
    if price not in _VALID_PRICE:
        return None, "invalid_price_value"
    profile["priceSensitivity"] = price

    # Require at least one non-empty field (all-general is OK)
    has_content = False
    for field in PROFILE_FIELDS:
        val = profile[field]
        if isinstance(val, list) and len(val) > 0:
            has_content = True
            break
        if isinstance(val, str) and val.strip():
            has_content = True
            break
    if not has_content:
        return None, "all_fields_empty"

    return obj, None


# =============================================================================
# SFT Construction
# =============================================================================

# JSON schema string shared across all instruction variants.
_PROFILE_JSON_SCHEMA = (
    '{"userShoppingProfile": {'
    '"shoppingGenderPreference": "string", '
    '"categoryPreferences": ["string"], '
    '"brandPreferences": ["string"], '
    '"retailerPreferences": ["string"], '
    '"priceSensitivity": "string", '
    '"fashionStyle": ["string"], '
    '"fashionFit": ["string"], '
    '"shoppingValues": ["string"], '
    '"negativePreferences": ["string"], '
    '"contextualShoppingInterests": ["string"], '
    '"suggestedRelatedBrands": ["string"]'
    "}}"     
)

# Instruction variants for the shopping profile generation task.
# Each variant conveys the same task semantics with different phrasing
# to improve the model's robustness to instruction wording at inference.
INSTRUCTION_VARIANTS = [
    # 0 - Original
    "Analyze the user's shopping event history (browsing, searching, and "
    "clicking behavior) and generate a structured shopping profile that "
    "reflects the user's medium-to-long-term personal shopping preferences. "
    "Extract preferences only from repeated patterns or strong behavioral "
    "signals — ignore one-time events, gifts, and purchases for others. "
    "Output strictly as JSON:\n" + _PROFILE_JSON_SCHEMA,

    # 1 - Concise
    "Based on the user's browsing, search, and click history below, "
    "produce a JSON shopping profile capturing their long-term personal "
    "preferences. Focus on recurring patterns; disregard isolated events "
    "and purchases made for others. "
    "Output format:\n" + _PROFILE_JSON_SCHEMA,

    # 2 - Analyst persona
    "You are a shopping behavior analyst. Given the chronological event "
    "log of a user's shopping activities, infer their stable personal "
    "shopping preferences — including preferred categories, brands, "
    "price tier, and style. Ignore one-off or gift-related events. "
    "Return a JSON profile:\n" + _PROFILE_JSON_SCHEMA,

    # 3 - Step-by-step
    "Review the user's shopping events step by step. Identify repeated "
    "categories, brands, and style signals that indicate genuine personal "
    "preferences. Filter out one-time purchases, gifts, and actions for "
    "others. Summarize the findings as a structured JSON shopping profile. "
    "Output:\n" + _PROFILE_JSON_SCHEMA,

    # 4 - Personalization focus
    "From the shopping event sequence provided, extract the user's "
    "personalized shopping profile for downstream recommendation. "
    "Only include preferences supported by multiple behavioral signals. "
    "Exclude temporary needs, gifts, and one-off purchases. "
    "Respond with JSON only:\n" + _PROFILE_JSON_SCHEMA,

    # 5 - Evidence-based
    "Examine the user's shopping journey below and build an evidence-based "
    "shopping profile. Every preference must be supported by at least two "
    "events. Do not infer from single occurrences or gift purchases. "
    "Output the profile as JSON:\n" + _PROFILE_JSON_SCHEMA,

    # 6 - Behavioral pattern emphasis
    "Detect behavioral patterns in the user's shopping events — recurring "
    "product categories, brand loyalty, price range consistency, and style "
    "tendencies. Compile these into a structured shopping profile. Ignore "
    "noise from one-time browsing or purchases for others. "
    "JSON output:\n" + _PROFILE_JSON_SCHEMA,

    # 7 - Direct and short
    "Given the user's shopping event history, generate their personal "
    "shopping profile as JSON. Include only medium-to-long-term preferences "
    "backed by repeated behavior. Exclude gifts and one-off events. "
    "Format:\n" + _PROFILE_JSON_SCHEMA,

    # 8 - Task-oriented
    "Task: Convert the user's raw shopping events into a structured "
    "shopping profile. Extract stable preferences (categories, brands, "
    "gender, price tier, style) from repeated patterns. Discard isolated "
    "actions and purchases intended for other people. "
    "Output JSON:\n" + _PROFILE_JSON_SCHEMA,

    # 9 - Recommendation system context
    "As part of a recommendation system, analyze the user's browsing, "
    "search, and click events to construct their shopping preference "
    "profile. Capture only persistent personal interests — not gifts, "
    "temporary needs, or single isolated events. "
    "Return strictly as JSON:\n" + _PROFILE_JSON_SCHEMA,
]


def create_instruction():
    """Randomly select an instruction variant for the shopping profile task.

    The original variant (index 0) is selected with ~50% probability;
    the remaining 9 variants share the other ~50% equally.

    Returns:
        Instruction string.
    """
    if random.random() < 0.5:
        return INSTRUCTION_VARIANTS[0]
    return random.choice(INSTRUCTION_VARIANTS[1:])


def build_input_text(events_text, max_events):
    """Build the input text from a user's event history.

    Keeps the original event lines (already formatted as
    "N | time_ago | action | desc") and truncates to the most recent
    max_events without re-numbering.

    Args:
        events_text: Multi-line string of user events (newline-separated).
        max_events: Maximum number of events to include.

    Returns:
        Tuple of (input_text, num_events) where input_text is the formatted
        string and num_events is the number of events included.
    """
    lines = [l.strip() for l in events_text.strip().split("\n") if l.strip()]
    if not lines:
        return "", 0

    # Keep only the most recent events (ordered newest-first in TSV)
    lines = lines[:max_events]

    input_parts = ["User Shopping Events:"]
    input_parts.extend(lines)
    input_parts.append("")
    input_parts.append("Generate the user's shopping profile:")

    return "\n".join(input_parts), len(lines)


def format_profile_output(profile_obj):
    """Format the profile JSON object as a compact output string.

    Args:
        profile_obj: Parsed profile dict (with "userShoppingProfile" key).

    Returns:
        Compact JSON string.
    """
    return json.dumps(profile_obj, ensure_ascii=False, separators=(",", ":"))


def create_sft_sample(user_id, events_text, profile_obj, max_events):
    """Create a single SFT training sample.

    Args:
        user_id: User identifier string.
        events_text: Raw multi-line event string.
        profile_obj: Validated profile JSON object.
        max_events: Maximum events to include in input.

    Returns:
        SFT sample dict with instruction, input, output, and metadata.
    """
    instruction = create_instruction()
    input_text, num_events = build_input_text(events_text, max_events)
    output_text = format_profile_output(profile_obj)

    profile = profile_obj.get("userShoppingProfile", {})

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": {
            "user_id": user_id,
            "num_events": num_events,
            "num_category_prefs": len(profile.get("categoryPreferences", [])),
            "num_brand_prefs": len(profile.get("brandPreferences", [])),
            "gender_pref": profile.get("shoppingGenderPreference", ""),
            "price_sensitivity": profile.get("priceSensitivity", ""),
        },
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def create_profile_sft_data(user_events, profiles, max_events):
    """Create SFT data from user events and shopping profiles.

    Args:
        user_events: Dict of UserId -> events_text.
        profiles: Dict of UserId -> profile JSON string.
        max_events: Maximum events per input sequence.

    Returns:
        Tuple of (sft_data, invalid_users) where invalid_users is a list
        of (user_id, reason) tuples.
    """
    sft_data = []
    skip_reasons = defaultdict(int)
    invalid_users = []  # (user_id, reason)

    # Statistics
    event_counts = []
    cat_counts = []
    brand_counts = []
    gender_dist = defaultdict(int)
    price_dist = defaultdict(int)
    field_nonempty = defaultdict(int)  # field -> count of non-empty

    matched_uids = set(user_events.keys()) & set(profiles.keys())
    print(f"  Users in events file:    {len(user_events):>10,}")
    print(f"  Users in profiles file:  {len(profiles):>10,}")
    print(f"  Matched users:           {len(matched_uids):>10,}")

    for user_id in tqdm(sorted(matched_uids), desc="Building profile SFT data"):
        events_text = user_events[user_id]
        profile_str = profiles[user_id]

        # Validate profile JSON (returns reason string or None)
        profile_obj, skip_reason = validate_profile_json(profile_str)
        if skip_reason is not None:
            skip_reasons[skip_reason] += 1
            invalid_users.append((user_id, skip_reason))
            continue

        # Check events are non-empty (>= 1 event)
        event_lines = [
            l.strip() for l in events_text.strip().split("\n") if l.strip()
        ]
        if len(event_lines) < 1:
            skip_reasons["no_events"] += 1
            invalid_users.append((user_id, "no_events"))
            continue

        # Create the SFT sample
        sample = create_sft_sample(user_id, events_text, profile_obj, max_events)

        if sample["metadata"]["num_events"] < 1:
            skip_reasons["no_events_after_parse"] += 1
            invalid_users.append((user_id, "no_events_after_parse"))
            continue

        sft_data.append(sample)

        # Track statistics
        meta = sample["metadata"]
        event_counts.append(meta["num_events"])
        cat_counts.append(meta["num_category_prefs"])
        brand_counts.append(meta["num_brand_prefs"])
        gender_dist[meta["gender_pref"]] += 1
        price_dist[meta["price_sensitivity"]] += 1

        # Track per-field non-empty coverage
        profile = profile_obj.get("userShoppingProfile", profile_obj)
        for field in PROFILE_FIELDS:
            val = profile.get(field)
            if isinstance(val, list) and len(val) > 0:
                field_nonempty[field] += 1
            elif isinstance(val, str) and val.strip():
                field_nonempty[field] += 1

    # ---- Print statistics ----
    total_valid = len(sft_data)
    print(f"\nData statistics:")
    print(f"  Generated samples:            {total_valid:>10,}")
    print(f"  Invalid/skipped users:        {len(invalid_users):>10,}")
    print(f"  Skip reasons:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"    - {reason:30s} {count:>10,}")

    if total_valid > 0:
        print(f"\n  Per-field non-empty coverage (of {total_valid:,} valid samples):")
        print(f"  {'Field':<30s} {'Count':>10s} {'Coverage':>10s}")
        print(f"  {'-'*30} {'-'*10} {'-'*10}")
        for field in PROFILE_FIELDS:
            cnt = field_nonempty.get(field, 0)
            pct = cnt / total_valid * 100
            print(f"  {field:<30s} {cnt:>10,} {pct:>9.1f}%")

    if event_counts:
        arr = np.array(event_counts)
        print(f"\n  User event count distribution:")
        print(f"    Min:    {arr.min()}")
        print(f"    Max:    {arr.max()}")
        print(f"    Mean:   {arr.mean():.2f}")
        print(f"    Median: {np.median(arr):.1f}")
        print(f"    P25:    {np.percentile(arr, 25):.1f}")
        print(f"    P75:    {np.percentile(arr, 75):.1f}")
        print(f"    P95:    {np.percentile(arr, 95):.1f}")

        # Histogram buckets
        buckets = [1, 5, 10, 20, 30, 50, 100]
        print(f"\n  Event count histogram:")
        for i, upper in enumerate(buckets):
            lower = buckets[i - 1] if i > 0 else 0
            count = int(np.sum((arr > lower) & (arr <= upper)))
            print(f"    ({lower:>3d}, {upper:>3d}]: {count:>8,}")
        overflow = int(np.sum(arr > buckets[-1]))
        if overflow > 0:
            print(f"    (>{buckets[-1]:>3d}):   {overflow:>8,}")

    if cat_counts:
        arr = np.array(cat_counts)
        print(f"\n  Category preferences per profile:")
        print(f"    Min: {arr.min()}, Max: {arr.max()}, "
              f"Mean: {arr.mean():.1f}, Median: {np.median(arr):.1f}")

    if brand_counts:
        arr = np.array(brand_counts)
        print(f"\n  Brand preferences per profile:")
        print(f"    Min: {arr.min()}, Max: {arr.max()}, "
              f"Mean: {arr.mean():.1f}, Median: {np.median(arr):.1f}")

    if gender_dist:
        print(f"\n  Shopping gender distribution:")
        total = sum(gender_dist.values())
        for g, cnt in sorted(gender_dist.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100 if total > 0 else 0
            label = g if g else "(empty)"
            print(f"    {label:20s} {cnt:>8,} ({pct:5.1f}%)")

    if price_dist:
        print(f"\n  Price sensitivity distribution:")
        total = sum(price_dist.values())
        for p, cnt in sorted(price_dist.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100 if total > 0 else 0
            label = p if p else "(empty)"
            print(f"    {label:25s} {cnt:>8,} ({pct:5.1f}%)")

    return sft_data, invalid_users


def save_sft_data(sft_data, output_file, invalid_users=None):
    """Save SFT data (full and training versions) and invalid user IDs.

    Full version (with metadata): <name>_full.json
    Training version (instruction/input/output only): <name>.json
    Invalid users: <name>_invalid.tsv (UserId, Reason)

    Args:
        sft_data: List of SFT sample dicts.
        output_file: Path to the training JSON file.
        invalid_users: Optional list of (user_id, reason) tuples.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Full version with all metadata
    full_file = output_file.replace(".json", "_full.json")
    with open(full_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"Full data saved: {full_file}")

    # Training version (instruction, input, output only)
    training_data = [
        {
            "instruction": s["instruction"],
            "input": s["input"],
            "output": s["output"],
        }
        for s in sft_data
    ]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    print(f"Training data saved: {output_file}")

    # Invalid user IDs
    if invalid_users:
        invalid_file = output_file.replace(".json", "_invalid.tsv")
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write("UserId\tReason\n")
            for uid, reason in invalid_users:
                f.write(f"{uid}\t{reason}\n")
        print(f"Invalid users saved: {invalid_file} ({len(invalid_users):,} users)")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build shopping profile SFT data from user events and "
        "pre-generated profiles"
    )
    parser.add_argument(
        "--events_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/0128_0301/ShoppingJourney_Input_500K_His50.tsv",
        help="Path to input TSV file with UserId and ReadableUserEvents columns "
             "(same as pre_s3 input)",
    )
    parser.add_argument(
        "--profiles_file",
        type=str,
        default="./raw_data/shopping_profiles.tsv",
        help="Path to shopping_profiles.tsv from pre_s3_construct_shopping_profile "
             "(default: ./raw_data/shopping_profiles.tsv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_data",
        help="Output directory. SFT data saved to <output_dir>/profile_sft.json "
             "(default: ./sft_data)",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help=f"Maximum number of user events per input sequence "
             f"(default: {DEFAULT_MAX_EVENTS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for instruction variant sampling (default: 42)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # =========================================================================
    # Step 1: Load input files
    # =========================================================================
    print("=" * 70)
    print("Step 1: Loading input files")
    print("=" * 70)

    random.seed(args.seed)

    print(f"  Loading user events: {args.events_file}")
    user_events = read_user_events_tsv(args.events_file)
    print(f"    Users: {len(user_events):,}")

    print(f"  Loading shopping profiles: {args.profiles_file}")
    profiles = read_shopping_profiles_tsv(args.profiles_file)
    print(f"    Profiles: {len(profiles):,}")

    # =========================================================================
    # Step 2: Build SFT data
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Building profile SFT data")
    print(f"  max_events = {args.max_events}")
    print("=" * 70)

    sft_data, invalid_users = create_profile_sft_data(
        user_events, profiles, max_events=args.max_events
    )

    # =========================================================================
    # Step 3: Save output
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Saving output")
    print("=" * 70)

    output_file = os.path.join(args.output_dir, "profile_sft.json")
    save_sft_data(sft_data, output_file, invalid_users=invalid_users)

    # =========================================================================
    # Step 4: Show example cases
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("Example cases (first 3):")
    print(f"{'=' * 70}")
    for idx, sample in enumerate(sft_data[:3]):
        meta = sample["metadata"]
        print(f"\n--- Example {idx + 1} ---")
        print(f"  User ID:          {meta['user_id']}")
        print(f"  Num events:       {meta['num_events']}")
        print(f"  Gender pref:      {meta['gender_pref']}")
        print(f"  Price sensitivity:{meta['price_sensitivity']}")
        print(f"  Categories:       {meta['num_category_prefs']}")
        print(f"  Brands:           {meta['num_brand_prefs']}")
        print(f"  Instruction:      {sample['instruction'][:100]}...")
        print(f"  Input (first 5 lines):")
        for line in sample["input"].split("\n")[:5]:
            print(f"    {line}")
        # Pretty-print the output
        try:
            out_obj = json.loads(sample["output"])
            profile = out_obj.get("userShoppingProfile", {})
            print(f"  Output profile:")
            for field in PROFILE_FIELDS[:6]:
                val = profile.get(field, "")
                print(f"    {field}: {val}")
            if len(PROFILE_FIELDS) > 6:
                print(f"    ... ({len(PROFILE_FIELDS) - 6} more fields)")
        except json.JSONDecodeError:
            print(f"  Output: {sample['output'][:200]}...")
    print(f"\n{'=' * 70}")

    print(f"\nDone! Generated {len(sft_data)} training samples")


if __name__ == "__main__":
    main()
