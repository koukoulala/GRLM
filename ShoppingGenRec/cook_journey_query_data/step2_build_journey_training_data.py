"""Step 2: Build Journey SFT Training Data (Event → Journey)

Reads the Step 1 output TSV (with LLM-generated journeys) and builds SFT
training data in the instruction / input / output format.

Input TSV columns (from Step 1):
  UserId, ReadableUserEvents, ShoppingProfile, RequestTime, HisCount, OUTPUT

Each row's OUTPUT is a JSON string:
  {"ContinuedJourneys":[{
      "JourneyType":"...",
      "Title":"...",
      "Description":"...",
      "ConversationStarter":["..."],
      "Queries":[{"Query":"..."},...],
      "Reason":"..."
  },...]}

Output: JSON list of SFT samples:
  [{"instruction":"...", "input":"...", "output":"..."}, ...]

Usage:
    python step2_build_journey_training_data.py --input_file <step1_output.tsv>
"""

import argparse
import csv
import json
import os
import re
import sys
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


# =============================================================================
# Default Paths & Constants
# =============================================================================

DEFAULT_INPUT_FILE = (
    "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/"
    "1225_0325/JourneyWithProfile/JourneyWithConversationStarterAndDesc/"
    "Step0_UserProfile_500KEnUsHisRandom0408_199K_Journey.tsv"
)
DEFAULT_OUTPUT_DIR = (
    "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260417_OnlyJourney/sft_data/"
)

DEFAULT_MAX_EVENTS = 500
DEFAULT_OUTPUT_SUFFIX = ""
DEFAULT_MIN_JOURNEYS = 1
DEFAULT_MAX_JOURNEYS = 20
DEFAULT_KEEP_EMPTY_RATIO = 0.0
DEFAULT_COUNT_RATIO = 0.5


# =============================================================================
# Time Normalization  (reused from reference)
# =============================================================================

def _normalize_time_expr(match):
    """Normalize a single time expression to days or hours."""
    text = match.group(0)
    parts = re.findall(
        r'(\d+)\s*(month|week|day|hour|minute|second)s?', text, re.IGNORECASE
    )
    if not parts:
        return text

    total_hours = 0
    total_minutes = 0
    for num_str, unit in parts:
        num = int(num_str)
        u = unit.lower()
        if u == 'month':
            total_hours += num * 30 * 24
        elif u == 'week':
            total_hours += num * 7 * 24
        elif u == 'day':
            total_hours += num * 24
        elif u == 'hour':
            total_hours += num
        elif u == 'minute':
            total_minutes += num

    total_days = total_hours // 24
    if total_days > 0:
        return f"{total_days} days ago"
    elif total_hours > 0:
        return f"{total_hours} hours ago"
    elif total_minutes > 0:
        return f"{total_minutes} minutes ago"
    return "0 minutes ago"


def normalize_event_times(text):
    """Normalize all time expressions (weeks/months → days) in event text."""
    pattern = r'(?:\d+\s*(?:month|week|day|hour|minute|second)s?\s*)+ago'
    return re.sub(pattern, _normalize_time_expr, text, flags=re.IGNORECASE)


# =============================================================================
# Data Loading
# =============================================================================

def read_input_tsv(filepath, max_rows=0):
    """Read step-1 output TSV and parse each row.

    Returns list of dicts: user_id, events (list[str]), profile, request_time,
    his_count, journeys (list[dict] from OUTPUT JSON).
    """
    rows = []
    parse_fail = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        print(f"  Columns: {reader.fieldnames}", flush=True)

        for row in reader:
            user_id = row.get("UserId", "").strip()
            events_raw = row.get("ReadableUserEvents", "").strip()
            profile = row.get("ShoppingProfile", "").strip()
            request_time = row.get("RequestTime", "").strip()
            his_count = row.get("HisCount", "").strip()
            output_raw = row.get("OUTPUT", "").strip()

            if not user_id or not events_raw:
                continue

            # Parse events (separated by #N#)
            events = [e.strip() for e in events_raw.split("#N#") if e.strip()]

            # Parse OUTPUT JSON (handle TSV backslash-escaped quotes)
            journeys = []
            if output_raw:
                text = output_raw
                # Unescape: \" -> " (TSV csv writer escapechar='\\')
                for _ in range(3):
                    try:
                        obj = json.loads(text)
                        journeys = obj.get("ContinuedJourneys", [])
                        if not isinstance(journeys, list):
                            journeys = []
                        break
                    except json.JSONDecodeError:
                        text = text.replace('\\"', '"')
                else:
                    parse_fail += 1

            rows.append({
                "user_id": user_id,
                "events": events,
                "profile": profile,
                "request_time": request_time,
                "his_count": his_count,
                "journeys": journeys,
            })

            if len(rows) % 50000 == 0:
                print(f"    ... loaded {len(rows):,} rows", flush=True)
            if max_rows > 0 and len(rows) >= max_rows:
                break

    print(f"  Loaded {len(rows):,} rows  (JSON parse failures: {parse_fail})")
    return rows


# =============================================================================
# Journey Validation
# =============================================================================

def validate_journey(j):
    """Return True if the journey dict has the required fields."""
    if not isinstance(j, dict):
        return False
    if not j.get("Title"):
        return False
    if not j.get("Queries") or not isinstance(j["Queries"], list):
        return False
    return True


# =============================================================================
# Instruction / Input / Output Builders
# =============================================================================

def create_instruction(num_journeys, count_ratio=DEFAULT_COUNT_RATIO):
    """Create instruction text for event2journey SFT.

    Returns (instruction, has_count, prompt_line).
    """
    has_count = num_journeys > 0 and random.random() < count_ratio

    if has_count:
        opening = (
            f"Based on the user's shopping event history, predict "
            f"{num_journeys} shopping journey(s) the user is likely to pursue."
        )
    else:
        opening = (
            "Based on the user's shopping event history, predict "
            "an appropriate number of shopping journey(s) the user is likely to pursue."
        )

    instruction = (
        f"{opening}"
        " Each journey has a JourneyType ('explicit' or 'related'),"
        " a short engaging Title,"
        " a Description (2-3 sentences in personal-shopper tone highlighting"
        " why this journey fits the user and what value exploring it brings),"
        " a list of ConversationStarters (3 natural first-person openings"
        " that resume the shopping journey),"
        " a set of Queries (3-7 concise product search queries),"
        " and a Reason (explains which user signals triggered this journey)."
        ' Output JSON:'
        ' {"ContinuedJourneys":[{"JourneyType":"...","Title":"...",'
        '"Description":"...","ConversationStarter":["...","...","..."],'
        '"Queries":[{"Query":"..."},...],"Reason":"..."},...]}'
    )

    if has_count:
        jword = "journey" if num_journeys == 1 else "journeys"
        prompt_line = (
            f"Predict the user's shopping journeys, "
            f"exactly {num_journeys} {jword}:"
        )
    else:
        prompt_line = "Predict an appropriate number of shopping journeys:"

    return instruction, has_count, prompt_line


def build_input_text(events, max_events, prompt_line):
    """Build input text from user event history.

    Returns (input_text, num_events_used).
    """
    used = events[:max_events]
    lines = ["User Event History:"]
    for idx, event in enumerate(used, 1):
        event = normalize_event_times(event)
        if len(event) > 150:
            event = event[:150] + "..."
        lines.append(f"{idx} | {event}")
    lines.append("")
    lines.append(prompt_line)
    return "\n".join(lines), len(used)


def build_output_json(journeys):
    """Build the output JSON string from a list of journey dicts.

    Keeps: JourneyType, Title, Description, ConversationStarter, Queries, Reason.
    """
    clean = []
    for j in journeys:
        entry = {
            "JourneyType": j.get("JourneyType", "explicit"),
            "Title": j.get("Title", ""),
            "Description": j.get("Description", ""),
            "ConversationStarter": j.get("ConversationStarter", []),
            "Queries": j.get("Queries", []),
            "Reason": j.get("Reason", ""),
        }
        clean.append(entry)
    return json.dumps({"ContinuedJourneys": clean}, ensure_ascii=False)


# =============================================================================
# Save
# =============================================================================

def save_sft_data(sft_data, output_file):
    """Save SFT data in two versions: full (with metadata) and training-only.
    Both use JSONL format (one JSON object per line).
    """
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Full version with metadata
    full_file = output_file.replace(".jsonl", "_full.jsonl")
    with open(full_file, "w", encoding="utf-8") as f:
        for s in sft_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    full_mb = os.path.getsize(full_file) / (1024 * 1024)
    print(f"  Full data saved:     {full_file} ({full_mb:.1f} MB)")

    # Training version (instruction / input / output only)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in sft_data:
            record = {"instruction": s["instruction"], "input": s["input"], "output": s["output"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    train_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Training data saved: {output_file} ({train_mb:.1f} MB)")


# =============================================================================
# Main Pipeline
# =============================================================================

def create_sft_data(
    rows,
    max_events=DEFAULT_MAX_EVENTS,
    min_journeys=DEFAULT_MIN_JOURNEYS,
    max_journeys=DEFAULT_MAX_JOURNEYS,
    keep_empty_ratio=DEFAULT_KEEP_EMPTY_RATIO,
    count_ratio=DEFAULT_COUNT_RATIO,
):
    """Build SFT samples from parsed TSV rows.

    Returns list of SFT sample dicts.
    """
    sft_data = []
    skip_reasons = defaultdict(int)

    # Statistics
    event_counts = []
    journey_counts = []
    queries_per_journey = []
    conversation_starters_per_journey = []
    empty_journey_total = 0
    empty_journey_kept = 0
    subsampled_users = 0
    original_journey_counts_before_subsample = []
    instruction_with_count = 0
    instruction_without_count = 0
    invalid_journey_count = 0

    for entry in tqdm(rows, desc="Building SFT data", mininterval=10):
        user_id = entry["user_id"]
        events = entry["events"]
        raw_journeys = entry["journeys"]

        if not events:
            skip_reasons["no_events"] += 1
            continue

        # Validate journeys
        valid_journeys = []
        for j in raw_journeys:
            if validate_journey(j):
                valid_journeys.append(j)
            else:
                invalid_journey_count += 1

        # Handle empty journeys
        if not valid_journeys:
            if not raw_journeys:
                # Originally had no journeys
                empty_journey_total += 1
                if random.random() >= keep_empty_ratio:
                    skip_reasons["empty_journeys_sampled_out"] += 1
                    continue
                empty_journey_kept += 1
            else:
                # Had journeys but all invalid
                skip_reasons["all_journeys_invalid"] += 1
                continue

        # Journey subsampling
        if max_journeys and len(valid_journeys) > max_journeys:
            subsampled_users += 1
            original_journey_counts_before_subsample.append(len(valid_journeys))
            valid_journeys = random.sample(valid_journeys, max_journeys)

        # Check min_journeys (allow empty-journey samples through)
        if valid_journeys and len(valid_journeys) < min_journeys:
            skip_reasons["below_min_journeys"] += 1
            continue

        num_journeys = len(valid_journeys)

        # Build instruction / input / output
        instruction, has_count, prompt_line = create_instruction(
            num_journeys, count_ratio
        )
        input_text, num_events_used = build_input_text(
            events, max_events, prompt_line
        )
        output_text = build_output_json(valid_journeys)

        if has_count:
            instruction_with_count += 1
        else:
            instruction_without_count += 1

        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "user_id": user_id,
                "num_events": num_events_used,
                "num_journeys": num_journeys,
                "journey_types": [
                    j.get("JourneyType", "explicit") for j in valid_journeys
                ],
            },
        }
        sft_data.append(sample)

        # Collect statistics
        event_counts.append(num_events_used)
        journey_counts.append(num_journeys)
        for j in valid_journeys:
            queries_per_journey.append(len(j.get("Queries", [])))
            conversation_starters_per_journey.append(
                len(j.get("ConversationStarter", []))
            )

    # =========================================================================
    # Statistics
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("Data Statistics (event2journey)")
    print(f"{'=' * 70}")
    print(f"  Total input rows:             {len(rows):>10,}")
    print(f"  Generated SFT samples:        {len(sft_data):>10,}")
    print(f"  Invalid journeys dropped:     {invalid_journey_count:>10,}")

    # Skip reasons
    print(f"\n  --- Skip Reasons ---")
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:35s} {count:>10,}")
    else:
        print(f"    (none)")

    # Empty journey handling
    print(f"\n  --- Empty Journey Handling ---")
    print(f"  Users with 0 valid journeys:  {empty_journey_total:>10,}")
    print(f"  Kept as empty samples:        {empty_journey_kept:>10,}")
    print(f"  Keep ratio (config):          {keep_empty_ratio:>10.1%}")

    # Journey subsampling
    if max_journeys:
        print(f"\n  --- Journey Subsampling ---")
        print(f"  max_journeys:                 {max_journeys:>10}")
        print(f"  Users subsampled:             {subsampled_users:>10,}")
        if original_journey_counts_before_subsample:
            arr = np.array(original_journey_counts_before_subsample)
            print(f"  Orig journeys (subsampled): "
                  f"Mean={arr.mean():.1f}, Max={arr.max()}")

    # Instruction variants
    total_instr = instruction_with_count + instruction_without_count
    print(f"\n  --- Instruction Variants ---")
    print(f"  With journey count:           {instruction_with_count:>10,} "
          f"({instruction_with_count / max(total_instr, 1) * 100:.1f}%)")
    print(f"  Without journey count:        {instruction_without_count:>10,} "
          f"({instruction_without_count / max(total_instr, 1) * 100:.1f}%)")

    # Event distribution
    if event_counts:
        arr = np.array(event_counts)
        print(f"\n  --- Events per Sample ---")
        print(f"    Min: {arr.min():>6}  P25: {int(np.percentile(arr, 25)):>6}  "
              f"P50: {int(np.percentile(arr, 50)):>6}  "
              f"P75: {int(np.percentile(arr, 75)):>6}  "
              f"P90: {int(np.percentile(arr, 90)):>6}  "
              f"Max: {arr.max():>6}  Mean: {arr.mean():.1f}")

    # Journey count distribution
    if journey_counts:
        arr = np.array(journey_counts)
        print(f"\n  --- Journeys per User ---")
        print(f"    Min: {arr.min():>6}  Max: {arr.max():>6}  "
              f"Mean: {arr.mean():.1f}  Median: {np.median(arr):.1f}")
        jc_dist = defaultdict(int)
        for c in journey_counts:
            jc_dist[c] += 1
        print(f"    Bucket distribution:")
        for cnt in sorted(jc_dist.keys()):
            label = f"{cnt} journey" if cnt == 1 else f"{cnt} journeys"
            pct = jc_dist[cnt] / len(journey_counts) * 100
            bar = "#" * int(pct / 2)
            print(f"      {label:>12s}: {jc_dist[cnt]:>8,} users "
                  f"({pct:5.1f}%) {bar}")

    # Journey type distribution
    if journey_counts:
        type_dist = defaultdict(int)
        for s in sft_data:
            for jt in s["metadata"]["journey_types"]:
                type_dist[jt] += 1
        total_j = sum(type_dist.values())
        print(f"\n  --- Journey Type Distribution ---")
        for jtype, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
            print(f"    {jtype:>15s}: {cnt:>10,} "
                  f"({cnt / max(total_j, 1) * 100:.1f}%)")

    # Queries per journey
    if queries_per_journey:
        arr = np.array(queries_per_journey)
        print(f"\n  --- Queries per Journey ---")
        print(f"    Min: {arr.min():>6}  P50: {int(np.percentile(arr, 50)):>6}  "
              f"Max: {arr.max():>6}  Mean: {arr.mean():.1f}")

    # Conversation starters per journey
    if conversation_starters_per_journey:
        arr = np.array(conversation_starters_per_journey)
        print(f"\n  --- ConversationStarters per Journey ---")
        print(f"    Min: {arr.min():>6}  P50: {int(np.percentile(arr, 50)):>6}  "
              f"Max: {arr.max():>6}  Mean: {arr.mean():.1f}")

    print(f"{'=' * 70}")
    return sft_data


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2: Build Journey SFT Training Data (event2journey)"
    )
    parser.add_argument(
        "--input_file", type=str, default=DEFAULT_INPUT_FILE,
        help="Path to Step 1 output TSV",
    )
    parser.add_argument(
        "--output_suffix", type=str, default=DEFAULT_OUTPUT_SUFFIX,
        help="Suffix appended to output filename (default: empty)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for SFT JSON files",
    )
    parser.add_argument(
        "--max_events", type=int, default=DEFAULT_MAX_EVENTS,
        help=f"Max events in input (default: {DEFAULT_MAX_EVENTS})",
    )
    parser.add_argument(
        "--min_journeys", type=int, default=DEFAULT_MIN_JOURNEYS,
        help=f"Min journeys per user after filtering (default: {DEFAULT_MIN_JOURNEYS})",
    )
    parser.add_argument(
        "--max_journeys", type=int, default=DEFAULT_MAX_JOURNEYS,
        help=f"Max journeys per sample; excess subsampled (default: {DEFAULT_MAX_JOURNEYS})",
    )
    parser.add_argument(
        "--keep_empty_ratio", type=float, default=DEFAULT_KEEP_EMPTY_RATIO,
        help=f"Fraction of zero-journey users to keep (default: {DEFAULT_KEEP_EMPTY_RATIO})",
    )
    parser.add_argument(
        "--count_ratio", type=float, default=DEFAULT_COUNT_RATIO,
        help=f"Probability of including journey count in instruction (default: {DEFAULT_COUNT_RATIO})",
    )
    parser.add_argument(
        "--max_rows", type=int, default=0,
        help="Max input rows to read; 0 = all (default: 0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Debug mode: read only first 100 rows",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.debug:
        args.max_rows = args.max_rows or 100
        print(f"*** DEBUG MODE: max_rows = {args.max_rows} ***\n")

    # ---- Step 1: Load input ----
    print("=" * 70)
    print("Step 1: Loading input data")
    print(f"  File: {args.input_file}")
    print("=" * 70)

    rows = read_input_tsv(args.input_file, max_rows=args.max_rows)
    if not rows:
        print("ERROR: No rows loaded.")
        sys.exit(1)

    # Quick summary
    has_journeys = sum(1 for r in rows if r["journeys"])
    total_journeys = sum(len(r["journeys"]) for r in rows)
    print(f"  Users with journeys:  {has_journeys:,} / {len(rows):,}")
    print(f"  Total journeys:       {total_journeys:,}")

    # ---- Step 2: Build SFT data ----
    print()
    print("=" * 70)
    print("Step 2: Building SFT data")
    print(f"  max_events       = {args.max_events}")
    print(f"  min_journeys     = {args.min_journeys}")
    print(f"  max_journeys     = {args.max_journeys}")
    print(f"  keep_empty_ratio = {args.keep_empty_ratio}")
    print(f"  count_ratio      = {args.count_ratio}")
    print(f"  seed             = {args.seed}")
    print("=" * 70)

    sft_data = create_sft_data(
        rows,
        max_events=args.max_events,
        min_journeys=args.min_journeys,
        max_journeys=args.max_journeys,
        keep_empty_ratio=args.keep_empty_ratio,
        count_ratio=args.count_ratio,
    )

    if not sft_data:
        print("WARNING: No SFT samples generated.")
        sys.exit(1)

    # ---- Step 3: Save ----
    print()
    print("=" * 70)
    print("Step 3: Saving output")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    input_base = os.path.splitext(os.path.basename(args.input_file))[0]
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_file = os.path.join(args.output_dir, f"event2journey_sft{suffix}.jsonl")
    save_sft_data(sft_data, output_file)

    # ---- Step 4: Example cases ----
    print(f"\n{'=' * 70}")
    print("Example cases (first 3):")
    print(f"{'=' * 70}")
    for idx, sample in enumerate(sft_data[:3]):
        meta = sample["metadata"]
        print(f"\n--- Example {idx + 1} ---")
        print(f"  User ID:      {meta['user_id']}")
        print(f"  Num events:   {meta['num_events']}")
        print(f"  Num journeys: {meta['num_journeys']}")
        print(f"  Journey types: {meta['journey_types']}")
        print(f"  Instruction:  {sample['instruction'][:200]}...")
        input_lines = sample["input"].split("\n")
        print(f"  Input (first 10 lines):")
        for line in input_lines[:10]:
            print(f"    {line[:150]}")
        if len(input_lines) > 10:
            print(f"    ... ({len(input_lines) - 10} more lines)")
        # Pretty-print output
        try:
            out_obj = json.loads(sample["output"])
            cj = out_obj.get("ContinuedJourneys", [])
            if not cj:
                print(f"  Output:       (empty journeys)")
            else:
                print(f"  Output ({len(cj)} journeys):")
                for ji, journey in enumerate(cj[:3], 1):
                    title = journey.get("Title", "")
                    desc = journey.get("Description", "")
                    queries = journey.get("Queries", [])
                    starters = journey.get("ConversationStarter", [])
                    print(f"    Journey {ji}: {title}")
                    print(f"      Description: {desc[:120]}")
                    print(f"      Queries: {len(queries)}, "
                          f"ConversationStarters: {len(starters)}")
                if len(cj) > 3:
                    print(f"    ... ({len(cj) - 3} more journeys)")
        except json.JSONDecodeError:
            print(f"  Output: {sample['output'][:200]}...")

    print(f"\n{'=' * 70}")
    print(f"Done! {len(sft_data):,} SFT samples generated.")


if __name__ == "__main__":
    main()
