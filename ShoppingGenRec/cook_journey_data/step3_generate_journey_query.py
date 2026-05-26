"""
Step 0: Generate Shopping Journeys from User Events and Shopping Profile.

Reads input TSV with user events and shopping profile, calls an LLM
(Copilot or Papyrus) with the JourneyGeneration prompt (which takes
ReadableUserEvents, Profile, and RequestTime as inputs) to predict
continued shopping journeys, validates the JSON output, and writes the
result as a TSV with columns:
    UserId, ReadableUserEvents, ShoppingProfile, RequestTime, HisCount, OUTPUT

Usage (Copilot):
    python step0_generate_journey.py \\
        --input_file /path/to/input.tsv \\
        --output_dir /path/to/output/ \\
        --token_file ../resources/tokens.txt \\
        --copilot_model gpt-5.4 \\
        --num_workers 20

Usage (Papyrus):
    python step0_generate_journey.py \\
        --input_file /path/to/input.tsv \\
        --output_dir /path/to/output/ \\
        --inference_backend papyrus \\
        --papyrus_model gpt-5-chat-shortco-2025-08-07-Bing \\
        --papyrus_workers 40

Split large input into chunks (no LLM inference):
    python step0_generate_journey.py \\
        --input_file /path/to/input.tsv \\
        --output_dir /path/to/output/ \\
        --max_users 50000
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

from tqdm import tqdm

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)

# Add resources directory to path for llm_utils / Infer_by_papyrus import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESOURCES_DIR = os.path.join(PROJECT_DIR, "resources")
sys.path.insert(0, RESOURCES_DIR)

from llm_utils import (run_llm_parallel_with_checkpoint,
                        load_checkpoint,
                        cleanup_checkpoint)


# =============================================================================
# Event Processing Utilities
# =============================================================================

def _normalize_time_expr(match):
    """Normalize a single time expression to days or hours."""
    text = match.group(0)
    parts = re.findall(r'(\d+)\s*(month|week|day|hour|minute|second)s?', text,
                       re.IGNORECASE)
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
    """Normalize all time expressions (weeks/months -> days) in event text."""
    pattern = r'(?:\d+\s*(?:month|week|day|hour|minute|second)s?\s*)+ago'
    return re.sub(pattern, _normalize_time_expr, text, flags=re.IGNORECASE)


def truncate_events(events_text, max_events):
    """Keep only the first max_events lines (most recent events)."""
    lines = [line for line in events_text.strip().split("\n") if line.strip()]
    if len(lines) <= max_events:
        return events_text
    return "\n".join(lines[:max_events])


def _parse_event_days_ago(time_part):
    """Parse 'X days/hours/minutes ago' to fractional days. Returns None on failure."""
    m = re.match(r'(\d+)\s*(days?|hours?|minutes?)\s*ago',
                 time_part.strip(), re.IGNORECASE)
    if not m:
        return None
    num = int(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith('day'):
        return num
    elif unit.startswith('hour'):
        return num / 24.0
    elif unit.startswith('minute'):
        return num / (24.0 * 60)
    return None


def filter_events_by_time_window(events_text, user_id, seed=42):
    """Randomly select a time window and filter events.

    With equal probability (1/4 each), keeps events within:
      7 days, 14 days, 30 days, or all events.

    Uses user_id + seed for deterministic per-user randomness.
    If filtering results in 0 events, falls back to all events.

    Args:
        events_text: Normalized event text (newline-separated).
        user_id: User ID string (used for deterministic seed).
        seed: Global seed.

    Returns:
        Tuple of (filtered_events_text, window_label).
    """
    # Use MD5 for deterministic per-user seed (Python hash() is randomized
    # by PYTHONHASHSEED and not reproducible across runs).
    uid_hash = int(hashlib.md5(user_id.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(uid_hash ^ seed)
    windows = [7, 14, 30, None]  # None = all
    window = rng.choice(windows)

    if window is None:
        return events_text, "all"

    lines = [line for line in events_text.strip().split("\n") if line.strip()]
    kept = []
    for line in lines:
        parts = line.split("|", 3)
        if len(parts) < 2:
            kept.append(line)
            continue
        days = _parse_event_days_ago(parts[1])
        if days is not None and days <= window:
            kept.append(line)

    if not kept:
        return events_text, f"{window}d->all(fallback)"

    return "\n".join(kept), f"{window}d"


# =============================================================================
# Constants
# =============================================================================

# Input column names
COL_USER_ID = "UserId"
COL_READABLE_EVENTS = "ReadableUserEvents"
COL_SHOPPING_PROFILE = "ShoppingProfile"
COL_REQUEST_TIME = "RequestTime"
COL_HIS_COUNT = "HisCount"
OUTPUT_COL_NAME = "OUTPUT"

# Output columns (in order, before OUTPUT)
OUTPUT_COLUMNS = [COL_USER_ID, COL_READABLE_EVENTS, COL_SHOPPING_PROFILE,
                  COL_REQUEST_TIME, COL_HIS_COUNT]


# =============================================================================
# User ID Filtering
# =============================================================================

def load_exclude_user_ids(exclude_files):
    """Load user IDs to exclude from a list of TSV file paths.

    Each file is expected to be a TSV with a header row containing a
    'UserId' column. All user IDs found across all files are collected
    into a set.

    Args:
        exclude_files: List of file paths to read user IDs from.

    Returns:
        set: Set of user ID strings to exclude.
    """
    exclude_ids = set()
    for fpath in exclude_files:
        if not os.path.exists(fpath):
            print(f"  [WARNING] Exclude file not found, skipping: {fpath}")
            continue

        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header is None:
                continue

            col_map = {name.strip(): idx for idx, name in enumerate(header)}
            uid_idx = col_map.get(COL_USER_ID)

            if uid_idx is None:
                # Fallback: try reading by position (first column)
                uid_idx = 0
                print(f"  [WARNING] 'UserId' column not found in {fpath}, "
                      f"using first column")

            for row in reader:
                if len(row) > uid_idx:
                    uid = row[uid_idx].strip()
                    if uid:
                        exclude_ids.add(uid)
                        count += 1

        print(f"  Loaded {count:,} user IDs from {fpath}")

    print(f"  Total unique user IDs to exclude: {len(exclude_ids):,}")
    return exclude_ids


# =============================================================================
# Data Loading
# =============================================================================

def read_raw_input(filepath, exclude_ids=None, max_events=100, max_rows=0,
                   random_event_window=False, seed=42):
    """Read input TSV and return rows with user events, profile, and metadata.

    Expected input columns: UserId, ReadableUserEvents, ShoppingProfile,
    RequestTime, HisCount.

    Args:
        filepath: Path to the input TSV file.
        exclude_ids: Optional set of user IDs to skip.
        max_events: Maximum number of events to keep per user.
        max_rows: If >0, return at most this many rows (for debug).
        random_event_window: If True, randomly filter events to 7/14/30/all days.
        seed: Random seed for event window filtering.

    Returns:
        Tuple of (rows, header).
    """
    if exclude_ids is None:
        exclude_ids = set()

    rows = []
    total_read = 0
    excluded_count = 0
    window_stats = {}  # track time window distribution

    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        col_map = {name.strip(): idx for idx, name in enumerate(header)}

        # Required columns
        if COL_USER_ID not in col_map:
            raise ValueError(
                f"Column '{COL_USER_ID}' not found in header: {header}")
        if COL_READABLE_EVENTS not in col_map:
            raise ValueError(
                f"Column '{COL_READABLE_EVENTS}' not found in header: {header}")

        uid_idx = col_map[COL_USER_ID]
        events_idx = col_map[COL_READABLE_EVENTS]
        profile_idx = col_map.get(COL_SHOPPING_PROFILE)
        time_idx = col_map.get(COL_REQUEST_TIME)
        hiscount_idx = col_map.get(COL_HIS_COUNT)

        for row in tqdm(reader, desc="Reading input",
                        mininterval=60, maxinterval=90):
            total_read += 1

            if len(row) <= max(uid_idx, events_idx):
                continue

            user_id = row[uid_idx].strip()
            if not user_id:
                continue

            # Filter out excluded users
            if user_id in exclude_ids:
                excluded_count += 1
                continue

            events_raw = row[events_idx].strip()
            if not events_raw:
                continue

            # Process events for LLM: replace #N# separators, normalize, truncate
            events_text = events_raw.replace("#N#", "\n")
            events_text = normalize_event_times(events_text)

            # Random event time-window filtering
            if random_event_window:
                events_text, window_label = filter_events_by_time_window(
                    events_text, user_id, seed=seed)
                window_stats[window_label] = window_stats.get(window_label, 0) + 1

            events_text = truncate_events(events_text, max_events)

            # Read profile
            profile_text = ""
            if profile_idx is not None and len(row) > profile_idx:
                profile_text = row[profile_idx].strip()

            # Read request time
            request_time = ""
            if time_idx is not None and len(row) > time_idx:
                request_time = row[time_idx].strip()

            # Read HisCount
            his_count = ""
            if hiscount_idx is not None and len(row) > hiscount_idx:
                his_count = row[hiscount_idx].strip()

            rows.append({
                "raw_parts": list(row),
                "user_id": user_id,
                "events_text": events_text,
                "profile_text": profile_text,
                "request_time": request_time,
                "his_count": his_count,
                "raw_events": events_raw,
            })

            if max_rows > 0 and len(rows) >= max_rows:
                break

    print(f"  Read {total_read:,} rows total")
    if exclude_ids:
        print(f"  Excluded {excluded_count:,} users")
    print(f"  Kept {len(rows):,} users after filtering")
    if window_stats:
        print(f"  Event time-window distribution:")
        for label in sorted(window_stats.keys()):
            print(f"    {label:<20s} {window_stats[label]:>10,}")
    return rows, [h.strip() for h in header]


# =============================================================================
# Prompt Construction
# =============================================================================

def load_prompt_template(prompt_file):
    """Load prompt template from a file (.md/.txt).

    Args:
        prompt_file: Path to the prompt template file.

    Returns:
        Prompt template string.
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def build_journey_prompt(events_text, profile_text, request_time,
                         prompt_template):
    """Build LLM prompt for journey generation.

    Uses ##placeholder## style substitution for:
      - ##ReadableUserEvents## -> events_text
      - ##Profile## -> profile_text
      - ##RequestTime## -> request_time

    Args:
        events_text: Cleaned event history string.
        profile_text: User shopping profile JSON string.
        request_time: Request time string.
        prompt_template: Prompt template string with ##placeholders##.

    Returns:
        Formatted prompt string.
    """
    result = prompt_template.replace("##ReadableUserEvents##", events_text)
    result = result.replace("##Profile##", profile_text)
    result = result.replace("##RequestTime##", request_time)
    return result


# =============================================================================
# JSON Validation
# =============================================================================

def extract_and_validate_journey_json(raw_text):
    """Extract and validate ContinuedJourneys JSON from LLM output.

    Attempts to parse the raw LLM output as JSON. If it contains markdown
    code fences or <OUTPUT> tags, extracts the JSON block first. Validates
    that the JSON has a 'ContinuedJourneys' key.

    Args:
        raw_text: Raw LLM output string.

    Returns:
        str: Cleaned, compact JSON string if valid, or empty string.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text.strip()

    # Strip <OUTPUT> tags if present
    output_match = re.search(
        r'<OUTPUT>\s*(.*?)\s*</OUTPUT>', text, re.DOTALL)
    if output_match:
        text = output_match.group(1).strip()

    # Strip markdown code fences if present
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to parse as JSON
    parsed = _try_parse_json(text)

    if parsed is None:
        # Try to find a JSON object in the text
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end > brace_start:
            parsed = _try_parse_json(text[brace_start:brace_end + 1])

    if parsed is None:
        return ""

    # Validate structure: must have ContinuedJourneys
    if "ContinuedJourneys" not in parsed:
        return ""

    journeys = parsed["ContinuedJourneys"]
    if not isinstance(journeys, list):
        return ""

    # Validate each journey has at least Title and Queries
    valid_journeys = []
    for j in journeys:
        if not isinstance(j, dict):
            continue
        if "Title" not in j or "Queries" not in j:
            continue
        queries = j.get("Queries", [])
        if not isinstance(queries, list) or not queries:
            continue
        valid_journeys.append(j)

    if not valid_journeys:
        return ""

    parsed["ContinuedJourneys"] = valid_journeys
    return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))


def _try_parse_json(text):
    """Try to parse text as JSON, return dict or None."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# =============================================================================
# LLM Inference
# =============================================================================

def _build_inputs(rows, prompt_template):
    """Build (user_id, prompt) inputs from rows list."""
    print("  Building prompts ...")
    inputs = []
    for row in rows:
        prompt = build_journey_prompt(
            row["events_text"],
            row["profile_text"],
            row["request_time"],
            prompt_template,
        )
        inputs.append((row["user_id"], prompt))
    return inputs


def run_journey_generation_copilot(
    rows,
    prompt_template,
    token_file,
    copilot_model,
    num_workers,
    max_tokens,
    checkpoint_dir,
    chunk_size=500,
):
    """Generate shopping journeys via Copilot API with checkpoint/resume."""
    inputs = _build_inputs(rows, prompt_template)
    return run_llm_parallel_with_checkpoint(
        inputs=inputs,
        token_file=token_file,
        checkpoint_dir=checkpoint_dir,
        num_workers=num_workers,
        model=copilot_model,
        temperature=0,
        max_tokens=max_tokens,
        chunk_size=chunk_size,
    )


def run_journey_generation_papyrus(
    rows,
    prompt_template,
    papyrus_endpoint,
    papyrus_model,
    papyrus_quota_id,
    papyrus_timeout_ms,
    papyrus_workers,
    max_tokens,
    checkpoint_dir,
    chunk_size=10000,
    debug=False,
):
    """Generate shopping journeys via Papyrus API with checkpoint/resume."""
    from Infer_by_papyrus import (run_papyrus_parallel,
                                   run_papyrus_parallel_with_checkpoint)

    inputs = _build_inputs(rows, prompt_template)

    if debug:
        return run_papyrus_parallel(
            inputs=inputs,
            papyrus_endpoint=papyrus_endpoint,
            model_name=papyrus_model,
            quota_id=papyrus_quota_id,
            timeout_ms=papyrus_timeout_ms,
            num_workers=papyrus_workers,
            max_tokens=max_tokens,
            max_retries=3,
        )
    else:
        return run_papyrus_parallel_with_checkpoint(
            inputs=inputs,
            checkpoint_dir=checkpoint_dir,
            papyrus_endpoint=papyrus_endpoint,
            model_name=papyrus_model,
            quota_id=papyrus_quota_id,
            timeout_ms=papyrus_timeout_ms,
            num_workers=papyrus_workers,
            max_tokens=max_tokens,
            max_retries=3,
            chunk_size=chunk_size,
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3: Generate shopping journeys from user events "
        "and shopping profile using Copilot or Papyrus API"
    )

    # --- I/O ---
    parser.add_argument(
        "--input_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
                "/Data/LLMTrainingData/20260513/raw_data"
                "/UserEvents_clean_profiles_results.tsv",
        help="Path to input TSV (output of step2, must have UserId, "
             "ReadableUserEvents, ShoppingProfile, RequestTime, HisCount)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
                "/Data/LLMTrainingData/20260513/raw_data",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Output file name (default: derived from input file name)",
    )
    parser.add_argument(
        "--exclude_files", type=str, nargs="*", default=None,
        help="List of TSV file paths whose UserId values will be excluded",
    )

    # --- Prompt ---
    parser.add_argument(
        "--prompt_file", type=str,
        default=os.path.join(
            PROJECT_DIR, "prompts",
            "ShoppingJourneyPromptV3.md"),
        help="Path to prompt template file (.md/.txt) with ##placeholders##",
    )

    # --- Inference backend ---
    parser.add_argument(
        "--inference_backend", type=str, default="copilot",
        choices=["copilot", "papyrus"],
        help="Inference backend: 'copilot' or 'papyrus' (default: copilot)",
    )

    # --- Copilot settings ---
    parser.add_argument(
        "--token_file", type=str,
        default=os.path.join(PROJECT_DIR, "resources", "tokens_full.txt"),
        help="Path to tokens.txt for Copilot API authentication",
    )
    parser.add_argument(
        "--copilot_model", type=str, default="gpt-5.2",
        help="Copilot model name",
    )
    parser.add_argument(
        "--num_workers", type=int, default=40,
        help="Number of parallel workers for Copilot API calls",
    )

    # --- Papyrus settings ---
    parser.add_argument(
        "--papyrus_endpoint", type=str,
        default="https://westus2batch.papyrus.binginternal.com",
        help="Papyrus API endpoint URL",
    )
    parser.add_argument(
        "--papyrus_model", type=str,
        default="gpt-54-2026-03-05-Eval",
        help="Papyrus model name",
    )
    parser.add_argument(
        "--papyrus_quota_id", type=str, default="",
        help="Papyrus quota ID",
    )
    parser.add_argument(
        "--papyrus_timeout_ms", type=int, default=600000,
        help="Papyrus request timeout in ms (default: 600000 = 10min)",
    )
    parser.add_argument(
        "--papyrus_workers", type=int, default=80,
        help="Number of parallel async workers for Papyrus",
    )

    # --- Common LLM settings ---
    parser.add_argument(
        "--max_tokens", type=int, default=10000,
        help="Maximum output tokens per API call",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=10000,
        help="Users per processing chunk for checkpoint saving",
    )
    parser.add_argument(
        "--max_events", type=int, default=500,
        help="Maximum number of events to keep per user",
    )
    parser.add_argument(
        "--random_event_window", action="store_true",
        default=True,
        dest="random_event_window",
        help="Randomly filter events to 7/14/30/all days with equal "
             "probability. Profile captures long-term interests while "
             "events capture variable-length short-term signals. "
             "(default: True, use --no-random_event_window to disable)",
    )
    parser.add_argument(
        "--no-random_event_window", action="store_false",
        dest="random_event_window",
        help="Disable random event window filtering.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for event window filtering (default: 42)",
    )

    parser.add_argument(
        "--max_users", type=int, default=0,
        help="If >0, split the filtered input into chunks of this size, "
             "save each chunk as a separate TSV file, and stop (no LLM "
             "inference). If <=0, read all data and run normally.",
    )
    parser.add_argument(
        "--split_subdir", type=str, default="split_chunks",
        help="Subdirectory name under output_dir for split chunk files. "
             "If provided, chunks are written to output_dir/split_subdir/.",
    )

    # --- Debug ---
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: process only first --debug_rows users",
    )
    parser.add_argument(
        "--debug_rows", type=int, default=50,
        help="Number of users in debug mode",
    )

    # --- Checkpoint & merge ---
    parser.add_argument(
        "--resume_checkpoint_dir", type=str, nargs="*",
        default=["/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/_journey_checkpoint_UserEvents_clean_profiles_results",],
        help="One or more checkpoint directories from previous runs. "
             "All completed results are loaded and merged. "
             "Pass flag with no arguments to disable loading. "
             "The working checkpoint is always auto-created in output_dir.",
    )
    parser.add_argument(
        "--cleanup_checkpoint", action="store_true",
        help="Delete the checkpoint directory after successful completion. "
             "By default, checkpoints are preserved for resume/inspection.",
    )
    parser.add_argument(
        "--merge_results_dir", type=str, default=None,
        help="Merge mode: path to a directory containing *_Results.tsv "
             "files. Finds and merges them into a single output TSV and "
             "exits. No LLM inference is run.",
    )
    parser.add_argument(
        "--combine_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
                "/Data/LLMTrainingData/20260513/raw_data"
                "/v3_Step0_UserProfile_500KEnUsHisRandom0408_500K_Journey.tsv",
        help="Path to an existing results TSV to combine with the current "
             "output. Produces an additional *_combined.tsv file that "
             "merges rows from this file and the current output (current "
             "results take priority for duplicate UserIds). "
             "Set to empty string to disable. (default: 500K_Journey.tsv)",
    )

    return parser.parse_args()


# =============================================================================
# Merge Results
# =============================================================================

def combine_with_existing(current_file, combine_file, combined_output):
    """Combine current results with an existing results file.

    Current results take priority for duplicate UserIds.
    Prints detailed row-count statistics.

    Args:
        current_file: Path to the current run's output TSV.
        combine_file: Path to the existing results TSV to combine with.
        combined_output: Path for the combined output TSV.

    Returns:
        int: Total rows in the combined output.
    """
    # Read combine file
    combine_rows = {}  # uid -> row
    combine_header = None
    with open(combine_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        combine_header = next(reader, None)
        for row in reader:
            if row and row[0].strip():
                combine_rows[row[0].strip()] = row

    # Read current file
    current_rows = {}  # uid -> row
    current_header = None
    with open(current_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        current_header = next(reader, None)
        for row in reader:
            if row and row[0].strip():
                current_rows[row[0].strip()] = row

    # Merge: combine_file first, then current overrides
    merged = dict(combine_rows)
    merged.update(current_rows)

    # Stats
    overlap = set(combine_rows.keys()) & set(current_rows.keys())
    only_combine = len(combine_rows) - len(overlap)
    only_current = len(current_rows) - len(overlap)

    print(f"  Combine file:    {len(combine_rows):>10,} rows  ({combine_file})")
    print(f"  Current output:  {len(current_rows):>10,} rows  ({current_file})")
    print(f"  Overlap (current wins): {len(overlap):>7,}")
    print(f"  Only in combine file:   {only_combine:>7,}")
    print(f"  Only in current output: {only_current:>7,}")
    print(f"  Combined total:  {len(merged):>10,} rows")

    # Write combined file
    header = current_header or combine_header
    with open(combined_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        if header:
            writer.writerow(header)
        for row in merged.values():
            writer.writerow(row)

    file_size_mb = os.path.getsize(combined_output) / (1024 * 1024)
    print(f"  Combined output: {combined_output} ({file_size_mb:.1f} MB)")
    return len(merged)


def merge_result_files(result_files, output_file):
    """Merge multiple Journey_Results TSV files into one.

    Deduplicates by UserId (later files overwrite earlier ones).
    Preserves the header from the first file.
    """
    print(f"\nMerging {len(result_files)} result files ...")
    merged = {}  # uid -> row (list of fields)
    header = None
    uid_col = 0  # UserId is first column

    for fpath in result_files:
        if not os.path.exists(fpath):
            print(f"  [WARNING] File not found, skipping: {fpath}")
            continue
        count = 0
        with open(fpath, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            file_header = next(reader, None)
            if file_header is None:
                continue
            if header is None:
                header = file_header
            for row in reader:
                if len(row) > uid_col:
                    uid = row[uid_col].strip()
                    if uid:
                        merged[uid] = row
                        count += 1
        print(f"  {os.path.basename(fpath)}: {count:,} rows")

    print(f"  Total unique users after merge: {len(merged):,}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        if header:
            writer.writerow(header)
        for row in merged.values():
            writer.writerow(row)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Output: {output_file} ({file_size_mb:.1f} MB, {len(merged):,} rows)")
    return len(merged)


# =============================================================================
# Results Output
# =============================================================================

def save_results_tsv(output_file, all_rows, result_map):
    """Write results TSV for all input rows, filling OUTPUT from result_map.

    Args:
        output_file: Output TSV file path.
        all_rows: List of row dicts (from read_raw_input).
        result_map: Dict of {user_id: raw_llm_output_text}.

    Returns:
        Tuple of (written, success_count, json_valid_count, json_invalid_ids).
    """
    output_header = OUTPUT_COLUMNS + [OUTPUT_COL_NAME]

    success_count = 0
    json_valid_count = 0
    json_invalid_ids = []
    written = 0

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(output_header)

        for row_data in all_rows:
            uid = row_data["user_id"]
            raw_output = result_map.get(uid, "")

            if raw_output:
                success_count += 1

            clean_json = extract_and_validate_journey_json(raw_output)
            if clean_json:
                json_valid_count += 1
                output_val = clean_json
            elif raw_output:
                json_invalid_ids.append(uid)
                output_val = raw_output.replace("\n", " ").replace("\t", " ")
            else:
                output_val = ""

            out_events = row_data["events_text"].replace("\n", "#N#")
            out_row = [
                uid,
                out_events,
                row_data["profile_text"],
                row_data["request_time"],
                row_data["his_count"],
                output_val,
            ]
            writer.writerow(out_row)
            written += 1

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Written {written:,} rows ({file_size_mb:.1f} MB)")
    print(f"  With results:    {success_count}/{len(all_rows)}")
    print(f"  Valid JSON:      {json_valid_count}/{success_count}")
    if json_invalid_ids:
        preview = json_invalid_ids[:10]
        print(f"  Invalid JSON ({len(json_invalid_ids)} users): "
              f"{preview}{'...' if len(json_invalid_ids) > 10 else ''}")

    return written, success_count, json_valid_count, json_invalid_ids


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("Step 3: Generate Shopping Journeys (with Profile)")
    print("=" * 70)
    print(f"  Input file:    {args.input_file}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Prompt file:   {args.prompt_file}")
    print(f"  Backend:       {args.inference_backend}")
    print(f"  Max events:    {args.max_events}")
    print(f"  Random event window: {args.random_event_window}")
    if args.max_users > 0:
        print(f"  Max users:     {args.max_users} (split mode, no inference)")
    if args.merge_results_dir:
        print(f"  Merge results: {args.merge_results_dir}")
    if args.exclude_files:
        print(f"  Exclude files: {len(args.exclude_files)} file(s)")
    if args.combine_file:
        print(f"  Combine file:  {args.combine_file}")
    print()

    if args.debug:
        print(f"*** DEBUG MODE: processing only {args.debug_rows} users ***\n")

    # ---- Load exclude user IDs ----
    exclude_ids = set()
    if args.exclude_files:
        print("Loading user IDs to exclude ...")
        exclude_ids = load_exclude_user_ids(args.exclude_files)
        print()

    # ---- Read input ----
    print("Reading input ...")
    max_rows = args.debug_rows if args.debug else 0  # 0 = read all
    rows, header = read_raw_input(
        args.input_file,
        exclude_ids=exclude_ids,
        max_events=args.max_events,
        max_rows=max_rows,
        random_event_window=args.random_event_window,
        seed=args.seed,
    )

    if not rows:
        print("No users to process after filtering!")
        return

    # ---- Resolve checkpoint dirs ----
    os.makedirs(args.output_dir, exist_ok=True)
    input_base = os.path.splitext(os.path.basename(args.input_file))[0]
    # Working checkpoint (always in output_dir, auto-created)
    checkpoint_dir = os.path.join(
        args.output_dir, f"_journey_checkpoint_{input_base}")

    # ---- Load completed users from resume checkpoints ----
    completed_results = {}  # uid -> result text (for output merging)
    resume_dirs = args.resume_checkpoint_dir or []
    for resume_dir in resume_dirs:
        if resume_dir and os.path.isdir(resume_dir):
            completed = load_checkpoint(resume_dir)
            new_count = len(set(completed.keys()) - set(completed_results.keys()))
            completed_results.update(completed)
            print(f"  Loaded checkpoint: +{new_count:,} users from {resume_dir}")
    # Also check the working checkpoint dir itself (for auto-resume)
    if os.path.isdir(checkpoint_dir) and checkpoint_dir not in resume_dirs:
        completed_work = load_checkpoint(checkpoint_dir)
        new_ids = set(completed_work.keys()) - set(completed_results.keys())
        if new_ids:
            completed_results.update(completed_work)
            print(f"  Working checkpoint: +{len(new_ids):,} users")
    if completed_results:
        print(f"  Total loaded from all checkpoints: {len(completed_results):,}")
        # Validate: only keep users with valid ContinuedJourneys JSON
        pre_count = len(completed_results)
        completed_results = {
            uid: text for uid, text in completed_results.items()
            if extract_and_validate_journey_json(text)
        }
        invalid_ckpt = pre_count - len(completed_results)
        if invalid_ckpt > 0:
            print(f"  [VALIDATE] Filtered {invalid_ckpt:,} invalid JSON, "
                  f"keeping {len(completed_results):,} valid")
        else:
            print(f"  [VALIDATE] All {len(completed_results):,} have valid JSON")

    # ---- Load completed results from merge_results_dir (TSV files) ----
    if args.merge_results_dir and os.path.isdir(args.merge_results_dir):
        pattern = os.path.join(args.merge_results_dir, "*_Results*.tsv")
        result_files = sorted(glob.glob(pattern))
        if result_files:
            print(f"\n[MERGE] Loading results from {len(result_files)} "
                  f"TSV files in {args.merge_results_dir} ...")
            merge_new = 0
            merge_override = 0
            merge_invalid = 0
            for fpath in result_files:
                file_count = 0
                with open(fpath, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f, delimiter="\t")
                    file_header = next(reader, None)
                    if file_header is None:
                        continue
                    col_map = {name.strip(): idx
                               for idx, name in enumerate(file_header)}
                    uid_idx = col_map.get(COL_USER_ID, 0)
                    output_idx = col_map.get(OUTPUT_COL_NAME)
                    if output_idx is None:
                        output_idx = len(file_header) - 1
                    for row in reader:
                        if len(row) > max(uid_idx, output_idx):
                            uid = row[uid_idx].strip()
                            output_text = row[output_idx]
                            if uid and output_text:
                                clean = extract_and_validate_journey_json(
                                    output_text)
                                if not clean:
                                    merge_invalid += 1
                                    continue
                                if uid in completed_results:
                                    merge_override += 1
                                else:
                                    merge_new += 1
                                completed_results[uid] = output_text
                                file_count += 1
                print(f"    {os.path.basename(fpath)}: {file_count:,} valid rows")
            print(f"  From merge dir: {merge_new:,} new, "
                  f"{merge_override:,} overriding checkpoint, "
                  f"{merge_invalid:,} skipped (invalid JSON)")
        else:
            print(f"\n[MERGE] No *_Results*.tsv files found in: "
                  f"{args.merge_results_dir}")

    if completed_results:
        print(f"  Total completed from all sources: {len(completed_results):,}")
    completed_ids = set(completed_results.keys())

    # ---- Determine output file path ----
    if args.output_name:
        out_name = args.output_name
    else:
        base = os.path.splitext(os.path.basename(args.input_file))[0]
        suffix = "_debug" if args.debug else ""
        out_name = f"{base}_Journey_Results{suffix}.tsv"
    output_file = os.path.join(args.output_dir, out_name)

    # ---- If max_users > 0, split REMAINING users into chunks ----
    if args.max_users > 0 and not args.debug:
        # Save preliminary results (checkpoint data) before splitting
        if completed_results:
            print(f"\nSaving preliminary results (from checkpoints) ...")
            save_results_tsv(output_file, rows, dict(completed_results))
            print(f"  Saved to: {output_file}")

        # Filter out already-completed users
        remaining_rows = [r for r in rows if r["user_id"] not in completed_ids]
        print(f"\n  Total users: {len(rows):,}")
        print(f"  Already completed (in checkpoint): {len(rows) - len(remaining_rows):,}")
        print(f"  Remaining to process: {len(remaining_rows):,}")

        if not remaining_rows:
            print("\n  All users already completed! Nothing to split.")
            return

        # Determine split output directory
        if args.split_subdir:
            split_dir = os.path.join(args.output_dir, args.split_subdir)
            os.makedirs(split_dir, exist_ok=True)
        else:
            split_dir = args.output_dir

        base_name = input_base
        chunk_size = args.max_users
        num_chunks = (len(remaining_rows) + chunk_size - 1) // chunk_size
        print(f"\nSplitting {len(remaining_rows):,} remaining rows into "
              f"{num_chunks} chunk(s) of up to {chunk_size:,} each ...")

        for ci in range(num_chunks):
            chunk = remaining_rows[ci * chunk_size : (ci + 1) * chunk_size]
            chunk_k = chunk_size // 1000
            if num_chunks > 1:
                suffix = f"_{chunk_k}K_{ci + 1}"
            else:
                suffix = f"_{chunk_k}K"
            chunk_file = os.path.join(
                split_dir, f"{base_name}{suffix}.tsv")
            with open(chunk_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t", lineterminator="\n")
                writer.writerow(header)
                for row_data in chunk:
                    writer.writerow(row_data["raw_parts"])
            chunk_mb = os.path.getsize(chunk_file) / (1024 * 1024)
            print(f"  Chunk {ci + 1}/{num_chunks}: {len(chunk):,} rows "
                  f"({chunk_mb:.1f} MB) -> {chunk_file}")

        print(f"\nDone! Split into {num_chunks} file(s) in {split_dir}")
        print(f"Run each chunk with:")
        print(f"  python step3_generate_journey_query.py \\")
        print(f"    --input_file <chunk_file> \\")
        print(f"    --output_dir {split_dir}")
        return

    # Show sample
    sample = rows[0]
    event_lines = sample["events_text"].strip().split("\n")
    print(f"\n  Sample user: {sample['user_id']}")
    print(f"  Request time: {sample['request_time'] or '(not available)'}")
    print(f"  Profile: {(sample['profile_text'] or '(empty)')[:200]}")
    print(f"  Events ({len(event_lines)}):")
    for line in event_lines[:3]:
        print(f"    {line[:120]}")
    if len(event_lines) > 3:
        print(f"    ... ({len(event_lines) - 3} more)")

    # ---- Filter out already-completed users before inference ----
    all_rows = rows  # keep all rows for final output
    if completed_ids:
        rows = [r for r in rows if r["user_id"] not in completed_ids]
        skipped = len(all_rows) - len(rows)
        if skipped > 0:
            print(f"\n  Skipped {skipped:,} already-completed users "
                  f"(from checkpoint), {len(rows):,} remaining")

    # ---- Save preliminary results (checkpoint snapshot) before inference ----
    if completed_results:
        print(f"\nSaving preliminary results (checkpoint snapshot) ...")
        save_results_tsv(output_file, all_rows, dict(completed_results))
        print(f"  Saved to: {output_file}")

    # ---- Load prompt template ----
    print(f"\nLoading prompt from: {args.prompt_file}")
    prompt_template = load_prompt_template(args.prompt_file)
    print("  Prompt template loaded successfully")

    # ---- Setup output ----
    # (checkpoint_dir already resolved above)

    # ---- Run LLM inference (only for non-completed users) ----
    if not rows:
        print("\n  All users already completed! Skipping inference.")
        results = []
    else:
        print(f"\nStarting journey generation ({len(rows):,} users) ...")
        start_time = time.time()

        if args.inference_backend == "copilot":
            print(f"  Model:   {args.copilot_model}")
            print(f"  Workers: {args.num_workers}")
            results = run_journey_generation_copilot(
                rows=rows,
                prompt_template=prompt_template,
                token_file=args.token_file,
                copilot_model=args.copilot_model,
                num_workers=args.num_workers,
                max_tokens=args.max_tokens,
                checkpoint_dir=checkpoint_dir,
                chunk_size=args.chunk_size,
            )
        else:
            print(f"  Model:   {args.papyrus_model}")
            print(f"  Workers: {args.papyrus_workers}")
            results = run_journey_generation_papyrus(
                rows=rows,
                prompt_template=prompt_template,
                papyrus_endpoint=args.papyrus_endpoint,
                papyrus_model=args.papyrus_model,
                papyrus_quota_id=args.papyrus_quota_id,
                papyrus_timeout_ms=args.papyrus_timeout_ms,
                papyrus_workers=args.papyrus_workers,
                max_tokens=args.max_tokens,
                checkpoint_dir=checkpoint_dir,
                chunk_size=args.chunk_size,
                debug=args.debug,
            )

        elapsed = time.time() - start_time
        print(f"\nLLM inference done in {elapsed:.1f}s "
              f"({len(rows) / elapsed:.1f} users/s)")

    # ---- Validate and save final output ----
    # Build user_id -> result mapping (checkpoint results + new results)
    result_map = dict(completed_results)
    result_map.update({uid: text for uid, text in results})

    print(f"\nSaving final results to: {output_file}")
    written, success_count, json_valid_count, json_invalid_ids = \
        save_results_tsv(output_file, all_rows, result_map)

    # ---- Show sample outputs ----
    num_show = min(3, len(results))
    print(f"\nSample outputs (first {num_show}):")
    for uid, raw_text in results[:num_show]:
        clean = extract_and_validate_journey_json(raw_text)
        label = "VALID" if clean else "INVALID"
        display = (clean or raw_text or "(empty)")[:300]
        print(f"  [{label}] {uid}: {display}")
        print()

    # ---- Combine with existing results file (if specified) ----
    if args.combine_file and os.path.exists(args.combine_file):
        combined_name = out_name.replace(".tsv", "_combined.tsv")
        combined_file = os.path.join(args.output_dir, combined_name)
        print(f"\nCombining with existing results file ...")
        combine_with_existing(output_file, args.combine_file, combined_file)
    elif args.combine_file:
        print(f"\n  [WARNING] Combine file not found, skipping: "
              f"{args.combine_file}")

    # ---- Cleanup checkpoint ----
    if args.cleanup_checkpoint:
        cleanup_checkpoint(checkpoint_dir)
        print(f"  Checkpoint cleaned up: {checkpoint_dir}")
    else:
        print(f"\n  Checkpoint preserved: {checkpoint_dir}")

    print(f"\nStep 3 Done!")
    print(f"  Output file: {output_file}")
    print(f"  Valid journeys: {json_valid_count}/{len(all_rows)}")


if __name__ == "__main__":
    main()
