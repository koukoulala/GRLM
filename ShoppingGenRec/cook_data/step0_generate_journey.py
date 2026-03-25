"""
Step 0: Generate Shopping Journeys from Raw User Event Data.

Reads the raw ShoppingJourney_Input TSV, filters out users that already
exist in previously processed files, normalizes user events, calls an LLM
(Copilot or Papyrus) with the JourneyGeneration prompt to predict continued
shopping journeys, validates the JSON output, and writes the result as a TSV
file with all original columns plus an OUTPUT column.

The OUTPUT column is consumed by step1_extract_query_and_infer.py.

Usage (Copilot):
    python step0_generate_journey.py \\
        --input_file /path/to/ShoppingJourney_Input.tsv \\
        --output_dir /path/to/output/ \\
        --token_file ../resources/tokens.txt \\
        --copilot_model gpt-5.4 \\
        --num_workers 20

Usage (Papyrus):
    python step0_generate_journey.py \\
        --input_file /path/to/ShoppingJourney_Input.tsv \\
        --output_dir /path/to/output/ \\
        --inference_backend papyrus \\
        --papyrus_model gpt-5-chat-shortco-2025-08-07-Bing \\
        --papyrus_workers 40

Filtering existing users:
    python step0_generate_journey.py \\
        --exclude_files /path/to/existing_results1.tsv /path/to/existing_results2.tsv \\
        ...
"""

import argparse
import csv
import json
import os
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
PREPROCESS_DIR = os.path.join(PROJECT_DIR, "preprocess_raw_data")
sys.path.insert(0, RESOURCES_DIR)
sys.path.insert(0, PREPROCESS_DIR)

from llm_utils import (load_prompts, run_llm_parallel_with_checkpoint,
                        cleanup_checkpoint)
from Infer_by_papyrus import (run_papyrus_parallel,
                               run_papyrus_parallel_with_checkpoint)

# Reuse event processing utilities from pre_s1
from pre_s1_construct_shopping_profile import (
    normalize_event_times,
    truncate_events,
)


# =============================================================================
# Constants
# =============================================================================

# Default columns to look for in the input TSV (by name from header)
COL_USER_ID = "UserId"
COL_READABLE_EVENTS = "ReadableUserEvents"
COL_REQUEST_TIME = "RequestTime"
OUTPUT_COL_NAME = "OUTPUT"


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

def read_raw_input(filepath, exclude_ids=None, max_events=100, max_rows=0):
    """Read raw input TSV and return rows with processed events.

    Reads the header row to find column indices by name.  Normalizes event
    time expressions and truncates events to max_events.

    Args:
        filepath: Path to the raw input TSV file.
        exclude_ids: Optional set of user IDs to skip.
        max_events: Maximum number of events to keep per user.
        max_rows: If >0, return at most this many rows (for debug).

    Returns:
        Tuple of (rows, header):
          - rows: list of dicts, each with:
              'raw_parts': list of original tab-separated field values,
              'user_id': str,
              'events_text': str (processed),
              'request_time': str (if available, else ""),
          - header: list of original column names from the TSV header.
    """
    if exclude_ids is None:
        exclude_ids = set()

    rows = []
    total_read = 0
    excluded_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        col_map = {name.strip(): idx for idx, name in enumerate(header)}

        if COL_USER_ID not in col_map:
            raise ValueError(
                f"Column '{COL_USER_ID}' not found in header: {header}")
        if COL_READABLE_EVENTS not in col_map:
            raise ValueError(
                f"Column '{COL_READABLE_EVENTS}' not found in header: {header}")

        uid_idx = col_map[COL_USER_ID]
        events_idx = col_map[COL_READABLE_EVENTS]
        time_idx = col_map.get(COL_REQUEST_TIME)  # may be None

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

            # Process events: replace #N# separators, normalize times, truncate
            events_text = events_raw.replace("#N#", "\n")
            events_text = normalize_event_times(events_text)
            events_text = truncate_events(events_text, max_events)

            request_time = ""
            if time_idx is not None and len(row) > time_idx:
                request_time = row[time_idx].strip()

            rows.append({
                "raw_parts": list(row),
                "user_id": user_id,
                "events_text": events_text,
                "request_time": request_time,
            })

            if max_rows > 0 and len(rows) >= max_rows:
                break

    print(f"  Read {total_read:,} rows total")
    if exclude_ids:
        print(f"  Excluded {excluded_count:,} users")
    print(f"  Kept {len(rows):,} users after filtering")
    return rows, [h.strip() for h in header]


# =============================================================================
# Prompt Construction
# =============================================================================

def build_journey_prompt(events_text, request_time, prompt_template):
    """Build LLM prompt for journey generation.

    Args:
        events_text: Cleaned event history string (newline-separated).
        request_time: Request time string (e.g. system time).
        prompt_template: Prompt template string with placeholders.

    Returns:
        Formatted prompt string.
    """
    return prompt_template.format(
        ReadableUserEvents=events_text,
        RequestTime=request_time,
    )


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
            row["events_text"], row["request_time"], prompt_template)
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
        description="Step 0: Generate shopping journeys from raw user event "
        "data using Copilot or Papyrus API"
    )

    # --- I/O ---
    parser.add_argument(
        "--input_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/ShoppingJourney_Input.tsv",
        help="Path to raw input TSV (must have UserId, ReadableUserEvents)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Output file name (default: derived from input file name)",
    )
    parser.add_argument(
        "--exclude_files", type=str, nargs="*", 
        default=None,
        #default=["/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/ShoppingJourney_Input_500K.tsv",
        #        "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/TestData/ShoppingJourney_Input_TestData_50K.tsv"],
        help="List of TSV file paths whose UserId values will be excluded",
    )

    # --- Prompt ---
    parser.add_argument(
        "--prompts_file", type=str,
        default=os.path.join(PROJECT_DIR, "resources", "prompts.yaml"),
        help="Path to prompts.yaml file",
    )
    parser.add_argument(
        "--prompt_key", type=str, default="JourneyGeneration",
        help="Key in prompts.yaml for the journey generation prompt",
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
        default=os.path.join(PROJECT_DIR, "resources", "tokens.txt"),
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
        default="gpt-5-chat-shortco-2025-08-07-Bing",
        help="Papyrus model name",
    )
    parser.add_argument(
        "--papyrus_quota_id", type=str, default="",
        help="Papyrus quota ID",
    )
    parser.add_argument(
        "--papyrus_timeout_ms", type=int, default=120000,
        help="Papyrus request timeout in ms (default: 120000)",
    )
    parser.add_argument(
        "--papyrus_workers", type=int, default=40,
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
        "--max_events", type=int, default=50,
        help="Maximum number of events to keep per user",
    )

    parser.add_argument(
        "--max_users", type=int, default=0,
        help="If >0, split the filtered input into chunks of this size, "
             "save each chunk as a separate TSV file, and stop (no LLM "
             "inference). If <=0, read all data and run normally.",
    )

    # --- Debug ---
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: process only first --debug_rows users",
    )
    parser.add_argument(
        "--debug_rows", type=int, default=1000,
        help="Number of users in debug mode",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("Step 0: Generate Shopping Journeys")
    print("=" * 70)
    print(f"  Input file:    {args.input_file}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Backend:       {args.inference_backend}")
    print(f"  Max events:    {args.max_events}")
    if args.max_users > 0:
        print(f"  Max users:     {args.max_users} (split mode, no inference)")
    else:
        print(f"  Max users:     unlimited")
    if args.exclude_files:
        print(f"  Exclude files: {len(args.exclude_files)} file(s)")
    print()

    if args.debug:
        print(f"*** DEBUG MODE: processing only {args.debug_rows} users ***\n")

    # ---- Step 0a: Load exclude user IDs ----
    exclude_ids = set()
    if args.exclude_files:
        print("Loading user IDs to exclude ...")
        exclude_ids = load_exclude_user_ids(args.exclude_files)
        print()

    # ---- Step 0b: Read raw input ----
    print("Reading raw input ...")
    max_rows = args.debug_rows if args.debug else 0  # 0 = read all
    rows, header = read_raw_input(
        args.input_file,
        exclude_ids=exclude_ids,
        max_events=args.max_events,
        max_rows=max_rows,
    )

    if not rows:
        print("No users to process after filtering!")
        return

    # ---- If max_users > 0, split into chunks and save, then stop ----
    if args.max_users > 0 and not args.debug:
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        chunk_size = args.max_users
        num_chunks = (len(rows) + chunk_size - 1) // chunk_size
        print(f"\nSplitting {len(rows):,} rows into {num_chunks} chunk(s) "
              f"of up to {chunk_size:,} each ...")

        for ci in range(num_chunks):
            chunk = rows[ci * chunk_size : (ci + 1) * chunk_size]
            chunk_k = len(chunk) // 1000
            if num_chunks > 1:
                suffix = f"_{chunk_k}K_{ci + 1}"
            else:
                suffix = f"_{chunk_k}K"
            chunk_file = os.path.join(
                args.output_dir, f"{base_name}{suffix}.tsv")
            with open(chunk_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t", lineterminator="\n")
                writer.writerow(header)
                for row_data in chunk:
                    writer.writerow(row_data["raw_parts"])
            chunk_mb = os.path.getsize(chunk_file) / (1024 * 1024)
            print(f"  Chunk {ci + 1}/{num_chunks}: {len(chunk):,} rows "
                  f"({chunk_mb:.1f} MB) -> {chunk_file}")

        print(f"\nDone! Split into {num_chunks} file(s).")
        print(f"Run step0 with --max_users 0 --input_file <chunk_file> "
              f"to generate journeys for each chunk.")
        return

    # Show sample
    sample = rows[0]
    event_lines = sample["events_text"].strip().split("\n")
    print(f"\n  Sample user: {sample['user_id']}")
    print(f"  Request time: {sample['request_time'] or '(not available)'}")
    print(f"  Events ({len(event_lines)}):")
    for line in event_lines[:3]:
        print(f"    {line[:120]}")
    if len(event_lines) > 3:
        print(f"    ... ({len(event_lines) - 3} more)")

    # ---- Step 0c: Load prompt template ----
    print(f"\nLoading prompt template from: {args.prompts_file}")
    prompts_config = load_prompts(args.prompts_file)
    prompt_template = prompts_config[args.prompt_key]["user"]
    print("  Prompt template loaded successfully")

    # ---- Step 0d: Setup output ----
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "_journey_checkpoint")

    # ---- Step 0e: Run LLM inference ----
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

    # ---- Step 0f: Validate and save output ----
    # Build user_id -> result mapping
    result_map = {uid: text for uid, text in results}

    # Determine output filename
    if args.output_name:
        out_name = args.output_name
    else:
        base = os.path.splitext(os.path.basename(args.input_file))[0]
        out_name = f"{base}_Journey_Results.tsv"

    output_file = os.path.join(args.output_dir, out_name)

    print(f"\nSaving results to: {output_file}")

    # Output: original columns + OUTPUT column
    output_header = header + [OUTPUT_COL_NAME]

    success_count = 0
    json_valid_count = 0
    json_invalid_ids = []
    written = 0

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(output_header)

        for row_data in rows:
            uid = row_data["user_id"]
            raw_output = result_map.get(uid, "")

            if raw_output:
                success_count += 1

            # Validate and clean JSON
            clean_json = extract_and_validate_journey_json(raw_output)
            if clean_json:
                json_valid_count += 1
                output_val = clean_json
            elif raw_output:
                json_invalid_ids.append(uid)
                # Save raw text with newlines/tabs escaped as fallback
                output_val = raw_output.replace("\n", " ").replace("\t", " ")
            else:
                output_val = ""

            # Write original columns + OUTPUT
            writer.writerow(row_data["raw_parts"] + [output_val])
            written += 1

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Written {written:,} rows ({file_size_mb:.1f} MB)")
    print(f"  API success:     {success_count}/{len(rows)}")
    print(f"  Valid JSON:      {json_valid_count}/{success_count}")
    if json_invalid_ids:
        preview = json_invalid_ids[:10]
        print(f"  Invalid JSON ({len(json_invalid_ids)} users): "
              f"{preview}{'...' if len(json_invalid_ids) > 10 else ''}")

    # ---- Show sample outputs ----
    num_show = min(3, len(results))
    print(f"\nSample outputs (first {num_show}):")
    for uid, raw_text in results[:num_show]:
        clean = extract_and_validate_journey_json(raw_text)
        label = "VALID" if clean else "INVALID"
        display = (clean or raw_text or "(empty)")[:300]
        print(f"  [{label}] {uid}: {display}")
        print()

    # ---- Cleanup checkpoint ----
    cleanup_checkpoint(checkpoint_dir)

    print(f"\nStep 0 Done!")
    print(f"  Output file: {output_file}")
    print(f"  Valid journeys: {json_valid_count}/{len(rows)}")


if __name__ == "__main__":
    main()
