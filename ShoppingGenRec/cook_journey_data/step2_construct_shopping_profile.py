"""Step 2: Generate Shopping Profiles from User Events

Reads a TSV file containing user shopping events (output of step2_0),
calls Copilot API or Papyrus API to generate a structured shopping profile
for each user, and outputs a TSV file that can be fed into step3.

Input:
    A TSV file (from step2_0) with columns:
      UserId | ReadableUserEvents | RequestTime | UserHistory | HisCount

Output:
    A TSV file with columns:
      UserId | ReadableUserEvents | ShoppingProfile | RequestTime | HisCount

    This matches the format expected by step3_generate_journey_query.py.

Supports two inference backends:
  1. GitHub Copilot API (default)
  2. Papyrus API (--inference_backend papyrus)

Both backends support checkpoint/resume: intermediate results are saved to
a checkpoint directory. If interrupted, re-running the script will resume
from where it left off. After all users are done, checkpoints are cleaned up.

Usage (Copilot):
    python cook_PG_journey_data/step2_construct_shopping_profile.py \\
        --input_file /path/to/UserEvents.tsv \\
        --output_dir /path/to/output/ \\
        --token_file ./resources/tokens_full.txt \\
        --copilot_model gpt-5.4 \\
        --num_workers 20

Usage (Papyrus):
    python cook_PG_journey_data/step2_construct_shopping_profile.py \\
        --input_file /path/to/UserEvents.tsv \\
        --output_dir /path/to/output/ \\
        --inference_backend papyrus \\
        --papyrus_model gpt-5-chat-shortco-2025-08-07-Bing \\
        --papyrus_workers 40
"""

import argparse
import csv
import json
import os
import re
import sys
import time

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)

# Add resources directory to path for llm_utils import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESOURCES_DIR = os.path.join(PROJECT_DIR, "resources")
sys.path.insert(0, RESOURCES_DIR)
from llm_utils import (run_llm_parallel_with_checkpoint,
                      cleanup_checkpoint)

# Column names
COL_USER_ID = "UserId"
COL_READABLE_EVENTS = "ReadableUserEvents"
COL_SHOPPING_PROFILE = "ShoppingProfile"
COL_REQUEST_TIME = "RequestTime"
COL_HIS_COUNT = "HisCount"

# Output columns (matching step3 input format)
OUTPUT_HEADER = [COL_USER_ID, COL_READABLE_EVENTS, COL_SHOPPING_PROFILE,
                 COL_REQUEST_TIME, COL_HIS_COUNT]


# =============================================================================
# Data Loading
# =============================================================================

def read_user_events_tsv(filepath):
    """Read TSV file and extract user data for profile generation.

    Reads all columns by name from the header. Preserves raw event text
    and metadata (RequestTime, HisCount) for the output TSV.

    Args:
        filepath: Path to the input TSV file.

    Returns:
        List of dicts with keys:
          - user_id: str
          - events_text: str (events with #N# replaced by newlines)
          - raw_events: str (original events string for output)
          - request_time: str
          - his_count: str
    """
    users = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        col_map = {name.strip(): idx for idx, name in enumerate(header)}
        if COL_USER_ID not in col_map:
            raise ValueError(f"Column '{COL_USER_ID}' not found in header: {header}")
        if COL_READABLE_EVENTS not in col_map:
            raise ValueError(
                f"Column '{COL_READABLE_EVENTS}' not found in header: {header}"
            )

        uid_idx = col_map[COL_USER_ID]
        events_idx = col_map[COL_READABLE_EVENTS]
        time_idx = col_map.get(COL_REQUEST_TIME)
        hiscount_idx = col_map.get(COL_HIS_COUNT)

        for row in reader:
            if len(row) <= max(uid_idx, events_idx):
                continue
            user_id = row[uid_idx].strip()
            events_raw = row[events_idx].strip()
            if not user_id:
                continue

            events_text = events_raw.replace("#N#", "\n") if events_raw else ""

            request_time = ""
            if time_idx is not None and len(row) > time_idx:
                request_time = row[time_idx].strip()

            his_count = ""
            if hiscount_idx is not None and len(row) > hiscount_idx:
                his_count = row[hiscount_idx].strip()

            users.append({
                "user_id": user_id,
                "events_text": events_text,
                "raw_events": events_raw,
                "request_time": request_time,
                "his_count": his_count,
            })

    return users


# =============================================================================
# Prompt Construction
# =============================================================================

def build_prompt(events_text, prompt_template):
    """Build LLM prompt for a single user's events.

    Args:
        events_text: Cleaned event history string (newline-separated).
        prompt_template: Prompt template string with {user_events} placeholder.

    Returns:
        Formatted prompt string.
    """
    return prompt_template.replace("{user_events}", events_text)


# =============================================================================
# JSON Validation
# =============================================================================

def extract_and_validate_json(raw_text):
    """Extract and validate JSON from LLM output.

    Attempts to parse the raw LLM output as JSON. If it contains markdown
    code fences (```json ... ```), extracts the JSON block first.

    Args:
        raw_text: Raw LLM output string.

    Returns:
        str: Cleaned, compact JSON string if valid, or empty string if
             the output is empty or not valid JSON.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to parse as JSON
    try:
        parsed = json.loads(text)
        # Re-serialize to compact, single-line JSON
        return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    # Look for the outermost { ... }
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            parsed = json.loads(candidate)
            return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
        except json.JSONDecodeError:
            pass

    # Could not extract valid JSON
    return ""


# =============================================================================
# Main Processing
# =============================================================================

def _build_inputs(users, prompt_template):
    """Build (user_id, prompt) inputs from users list."""
    print("  Building prompts ...")
    inputs = []
    for user in users:
        prompt = build_prompt(user["events_text"], prompt_template)
        inputs.append((user["user_id"], prompt))
    return inputs


def run_profile_generation_copilot(
    users,
    prompt_template,
    token_file,
    copilot_model,
    num_workers,
    max_tokens,
    checkpoint_dir,
    chunk_size=500,
):
    """Generate shopping profiles via Copilot API with checkpoint/resume.

    Args:
        users: List of user dicts (from read_user_events_tsv).
        prompt_template: Prompt template string with {user_events} placeholder.
        token_file: Path to tokens.txt file.
        copilot_model: Copilot model name.
        num_workers: Number of parallel worker threads.
        max_tokens: Maximum output tokens per API call.
        checkpoint_dir: Directory for checkpoint files.
        chunk_size: Number of users per processing chunk.

    Returns:
        List of (user_id, profile_json_string) tuples in original order.
    """
    inputs = _build_inputs(users, prompt_template)
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


def run_profile_generation_papyrus(
    users,
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
    """Generate shopping profiles via Papyrus API with checkpoint/resume.

    Args:
        users: List of user dicts (from read_user_events_tsv).
        prompt_template: Prompt template string with {user_events} placeholder.
        papyrus_endpoint: Papyrus API endpoint URL.
        papyrus_model: Papyrus model name.
        papyrus_quota_id: Papyrus quota ID.
        papyrus_timeout_ms: Request timeout in ms.
        papyrus_workers: Number of parallel async workers.
        max_tokens: Maximum output tokens per API call.
        checkpoint_dir: Directory for checkpoint files.
        chunk_size: Number of users per processing chunk.
        debug: If True, skip checkpoint and run directly.

    Returns:
        List of (user_id, profile_json_string) tuples in original order.
    """
    from Infer_by_papyrus import (run_papyrus_parallel,
                                   run_papyrus_parallel_with_checkpoint)

    inputs = _build_inputs(users, prompt_template)

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
        description="Generate shopping profiles from user event histories "
        "using Copilot or Papyrus API"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
                "/Data/LLMTrainingData/20260513/raw_data/UserEvents_clean.tsv",
        help="Path to input TSV file (output of step2_0) with UserId, "
             "ReadableUserEvents, RequestTime, HisCount columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
                "/Data/LLMTrainingData/20260513/raw_data",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=os.path.join(
            PROJECT_DIR, "prompts",
            "ShoppingProfile_Step1_GenUserProfile.md"),
        help="Path to prompt template file (.md) with {user_events} placeholder",
    )
    parser.add_argument(
        "--inference_backend",
        type=str,
        default="copilot",
        choices=["copilot", "papyrus"],
        help="Inference backend: 'copilot' or 'papyrus'",
    )
    # --- Copilot API-specific args ---
    parser.add_argument(
        "--token_file",
        type=str,
        default=os.path.join(PROJECT_DIR, "resources", "tokens_full.txt"),
        help="Path to tokens.txt file for Copilot API authentication",
    )
    parser.add_argument(
        "--copilot_model",
        type=str,
        default="gpt-5.4",
        help="Copilot model name",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=60,
        help="Number of parallel workers for Copilot API calls",
    )
    # --- Papyrus API-specific args ---
    parser.add_argument(
        "--papyrus_endpoint",
        type=str,
        default="https://westus2batch.papyrus.binginternal.com",
        help="Papyrus API base endpoint (without /chat/completions)",
    )
    parser.add_argument(
        "--papyrus_model",
        type=str,
        default="gpt-5-chat-shortco-2025-08-07-Bing",
        help="Papyrus model name",
    )
    parser.add_argument(
        "--papyrus_quota_id",
        type=str,
        default="",
        help="Papyrus quota ID (default: empty)",
    )
    parser.add_argument(
        "--papyrus_timeout_ms",
        type=int,
        default=120000,
        help="Papyrus request timeout in ms (default: 120000)",
    )
    parser.add_argument(
        "--papyrus_workers",
        type=int,
        default=3,
        help="Number of parallel async workers for Papyrus API",
    )
    parser.add_argument(
        "--papyrus_chunk_size",
        type=int,
        default=20000,
        help="Number of users per Papyrus processing chunk for checkpoint "
             "saving",
    )
    # --- Common args ---
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10000,
        help="Maximum output tokens per API call",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=20000,
        help="Number of users per processing chunk for checkpoint saving",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode: only process first 50 users",
    )
    parser.add_argument(
        "--prompt_results_dir",
        type=str,
        default=None,
        help="Directory containing *_profiles_results.tsv files from previous "
             "runs. If provided, merges all results, validates JSON, filters "
             "by users in --input_file, and writes merged output to "
             "--output_dir. No LLM inference is run.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output file name (default: derived from input file name)",
    )
    return parser.parse_args()


# =============================================================================
# Merge Results Mode
# =============================================================================

def _merge_results(args):
    """Merge profiles from multiple result files, validate JSON, combine
    with original input data, and write a single output TSV matching the
    step3-compatible format:
      UserId | ReadableUserEvents | ShoppingProfile | RequestTime | HisCount
    """
    import glob

    results_dir = args.prompt_results_dir
    print("=" * 60)
    print("Merge Results Mode (--prompt_results_dir)")
    print(f"  Results dir:  {results_dir}")
    print(f"  Input file:   {args.input_file}")
    print(f"  Output dir:   {args.output_dir}")
    print("=" * 60)

    if not os.path.isdir(results_dir):
        print(f"Error: results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- Load input data (preserving all columns) ----
    print(f"\nLoading input data from: {args.input_file}")
    all_users = read_user_events_tsv(args.input_file)
    user_data_map = {u["user_id"]: u for u in all_users}
    print(f"  Loaded {len(all_users):,} users")

    # ---- Find and load all results files ----
    pattern = os.path.join(results_dir, "*_profiles_results*.tsv")
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        pattern = os.path.join(results_dir, "*_results*.tsv")
        result_files = sorted(glob.glob(pattern))
    print(f"\nFound {len(result_files)} result files:")
    for rf in result_files:
        print(f"  {os.path.basename(rf)}")

    # ---- Read and merge (later files overwrite earlier for duplicate UIDs) ----
    merged = {}  # uid -> raw_profile_text
    total_rows = 0
    for rf in result_files:
        file_count = 0
        with open(rf, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if not header:
                continue
            col_map = {name.strip(): idx for idx, name in enumerate(header)}

            if COL_SHOPPING_PROFILE in col_map:
                uid_col = col_map[COL_USER_ID]
                profile_col = col_map[COL_SHOPPING_PROFILE]
            elif "OUTPUT" in col_map:
                uid_col = col_map[COL_USER_ID]
                profile_col = col_map["OUTPUT"]
            else:
                print(f"  WARNING: Skipping {os.path.basename(rf)} — "
                      f"no ShoppingProfile or OUTPUT column found. "
                      f"Header: {header}")
                continue

            for row in reader:
                if len(row) <= max(uid_col, profile_col):
                    continue
                uid = row[uid_col].strip()
                profile = row[profile_col].strip()
                if uid and profile:
                    merged[uid] = profile
                    file_count += 1

        total_rows += file_count
        print(f"  {os.path.basename(rf)}: {file_count:,} rows with profiles")

    print(f"\nTotal merged: {len(merged):,} unique users "
          f"(from {total_rows:,} total rows)")

    # ---- Filter to valid users and validate JSON ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "shopping_profiles_merged.tsv")

    json_valid = 0
    json_invalid = 0
    json_invalid_ids = []
    not_in_input = 0
    written = 0

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(OUTPUT_HEADER)

        for uid, raw_profile in merged.items():
            if uid not in user_data_map:
                not_in_input += 1
                continue

            clean_json = extract_and_validate_json(raw_profile)
            if clean_json:
                json_valid += 1
                user = user_data_map[uid]
                writer.writerow([
                    uid,
                    user["raw_events"],
                    clean_json,
                    user["request_time"],
                    user["his_count"],
                ])
                written += 1
            else:
                json_invalid += 1
                json_invalid_ids.append(uid)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    missing = len(user_data_map) - written
    print(f"\n{'='*60}")
    print(f"Merge Results Summary")
    print(f"{'='*60}")
    print(f"  Input users:          {len(user_data_map):,}")
    print(f"  Users with profiles:  {len(merged):,}")
    print(f"  Not in input (skip):  {not_in_input:,}")
    print(f"  Valid JSON written:   {json_valid:,}")
    print(f"  Invalid JSON (drop):  {json_invalid:,}")
    print(f"  Missing (no profile): {missing:,}")
    print(f"  Output: {output_file} ({file_size_mb:.1f} MB)")
    print(f"  Output columns: {OUTPUT_HEADER}")
    if json_invalid_ids:
        print(f"  Invalid JSON UIDs:    {json_invalid_ids[:10]}"
              f"{'...' if len(json_invalid_ids) > 10 else ''}")
    print(f"{'='*60}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    backend = args.inference_backend

    print("=" * 60)
    print("Step 2: Generate Shopping Profiles from User Events")
    print(f"  Input file:    {args.input_file}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Prompt file:   {args.prompt_file}")
    print(f"  Backend:       {backend}")
    print(f"  Max tokens:    {args.max_tokens}")
    if backend == "copilot":
        print(f"  Model:         {args.copilot_model}")
        print(f"  Workers:       {args.num_workers}")
        print(f"  Chunk size:    {args.chunk_size}")
    else:  # papyrus
        print(f"  Endpoint:      {args.papyrus_endpoint}")
        print(f"  Model:         {args.papyrus_model}")
        print(f"  Workers:       {args.papyrus_workers}")
        print(f"  Quota ID:      {args.papyrus_quota_id or '(default)'}")
        print(f"  Chunk size:    {args.papyrus_chunk_size}")
    print("=" * 60)

    # ---- Merge results mode (--prompt_results_dir) ----
    if args.prompt_results_dir:
        _merge_results(args)
        return

    if args.debug:
        print("\n*** DEBUG MODE: processing only 50 users ***\n")

    # ---- Load prompt template from .md file ----
    print(f"\nLoading prompt template from: {args.prompt_file}")
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    print("  Prompt template loaded successfully")

    # ---- Load data ----
    print(f"\nLoading user events from: {args.input_file}")
    all_users = read_user_events_tsv(args.input_file)
    total_loaded = len(all_users)
    print(f"  Loaded {total_loaded:,} users")

    # Split into users with/without events — skip empty entirely
    all_users_raw = all_users
    all_users = [u for u in all_users_raw if u["events_text"].strip()]
    num_empty = len(all_users_raw) - len(all_users)
    if num_empty > 0:
        print(f"  Skipped {num_empty:,} users with empty events")
        print(f"  Remaining users to process: {len(all_users):,}")

    if args.debug:
        all_users = all_users[:50]
        print(f"  DEBUG: trimmed to {len(all_users)} users")

    # Show sample
    if all_users:
        sample = all_users[0]
        event_lines = sample["events_text"].strip().split("\n")
        print(f"\n  Sample user: {sample['user_id']}")
        print(f"  Request time: {sample['request_time'] or '(not available)'}")
        print(f"  HisCount: {sample['his_count'] or '(not available)'}")
        print(f"  Total events: {len(event_lines)}")
        for line in event_lines[:3]:
            print(f"    {line[:120]}")
        if len(event_lines) > 3:
            print(f"    ... ({len(event_lines) - 3} more)")

    # ---- Checkpoint setup ----
    os.makedirs(args.output_dir, exist_ok=True)
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    checkpoint_dir = os.path.join(
        args.output_dir, f"_profile_checkpoint_{input_basename}")

    # ---- Run profile generation ----
    start_time = time.time()

    if backend == "papyrus":
        results = run_profile_generation_papyrus(
            users=all_users,
            prompt_template=prompt_template,
            papyrus_endpoint=args.papyrus_endpoint,
            papyrus_model=args.papyrus_model,
            papyrus_quota_id=args.papyrus_quota_id,
            papyrus_timeout_ms=args.papyrus_timeout_ms,
            papyrus_workers=args.papyrus_workers,
            max_tokens=args.max_tokens,
            checkpoint_dir=checkpoint_dir,
            chunk_size=args.papyrus_chunk_size,
            debug=args.debug,
        )
    else:  # copilot
        results = run_profile_generation_copilot(
            users=all_users,
            prompt_template=prompt_template,
            token_file=args.token_file,
            copilot_model=args.copilot_model,
            num_workers=args.num_workers,
            max_tokens=args.max_tokens,
            checkpoint_dir=checkpoint_dir,
            chunk_size=args.chunk_size,
        )

    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed:.1f}s "
          f"({len(all_users) / max(elapsed, 0.1):.1f} users/s)")

    # ---- Build uid -> profile mapping ----
    result_map = {uid: text for uid, text in results}

    # ---- Save output TSV ----
    # Output format: UserId | ReadableUserEvents | ShoppingProfile |
    #                RequestTime | HisCount
    if args.output_name:
        out_name = args.output_name
    else:
        suffix = "_debug" if args.debug else ""
        out_name = f"{input_basename}_profiles_results{suffix}.tsv"

    output_file = os.path.join(args.output_dir, out_name)
    print(f"\nSaving results to: {output_file}")

    success_count = 0
    json_valid_count = 0
    json_invalid_ids = []
    written = 0

    # Only write users that were processed (skip empty-event users entirely)
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(OUTPUT_HEADER)

        for user in all_users:
            uid = user["user_id"]
            raw_profile = result_map.get(uid, "")

            if raw_profile:
                success_count += 1

            # Validate and extract JSON
            clean_json = extract_and_validate_json(raw_profile)
            if clean_json:
                json_valid_count += 1
                profile_val = clean_json
            elif raw_profile:
                json_invalid_ids.append(uid)
                profile_val = raw_profile.replace("\n", " ").replace("\t", " ")
            else:
                profile_val = ""

            writer.writerow([
                uid,
                user["raw_events"],
                profile_val,
                user["request_time"],
                user["his_count"],
            ])
            written += 1

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Written {written:,} rows ({file_size_mb:.1f} MB)")
    print(f"  Empty events (no LLM call): {num_empty:,}")
    print(f"  API success:     {success_count}/{len(all_users)}")
    print(f"  Valid JSON:      {json_valid_count}/{success_count}")
    if json_invalid_ids:
        print(f"  Invalid JSON ({len(json_invalid_ids)} users): "
              f"{json_invalid_ids[:10]}{'...' if len(json_invalid_ids) > 10 else ''}")

    # ---- Show sample outputs ----
    num_show = min(3, len(results))
    print(f"\nSample outputs (first {num_show}):")
    for uid, profile in results[:num_show]:
        clean = extract_and_validate_json(profile)
        label = "VALID" if clean else "INVALID"
        display = (clean or profile or "(empty)")[:300]
        print(f"  [{label}] {uid}: {display}")
        print()

    # ---- Clean up checkpoint ----
    cleanup_checkpoint(checkpoint_dir)

    print(f"\nStep 2 Done!")
    print(f"  Output file: {output_file}")
    print(f"  Valid profiles: {json_valid_count}/{written}")
    print(f"  Output columns: {OUTPUT_HEADER}")


if __name__ == "__main__":
    main()
