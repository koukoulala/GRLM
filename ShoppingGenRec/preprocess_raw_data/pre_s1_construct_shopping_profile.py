"""Step 3: Generate Shopping Profiles from User Events

Reads a TSV file containing user shopping events (Browsed, Searched, Clicked),
calls Copilot API or Papyrus API to generate a structured shopping profile for
each user, and outputs results as a TSV file.

Input:
    A TSV file with a header row. The script reads columns by name from
    the header. Required columns:
      - UserId      : unique user identifier string
      - ReadableUserEvents : pipe-delimited event history string, with
                             "#N#" used as line separators between events

    Event format (after "#N#" → newline conversion):
      "1 | 1 days ago | Browsed | Blue Mountain Unisex PVC Rainsuit , Green..."
      "2 | 19 days ago | Clicked | Kinder's Cowboy Butter Seasoning, 6.4oz..."
      "3 | 28 days ago | Searched | viking backpack..."

    Time expressions are normalized: weeks/months are converted to days,
    and "0 days ago" is displayed as "X hours ago".

Output:
    shopping_profiles.tsv - two columns:
      - UserId          : user identifier
      - ShoppingProfile : JSON string of the generated shopping profile

Supports two inference backends:
  1. GitHub Copilot API (default)
  2. Papyrus API (--inference_backend papyrus)

Both backends support checkpoint/resume: intermediate results are saved to
a checkpoint directory. If interrupted, re-running the script will resume
from where it left off. After all users are done, checkpoints are cleaned up.

Usage (Copilot):
    python pre_s1_construct_shopping_profile.py \\
        --input_file ./raw_data/ShoppingJourney_Input.tsv \\
        --output_dir ./raw_data/ \\
        --token_file ./resources/tokens.txt \\
        --copilot_model gpt-5.4 \\
        --num_workers 20

Usage (Papyrus):
    python pre_s1_construct_shopping_profile.py \\
        --input_file ./raw_data/ShoppingJourney_Input.tsv \\
        --output_dir ./raw_data/ \\
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

from tqdm import tqdm

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)

# Add resources directory to path for llm_utils import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESOURCES_DIR = os.path.join(PROJECT_DIR, "resources")
sys.path.insert(0, RESOURCES_DIR)
from llm_utils import (load_prompts, run_llm_parallel_with_checkpoint,
                      cleanup_checkpoint)
from Infer_by_papyrus import (run_papyrus_parallel,
                              run_papyrus_parallel_with_checkpoint)


# =============================================================================
# Time Normalization
# =============================================================================

def _normalize_time_expr(match):
    """Normalize a single time expression to days or hours.

    Converts weeks/months to days. If result is 0 days and hours are
    available, uses "X hours ago" instead.
    """
    text = match.group(0)

    # Parse all numeric + unit pairs in the expression
    parts = re.findall(r'(\d+)\s*(month|week|day|hour|minute|second)s?', text,
                       re.IGNORECASE)
    if not parts:
        return text

    total_hours = 0
    total_minutes = 0
    for num_str, unit in parts:
        num = int(num_str)
        unit_lower = unit.lower()
        if unit_lower == 'month':
            total_hours += num * 30 * 24
        elif unit_lower == 'week':
            total_hours += num * 7 * 24
        elif unit_lower == 'day':
            total_hours += num * 24
        elif unit_lower == 'hour':
            total_hours += num
        elif unit_lower == 'minute':
            total_minutes += num
        elif unit_lower == 'second':
            total_minutes += 0

    total_days = total_hours // 24

    if total_days > 0:
        return f"{total_days} days ago"
    elif total_hours > 0:
        return f"{total_hours} hours ago"
    elif total_minutes > 0:
        return f"{total_minutes} minutes ago"
    else:
        return "0 minutes ago"


def normalize_event_times(events_text):
    """Normalize all time expressions in event text to days/hours.

    Converts patterns like "2 weeks 3 days ago", "1 month ago" to
    "17 days ago", "30 days ago". If result is 0 days, shows hours.
    """
    # Match time expressions like "X weeks Y days ago", "X months ago", etc.
    pattern = r'(?:\d+\s*(?:month|week|day|hour|minute|second)s?\s*)+ago'
    return re.sub(pattern, _normalize_time_expr, events_text, flags=re.IGNORECASE)


def truncate_events(events_text, max_events):
    """Keep only the most recent max_events events.

    Events are newline-separated lines. Each line starts with an index number.
    Keeps the first max_events lines (most recent events come first).
    """
    lines = [line for line in events_text.strip().split("\n") if line.strip()]
    if len(lines) <= max_events:
        return events_text
    return "\n".join(lines[:max_events])


# =============================================================================
# Data Loading
# =============================================================================

def read_user_events_tsv(filepath, max_events=100):
    """Read TSV file and extract UserId and ReadableUserEvents columns.

    Reads column names from the header row so the script is resilient to
    column ordering and extra columns. Normalizes time expressions and
    truncates events to max_events.

    Args:
        filepath: Path to the input TSV file.
        max_events: Maximum number of events to keep per user.

    Returns:
        List of (user_id, events_text) tuples, where events_text has
        "#N#" replaced with newline characters, time expressions
        normalized, and events truncated.
    """
    users = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        # Find column indices by name
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
            # Normalize time expressions (weeks/months -> days)
            events_text = normalize_event_times(events_text)
            # Truncate to max_events
            events_text = truncate_events(events_text, max_events)
            users.append((user_id, events_text))

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
    return prompt_template.format(user_events=events_text)


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
    import re
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
    for user_id, events_text in users:
        prompt = build_prompt(events_text, prompt_template)
        inputs.append((user_id, prompt))
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
        users: List of (user_id, events_text) tuples.
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
        users: List of (user_id, events_text) tuples.
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
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/ShoppingJourney_Input.tsv",
        help="Path to input TSV file with UserId and ReadableUserEvents columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/raw_data/",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="./resources/prompts.yaml",
        help="Path to prompts.yaml file with prompt templates",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=100,
        help="Maximum number of events to keep per user (default: 100)",
    )
    parser.add_argument(
        "--inference_backend",
        type=str,
        default="papyrus",
        choices=["copilot", "papyrus"],
        help="Inference backend: 'copilot' or 'papyrus'",
    )
    # --- Copilot API-specific args ---
    parser.add_argument(
        "--token_file",
        type=str,
        default="./resources/tokens.txt",
        help="Path to tokens.txt file for Copilot API authentication",
    )
    parser.add_argument(
        "--copilot_model",
        type=str,
        default="gpt-5.2",
        help="Copilot model name (default: gpt-5.2)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=40,
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
        default=20,
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
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    backend = args.inference_backend

    print("=" * 60)
    print("Generate Shopping Profiles from User Events")
    print(f"  Input file:    {args.input_file}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Backend:       {backend}")
    print(f"  Max events:    {args.max_events}")
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

    if args.debug:
        print("\n*** DEBUG MODE: processing only 50 users ***\n")

    # ---- Load prompt template ----
    print(f"\nLoading prompt template from: {args.prompts_file}")
    prompts_config = load_prompts(args.prompts_file)
    prompt_template = prompts_config["generate_shopping_profile_from_events"]["user"]
    print("  Prompt template loaded successfully")

    # ---- Load data ----
    print(f"\nLoading user events from: {args.input_file}")
    all_users = read_user_events_tsv(args.input_file, max_events=args.max_events)
    total_loaded = len(all_users)
    print(f"  Loaded {total_loaded} users (max {args.max_events} events each)")

    # Filter out users with empty events
    empty_event_users = [(uid, ev) for uid, ev in all_users if not ev.strip()]
    all_users = [(uid, ev) for uid, ev in all_users if ev.strip()]
    num_empty = len(empty_event_users)
    if num_empty > 0:
        print(f"  Skipped {num_empty} users with empty events")
        print(f"  Remaining users to process: {len(all_users)}")

    if args.debug:
        all_users = all_users[:50]
        print(f"  DEBUG: trimmed to {len(all_users)} users")

    # Show sample
    if all_users:
        sample_uid, sample_events = all_users[0]
        event_lines = sample_events.strip().split("\n")
        print(f"\n  Sample user: {sample_uid}")
        print(f"  Total events: {len(event_lines)}")
        for line in event_lines[:3]:
            print(f"    {line[:120]}...")
        if len(event_lines) > 3:
            print(f"    ... ({len(event_lines) - 3} more events)")

    # ---- Checkpoint setup ----
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "_profile_checkpoint")

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
          f"({len(all_users) / elapsed:.1f} users/s)")

    # ---- Save output TSV ----
    # Merge empty-event users (with empty profile) into results
    all_results = list(results)
    for uid, _ in empty_event_users:
        all_results.append((uid, ""))

    output_file = os.path.join(args.output_dir, "shopping_profiles.tsv")
    print(f"\nSaving results to: {output_file}")

    success_count = 0
    json_valid_count = 0
    json_invalid_ids = []
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["UserId", "ShoppingProfile"])
        for user_id, profile in all_results:
            if not profile:
                writer.writerow([user_id, ""])
                continue
            success_count += 1

            # Validate and extract JSON
            clean_json = extract_and_validate_json(profile)
            if clean_json:
                json_valid_count += 1
                writer.writerow([user_id, clean_json])
            else:
                json_invalid_ids.append(user_id)
                # Save raw text with newlines/tabs escaped as fallback
                fallback = profile.replace("\n", " ").replace("\t", " ")
                writer.writerow([user_id, fallback])

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved {len(all_results)} users ({file_size_mb:.1f} MB)")
    print(f"  Empty events (skipped): {num_empty}")
    print(f"  API success: {success_count}/{len(results)}")
    print(f"  Valid JSON:  {json_valid_count}/{success_count}")
    if json_invalid_ids:
        print(f"  Invalid JSON ({len(json_invalid_ids)} users): "
              f"{json_invalid_ids[:10]}{'...' if len(json_invalid_ids) > 10 else ''}")

    # ---- Show sample outputs ----
    num_show = min(3, len(results))
    print(f"\nSample outputs (first {num_show}):")
    for uid, profile in results[:num_show]:
        print(f"  UserId: {uid}")
        print(f"  Profile: {profile[:300]}...")
        print()

    # ---- Clean up checkpoint ----
    cleanup_checkpoint(checkpoint_dir)

    print(f"\nCompleted! {json_valid_count}/{len(all_results)} valid profiles "
          f"({num_empty} skipped due to empty events).")


if __name__ == "__main__":
    main()
