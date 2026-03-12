"""Step 3: Generate Shopping Profiles from User Events

Reads a TSV file containing user shopping events (Browsed, Searched, Clicked),
calls GitHub Copilot API to generate a structured shopping profile for each
user, and outputs results as a TSV file.

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

Output:
    shopping_profiles.tsv - two columns:
      - UserId          : user identifier
      - ShoppingProfile : JSON string of the generated shopping profile

Supports checkpoint/resume: intermediate results are saved to a checkpoint
directory. If interrupted, re-running the script will resume from where it
left off. After all users are done, checkpoints are cleaned up.

Usage:
    python s3_generate_shopping_profile.py \\
        --input_file ./raw_data/ShoppingJourney_Input_500K_His50.tsv \\
        --output_dir ./processed/ \\
        --token_file ./resources/tokens.txt \\
        --copilot_model gpt-5.4 \\
        --num_workers 20
"""

import argparse
import csv
import json
import os
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


# =============================================================================
# Data Loading
# =============================================================================

def read_user_events_tsv(filepath):
    """Read TSV file and extract UserId and ReadableUserEvents columns.

    Reads column names from the header row so the script is resilient to
    column ordering and extra columns.

    Args:
        filepath: Path to the input TSV file.

    Returns:
        List of (user_id, events_text) tuples, where events_text has
        "#N#" replaced with newline characters.
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

def run_profile_generation(
    users,
    prompt_template,
    token_file,
    copilot_model,
    num_workers,
    max_tokens,
    checkpoint_dir,
    chunk_size=500,
):
    """Generate shopping profiles for all users with checkpoint/resume.

    Builds (user_id, prompt) inputs and delegates to the shared
    run_llm_parallel_with_checkpoint function.

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
    # Build (user_id, prompt) inputs
    print("  Building prompts ...")
    inputs = []
    for user_id, events_text in users:
        prompt = build_prompt(events_text, prompt_template)
        inputs.append((user_id, prompt))

    # Delegate to shared checkpoint-enabled parallel runner
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


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate shopping profiles from user event histories "
        "using GitHub Copilot API"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/ShoppingJourney_Input_500K_His50.tsv",
        help="Path to input TSV file with UserId and ReadableUserEvents columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw_data/",
        help="Directory to save output files (default: ./raw_data/)",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="./resources/prompts.yaml",
        help="Path to prompts.yaml file with prompt templates",
    )
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
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2000,
        help="Maximum output tokens per API call (default: 2000)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of users per processing chunk for checkpoint saving "
        "(default: 10000)",
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

    print("=" * 60)
    print("Step 3: Generate Shopping Profiles from User Events")
    print(f"  Input file:    {args.input_file}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Model:         {args.copilot_model}")
    print(f"  Workers:       {args.num_workers}")
    print(f"  Max tokens:    {args.max_tokens}")
    print(f"  Chunk size:    {args.chunk_size}")
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
    all_users = read_user_events_tsv(args.input_file)
    print(f"  Loaded {len(all_users)} users")

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

    results = run_profile_generation(
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
    output_file = os.path.join(args.output_dir, "shopping_profiles.tsv")
    print(f"\nSaving results to: {output_file}")

    success_count = 0
    json_valid_count = 0
    json_invalid_ids = []
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["UserId", "ShoppingProfile"])
        for user_id, profile in results:
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
    print(f"  Saved {len(results)} users ({file_size_mb:.1f} MB)")
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

    print(f"\nCompleted! {json_valid_count}/{len(results)} valid profiles generated.")


if __name__ == "__main__":
    main()
