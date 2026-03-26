"""
Step 2.8: Call LLM Ranker for Shopping Journey Products

Reads the JWP (JourneyWithProducts) TSV, fills in the ranker prompt template
with per-user data (ReadableUserEvents, RequestTime, JourneyWithProducts),
calls GitHub Copilot API to rank products, and saves the results.

Input:
  TSV with columns: UserId, ReadableUserEvents, RequestTime, UserHistory,
  JourneyWithProducts

Prompt template (journey_ranker.md) has placeholders:
  ##ReadableUserEvents##  → user's browsing/search/click history
  ##RequestTime##         → current system time
  ##JourneyWithProducts## → JSON of journeys with candidate products

Output:
  TSV with columns: UserId, ReadableUserEvents, RequestTime, JourneyWithProducts, OUTPUT

Usage:
    python step2.8_call_LLM_ranker.py --input_file /path/to/JWP.tsv [--debug]
"""

import argparse
import csv
import json
import os
import re
import sys
import time

from tqdm import tqdm

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# Add llm_utils directory to path
LLM_UTILS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "resources"
)
sys.path.insert(0, LLM_UTILS_DIR)
from llm_utils import run_llm_parallel_with_checkpoint, cleanup_checkpoint


# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_INPUT_FILE = (
    "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/ShoppingJourney_Input_80K_1_results_JWP.tsv"
)
PROMPT_FILE = os.path.join(PROJECT_DIR, "resources", "journey_ranker.md")
TOKEN_FILE = os.path.join(PROJECT_DIR, "resources", "tokens.txt")


# =============================================================================
# Data Loading
# =============================================================================

def load_prompt_template(prompt_file):
    """Load the ranker prompt template from a markdown file.

    The template uses ##FieldName## as placeholders.

    Returns:
        str: Raw prompt template string.
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def read_jwp_tsv(filepath, max_users=0):
    """Read JWP TSV and extract per-user data.

    Columns: UserId, ReadableUserEvents, RequestTime, UserHistory,
             JourneyWithProducts

    Args:
        filepath: Path to the JWP TSV file.
        max_users: If >0, stop after this many users.

    Returns:
        List of dicts with keys: user_id, events, request_time, jwp_str
    """
    users = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            user_id = row.get("UserId", "").strip()
            events_raw = row.get("ReadableUserEvents", "").strip()
            request_time = row.get("RequestTime", "").strip()
            jwp_str = row.get("JourneyWithProducts", "").strip()

            if not user_id or not jwp_str:
                continue

            # Replace "#N#" separator with newlines
            events_text = events_raw.replace("#N#", "\n")

            users.append({
                "user_id": user_id,
                "events": events_text,
                "request_time": request_time,
                "jwp_str": jwp_str,
            })

            if max_users > 0 and len(users) >= max_users:
                break

    return users


# =============================================================================
# Prompt Construction
# =============================================================================

def build_prompt(user_data, prompt_template):
    """Build LLM prompt by replacing ##xxx## placeholders.

    Replacements:
      ##ReadableUserEvents## → user events text
      ##RequestTime##        → request time string
      ##JourneyWithProducts## → JWP JSON string

    Args:
        user_data: Dict with keys: events, request_time, jwp_str.
        prompt_template: Template string with ##xxx## placeholders.

    Returns:
        str: Filled prompt.
    """
    prompt = prompt_template
    prompt = prompt.replace("##ReadableUserEvents##", user_data["events"])
    prompt = prompt.replace("##RequestTime##", user_data["request_time"])
    prompt = prompt.replace("##JourneyWithProducts##", user_data["jwp_str"])
    return prompt


# =============================================================================
# Result Extraction
# =============================================================================

def extract_output_json(raw_text):
    """Extract JSON from LLM output wrapped in <OUTPUT>...</OUTPUT> tags.

    Falls back to finding the outermost { ... } if tags are not present.

    Args:
        raw_text: Raw LLM response string.

    Returns:
        str: Compact JSON string, or empty string if extraction fails.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text.strip()

    # Try to extract from <OUTPUT>...</OUTPUT> tags
    tag_match = re.search(r'<OUTPUT>\s*(.*?)\s*</OUTPUT>', text, re.DOTALL)
    if tag_match:
        text = tag_match.group(1).strip()

    # Strip markdown code fences if present
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to parse as JSON
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
    except json.JSONDecodeError:
        pass

    # Fallback: find outermost { ... }
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            parsed = json.loads(candidate)
            return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
        except json.JSONDecodeError:
            pass

    return ""


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2.8: Call LLM to rank products in shopping journeys"
    )
    parser.add_argument(
        "--input_file", type=str, default=DEFAULT_INPUT_FILE,
        help="Path to JWP TSV file",
    )
    parser.add_argument(
        "--prompt_file", type=str, default=PROMPT_FILE,
        help="Path to journey_ranker.md prompt template",
    )
    parser.add_argument(
        "--token_file", type=str, default=TOKEN_FILE,
        help="Path to GitHub tokens file",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: same dir as input_file)",
    )
    parser.add_argument(
        "--copilot_model", type=str, default="gpt-5.2",
        help="Copilot model name (default: gpt-5.2)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=40,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=10000,
        help="Maximum output tokens per API call ",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=10000,
        help="Users per checkpoint chunk",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Debug mode: process only first 50 users",
    )
    parser.add_argument(
        "--debug_rows", type=int, default=50,
        help="Number of users in debug mode (default: 50)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("Step 2.8: Call LLM Ranker for Shopping Journeys")
    print(f"  Input:    {args.input_file}")
    print(f"  Prompt:   {args.prompt_file}")
    print(f"  Model:    {args.copilot_model}")
    print(f"  Workers:  {args.num_workers}")
    print(f"  Max tok:  {args.max_tokens}")
    print("=" * 70)

    if args.debug:
        print(f"\n*** DEBUG MODE: processing only {args.debug_rows} users ***\n")

    max_users = args.debug_rows if args.debug else 0

    # ---- Load prompt template ----
    print(f"\nLoading prompt template: {args.prompt_file}")
    prompt_template = load_prompt_template(args.prompt_file)
    print(f"  Template length: {len(prompt_template):,} chars")

    # ---- Load data ----
    print(f"\nLoading JWP data: {args.input_file}")
    users = read_jwp_tsv(args.input_file, max_users=max_users)
    print(f"  Loaded {len(users):,} users")

    if not users:
        print("No users found!")
        return

    # Show sample
    sample = users[0]
    print(f"\n  Sample user: {sample['user_id']}")
    print(f"  Events length: {len(sample['events'])} chars")
    print(f"  RequestTime: {sample['request_time']}")
    try:
        jwp = json.loads(sample["jwp_str"])
        num_j = len(jwp.get("ContinuedJourneys", []))
        num_q = sum(
            len(j.get("Queries", []))
            for j in jwp.get("ContinuedJourneys", [])
        )
        num_p = sum(
            len(q.get("Products", []))
            for j in jwp.get("ContinuedJourneys", [])
            for q in j.get("Queries", [])
        )
        print(f"  Journeys: {num_j}, Queries: {num_q}, Products: {num_p}")
    except json.JSONDecodeError:
        print(f"  JWP: (invalid JSON)")

    # ---- Build prompts ----
    print("\nBuilding prompts ...")
    inputs = []
    for u in users:
        prompt = build_prompt(u, prompt_template)
        inputs.append((u["user_id"], prompt))

    # Show prompt length stats
    prompt_lens = [len(p) for _, p in inputs]
    print(f"  Prompt lengths: min={min(prompt_lens):,}, "
          f"max={max(prompt_lens):,}, avg={sum(prompt_lens)/len(prompt_lens):,.0f}")

    # ---- Call LLM ----
    output_dir = args.output_dir or os.path.dirname(args.input_file)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "_ranker_checkpoint")

    print(f"\nCalling LLM ranker ({args.copilot_model}) ...")
    start_time = time.time()

    results = run_llm_parallel_with_checkpoint(
        inputs=inputs,
        token_file=args.token_file,
        checkpoint_dir=checkpoint_dir,
        num_workers=args.num_workers,
        model=args.copilot_model,
        temperature=0,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
    )

    elapsed = time.time() - start_time
    throughput = len(users) / elapsed if elapsed > 0 else 0
    print(f"\nLLM calls done: {elapsed:.1f}s ({throughput:.1f} users/s)")

    # ---- Save results ----
    # Derive output file name from input file base
    base = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = args.output_dir or os.path.dirname(args.input_file)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{base}_Ranker.tsv")
    print(f"\nSaving results to: {output_file}")

    # Build user_id -> user_data map for quick lookup
    user_map = {u["user_id"]: u for u in users}

    success_count = 0
    json_valid_count = 0
    json_invalid_ids = []

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE,
                            escapechar="\\")
        writer.writerow([
            "UserId", "ReadableUserEvents", "RequestTime",
            "JourneyWithProducts", "OUTPUT"
        ])

        for user_id, raw_result in results:
            u = user_map[user_id]
            events_out = u["events"].replace("\n", "#N#")

            if not raw_result:
                writer.writerow([
                    user_id, events_out, u["request_time"],
                    u["jwp_str"], ""
                ])
                continue
            success_count += 1

            clean_json = extract_output_json(raw_result)
            if clean_json:
                json_valid_count += 1
                writer.writerow([
                    user_id, events_out, u["request_time"],
                    u["jwp_str"], clean_json
                ])
            else:
                json_invalid_ids.append(user_id)
                fallback = raw_result.replace("\n", " ").replace("\t", " ")
                writer.writerow([
                    user_id, events_out, u["request_time"],
                    u["jwp_str"], fallback
                ])

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved {len(results):,} users ({file_size_mb:.1f} MB)")
    print(f"  API success:  {success_count}/{len(results)}")
    print(f"  Valid JSON:   {json_valid_count}/{success_count}")
    if json_invalid_ids:
        print(f"  Invalid JSON ({len(json_invalid_ids)}): "
              f"{json_invalid_ids[:10]}{'...' if len(json_invalid_ids) > 10 else ''}")

    # ---- Show samples ----
    num_show = min(2, len(results))
    print(f"\nSample outputs (first {num_show}):")
    for uid, raw in results[:num_show]:
        clean = extract_output_json(raw)
        print(f"\n  UserId: {uid}")
        if clean:
            try:
                obj = json.loads(clean)
                cj = obj.get("ContinuedJourneys", [])
                print(f"  Output journeys: {len(cj)}")
                for j in cj[:2]:
                    prods = j.get("Products", [])
                    print(f"    Journey: {j.get('Title', '')[:80]}")
                    print(f"    Ranked products: {len(prods)}")
                    if prods:
                        print(f"    Top product: {prods[0].get('Title', '')[:80]}")
            except json.JSONDecodeError:
                print(f"  Output: {clean[:200]}...")
        else:
            print(f"  Raw (truncated): {(raw or '')[:200]}...")

    # ---- Cleanup ----
    cleanup_checkpoint(checkpoint_dir)

    print(f"\nDone! {json_valid_count}/{len(results)} valid ranked results.")


if __name__ == "__main__":
    main()
