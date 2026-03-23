"""Step 5 Journey: Evaluate Journey Prediction Tasks

Reads a test TSV file (same format as pre_s3's journey_file), samples N
users, generates shopping profiles via Copilot API, then evaluates the
trained SFT model on event2journey and profile2journey tasks using vLLM.

Produces three TSV output files:

  1. llm_output.tsv           - Ground truth from FinalJourney column.
  2. slm_output.tsv           - Model predictions on event2journey inputs.
  3. slm_output_w_profile.tsv - Model predictions on profile2journey inputs.

Each output file contains:
  user_id, user_events, user_events_readable, user_profile,
  journey_model_output, journey_joined_products

For llm_output, the last two columns are identical since FinalJourney
contains product OfferIds (GlobalOfferIds) directly.

Pipeline:
  1. Read test TSV file (UserId, ReadableUserEvents, UserHistory,
     JourneyWithProducts, FinalJourney).
  2. Sample N user_ids.
  3. Generate shopping profiles via Copilot API (profile_pre_construct).
  4. Filter to users with valid profiles.
  5. Build ground truth from FinalJourney (llm_output).
  6. Run trained model on event2journey prompts -> slm_output.
  7. Run trained model on profile2journey prompts -> slm_output_w_profile.
  8. Map ProductTIDs back to GlobalOfferIds via exact + fuzzy matching.
  9. Output three TSV files with statistics.

Usage:
    python s5_journey_eval.py \\
        --model_path /path/to/sft_checkpoint \\
        --test_file /path/to/journey_test.tsv \\
        --tid2item_id_file ./sft_data/item_id2tid/tid2item_id.json \\
        --output_dir ./eval_results/journey \\
        --sample_n 500 \\
        --token_file ./resources/tokens.txt
"""

import os
import re
import csv
import json
import random
import argparse
import sys
from collections import defaultdict

import numpy as np

# Handle very large fields in TSV
csv.field_size_limit(sys.maxsize)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Add preprocess_raw_data and resources to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_DIR = os.path.join(SCRIPT_DIR, "preprocess_raw_data")
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "resources")
sys.path.insert(0, PREPROCESS_DIR)
sys.path.insert(0, RESOURCES_DIR)

from pre_s3_construct_shopping_journey import (
    parse_readable_user_events,
    parse_final_journey,
)
from profile_pre_construct_shopping_profile import (
    run_profile_generation,
    extract_and_validate_json,
)
from llm_utils import load_prompts, cleanup_checkpoint


# =============================================================================
# TID <-> GlobalOfferId Mapping (adapted from s5_eval.py)
# =============================================================================

def create_reverse_mapping(original_dict):
    """Create reverse mapping and word-level index for fuzzy matching."""
    reverse_mapping = {}
    word_to_keys = defaultdict(list)

    for key_str, ids in original_dict.items():
        words = [word.strip().lower() for word in key_str.split(",")]
        reverse_mapping[key_str] = {"words": words, "ids": ids}
        for word in words:
            word_to_keys[word].append(key_str)

    return reverse_mapping, word_to_keys


def get_iid_by_tid(tid_words, tid2item_id, reverse_mapping, word_to_keys):
    """Map a single TID (list of 7 words) to GlobalOfferIds.

    Returns:
        Tuple of (list_of_iids, match_type): "exact", "fuzzy", or "none".
    """
    tid_key = ",".join(tid_words)

    if tid_key in tid2item_id:
        return list(tid2item_id[tid_key]), "exact"

    candidate_scores = defaultdict(float)
    for i, query_word in enumerate(tid_words):
        position_weight = 1.0 / (i + 1)
        query_word_lower = query_word.lower().strip()
        for candidate_word, candidate_keys in word_to_keys.items():
            similarity = 0.0
            if query_word_lower == candidate_word:
                similarity = 1.0
            elif query_word_lower in candidate_word or candidate_word in query_word_lower:
                similarity = 0.8
            if similarity > 0:
                for candidate_key in candidate_keys:
                    candidate_scores[candidate_key] += similarity * position_weight

    if candidate_scores:
        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        iids = []
        for candidate_key, _ in sorted_candidates:
            iids.extend(reverse_mapping[candidate_key]["ids"])
        return iids[:1], "fuzzy"

    return [], "none"


# =============================================================================
# Journey JSON Parsing & Product Mapping
# =============================================================================

def parse_journey_json(raw_output):
    """Parse model output into ContinuedJourneys structure.

    Handles: markdown code fences, thinking blocks, extra text.

    Returns:
        Parsed dict with "ContinuedJourneys" key, or None on failure.
    """
    if not raw_output or not raw_output.strip():
        return None

    text = raw_output.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    brace_start = text.find("{")
    if brace_start == -1:
        return None

    depth = 0
    brace_end = -1
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                brace_end = i
                break

    candidate = (
        text[brace_start:brace_end + 1] if brace_end != -1
        else text[brace_start:] + "}"
    )

    try:
        data = json.loads(candidate)
        if "ContinuedJourneys" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        data = json.loads(text)
        if "ContinuedJourneys" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    return None


def map_journey_products(journey_data, tid2item_id, reverse_mapping, word_to_keys):
    """Map all ProductTIDs in a parsed journey to GlobalOfferIds.

    Returns:
        Tuple of (mapped_journey_data, mapping_stats).
    """
    stats = {"total_products": 0, "exact_matches": 0, "fuzzy_matches": 0, "no_matches": 0}

    if journey_data is None:
        return None, stats

    mapped = {"ContinuedJourneys": []}

    for journey in journey_data.get("ContinuedJourneys", []):
        mapped_journey = {
            "Title": journey.get("Title", ""),
            "Reason": journey.get("Reason", ""),
            "Products": [],
        }

        for tid_words in journey.get("ProductTIDs", []):
            stats["total_products"] += 1

            if not isinstance(tid_words, list) or len(tid_words) == 0:
                stats["no_matches"] += 1
                continue

            iids, match_type = get_iid_by_tid(
                tid_words, tid2item_id, reverse_mapping, word_to_keys
            )

            if match_type == "exact":
                stats["exact_matches"] += 1
            elif match_type == "fuzzy":
                stats["fuzzy_matches"] += 1
            else:
                stats["no_matches"] += 1

            mapped_journey["Products"].append({
                "TID": tid_words,
                "GlobalOfferIds": iids[:1] if iids else [],
                "match_type": match_type,
            })

        mapped["ContinuedJourneys"].append(mapped_journey)

    return mapped, stats


# =============================================================================
# Test Data Loading
# =============================================================================

def read_test_tsv(filepath):
    """Read the test TSV file (same format as pre_s3's journey_file).

    Expected columns: UserId, ReadableUserEvents, UserHistory,
                      JourneyWithProducts, FinalJourney

    Returns:
        List of row dicts, deduplicated by UserId (keep first).
    """
    rows = []
    seen_uids = set()

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        col_map = {name.strip(): idx for idx, name in enumerate(header)}

        for row in reader:
            uid_idx = col_map.get("UserId", 0)
            if len(row) <= uid_idx:
                continue
            uid = row[uid_idx].strip()
            if not uid or uid in seen_uids:
                continue
            seen_uids.add(uid)

            row_dict = {}
            for name, idx in col_map.items():
                row_dict[name] = row[idx].strip() if idx < len(row) else ""
            rows.append(row_dict)

    return rows


# =============================================================================
# Prompt Construction for Journey Tasks
# =============================================================================

def make_event2journey_instruction(num_journeys):
    """Create event2journey instruction with specific journey count."""
    return (
        f"Based on the user's shopping event history, predict {num_journeys} shopping "
        "journey(s) the user is likely to pursue. Each journey includes a "
        "title, a reason, and recommended products as text IDs (7 slots each). "
        '{"ContinuedJourneys":[{"Title":"...","Reason":"...",'
        '"ProductTIDs":[["s1","s2","s3","s4","s5","s6","s7"],...]},...]}.'
    )


def make_profile2journey_instruction(num_journeys):
    """Create profile2journey instruction with specific journey count."""
    return (
        f"Based on the user's shopping profile and recent shopping events, "
        f"predict {num_journeys} shopping journey(s) the user is likely to pursue. "
        "Each journey includes a title, a reason, and recommended products as "
        'text IDs (7 slots each). Output JSON: '
        '{"ContinuedJourneys":[{"Title":"...","Reason":"...",'
        '"ProductTIDs":[["s1","s2","s3","s4","s5","s6","s7"],...]},...]}.'
    )


def build_event2journey_input(events, max_events=50):
    """Build event2journey input text from event list."""
    truncated = events[:max_events]
    lines = ["User Event History:"]
    for idx, event in enumerate(truncated, 1):
        if len(event) > 150:
            event = event[:150] + "..."
        lines.append(f"{idx} | {event}")
    lines.append("")
    lines.append("Predict the user's shopping journeys:")
    return "\n".join(lines)


def build_profile2journey_input(profile_text, events, max_recent_events=10):
    """Build profile2journey input text from profile and events."""
    recent = events[:max_recent_events]
    lines = ["User Shopping Profile:", profile_text, ""]
    lines.append("Recent Shopping Events:")
    for idx, event in enumerate(recent, 1):
        if len(event) > 150:
            event = event[:150] + "..."
        lines.append(f"{idx} | {event}")
    lines.append("")
    lines.append("Predict the user's shopping journeys:")
    return "\n".join(lines)


# =============================================================================
# vLLM Inference
# =============================================================================

def run_vllm_inference(prompts, model_path, num_gpus, gpu_memory_utilization,
                       max_model_len, max_tokens):
    """Run vLLM offline inference on formatted prompt strings."""
    from vllm import LLM, SamplingParams

    print(f"\nInitializing vLLM engine ...")
    print(f"  Model: {model_path}")
    print(f"  Tensor parallel size: {num_gpus}")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    print(f"  Max model length: {max_model_len}")
    print(f"  Max output tokens: {max_tokens}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        seed=SEED,
    )

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0, top_p=1.0)

    import time
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time

    throughput = len(prompts) / elapsed if elapsed > 0 else 0
    print(f"  Inference done in {elapsed:.1f}s ({throughput:.1f} items/s)")

    return [output.outputs[0].text.strip() for output in outputs]


def build_chat_prompts(instructions_and_inputs, tokenizer, task_type,
                       enable_thinking=False):
    """Build chat-formatted prompts for vLLM.

    Args:
        instructions_and_inputs: List of (instruction, input_text) tuples.
    """
    prompts = []
    for instruction, input_text in instructions_and_inputs:
        content = instruction + "\n" + input_text
        messages = [{"role": "user", "content": content}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(formatted)

    print(f"  Built {len(prompts)} prompts for {task_type}")
    return prompts


# =============================================================================
# Ground Truth Construction
# =============================================================================

def format_ground_truth_journey(journey_text):
    """Parse FinalJourney text and format as ground truth JSON.

    Uses parse_final_journey with a permissive set that accepts all
    OfferIds (no filtering).

    Returns:
        Tuple of (json_string, journeys_list). Returns ("", []) on failure.
    """
    if not journey_text or not journey_text.strip():
        return "", []

    class _AcceptAll:
        """A set-like object that accepts any membership test."""
        def __contains__(self, item):
            return True

    journeys, _, _ = parse_final_journey(journey_text, _AcceptAll())

    if not journeys:
        return "", []

    continued = []
    for j in journeys:
        continued.append({
            "Title": j.get("title", ""),
            "Reason": j.get("reason", ""),
            "ProductIds": j.get("product_ids", []),
        })
    gt_json = json.dumps({"ContinuedJourneys": continued}, ensure_ascii=False)
    return gt_json, journeys


# =============================================================================
# Output Construction & Statistics
# =============================================================================

def build_output_rows(
    user_data_list,
    model_outputs,
    tid2item_id,
    reverse_mapping,
    word_to_keys,
    task_type,
    is_ground_truth=False,
):
    """Build output rows for one output file.

    Args:
        user_data_list: List of dicts with user_id, user_events,
            user_events_readable, user_profile, ground_truth_json,
            ground_truth_journeys.
        model_outputs: List of raw model output strings (or None for GT).
        tid2item_id, reverse_mapping, word_to_keys: TID mapping data.
        task_type: "event2journey" or "profile2journey".
        is_ground_truth: If True, use ground_truth_json field.

    Returns:
        Tuple of (rows, aggregate_stats).
    """
    rows = []
    agg_stats = {
        "total_users": len(user_data_list),
        "json_parse_success": 0,
        "json_parse_fail": 0,
        "total_products": 0,
        "exact_matches": 0,
        "fuzzy_matches": 0,
        "no_matches": 0,
        "users_with_all_fields": 0,
        "per_user_exact_ratios": [],
    }

    for idx, ud in enumerate(user_data_list):
        uid = ud["user_id"]
        user_events = ud["user_events"]
        user_events_readable = ud["user_events_readable"]
        user_profile = ud.get("user_profile", "")

        if is_ground_truth:
            raw_output = ud.get("ground_truth_json", "")
            journey_joined = raw_output  # identical for GT

            gt_journeys = ud.get("ground_truth_journeys", [])
            total_prods = sum(len(j.get("product_ids", [])) for j in gt_journeys)
            agg_stats["total_products"] += total_prods
            agg_stats["exact_matches"] += total_prods
            if total_prods > 0:
                agg_stats["per_user_exact_ratios"].append(1.0)

            if gt_journeys:
                agg_stats["json_parse_success"] += 1
            else:
                agg_stats["json_parse_fail"] += 1

            has_all = bool(uid and user_events and gt_journeys)
            if has_all:
                agg_stats["users_with_all_fields"] += 1
        else:
            raw_output = model_outputs[idx] if model_outputs else ""

            journey_data = parse_journey_json(raw_output)
            if journey_data is not None:
                agg_stats["json_parse_success"] += 1
            else:
                agg_stats["json_parse_fail"] += 1

            mapped_data, map_stats = map_journey_products(
                journey_data, tid2item_id, reverse_mapping, word_to_keys
            )
            agg_stats["total_products"] += map_stats["total_products"]
            agg_stats["exact_matches"] += map_stats["exact_matches"]
            agg_stats["fuzzy_matches"] += map_stats["fuzzy_matches"]
            agg_stats["no_matches"] += map_stats["no_matches"]

            user_total = map_stats["total_products"]
            user_exact = map_stats["exact_matches"]
            if user_total > 0:
                agg_stats["per_user_exact_ratios"].append(user_exact / user_total)

            has_all = bool(uid and user_events and raw_output and mapped_data)
            if has_all:
                agg_stats["users_with_all_fields"] += 1

            journey_joined = (
                json.dumps(mapped_data, ensure_ascii=False) if mapped_data else ""
            )

        rows.append({
            "user_id": uid,
            "user_events": user_events,
            "user_events_readable": user_events_readable,
            "user_profile": user_profile,
            "journey_model_output": raw_output,
            "journey_joined_products": journey_joined,
        })

    return rows, agg_stats


def save_tsv(rows, output_file, columns):
    """Save rows as a TSV file."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, delimiter="\t",
            quoting=csv.QUOTE_ALL, extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Saved {len(rows)} rows to: {output_file}")


def print_stats(label, stats):
    """Print aggregate statistics for one output file."""
    print(f"\n  --- {label} Statistics ---")
    print(f"    Total users:              {stats['total_users']:>10,}")
    print(f"    JSON parse success:       {stats['json_parse_success']:>10,}")
    print(f"    JSON parse fail:          {stats['json_parse_fail']:>10,}")
    print(f"    Users with all fields:    {stats['users_with_all_fields']:>10,}")
    print(f"    Total products:           {stats['total_products']:>10,}")
    print(f"    Exact matches:            {stats['exact_matches']:>10,} "
          f"({stats['exact_matches'] / max(stats['total_products'], 1) * 100:.1f}%)")
    print(f"    Fuzzy matches:            {stats['fuzzy_matches']:>10,} "
          f"({stats['fuzzy_matches'] / max(stats['total_products'], 1) * 100:.1f}%)")
    print(f"    No matches:               {stats['no_matches']:>10,} "
          f"({stats['no_matches'] / max(stats['total_products'], 1) * 100:.1f}%)")

    ratios = stats.get("per_user_exact_ratios", [])
    if ratios:
        arr = np.array(ratios)
        print(f"    Per-user exact match rate:")
        print(f"      Mean:   {arr.mean():.2%}")
        print(f"      Median: {np.median(arr):.2%}")
        print(f"      Min:    {arr.min():.2%}")
        print(f"      Max:    {arr.max():.2%}")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate journey prediction tasks from a test TSV file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the trained SFT model checkpoint (Qwen3.5)",
    )
    parser.add_argument(
        "--test_file", type=str, required=True,
        help="Path to test TSV file (same format as pre_s3 journey_file: "
             "UserId, ReadableUserEvents, UserHistory, JourneyWithProducts, "
             "FinalJourney)",
    )
    parser.add_argument(
        "--tid2item_id_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/sft_data/item_id2tid/tid2item_id.json",
        help="Path to tid2item_id.json for TID -> GlobalOfferId mapping",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/eval_results/journey",
        help="Directory to save evaluation output files",
    )
    parser.add_argument(
        "--sample_n", type=int, default=500,
        help="Number of user_ids to sample for evaluation (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max_events", type=int, default=50,
        help="Max events for event2journey input (default: 50)",
    )
    parser.add_argument(
        "--max_recent_events", type=int, default=10,
        help="Max recent events for profile2journey input (default: 10)",
    )
    # vLLM args
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="Number of GPUs for vLLM tensor parallelism (default: all)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.90,
        help="Fraction of GPU memory for vLLM (default: 0.90)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=8192,
        help="Maximum model context length (default: 8192)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048,
        help="Maximum output tokens per sample (default: 2048)",
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", default=False,
        help="Enable thinking/reasoning mode in chat template",
    )
    # Profile generation args (Copilot API)
    parser.add_argument(
        "--prompts_file", type=str,
        default="./resources/prompts.yaml",
        help="Path to prompts.yaml (for profile generation template)",
    )
    parser.add_argument(
        "--token_file", type=str,
        default="./resources/tokens.txt",
        help="Path to tokens.txt for Copilot API authentication",
    )
    parser.add_argument(
        "--copilot_model", type=str, default="gpt-5.2",
        help="Copilot model name for profile generation (default: gpt-5.2)",
    )
    parser.add_argument(
        "--profile_workers", type=int, default=20,
        help="Number of parallel workers for profile generation (default: 20)",
    )
    parser.add_argument(
        "--profile_max_tokens", type=int, default=2000,
        help="Max tokens for profile generation API calls (default: 2000)",
    )
    parser.add_argument(
        "--profile_chunk_size", type=int, default=500,
        help="Chunk size for profile generation checkpoints (default: 500)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_gpus = args.num_gpus if args.num_gpus is not None else max(available_gpus, 1)

    print("=" * 70)
    print("Step 5 Journey: Evaluate Journey Prediction Tasks")
    print("=" * 70)
    print(f"  Model: {args.model_path}")
    print(f"  Test file: {args.test_file}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Sample N: {args.sample_n}")
    print(f"  Seed: {args.seed}")

    # =========================================================================
    # Step 1: Read test TSV file
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 1: Reading test TSV file")
    print("=" * 70)

    print(f"  Loading: {args.test_file}")
    test_rows = read_test_tsv(args.test_file)
    print(f"    Total unique users: {len(test_rows):,}")

    if not test_rows:
        print("ERROR: No data found in test file.")
        return

    # =========================================================================
    # Step 2: Sample N user_ids
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Sampling user_ids")
    print("=" * 70)

    sample_n = min(args.sample_n, len(test_rows))
    sampled_rows = random.sample(test_rows, sample_n)
    sampled_rows.sort(key=lambda r: r.get("UserId", ""))
    print(f"  Sampled {len(sampled_rows):,} users from {len(test_rows):,}")

    # =========================================================================
    # Step 3: Process user data (events, ground truth)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Processing user data")
    print("=" * 70)

    user_data = []
    no_events = 0
    no_final_journey = 0
    gt_parse_success = 0
    gt_parse_fail = 0

    for row in sampled_rows:
        uid = row["UserId"]

        # user_events: UserHistory column (raw text)
        user_events_raw = row.get("UserHistory", "")

        # user_events_readable: ReadableUserEvents processed via parse_readable_user_events
        readable_raw = row.get("ReadableUserEvents", "")
        events_list, _ = parse_readable_user_events(readable_raw)
        if not events_list:
            no_events += 1

        # Format readable events as numbered list
        readable_lines = []
        for idx, event in enumerate(events_list, 1):
            if len(event) > 150:
                event = event[:150] + "..."
            readable_lines.append(f"{idx} | {event}")
        user_events_readable = "\n".join(readable_lines)

        # Ground truth from FinalJourney
        final_journey_text = row.get("FinalJourney", "")
        gt_json, gt_journeys = format_ground_truth_journey(final_journey_text)
        if gt_journeys:
            gt_parse_success += 1
        else:
            gt_parse_fail += 1
            if not final_journey_text:
                no_final_journey += 1

        user_data.append({
            "user_id": uid,
            "user_events": user_events_raw,
            "user_events_readable": user_events_readable,
            "events_list": events_list,
            "ground_truth_json": gt_json,
            "ground_truth_journeys": gt_journeys,
            "num_journeys": len(gt_journeys),
        })

    print(f"    Users with events: {len(user_data) - no_events:,} / {len(user_data):,}")
    print(f"    Users without events: {no_events:,}")
    print(f"    Ground truth parse success: {gt_parse_success:,}")
    print(f"    Ground truth parse fail: {gt_parse_fail:,}")
    print(f"    No FinalJourney column: {no_final_journey:,}")

    # =========================================================================
    # Step 4: Generate shopping profiles via Copilot API
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Generating shopping profiles via Copilot API")
    print("=" * 70)

    print(f"  Loading prompt template from: {args.prompts_file}")
    prompts_config = load_prompts(args.prompts_file)
    profile_prompt_template = prompts_config["generate_shopping_profile_from_events"]["user"]

    # Prepare inputs: (user_id, events_text) where events_text is
    # newline-separated event strings
    profile_inputs = []
    for ud in user_data:
        uid = ud["user_id"]
        events_text = "\n".join(ud["events_list"])
        profile_inputs.append((uid, events_text))

    os.makedirs(args.output_dir, exist_ok=True)
    profile_checkpoint_dir = os.path.join(args.output_dir, "_profile_checkpoint")

    print(f"  Generating profiles for {len(profile_inputs)} users ...")
    print(f"    Copilot model: {args.copilot_model}")
    print(f"    Workers: {args.profile_workers}")

    profile_results = run_profile_generation(
        users=profile_inputs,
        prompt_template=profile_prompt_template,
        token_file=args.token_file,
        copilot_model=args.copilot_model,
        num_workers=args.profile_workers,
        max_tokens=args.profile_max_tokens,
        checkpoint_dir=profile_checkpoint_dir,
        chunk_size=args.profile_chunk_size,
    )

    # Parse and validate profiles
    profile_map = {}
    profile_valid = 0
    profile_invalid = 0
    profile_empty = 0

    for uid, raw_profile in profile_results:
        if not raw_profile:
            profile_empty += 1
            continue
        clean_json = extract_and_validate_json(raw_profile)
        if clean_json:
            profile_map[uid] = clean_json
            profile_valid += 1
        else:
            profile_invalid += 1

    print(f"\n  Profile generation results:")
    print(f"    Total users:    {len(profile_results):>10,}")
    print(f"    Valid profiles: {profile_valid:>10,}")
    print(f"    Invalid JSON:   {profile_invalid:>10,}")
    print(f"    Empty response: {profile_empty:>10,}")

    # Attach profiles to user_data
    for ud in user_data:
        ud["user_profile"] = profile_map.get(ud["user_id"], "")

    cleanup_checkpoint(profile_checkpoint_dir)

    # Filter to users with valid profiles
    users_with_profile = [ud for ud in user_data if ud["user_profile"]]
    users_without_profile = len(user_data) - len(users_with_profile)
    print(f"\n  Users with valid profile: {len(users_with_profile):,}")
    print(f"  Users without valid profile: {users_without_profile:,}")

    if not users_with_profile:
        print("ERROR: No users with valid profiles. Cannot run journey tasks.")
        return

    # =========================================================================
    # Step 5: Load TID -> GlobalOfferId mapping
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 5: Loading TID mapping")
    print("=" * 70)

    print(f"  Loading: {args.tid2item_id_file}")
    with open(args.tid2item_id_file, "r", encoding="utf-8") as f:
        tid2item_id = json.load(f)
    reverse_mapping, word_to_keys = create_reverse_mapping(tid2item_id)
    print(f"    Unique TIDs: {len(tid2item_id):,}")
    print(f"    Unique words in index: {len(word_to_keys):,}")

    # =========================================================================
    # Step 6: Build ground truth output (llm_output)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 6: Building ground truth output (llm_output)")
    print("=" * 70)

    llm_rows, llm_stats = build_output_rows(
        user_data_list=users_with_profile,
        model_outputs=None,
        tid2item_id=tid2item_id,
        reverse_mapping=reverse_mapping,
        word_to_keys=word_to_keys,
        task_type="event2journey",
        is_ground_truth=True,
    )
    print_stats("llm_output (ground truth)", llm_stats)

    # =========================================================================
    # Step 7: Run model inference with vLLM
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 7: Running model inference via vLLM")
    print("=" * 70)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    # --- 7a: event2journey prompts ---
    print("\n  Building event2journey prompts ...")
    e2j_instr_inputs = []
    for ud in users_with_profile:
        input_text = build_event2journey_input(
            ud["events_list"], max_events=args.max_events
        )
        instruction = make_event2journey_instruction(ud["num_journeys"])
        e2j_instr_inputs.append((instruction, input_text))

    e2j_prompts = build_chat_prompts(
        e2j_instr_inputs, tokenizer, "event2journey",
        enable_thinking=args.enable_thinking,
    )

    # --- 7b: profile2journey prompts ---
    print("  Building profile2journey prompts ...")
    p2j_instr_inputs = []
    for ud in users_with_profile:
        input_text = build_profile2journey_input(
            ud["user_profile"], ud["events_list"],
            max_recent_events=args.max_recent_events,
        )
        instruction = make_profile2journey_instruction(ud["num_journeys"])
        p2j_instr_inputs.append((instruction, input_text))

    p2j_prompts = build_chat_prompts(
        p2j_instr_inputs, tokenizer, "profile2journey",
        enable_thinking=args.enable_thinking,
    )

    # --- 7c: Run all prompts ---
    all_prompts = e2j_prompts + p2j_prompts
    print(f"\n  Total prompts: {len(all_prompts)} "
          f"({len(e2j_prompts)} event2journey + {len(p2j_prompts)} profile2journey)")

    all_outputs = run_vllm_inference(
        prompts=all_prompts,
        model_path=args.model_path,
        num_gpus=num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
    )

    e2j_outputs = all_outputs[:len(e2j_prompts)]
    p2j_outputs = all_outputs[len(e2j_prompts):]

    print(f"  event2journey outputs: {len(e2j_outputs)}")
    print(f"  profile2journey outputs: {len(p2j_outputs)}")

    # =========================================================================
    # Step 8: Build slm_output (event2journey predictions)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 8: Building slm_output (event2journey predictions)")
    print("=" * 70)

    slm_rows, slm_stats = build_output_rows(
        user_data_list=users_with_profile,
        model_outputs=e2j_outputs,
        tid2item_id=tid2item_id,
        reverse_mapping=reverse_mapping,
        word_to_keys=word_to_keys,
        task_type="event2journey",
        is_ground_truth=False,
    )
    print_stats("slm_output (event2journey)", slm_stats)

    # =========================================================================
    # Step 9: Build slm_output_w_profile (profile2journey predictions)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 9: Building slm_output_w_profile (profile2journey predictions)")
    print("=" * 70)

    slm_profile_rows, slm_profile_stats = build_output_rows(
        user_data_list=users_with_profile,
        model_outputs=p2j_outputs,
        tid2item_id=tid2item_id,
        reverse_mapping=reverse_mapping,
        word_to_keys=word_to_keys,
        task_type="profile2journey",
        is_ground_truth=False,
    )
    print_stats("slm_output_w_profile (profile2journey)", slm_profile_stats)

    # =========================================================================
    # Step 10: Save output files
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 10: Saving output files")
    print("=" * 70)

    columns = [
        "user_id", "user_events", "user_events_readable", "user_profile",
        "journey_model_output", "journey_joined_products",
    ]

    llm_file = os.path.join(args.output_dir, "llm_output.tsv")
    save_tsv(llm_rows, llm_file, columns)

    slm_file = os.path.join(args.output_dir, "slm_output.tsv")
    save_tsv(slm_rows, slm_file, columns)

    slm_profile_file = os.path.join(args.output_dir, "slm_output_w_profile.tsv")
    save_tsv(slm_profile_rows, slm_profile_file, columns)

    # =========================================================================
    # Step 11: Example outputs
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 11: Example outputs (first 2 users)")
    print("=" * 70)

    for idx in range(min(2, len(users_with_profile))):
        uid = users_with_profile[idx]["user_id"]
        print(f"\n--- User {idx + 1}: {uid} ---")

        gt_row = llm_rows[idx]
        gt_journey = parse_journey_json(gt_row["journey_model_output"])
        print(f"  [Ground Truth]")
        if gt_journey:
            for ji, j in enumerate(gt_journey.get("ContinuedJourneys", [])[:2]):
                print(f"    Journey {ji+1}: {j.get('Title', 'N/A')}")
                print(f"      Products: {len(j.get('ProductIds', []))}")

        slm_row = slm_rows[idx]
        print(f"  [SLM event2journey]")
        print(f"    Raw (first 200): {slm_row['journey_model_output'][:200]}")

        sp_row = slm_profile_rows[idx]
        print(f"  [SLM profile2journey]")
        print(f"    Raw (first 200): {sp_row['journey_model_output'][:200]}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"  Total users in test file:       {len(test_rows):>10,}")
    print(f"  Sampled users:                  {len(sampled_rows):>10,}")
    print(f"  Valid profiles generated:       {profile_valid:>10,}")
    print(f"  Invalid/empty profiles:         {profile_invalid + profile_empty:>10,}")
    print(f"  Users evaluated (with profile): {len(users_with_profile):>10,}")

    print(f"\n  {'Metric':<40s} {'GT':>10s} {'SLM':>10s} {'SLM+Prof':>10s}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
    for label, st in [
        ("JSON parse success", "json_parse_success"),
        ("JSON parse fail", "json_parse_fail"),
        ("Total products", "total_products"),
        ("Exact matches", "exact_matches"),
        ("Fuzzy matches", "fuzzy_matches"),
        ("No matches", "no_matches"),
        ("Users with all fields", "users_with_all_fields"),
    ]:
        print(f"  {label:<40s} {llm_stats[st]:>10,} "
              f"{slm_stats[st]:>10,} {slm_profile_stats[st]:>10,}")

    for name, stats in [
        ("GT", llm_stats), ("SLM", slm_stats), ("SLM+Profile", slm_profile_stats),
    ]:
        ratios = stats.get("per_user_exact_ratios", [])
        if ratios:
            arr = np.array(ratios)
            print(f"  {name} per-user exact rate: "
                  f"mean={arr.mean():.2%}, median={np.median(arr):.2%}")

    # Save summary JSON
    summary = {
        "test_file": args.test_file,
        "total_test_users": len(test_rows),
        "sample_n": len(sampled_rows),
        "valid_profiles": profile_valid,
        "invalid_profiles": profile_invalid,
        "empty_profiles": profile_empty,
        "evaluated_users": len(users_with_profile),
        "model_path": args.model_path,
        "seed": args.seed,
        "llm_stats": {k: v for k, v in llm_stats.items()
                      if k != "per_user_exact_ratios"},
        "slm_stats": {k: v for k, v in slm_stats.items()
                      if k != "per_user_exact_ratios"},
        "slm_profile_stats": {k: v for k, v in slm_profile_stats.items()
                              if k != "per_user_exact_ratios"},
    }
    summary_file = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary saved to: {summary_file}")

    print(f"\nDone! Evaluated {len(users_with_profile)} users on 2 tasks, "
          f"saved 3 output files to {args.output_dir}")


if __name__ == "__main__":
    main()
