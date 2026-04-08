"""Step 5 Journey: Standalone Evaluate Journey Prediction Tasks

Reads a test TSV file, samples N users, generates shopping profiles via
vLLM (using a dedicated profile model), then evaluates the trained SFT
journey model on event2journey and profile2journey tasks using vLLM.

Standalone: no external imports from preprocess_raw_data or resources.

Produces three TSV output files:

  1. llm_output.tsv           - Ground truth from FinalJourney column.
  2. slm_output.tsv           - Model predictions on event2journey inputs.
  3. slm_output_w_profile.tsv - Model predictions on profile2journey inputs.

Pipeline:
  1. Read test TSV file.
  2. Sample N user_ids.
  3. Convert UserHistory -> ReadableUserEvents if column missing.
  4. Generate shopping profiles via vLLM (profile model).
  5. Filter to users with valid profiles.
  6. Build ground truth from FinalJourney (llm_output).
  7. Run journey model on event2journey prompts -> slm_output.
  8. Run journey model on profile2journey prompts -> slm_output_w_profile.
  9. Map ProductTIDs back to GlobalOfferIds via exact + fuzzy matching.
  10. Output three TSV files with statistics.

Usage:
    python s5_journey_eval_standalone.py \\
        --model_path /path/to/journey_sft_checkpoint \\
        --profile_model_path /path/to/profile_checkpoint \\
        --test_file /path/to/journey_test.tsv \\
        --tid2item_id_file ./sft_data/item_id2tid/tid2item_id.json \\
        --output_dir ./eval_results/journey \\
        --sample_n 500
"""

import os
import re
import csv
import json
import math
import random
import argparse
import sys
import time
from collections import defaultdict

import numpy as np

# Handle very large fields in TSV
csv.field_size_limit(sys.maxsize)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Inlined helpers (previously imported from preprocess_raw_data/)
# =============================================================================

_RE_EVENT_NUMBER = re.compile(r"^\d+\s*\|\s*(.*)")
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s|]")
_RE_MULTI_SPACE = re.compile(r"\s+")


def _normalize_event_key(event):
    """Normalize an event string for deduplication."""
    key = event.lower()
    key = _RE_NON_ALNUM.sub(" ", key)
    key = _RE_MULTI_SPACE.sub(" ", key)
    return key.strip()


def parse_readable_user_events(events_text):
    """Parse ReadableUserEvents text (#N# separated) into list of events.

    Returns:
        Tuple of (deduplicated event list, total raw events before dedup).
    """
    if not events_text or not events_text.strip():
        return [], 0

    text = events_text.replace("#N#", "\n")
    lines = text.strip().split("\n")

    raw_events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = _RE_EVENT_NUMBER.match(line)
        if match:
            event = match.group(1).strip()
            if event:
                raw_events.append(event)

    seen_keys = set()
    deduped = []
    for event in raw_events:
        key = _normalize_event_key(event)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(event)

    return deduped, len(raw_events)


def parse_final_journey(journey_text, valid_offer_ids):
    """Parse FinalJourney JSON and extract journeys with validated OfferIds.

    Returns:
        Tuple of (journeys_list, stats_dict, missing_ids_set).
    """
    stats = {
        "total_offer_ids": 0, "found_offer_ids": 0, "missing_offer_ids": 0,
        "total_journeys": 0, "kept_journeys": 0, "empty_product_journeys": 0,
    }
    missing_ids = set()

    if not journey_text or not journey_text.strip():
        return [], stats, missing_ids

    data = None
    for attempt_text in [journey_text, journey_text.replace('\\"', '"')]:
        try:
            data = json.loads(attempt_text)
            break
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if data is None:
        return [], stats, missing_ids

    journeys = []
    for j_raw in data.get("ContinuedJourneys", []):
        title = j_raw.get("Title", "").strip()
        reason = j_raw.get("Reason", "").strip()
        products = j_raw.get("Products", [])
        stats["total_journeys"] += 1

        product_ids = []
        for product in products:
            offer_id = str(product.get("OfferId", "")).strip()
            if not offer_id:
                continue
            stats["total_offer_ids"] += 1
            if offer_id in valid_offer_ids:
                product_ids.append(offer_id)
                stats["found_offer_ids"] += 1
            else:
                stats["missing_offer_ids"] += 1
                missing_ids.add(offer_id)

        if not product_ids:
            stats["empty_product_journeys"] += 1
            continue

        journey_entry = {
            "title": title, "reason": reason, "product_ids": product_ids,
        }
        journey_type = j_raw.get("JourneyType", "").strip()
        if journey_type:
            journey_entry["journey_type"] = journey_type
        journeys.append(journey_entry)
        stats["kept_journeys"] += 1

    return journeys, stats, missing_ids


def extract_and_validate_json(raw_text):
    """Extract and validate JSON from LLM output."""
    if not raw_text or not raw_text.strip():
        return ""
    text = raw_text.strip()
    # Strip thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
    except json.JSONDecodeError:
        pass
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
# UserHistory -> ReadableUserEvents conversion
# =============================================================================

def classify_vertical(vertical):
    """Classify a Vertical/Type string into (action, source)."""
    v = vertical.strip().lower()
    if "query" in v:
        return "Searched", "query"
    elif "click" in v:
        return "Clicked", "offerid"
    elif "pagetitle" in v:
        return "Browsed", "pagetitle"
    else:
        return "Browsed", "pagetitle"


def format_time_ago(event_ts, reference_ts):
    """Format a timestamp as 'X days/hours/minutes ago'."""
    diff_seconds = max(reference_ts - event_ts, 0)
    diff_days = diff_seconds // 86400
    diff_hours = diff_seconds // 3600
    diff_minutes = diff_seconds // 60
    if diff_days > 0:
        return f"{diff_days} days ago"
    elif diff_hours > 0:
        return f"{diff_hours} hours ago"
    elif diff_minutes > 0:
        return f"{diff_minutes} minutes ago"
    else:
        return "0 minutes ago"


def user_history_to_readable_events(user_history_json, reference_ts=None):
    """Convert UserHistory JSON to ReadableUserEvents format.

    UserHistory is a JSON array of objects:
      [{"TimeStamp": int, "GlobalOfferId": str, "PageTitle": str,
        "Query": str, "Type": str}, ...]

    Returns:
        Tuple of (readable_text with #N# separators, events_list).
    """
    if not user_history_json or not user_history_json.strip():
        return "", []

    try:
        history = json.loads(user_history_json)
    except (json.JSONDecodeError, TypeError):
        return "", []

    if not isinstance(history, list):
        return "", []

    # Determine reference timestamp if not provided
    if reference_ts is None:
        # Use the most recent event timestamp + 1 day as reference
        timestamps = [e.get("TimeStamp", 0) for e in history
                      if isinstance(e.get("TimeStamp"), (int, float)) and e.get("TimeStamp", 0) > 0]
        if timestamps:
            reference_ts = max(timestamps) + 86400
        else:
            reference_ts = int(time.time())

    readable_parts = []
    events_list = []
    idx = 0

    for entry in history:
        if not isinstance(entry, dict):
            continue

        ts = entry.get("TimeStamp", 0)
        if not isinstance(ts, (int, float)) or ts <= 0:
            continue

        vertical = entry.get("Type", "")
        action, source = classify_vertical(vertical)

        # Determine content based on source
        if source == "query":
            content = entry.get("Query", "")
        elif source == "offerid":
            content = entry.get("PageTitle", "")
        else:
            content = entry.get("PageTitle", "")

        if not content or not content.strip():
            continue

        content = content.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
        time_ago = format_time_ago(int(ts), reference_ts)

        idx += 1
        readable = f"{idx} | {time_ago} | {action} | {content}"
        readable_parts.append(readable)
        events_list.append(f"{time_ago} | {action} | {content}")

    readable_text = "#N#".join(readable_parts)
    return readable_text, events_list


# =============================================================================
# TID <-> GlobalOfferId Mapping
# =============================================================================

def load_item_titles(item_file):
    """Load item titles from merged_clean_item JSON file."""
    print(f"  Loading item titles from: {item_file}")
    id2title = {}
    with open(item_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item_id, item_data in data.items():
        if isinstance(item_data, dict) and "title" in item_data:
            id2title[str(item_id)] = item_data["title"]
    print(f"    Loaded titles for {len(id2title):,} items")
    return id2title


def create_reverse_mapping(original_dict):
    """Create reverse mapping and word-level index for fuzzy matching."""
    reverse_mapping = {}
    word_to_keys = defaultdict(list)
    normalized_key_map = {}
    sorted_key_map = {}

    for key_str, ids in original_dict.items():
        words = [word.strip().lower() for word in key_str.split(",")]
        reverse_mapping[key_str] = {"words": words, "ids": ids}
        for word in words:
            word_to_keys[word].append(key_str)
        norm_key = ",".join(words)
        if norm_key not in normalized_key_map:
            normalized_key_map[norm_key] = key_str
        sorted_key = ",".join(sorted(words))
        if sorted_key not in sorted_key_map:
            sorted_key_map[sorted_key] = key_str

    return reverse_mapping, word_to_keys, normalized_key_map, sorted_key_map


def get_iid_by_tid(tid_words, tid2item_id, reverse_mapping, word_to_keys,
                   normalized_key_map, sorted_key_map):
    """Map a single TID (list of 7 words) to GlobalOfferIds."""
    tid_key = ",".join(tid_words)

    if tid_key in tid2item_id:
        return list(tid2item_id[tid_key]), "exact", len(tid_words), 0.0

    words_lower = [w.strip().lower() for w in tid_words]
    tid_key_normalized = ",".join(words_lower)
    if tid_key_normalized in normalized_key_map:
        orig_key = normalized_key_map[tid_key_normalized]
        return list(tid2item_id[orig_key]), "exact", len(tid_words), 0.0

    tid_key_sorted = ",".join(sorted(words_lower))
    if tid_key_sorted in sorted_key_map:
        orig_key = sorted_key_map[tid_key_sorted]
        return list(tid2item_id[orig_key]), "exact", len(tid_words), 0.0

    first_word = words_lower[0]
    if first_word not in word_to_keys:
        return [], "none", 0, 0.0

    best_prefix_len = 0
    best_key = None
    for candidate_key in word_to_keys[first_word]:
        cand_words = reverse_mapping[candidate_key]["words"]
        prefix_len = 0
        for qi, qw in enumerate(words_lower):
            if qi < len(cand_words) and cand_words[qi] == qw:
                prefix_len += 1
            else:
                break
        if prefix_len > best_prefix_len:
            best_prefix_len = prefix_len
            best_key = candidate_key

    if best_key is not None and best_prefix_len > 0:
        iids = reverse_mapping[best_key]["ids"]
        return iids[:1], "fuzzy", best_prefix_len, float(best_prefix_len)

    return [], "none", 0, 0.0


# =============================================================================
# Journey JSON Parsing & Product Mapping
# =============================================================================

def parse_journey_json(raw_output):
    """Parse model output into ContinuedJourneys structure."""
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


def map_journey_products(journey_data, tid2item_id, reverse_mapping,
                         word_to_keys, normalized_key_map, sorted_key_map,
                         id2title=None, fuzzy_score_threshold=0.0):
    """Map all ProductTIDs in a parsed journey to GlobalOfferIds."""
    stats = {
        "total_products": 0, "exact_matches": 0, "fuzzy_matches": 0,
        "no_matches": 0, "fuzzy_filtered": 0, "journeys_dropped": 0,
        "fuzzy_matched_words": [], "fuzzy_best_scores": [],
    }

    if journey_data is None:
        return None, stats

    mapped = {"ContinuedJourneys": []}

    for journey in journey_data.get("ContinuedJourneys", []):
        mapped_journey = {
            "Title": journey.get("Title", ""),
            "Reason": journey.get("Reason", ""),
            "Products": [],
        }
        used_gids_in_journey = set()

        for tid_words in journey.get("ProductTIDs", []):
            stats["total_products"] += 1
            if not isinstance(tid_words, list) or len(tid_words) == 0:
                stats["no_matches"] += 1
                continue

            iids, match_type, matched_words, best_score = get_iid_by_tid(
                tid_words, tid2item_id, reverse_mapping, word_to_keys,
                normalized_key_map, sorted_key_map,
            )

            if match_type == "exact":
                stats["exact_matches"] += 1
            elif match_type == "fuzzy":
                stats["fuzzy_matches"] += 1
                stats["fuzzy_matched_words"].append(matched_words)
                stats["fuzzy_best_scores"].append(best_score)
                if fuzzy_score_threshold > 0 and best_score < fuzzy_score_threshold:
                    stats["fuzzy_filtered"] += 1
                    continue
            else:
                stats["no_matches"] += 1
                continue

            chosen_gid = None
            for candidate_gid in (iids or []):
                if candidate_gid not in used_gids_in_journey:
                    chosen_gid = candidate_gid
                    break
            if chosen_gid is None and iids:
                chosen_gid = iids[0]
            if chosen_gid is not None:
                used_gids_in_journey.add(chosen_gid)

            mapped_journey["Products"].append({
                "TID": tid_words,
                "GlobalOfferIds": [chosen_gid] if chosen_gid else [],
                "match_type": match_type,
                "title": id2title.get(str(chosen_gid), "") if (chosen_gid and id2title) else "",
            })

        # Dedup products by GlobalOfferId
        seen_gids = set()
        deduped_products = []
        for p in mapped_journey["Products"]:
            gid = p["GlobalOfferIds"][0] if p["GlobalOfferIds"] else None
            if gid is not None and gid in seen_gids:
                stats["products_deduped"] = stats.get("products_deduped", 0) + 1
                continue
            if gid is not None:
                seen_gids.add(gid)
            deduped_products.append(p)
        mapped_journey["Products"] = deduped_products

        if mapped_journey["Products"]:
            mapped["ContinuedJourneys"].append(mapped_journey)
        else:
            stats["journeys_dropped"] += 1

    # Dedup journeys by title
    seen_titles = set()
    deduped_journeys = []
    for j in mapped["ContinuedJourneys"]:
        title_key = j["Title"].strip().lower()
        if title_key in seen_titles:
            stats["journeys_title_deduped"] = stats.get("journeys_title_deduped", 0) + 1
            continue
        seen_titles.add(title_key)
        deduped_journeys.append(j)
    mapped["ContinuedJourneys"] = deduped_journeys

    if not mapped["ContinuedJourneys"]:
        return None, stats

    return mapped, stats


# =============================================================================
# Profile Generation via vLLM
# =============================================================================

# Profile instruction matching SFT training format (variant 0 from
# sft_enus_gpt52_1000_forInfer_promptv2.json)
PROFILE_INSTRUCTION = (
    "\nAnalyze the user's shopping event history and generate a JSON shopping "
    "profile capturing their medium-to-long-term personal preferences.\n\n"
    "## Rules\n"
    "- Extract preferences ONLY from the user's own repeated patterns "
    "(2+ events in same category/brand). Ignore isolated single events.\n"
    "- Ignore: gifts/purchases for others, one-time needs, non-product pages "
    "(sale banners, cart pages, homepage, navigation elements).\n"
    "- Exclude categories: digital/software, apps, subscriptions, "
    "medical/health/vitamin supplements,"
    "weapons, firearms, hunting gear, alcohol, tobacco, vaping, "
    "vehicles (car accessories OK), services,"
    "restaurants, raw food, everyday commodities (batteries, cables, trash bags), "
    "funeral products.\n"
    "- Signal strength: Clicked > Browsed > Searched.\n"
    "- When evidence is weak or unclear, return \"general\" (for string fields) "
    "or empty list (for array fields).\n\n"
    "## Output\n"
    "Return ONLY a JSON object, no explanation:\n"
    '{"userShoppingProfile": {"shoppingGenderPreference": "", '
    '"categoryPreferences": [], "brandPreferences": [], '
    '"retailerPreferences": [], "priceSensitivity": "", '
    '"fashionStyle": [], "fashionFit": [], "shoppingValues": [], '
    '"contextualShoppingInterests": [], "suggestedRelatedBrands": []}}\n\n'
    "## User Shopping Events\n"
    "Each event: index | time_ago | action_type | product_description\n"
)


def build_profile_input(events_list, max_events=100):
    """Build profile generation input from event list."""
    truncated = events_list[:max_events]
    lines = []
    for idx, event in enumerate(truncated, 1):
        if len(event) > 150:
            event = event[:150] + "..."
        lines.append(f"{idx} | {event}")
    lines.append("")
    lines.append("Generate the user's shopping profile:")
    return "\n".join(lines)


def run_profile_generation_vllm(users_with_events, profile_model_path,
                                num_gpus, gpu_memory_utilization,
                                max_model_len, max_tokens,
                                enable_thinking=False):
    """Generate shopping profiles for all users using vLLM.

    Args:
        users_with_events: List of dicts with "UserId" and "events_list".

    Returns:
        Dict mapping UserId -> profile JSON string (or "").
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n  Initializing vLLM for profile generation ...")
    print(f"    Model: {profile_model_path}")
    print(f"    GPUs: {num_gpus}, mem: {gpu_memory_utilization}")

    tokenizer = AutoTokenizer.from_pretrained(
        profile_model_path, trust_remote_code=True,
    )

    # Build prompts
    prompts = []
    uids = []
    for ud in users_with_events:
        instruction = PROFILE_INSTRUCTION
        input_text = build_profile_input(ud["events_list"], max_events=100)
        content = instruction + "\n" + input_text
        messages = [{"role": "user", "content": content}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(formatted)
        uids.append(ud["UserId"])

    print(f"    Built {len(prompts)} profile prompts")

    llm = LLM(
        model=profile_model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        seed=SEED,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time
    throughput = len(prompts) / elapsed if elapsed > 0 else 0
    print(f"    Inference done in {elapsed:.1f}s ({throughput:.1f} items/s)")

    # Parse results
    profile_map = {}
    valid = 0
    invalid = 0
    for uid, output in zip(uids, outputs):
        raw = output.outputs[0].text.strip()
        clean = extract_and_validate_json(raw)
        if clean:
            profile_map[uid] = clean
            valid += 1
        else:
            invalid += 1

    print(f"    Valid profiles: {valid:,}")
    print(f"    Invalid/empty:  {invalid:,}")

    # Free GPU memory before loading journey model
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return profile_map


# =============================================================================
# Test Data Loading
# =============================================================================

def load_llm_output_tsv(filepath):
    """Load a previous llm_output.tsv and reconstruct user_data."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_ALL)
        for row in reader:
            restored = {}
            for k, v in row.items():
                if isinstance(v, str):
                    restored[k] = v.replace("\\n", "\n").replace("\\r", "\r")
                else:
                    restored[k] = v
            rows.append(restored)

    print(f"  Loaded {len(rows):,} rows from previous llm_output: {filepath}")

    users_with_profile = []
    llm_rows = []
    llm_stats = {
        "total_users": len(rows), "json_parse_success": 0,
        "json_parse_fail": 0, "total_products": 0, "exact_matches": 0,
        "fuzzy_matches": 0, "no_matches": 0, "users_with_all_fields": 0,
        "per_user_exact_ratios": [],
    }

    for row in rows:
        uid = row.get("UserId", "")
        user_events = row.get("UserSignals", "")
        user_events_readable = row.get("ReadableUserSignals", "")
        user_profile = row.get("UserProfile", "")
        gt_json = row.get("RawShoppingJourneys", "")

        events_list = []
        if user_events_readable:
            for line in user_events_readable.split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" | ", 1)
                if len(parts) == 2:
                    events_list.append(parts[1])
                else:
                    events_list.append(line)

        gt_data = parse_journey_json(gt_json)
        gt_journeys = []
        if gt_data:
            for j in gt_data.get("ContinuedJourneys", []):
                gt_journeys.append({
                    "title": j.get("Title", ""),
                    "reason": j.get("Reason", ""),
                    "product_ids": j.get("ProductIds", []),
                })

        total_prods = sum(len(j.get("product_ids", [])) for j in gt_journeys)
        if gt_journeys:
            llm_stats["json_parse_success"] += 1
        else:
            llm_stats["json_parse_fail"] += 1
        llm_stats["total_products"] += total_prods
        llm_stats["exact_matches"] += total_prods
        if total_prods > 0:
            llm_stats["per_user_exact_ratios"].append(1.0)
        if uid and user_events and gt_journeys:
            llm_stats["users_with_all_fields"] += 1

        ud = {
            "UserId": uid, "UserSignals": user_events,
            "ReadableUserSignals": user_events_readable,
            "events_list": events_list, "UserProfile": user_profile,
            "ground_truth_json": gt_json, "ground_truth_journeys": gt_journeys,
            "num_journeys": len(gt_journeys),
        }
        if user_profile:
            users_with_profile.append(ud)

        llm_rows.append({
            "UserId": uid, "UserSignals": user_events,
            "ReadableUserSignals": user_events_readable,
            "UserProfile": user_profile,
            "RawShoppingJourneys": gt_json,
            "ShoppingJourneys": row.get("ShoppingJourneys", gt_json),
        })

    print(f"  Users with valid profile: {len(users_with_profile):,}")

    return users_with_profile, llm_rows, llm_stats


def read_test_tsv(filepath):
    """Read the test TSV file. Handles missing ReadableUserEvents column."""
    rows = []
    seen_uids = set()

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        col_map = {name.strip(): idx for idx, name in enumerate(header)}
        has_readable = "ReadableUserEvents" in col_map

        if not has_readable:
            print(f"  [INFO] ReadableUserEvents column not found; "
                  f"will convert from UserHistory")

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

            # Convert UserHistory -> ReadableUserEvents if needed
            if not has_readable and "UserHistory" in col_map:
                uh = row_dict.get("UserHistory", "")
                readable_text, _ = user_history_to_readable_events(uh)
                row_dict["ReadableUserEvents"] = readable_text

            rows.append(row_dict)

    return rows


# =============================================================================
# Prompt Construction for Journey Tasks
# =============================================================================

def make_journey_instruction(task, num_journeys, min_products):
    """Create instruction + prompt_line matching s3's format."""
    if task == "event2journey":
        opening = (f"Based on the user's shopping event history, predict "
                   f"{num_journeys} shopping journey(s) the user is likely to pursue.")
    else:
        opening = (f"Based on the user's shopping profile and shopping event history, predict "
                   f"{num_journeys} shopping journey(s) the user is likely to pursue.")

    instruction = (
        f"{opening}"
        f" Each journey represents a different product category."
        f" Each journey has a short, engaging title, a brief user-centric reason"
        f" referencing the user's history, and at least {min_products} recommended products"
        f" as text IDs (7 slots each)."
        f" Products within each journey should cover different brands, styles, use cases"
        f" and subcategories -- avoid recommending near-identical items."
        f' Output JSON:'
        f' {{"ContinuedJourneys":[{{"Title":"...","Reason":"...",'
        f'"ProductTIDs":[["s1","s2","s3","s4","s5","s6","s7"],...]}},...]}}.'
    )

    jword = "journey" if num_journeys == 1 else "journeys"
    prompt_line = (f"Predict the user's shopping journeys, "
                   f"exactly {num_journeys} {jword}, "
                   f"at least {min_products} products in each journey:")

    return instruction, prompt_line


def build_event2journey_input(events, max_events=50, prompt_line=None):
    """Build event2journey input text from event list."""
    final_prompt = prompt_line or "Predict the user's shopping journeys:"
    truncated = events[:max_events]
    lines = ["User Event History:"]
    for idx, event in enumerate(truncated, 1):
        if len(event) > 150:
            event = event[:150] + "..."
        lines.append(f"{idx} | {event}")
    lines.append("")
    lines.append(final_prompt)
    return "\n".join(lines)


def build_profile2journey_input(profile_text, events, max_recent_events=10,
                                prompt_line=None):
    """Build profile2journey input text from profile and events."""
    final_prompt = prompt_line or "Predict the user's shopping journeys:"
    recent = events[:max_recent_events]
    lines = ["User Shopping Profile:", profile_text, ""]
    lines.append("Recent Shopping Events:")
    for idx, event in enumerate(recent, 1):
        if len(event) > 150:
            event = event[:150] + "..."
        lines.append(f"{idx} | {event}")
    lines.append("")
    lines.append(final_prompt)
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

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
    )

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time

    throughput = len(prompts) / elapsed if elapsed > 0 else 0
    print(f"  Inference done in {elapsed:.1f}s ({throughput:.1f} items/s)")

    return [output.outputs[0].text.strip() for output in outputs]


def build_chat_prompts(instructions_and_inputs, tokenizer, task_type,
                       enable_thinking=False):
    """Build chat-formatted prompts for vLLM."""
    prompts = []
    for instruction, input_text in instructions_and_inputs:
        content = instruction + "\n" + input_text
        messages = [{"role": "user", "content": content}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(formatted)

    print(f"  Built {len(prompts)} prompts for {task_type}")
    return prompts


# =============================================================================
# Ground Truth Construction
# =============================================================================

def format_ground_truth_journey(journey_text):
    """Parse FinalJourney text and format as ground truth JSON."""
    if not journey_text or not journey_text.strip():
        return "", []

    class _AcceptAll:
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

def build_output_rows(user_data_list, model_outputs, tid2item_id,
                      reverse_mapping, word_to_keys, normalized_key_map,
                      sorted_key_map, task_type, is_ground_truth=False,
                      id2title=None, fuzzy_score_threshold=0.0):
    """Build output rows for one output file."""
    rows = []
    agg_stats = {
        "total_users": len(user_data_list), "json_parse_success": 0,
        "json_parse_fail": 0, "total_products": 0, "exact_matches": 0,
        "fuzzy_matches": 0, "no_matches": 0, "users_with_all_fields": 0,
        "per_user_exact_ratios": [],
    }

    for idx, ud in enumerate(user_data_list):
        uid = ud["UserId"]
        user_events = ud["UserSignals"]
        user_events_readable = ud["ReadableUserSignals"]
        user_profile = ud.get("UserProfile", "")

        if is_ground_truth:
            raw_output = ud.get("ground_truth_json", "")
            gt_journeys = ud.get("ground_truth_journeys", [])

            if gt_journeys:
                agg_stats["json_parse_success"] += 1
            else:
                agg_stats["json_parse_fail"] += 1

            mapped = {"ContinuedJourneys": []}
            user_total = 0
            user_has_title = 0
            for j in gt_journeys:
                mapped_j = {
                    "Title": j.get("title", ""),
                    "Reason": j.get("reason", ""),
                    "Products": [],
                }
                for gid in j.get("product_ids", []):
                    gid_str = str(gid)
                    user_total += 1
                    title = id2title.get(gid_str, "") if id2title else ""
                    if title:
                        user_has_title += 1
                    mapped_j["Products"].append({
                        "GlobalOfferIds": [gid_str],
                        "match_type": "exact",
                        "title": title,
                    })
                if mapped_j["Products"]:
                    mapped["ContinuedJourneys"].append(mapped_j)

            agg_stats["exact_matches"] += user_total
            agg_stats.setdefault("gt_has_title", 0)
            agg_stats["gt_has_title"] += user_has_title
            agg_stats.setdefault("gt_no_title", 0)
            agg_stats["gt_no_title"] += (user_total - user_has_title)
            if user_total > 0:
                agg_stats["per_user_exact_ratios"].append(1.0)
            agg_stats["total_products"] += user_total

            journey_joined = (
                json.dumps(mapped, ensure_ascii=False)
                if mapped["ContinuedJourneys"] else ""
            )
            if uid and user_events and mapped["ContinuedJourneys"]:
                agg_stats["users_with_all_fields"] += 1
        else:
            raw_output = model_outputs[idx] if model_outputs else ""

            journey_data = parse_journey_json(raw_output)
            if journey_data is not None:
                agg_stats["json_parse_success"] += 1
            else:
                agg_stats["json_parse_fail"] += 1

            mapped_data, map_stats = map_journey_products(
                journey_data, tid2item_id, reverse_mapping, word_to_keys,
                normalized_key_map, sorted_key_map, id2title,
                fuzzy_score_threshold=fuzzy_score_threshold,
            )
            agg_stats["total_products"] += map_stats["total_products"]
            agg_stats["exact_matches"] += map_stats["exact_matches"]
            agg_stats["fuzzy_matches"] += map_stats["fuzzy_matches"]
            agg_stats["no_matches"] += map_stats["no_matches"]
            agg_stats["fuzzy_filtered"] = agg_stats.get("fuzzy_filtered", 0) + map_stats.get("fuzzy_filtered", 0)
            agg_stats["journeys_dropped"] = agg_stats.get("journeys_dropped", 0) + map_stats.get("journeys_dropped", 0)
            agg_stats["products_deduped"] = agg_stats.get("products_deduped", 0) + map_stats.get("products_deduped", 0)
            agg_stats["journeys_title_deduped"] = agg_stats.get("journeys_title_deduped", 0) + map_stats.get("journeys_title_deduped", 0)
            agg_stats.setdefault("fuzzy_matched_words", []).extend(
                map_stats.get("fuzzy_matched_words", []))
            agg_stats.setdefault("fuzzy_best_scores", []).extend(
                map_stats.get("fuzzy_best_scores", []))

            user_total = map_stats["total_products"]
            user_exact = map_stats["exact_matches"]
            if user_total > 0:
                agg_stats["per_user_exact_ratios"].append(user_exact / user_total)

            if uid and user_events and raw_output and mapped_data:
                agg_stats["users_with_all_fields"] += 1
            if not mapped_data:
                agg_stats["users_no_valid_result"] = agg_stats.get("users_no_valid_result", 0) + 1

            journey_joined = (
                json.dumps(mapped_data, ensure_ascii=False) if mapped_data else ""
            )

        rows.append({
            "UserId": uid, "UserSignals": user_events,
            "ReadableUserSignals": user_events_readable,
            "UserProfile": user_profile,
            "RawShoppingJourneys": raw_output,
            "ShoppingJourneys": journey_joined,
        })

    return rows, agg_stats


def compute_diversity_stats(rows, label):
    """Compute and print diversity statistics for output rows."""
    print(f"\n  --- {label} Diversity & Size Statistics ---")

    all_journeys_per_user = []
    all_products_per_journey = []
    all_unique_products_per_user = []
    all_unique_titles_per_user = []
    all_journey_titles = []
    total_users_with_data = 0

    for row in rows:
        joined = row.get("ShoppingJourneys", "")
        if not joined:
            continue
        try:
            data = json.loads(joined)
        except (json.JSONDecodeError, TypeError):
            continue
        journeys = data.get("ContinuedJourneys", [])
        if not journeys:
            continue

        total_users_with_data += 1
        all_journeys_per_user.append(len(journeys))
        user_gids = []
        journey_titles_this_user = []

        for j in journeys:
            title = j.get("Title", "")
            journey_titles_this_user.append(title.strip().lower())
            all_journey_titles.append(title.strip().lower())
            products = j.get("Products", [])
            product_ids = j.get("ProductIds", [])
            if product_ids and not products:
                all_products_per_journey.append(len(product_ids))
                user_gids.extend(str(pid) for pid in product_ids)
                continue
            all_products_per_journey.append(len(products))
            for p in products:
                gids = p.get("GlobalOfferIds", [])
                if gids:
                    user_gids.append(str(gids[0]))

        all_unique_products_per_user.append(len(set(user_gids)))
        all_unique_titles_per_user.append(len(set(journey_titles_this_user)))

    if not total_users_with_data:
        print(f"    No users with valid data.")
        return {}

    print(f"    Users with data:          {total_users_with_data:>10,}")

    j_arr = np.array(all_journeys_per_user)
    print(f"    Journeys per user:")
    print(f"      Mean: {j_arr.mean():.2f}  Median: {np.median(j_arr):.1f}  "
          f"Min: {j_arr.min()}  Max: {j_arr.max()}")

    jt_arr = np.array(all_unique_titles_per_user)
    print(f"    Unique journey titles per user:")
    print(f"      Mean: {jt_arr.mean():.2f}  Median: {np.median(jt_arr):.1f}  "
          f"Min: {jt_arr.min()}  Max: {jt_arr.max()}")

    if all_products_per_journey:
        p_arr = np.array(all_products_per_journey)
        print(f"    Products per journey:")
        print(f"      Mean: {p_arr.mean():.2f}  Median: {np.median(p_arr):.1f}  "
              f"Min: {p_arr.min()}  Max: {p_arr.max()}")

    if all_unique_products_per_user:
        up_arr = np.array(all_unique_products_per_user)
        print(f"    Unique products per user:")
        print(f"      Mean: {up_arr.mean():.2f}  Median: {np.median(up_arr):.1f}  "
              f"Min: {up_arr.min()}  Max: {up_arr.max()}")

    unique_global = len(set(all_journey_titles))
    print(f"    Global unique journey titles: {unique_global:,} / {len(all_journey_titles):,} "
          f"({unique_global / max(len(all_journey_titles), 1) * 100:.1f}%)")

    def _arr_stats(arr):
        return {"mean": round(float(arr.mean()), 2),
                "median": round(float(np.median(arr)), 1),
                "min": int(arr.min()), "max": int(arr.max())}

    diversity = {
        "users_with_data": total_users_with_data,
        "journeys_per_user": _arr_stats(j_arr),
        "unique_journey_titles_per_user": _arr_stats(jt_arr),
        "global_unique_journey_titles": unique_global,
        "global_total_journey_titles": len(all_journey_titles),
        "global_journey_title_uniqueness_pct": round(
            unique_global / max(len(all_journey_titles), 1) * 100, 1),
    }
    if all_products_per_journey:
        diversity["products_per_journey"] = _arr_stats(np.array(all_products_per_journey))
    if all_unique_products_per_user:
        diversity["unique_products_per_user"] = _arr_stats(np.array(all_unique_products_per_user))

    return diversity


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
            clean_row = {}
            for k, v in row.items():
                if isinstance(v, str):
                    clean_row[k] = v.replace("\n", "\\n").replace("\r", "\\r")
                else:
                    clean_row[k] = v
            writer.writerow(clean_row)
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
    gt_has_title = stats.get('gt_has_title', 0)
    gt_no_title = stats.get('gt_no_title', 0)
    if gt_has_title or gt_no_title:
        gt_total = gt_has_title + gt_no_title
        print(f"    GT products w/ title:     {gt_has_title:>10,} "
              f"({gt_has_title / max(gt_total, 1) * 100:.1f}%)")
        print(f"    GT products w/o title:    {gt_no_title:>10,} "
              f"({gt_no_title / max(gt_total, 1) * 100:.1f}%)")
    fuzzy_filtered = stats.get('fuzzy_filtered', 0)
    journeys_dropped = stats.get('journeys_dropped', 0)
    users_no_valid = stats.get('users_no_valid_result', 0)
    if fuzzy_filtered or journeys_dropped or users_no_valid:
        print(f"    --- Threshold filtering ---")
        print(f"    Fuzzy products filtered:  {fuzzy_filtered:>10,}")
        print(f"    Journeys dropped (empty): {journeys_dropped:>10,}")
        print(f"    Users w/o valid result:   {users_no_valid:>10,}")
    products_deduped = stats.get('products_deduped', 0)
    journeys_title_deduped = stats.get('journeys_title_deduped', 0)
    if products_deduped or journeys_title_deduped:
        print(f"    --- Deduplication ---")
        print(f"    Products deduped (by GID):{products_deduped:>10,}")
        print(f"    Journeys deduped (title): {journeys_title_deduped:>10,}")

    ratios = stats.get("per_user_exact_ratios", [])
    if ratios:
        arr = np.array(ratios)
        print(f"    Per-user exact match rate:")
        print(f"      Mean:   {arr.mean():.2%}")
        print(f"      Median: {np.median(arr):.2%}")
        print(f"      Min:    {arr.min():.2%}")
        print(f"      Max:    {arr.max():.2%}")

    fuzzy_words = stats.get("fuzzy_matched_words", [])
    fuzzy_scores = stats.get("fuzzy_best_scores", [])
    if fuzzy_words:
        words_arr = np.array(fuzzy_words)
        scores_arr = np.array(fuzzy_scores)
        print(f"    Fuzzy match details ({len(fuzzy_words)} products):")
        print(f"      Matched words (out of 7):")
        print(f"        Mean:   {words_arr.mean():.2f}")
        print(f"        Median: {np.median(words_arr):.1f}")
        print(f"        Min:    {words_arr.min()}")
        print(f"        Max:    {words_arr.max()}")
        for n_words in range(0, 8):
            count = int((words_arr == n_words).sum())
            if count > 0:
                print(f"        {n_words} words matched: {count:>6,} "
                      f"({count / len(fuzzy_words) * 100:.1f}%)")
        print(f"      Best candidate score:")
        print(f"        Mean:   {scores_arr.mean():.3f}")
        print(f"        Median: {np.median(scores_arr):.3f}")
        for p in [10, 25, 50, 75, 90]:
            print(f"        P{p}:    {np.percentile(scores_arr, p):.3f}")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone evaluation of journey prediction tasks"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/"
                "journey_v3_cp1200/lora_journey_v3/sft_4gpus_lr5e-5_batch12_"
                "gradacc2_lorarank32_cut4096_packing_enablethinkingfalse/"
                "checkpoint-8000-merged",
        help="Path to the trained journey SFT model checkpoint",
    )
    parser.add_argument(
        "--profile_model_path", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/"
                "Model/UserProfile_keppsuggestionBrandScore1_Ckpt1834/",
        help="Path to the profile generation model checkpoint",
    )
    parser.add_argument(
        "--test_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/0128_0301/TestData/ranker_results_5000_cleaned.tsv",
        help="Path to test TSV file",
    )
    parser.add_argument(
        "--tid2item_id_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260324/sft_data_v4/item_id2tid/tid2item_id.json",
        help="Path to tid2item_id.json",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/eval_results/standalone_eval/",
        help="Output directory",
    )
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_events", type=int, default=100)
    parser.add_argument("--max_recent_events", type=int, default=100)

    # vLLM args (journey model)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--tokenizer_path", type=str, default=None)

    # vLLM args (profile model)
    parser.add_argument("--profile_num_gpus", type=int, default=None,
                        help="GPUs for profile model (default: same as --num_gpus)")
    parser.add_argument("--profile_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--profile_max_model_len", type=int, default=8192)
    parser.add_argument("--profile_max_tokens", type=int, default=2000)

    parser.add_argument(
        "--item_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260324/raw_data/item.json",
        help="Path to item JSON for product titles",
    )
    parser.add_argument("--fuzzy_score_threshold", type=float, default=7.0)
    parser.add_argument(
        "--llm_output_file", type=str, default=None,
        help="Path to a previous llm_output.tsv to skip steps 1-6",
    )
    parser.add_argument("--debug", action="store_true", default=False)
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
    profile_num_gpus = args.profile_num_gpus if args.profile_num_gpus is not None else num_gpus

    print("=" * 70)
    print("Step 5 Journey: Standalone Evaluation")
    print("=" * 70)
    print(f"  Journey model:  {args.model_path}")
    print(f"  Profile model:  {args.profile_model_path}")
    print(f"  Test file:      {args.test_file}")
    print(f"  GPUs (journey): {num_gpus}")
    print(f"  GPUs (profile): {profile_num_gpus}")
    print(f"  Sample N:       {args.sample_n}")
    print(f"  Seed:           {args.seed}")

    # =========================================================================
    # Check if we can reuse a previous llm_output file
    # =========================================================================
    if args.llm_output_file and os.path.isfile(args.llm_output_file):
        print()
        print("=" * 70)
        print("Loading previous llm_output (skipping steps 1-6)")
        print("=" * 70)

        users_with_profile, llm_rows, llm_stats = load_llm_output_tsv(
            args.llm_output_file
        )
        print_stats("llm_output (from previous run)", llm_stats)

        if not users_with_profile:
            print("ERROR: No users with valid profiles in llm_output file.")
            return

        if not args.debug:
            print(f"\n  Loading TID mapping: {args.tid2item_id_file}")
            with open(args.tid2item_id_file, "r", encoding="utf-8") as f:
                tid2item_id = json.load(f)
            reverse_mapping, word_to_keys, normalized_key_map, sorted_key_map = \
                create_reverse_mapping(tid2item_id)
            print(f"    Unique TIDs: {len(tid2item_id):,}")

            id2title = None
            if args.item_file and os.path.isfile(args.item_file):
                id2title = load_item_titles(args.item_file)

    else:
        # =================================================================
        # Step 1: Read test TSV file
        # =================================================================
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

        # =================================================================
        # Step 2: Sample N user_ids
        # =================================================================
        print()
        print("=" * 70)
        print("Step 2: Sampling user_ids")
        print("=" * 70)

        sample_n = min(args.sample_n, len(test_rows))
        sampled_rows = random.sample(test_rows, sample_n)
        sampled_rows.sort(key=lambda r: r.get("UserId", ""))
        print(f"  Sampled {len(sampled_rows):,} users from {len(test_rows):,}")

        # =================================================================
        # Step 3: Process user data
        # =================================================================
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
            user_events_raw = row.get("UserHistory", "")

            readable_raw = row.get("ReadableUserEvents", "")
            events_list, _ = parse_readable_user_events(readable_raw)

            # Fallback: convert UserHistory if ReadableUserEvents is empty
            if not events_list and user_events_raw:
                _, events_list = user_history_to_readable_events(user_events_raw)

            if not events_list:
                no_events += 1

            readable_lines = []
            for idx, event in enumerate(events_list, 1):
                if len(event) > 150:
                    event = event[:150] + "..."
                readable_lines.append(f"{idx} | {event}")
            user_events_readable = "\n".join(readable_lines)

            final_journey_text = row.get("FinalJourney", "")
            gt_json, gt_journeys = format_ground_truth_journey(final_journey_text)
            if gt_journeys:
                gt_parse_success += 1
            else:
                gt_parse_fail += 1
                if not final_journey_text:
                    no_final_journey += 1

            user_data.append({
                "UserId": uid,
                "UserSignals": user_events_raw,
                "ReadableUserSignals": user_events_readable,
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

        # =================================================================
        # Step 4: Generate shopping profiles via vLLM
        # =================================================================
        print()
        print("=" * 70)
        print("Step 4: Generating shopping profiles via vLLM")
        print("=" * 70)

        profile_map = run_profile_generation_vllm(
            users_with_events=user_data,
            profile_model_path=args.profile_model_path,
            num_gpus=profile_num_gpus,
            gpu_memory_utilization=args.profile_gpu_memory_utilization,
            max_model_len=args.profile_max_model_len,
            max_tokens=args.profile_max_tokens,
            enable_thinking=args.enable_thinking,
        )

        for ud in user_data:
            ud["UserProfile"] = profile_map.get(ud["UserId"], "")

        users_with_profile = [ud for ud in user_data if ud["UserProfile"]]
        print(f"\n  Users with valid profile: {len(users_with_profile):,}")
        print(f"  Users without valid profile: {len(user_data) - len(users_with_profile):,}")

        if not users_with_profile:
            print("ERROR: No users with valid profiles.")
            return

        # =================================================================
        # Step 5: Load TID mapping
        # =================================================================
        print()
        print("=" * 70)
        print("Step 5: Loading TID mapping")
        print("=" * 70)

        print(f"  Loading: {args.tid2item_id_file}")
        with open(args.tid2item_id_file, "r", encoding="utf-8") as f:
            tid2item_id = json.load(f)
        reverse_mapping, word_to_keys, normalized_key_map, sorted_key_map = \
            create_reverse_mapping(tid2item_id)
        print(f"    Unique TIDs: {len(tid2item_id):,}")
        print(f"    Unique words in index: {len(word_to_keys):,}")

        id2title = None
        if args.item_file and os.path.isfile(args.item_file):
            id2title = load_item_titles(args.item_file)

        # =================================================================
        # Step 6: Build ground truth output (llm_output)
        # =================================================================
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
            normalized_key_map=normalized_key_map,
            sorted_key_map=sorted_key_map,
            task_type="event2journey",
            is_ground_truth=True,
            id2title=id2title,
            fuzzy_score_threshold=0.0,
        )
        print_stats("llm_output (ground truth)", llm_stats)

        columns = [
            "UserId", "UserSignals", "ReadableUserSignals", "UserProfile",
            "RawShoppingJourneys", "ShoppingJourneys",
        ]
        os.makedirs(args.output_dir, exist_ok=True)
        llm_file = os.path.join(args.output_dir, "llm_output.tsv")
        save_tsv(llm_rows, llm_file, columns)

    # =========================================================================
    # User event statistics
    # =========================================================================
    event_counts = np.array([len(ud.get("events_list", [])) for ud in users_with_profile])
    print()
    print("=" * 70)
    print("User Data Statistics")
    print("=" * 70)
    print(f"  Users: {len(users_with_profile):,}")
    print(f"\n  --- Events per User ---")
    print(f"    Min: {event_counts.min():>6}  P25: {int(np.percentile(event_counts, 25)):>6}  "
          f"P50: {int(np.percentile(event_counts, 50)):>6}  P75: {int(np.percentile(event_counts, 75)):>6}  "
          f"P90: {int(np.percentile(event_counts, 90)):>6}  Max: {event_counts.max():>6}  "
          f"Mean: {event_counts.mean():.1f}")

    # =========================================================================
    # Step 7: Run model inference with vLLM
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 7: Running journey model inference via vLLM")
    print("=" * 70)

    from transformers import AutoTokenizer

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True,
    )

    # --- 7a: event2journey prompts ---
    print("\n  Building event2journey prompts ...")
    e2j_instr_inputs = []
    for ud in users_with_profile:
        gt_j = ud.get("ground_truth_journeys", [])
        if gt_j:
            min_prods = min(len(j.get("product_ids", [])) for j in gt_j)
            min_prods = max(min_prods, 5)
        else:
            min_prods = 5
        instruction, prompt_line = make_journey_instruction(
            "event2journey", ud["num_journeys"], min_prods,
        )
        input_text = build_event2journey_input(
            ud["events_list"], max_events=args.max_events,
            prompt_line=prompt_line,
        )
        e2j_instr_inputs.append((instruction, input_text))

    e2j_prompts = build_chat_prompts(
        e2j_instr_inputs, tokenizer, "event2journey",
        enable_thinking=args.enable_thinking,
    )

    # --- 7b: profile2journey prompts ---
    print("  Building profile2journey prompts ...")
    p2j_instr_inputs = []
    for ud in users_with_profile:
        gt_j = ud.get("ground_truth_journeys", [])
        if gt_j:
            min_prods = min(len(j.get("product_ids", [])) for j in gt_j)
            min_prods = max(min_prods, 5)
        else:
            min_prods = 5
        instruction, prompt_line = make_journey_instruction(
            "profile2journey", ud["num_journeys"], min_prods,
        )
        input_text = build_profile2journey_input(
            ud["UserProfile"], ud["events_list"],
            max_recent_events=args.max_recent_events,
            prompt_line=prompt_line,
        )
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
    # Debug mode
    # =========================================================================
    if args.debug:
        print()
        print("=" * 70)
        print("Debug mode: saving prompt/output TSV files")
        print("=" * 70)

        os.makedirs(args.output_dir, exist_ok=True)
        debug_columns = ["UserId", "Prompt", "ModelOutput"]

        e2j_debug_rows = []
        for i, ud in enumerate(users_with_profile):
            e2j_debug_rows.append({
                "UserId": ud["UserId"],
                "Prompt": e2j_prompts[i],
                "ModelOutput": e2j_outputs[i],
            })
        save_tsv(e2j_debug_rows,
                 os.path.join(args.output_dir, "event2journey_debug.tsv"),
                 debug_columns)

        p2j_debug_rows = []
        for i, ud in enumerate(users_with_profile):
            p2j_debug_rows.append({
                "UserId": ud["UserId"],
                "Prompt": p2j_prompts[i],
                "ModelOutput": p2j_outputs[i],
            })
        save_tsv(p2j_debug_rows,
                 os.path.join(args.output_dir, "profile2journey_debug.tsv"),
                 debug_columns)

        print(f"\nDebug mode done. Saved 2 files to {args.output_dir}")
        return

    # =========================================================================
    # Step 8: Build slm_output (event2journey)
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
        normalized_key_map=normalized_key_map,
        sorted_key_map=sorted_key_map,
        task_type="event2journey",
        is_ground_truth=False,
        id2title=id2title,
        fuzzy_score_threshold=args.fuzzy_score_threshold,
    )
    print_stats("slm_output (event2journey)", slm_stats)

    # =========================================================================
    # Step 9: Build slm_output_w_profile (profile2journey)
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
        normalized_key_map=normalized_key_map,
        sorted_key_map=sorted_key_map,
        task_type="profile2journey",
        is_ground_truth=False,
        id2title=id2title,
        fuzzy_score_threshold=args.fuzzy_score_threshold,
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
        "UserId", "UserSignals", "ReadableUserSignals", "UserProfile",
        "RawShoppingJourneys", "ShoppingJourneys",
    ]

    llm_file = os.path.join(args.output_dir, "llm_output.tsv")
    save_tsv(llm_rows, llm_file, columns)

    slm_file = os.path.join(args.output_dir, "slm_output.tsv")
    save_tsv(slm_rows, slm_file, columns)

    slm_profile_file = os.path.join(args.output_dir, "slm_output_w_profile.tsv")
    save_tsv(slm_profile_rows, slm_profile_file, columns)

    llm_diversity = compute_diversity_stats(llm_rows, "llm_output (ground truth)")
    slm_diversity = compute_diversity_stats(slm_rows, "slm_output (event2journey)")
    slm_profile_diversity = compute_diversity_stats(
        slm_profile_rows, "slm_output_w_profile (profile2journey)")

    # =========================================================================
    # Step 11: Example outputs
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 11: Example outputs (first 2 users)")
    print("=" * 70)

    for idx in range(min(2, len(users_with_profile))):
        uid = users_with_profile[idx]["UserId"]
        print(f"\n--- User {idx + 1}: {uid} ---")

        gt_row = llm_rows[idx]
        gt_journey = parse_journey_json(gt_row["RawShoppingJourneys"])
        print(f"  [Ground Truth]")
        if gt_journey:
            for ji, j in enumerate(gt_journey.get("ContinuedJourneys", [])[:2]):
                print(f"    Journey {ji+1}: {j.get('Title', 'N/A')}")
                print(f"      Products: {len(j.get('ProductIds', []))}")

        slm_row = slm_rows[idx]
        print(f"  [SLM event2journey]")
        print(f"    Raw (first 200): {slm_row['RawShoppingJourneys'][:200]}")

        sp_row = slm_profile_rows[idx]
        print(f"  [SLM profile2journey]")
        print(f"    Raw (first 200): {sp_row['RawShoppingJourneys'][:200]}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    if args.llm_output_file and os.path.isfile(args.llm_output_file):
        print(f"  Loaded from previous llm_output: {args.llm_output_file}")
    else:
        print(f"  Total users in test file:       {len(test_rows):>10,}")
        print(f"  Sampled users:                  {len(sampled_rows):>10,}")
        print(f"  Valid profiles (vLLM):          {len(profile_map):>10,}")
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
        ("GT", llm_stats), ("SLM", slm_stats),
        ("SLM+Profile", slm_profile_stats),
    ]:
        ratios = stats.get("per_user_exact_ratios", [])
        if ratios:
            arr = np.array(ratios)
            print(f"  {name} per-user exact rate: "
                  f"mean={arr.mean():.2%}, median={np.median(arr):.2%}")

    summary = {
        "model_path": args.model_path,
        "profile_model_path": args.profile_model_path,
        "seed": args.seed,
        "evaluated_users": len(users_with_profile),
        "llm_stats": {k: v for k, v in llm_stats.items()
                      if k not in ("per_user_exact_ratios",
                                   "fuzzy_matched_words", "fuzzy_best_scores")},
        "slm_stats": {k: v for k, v in slm_stats.items()
                      if k not in ("per_user_exact_ratios",
                                   "fuzzy_matched_words", "fuzzy_best_scores")},
        "slm_profile_stats": {k: v for k, v in slm_profile_stats.items()
                              if k not in ("per_user_exact_ratios",
                                           "fuzzy_matched_words",
                                           "fuzzy_best_scores")},
        "llm_diversity": llm_diversity or {},
        "slm_diversity": slm_diversity or {},
        "slm_profile_diversity": slm_profile_diversity or {},
    }
    if not (args.llm_output_file and os.path.isfile(args.llm_output_file)):
        summary.update({
            "test_file": args.test_file,
            "total_test_users": len(test_rows),
            "sample_n": len(sampled_rows),
            "valid_profiles_vllm": len(profile_map),
        })
    summary_file = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary saved to: {summary_file}")

    print(f"\nDone! Evaluated {len(users_with_profile)} users on 2 tasks, "
          f"saved 3 output files to {args.output_dir}")


if __name__ == "__main__":
    main()
