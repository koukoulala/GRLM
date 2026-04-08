"""
Construct shopping journey data from journey prediction TSV and item metadata.

Reads:
  1. Shopping journey TSV files (*_cleaned.tsv) with columns:
     UserId, ReadableUserEvents, UserHistory, JourneyWithProducts, FinalJourney
  2. Item metadata JSON (from s0_init_emb.py), keyed by GlobalOfferId.
     Used to validate that product OfferIds in journeys actually exist.

Produces:
  shopping_journeys.json - keyed by UserId, each entry containing:
    - user_shopping_events: list of event strings
        (format: "time_ago | action | description")
    - journeys: list of journey dicts, each with:
        - title: journey title string
        - reason: journey reason string
        - product_ids: list of GlobalOfferId strings (validated)

Pipeline:
  1. Load item metadata JSON to build valid OfferId set
  2. Read journey prediction TSV
  3. Parse ReadableUserEvents (replace #N# with newlines, split events)
  4. Parse FinalJourney JSON: extract ContinuedJourneys with OfferIds
  5. Validate OfferIds against item metadata
  6. Deduplicate by UserId (keep first occurrence)
  7. Compute statistics and write output
"""

import argparse
import csv
import gc
import json
import multiprocessing
import os
import re
import sys
from collections import defaultdict

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)

# Try to use orjson for faster JSON parsing/writing (3-10x speedup)
try:
    import orjson
    _json_loads = orjson.loads
    _HAS_ORJSON = True
except ImportError:
    _json_loads = json.loads
    _HAS_ORJSON = False

# Pre-compiled regex patterns for hot paths
_RE_EVENT_NUMBER = re.compile(r"^\d+\s*\|\s*(.*)")
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s|]")
_RE_MULTI_SPACE = re.compile(r"\s+")

# Cap the number of tracked missing OfferIds to avoid unbounded memory
_MAX_MISSING_IDS = 10_000


# =============================================================================
# Utility Functions
# =============================================================================

def _clean_profile_json(raw):
    """Unescape multi-layer escaped profile JSON from TSV.

    TSV writers using escapechar='\\' can produce layered escaping:
      \\\\\"key\\\\\" -> after csv read -> \\\"key\\\" -> we need -> "key"
    Try to parse as JSON after progressively unescaping. If successful,
    re-serialize as clean compact JSON.
    """
    if not raw or not raw.strip():
        return raw

    # Quick check: if it already parses as valid JSON, just re-format
    try:
        obj = _json_loads(raw)
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False, separators=(',', ': '))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try progressively unescaping (up to 3 layers)
    text = raw
    for _ in range(3):
        text = text.replace('\\\\', '\x00__BS__\x00')
        text = text.replace('\\"', '"')
        text = text.replace('\x00__BS__\x00', '\\')
        try:
            obj = _json_loads(text)
            if isinstance(obj, (dict, list)):
                return json.dumps(obj, ensure_ascii=False, separators=(',', ': '))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # Fallback: return with basic cleanup
    return raw


def _normalize_event_key(event):
    """Normalize an event string for deduplication."""
    key = event.lower()
    key = _RE_NON_ALNUM.sub(" ", key)
    key = _RE_MULTI_SPACE.sub(" ", key)
    return key.strip()


def parse_readable_user_events(events_text):
    """Parse ReadableUserEvents text into a list of event strings.

    The events are separated by #N# in the raw text. Each event has
    format: "N | time_ago | action | description"
    We strip the leading event number.

    Args:
        events_text: Raw ReadableUserEvents string with #N# separators.

    Returns:
        Tuple of (deduplicated event list, total raw events before dedup).
    """
    if not events_text or not events_text.strip():
        return [], 0

    # Replace #N# with newlines
    text = events_text.replace("#N#", "\n")
    lines = text.strip().split("\n")

    raw_events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading event number: "1 | time_ago | action | desc"
        match = _RE_EVENT_NUMBER.match(line)
        if match:
            event = match.group(1).strip()
            if event:
                raw_events.append(event)

    # Deduplicate while preserving order
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

    Args:
        journey_text: Raw FinalJourney JSON string containing
            ContinuedJourneys array.
        valid_offer_ids: Set of valid GlobalOfferIds from item metadata.

    Returns:
        Tuple of (journeys_list, stats_dict, missing_ids_set):
          - journeys_list: list of journey dicts with title, reason,
            product_ids
          - stats_dict: dict with counts of found/missing OfferIds
          - missing_ids_set: set of OfferIds not found in item metadata
    """
    stats = {
        "total_offer_ids": 0,
        "found_offer_ids": 0,
        "missing_offer_ids": 0,
        "total_journeys": 0,
        "kept_journeys": 0,
        "empty_product_journeys": 0,
    }
    missing_ids = set()

    if not journey_text or not journey_text.strip():
        return [], stats, missing_ids

    # Try parsing the JSON
    data = None
    for attempt_text in [journey_text, journey_text.replace('\\"', '"')]:
        try:
            data = _json_loads(attempt_text)
            break
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if data is None:
        return [], stats, missing_ids

    continued_journeys = data.get("ContinuedJourneys", [])
    journeys = []

    for j_raw in continued_journeys:
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
            "title": title,
            "reason": reason,
            "product_ids": product_ids,
        }
        journey_type = j_raw.get("JourneyType", "").strip()
        if journey_type:
            journey_entry["journey_type"] = journey_type

        journeys.append(journey_entry)
        stats["kept_journeys"] += 1

    return journeys, stats, missing_ids


# =============================================================================
# Multiprocessing Worker
# =============================================================================

# Set in main() before Pool creation; inherited via fork copy-on-write (Linux)
_worker_valid_offer_ids = None


def _worker_init(valid_ids):
    """Safety-net initializer for non-fork contexts (e.g. spawn)."""
    global _worker_valid_offer_ids
    _worker_valid_offer_ids = valid_ids


def _process_row(row_tuple):
    """Process a single (user_id, events_text, journey_text, profile_text) tuple.

    Runs in a worker process. Returns a dict with parsed result and stats.
    """
    user_id, events_text, journey_text, profile_text = row_tuple

    user_events, raw_event_count = parse_readable_user_events(events_text)
    deduped_count = len(user_events)
    has_events = bool(user_events)

    if not journey_text:
        return {
            "user_id": user_id,
            "result": None,
            "raw_event_count": raw_event_count,
            "deduped_event_count": deduped_count,
            "has_events": has_events,
            "no_final_journey": True,
            "stats": None,
            "missing_ids": set(),
        }

    journeys, stats, missing_ids = parse_final_journey(
        journey_text, _worker_valid_offer_ids
    )

    result = None
    if has_events:
        result = {
            "user_shopping_events": user_events,
            "journeys": journeys,
        }
        if profile_text:
            result["user_profile"] = _clean_profile_json(profile_text)

    return {
        "user_id": user_id,
        "result": result,
        "raw_event_count": raw_event_count,
        "deduped_event_count": deduped_count,
        "has_events": has_events,
        "no_final_journey": False,
        "stats": stats,
        "missing_ids": missing_ids,
    }


# =============================================================================
# Data Loading — yields lightweight tuples with only needed columns
# =============================================================================

def _load_journey_rows_from_dir(prompt_results_dir, require_profile=False):
    """Load step3 *_cleaned.tsv files, yielding only needed columns as tuples.

    Yields:
        (user_id, events_text, journey_text, profile_text) tuples.
        Skips UserHistory and JourneyWithProducts entirely.
    """
    base_required = {"UserId", "ReadableUserEvents", "FinalJourney"}
    if require_profile:
        base_required.add("Profile")

    tsv_files = sorted(
        f for f in os.listdir(prompt_results_dir)
        if f.endswith("_cleaned.tsv") or f.endswith("_clean.tsv")
    )
    if not tsv_files:
        print(f"  ERROR: No *_cleaned.tsv or *_clean.tsv files found in {prompt_results_dir}")
        sys.exit(1)

    print(f"  Found {len(tsv_files)} TSV file(s) in: {prompt_results_dir}")

    total_rows = 0
    for fname in tsv_files:
        fpath = os.path.join(prompt_results_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header is None:
                print(f"    SKIP (empty): {fname}")
                continue

            header_set = set(header)
            if not base_required.issubset(header_set):
                print(f"    SKIP (missing columns): {fname}")
                print(f"      Have: {header}")
                print(f"      Need: {sorted(base_required)}")
                continue

            # Only index the columns we actually need
            uid_idx = header.index("UserId")
            events_idx = header.index("ReadableUserEvents")
            journey_idx = header.index("FinalJourney")
            profile_idx = header.index("Profile") if "Profile" in header_set else -1
            max_idx = max(uid_idx, events_idx, journey_idx,
                          profile_idx if profile_idx >= 0 else 0)

            file_rows = 0
            for row in reader:
                if len(row) <= max_idx:
                    continue
                yield (
                    row[uid_idx],
                    row[events_idx],
                    row[journey_idx],
                    row[profile_idx] if profile_idx >= 0 else "",
                )
                file_rows += 1
                total_rows += 1

            print(f"    Loaded {file_rows:>10,} rows from: {fname}")

    print(f"  Total merged rows: {total_rows:,}")


def _load_journey_rows_from_file(filepath):
    """Read single journey TSV file, yielding only needed columns as tuples.

    Yields:
        (user_id, events_text, journey_text, profile_text) tuples.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return

        header_set = set(header)
        uid_idx = header.index("UserId") if "UserId" in header_set else 0
        events_idx = (header.index("ReadableUserEvents")
                      if "ReadableUserEvents" in header_set else 1)
        journey_idx = (header.index("FinalJourney")
                       if "FinalJourney" in header_set else -1)
        if journey_idx < 0:
            print(f"  WARNING: FinalJourney column not found in {filepath}")
            return
        profile_idx = header.index("Profile") if "Profile" in header_set else -1
        max_idx = max(uid_idx, events_idx, journey_idx,
                      profile_idx if profile_idx >= 0 else 0)

        for row in reader:
            if len(row) <= max_idx:
                continue
            yield (
                row[uid_idx],
                row[events_idx],
                row[journey_idx],
                row[profile_idx] if profile_idx >= 0 else "",
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct shopping journey data from journey prediction "
                    "TSV and item metadata"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="event2journey",
        choices=["event2journey", "profile2journey"],
        help="Task type. profile2journey requires Profile column in input "
             "and outputs shopping_journeys_Profile.json (default: event2journey)",
    )
    parser.add_argument(
        "--journey_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/0128_0301/ShoppingJourney_Input_500K_His50_Final_Training_clean.tsv",
        help="Path to the shopping journey TSV file",
    )
    parser.add_argument(
        "--prompt_results_dir",
        type=str,
        #default=None,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/CookData_merged/",
        help="Directory containing step3 *_cleaned.tsv files to merge. "
             "Only files ending with '_cleaned.tsv' are loaded. "
             "When set, overrides --journey_file.",
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/raw_data/item.json",
        help="Path to item metadata JSON file (from pre_s0_combine_item_data.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260407/raw_data/",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for parallel parsing "
             "(0 = auto-detect cpu_count-1, default: 0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Chunk size for multiprocessing pool dispatch (default: 5000)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # =========================================================================
    # Step 1: Load item metadata to build valid OfferId set
    # =========================================================================
    print("=" * 70)
    print("Step 1: Loading item metadata")
    print("=" * 70)

    # Stream-parse only the top-level keys from item.json to save memory.
    # The full dict can be 50+GB; we only need the key set (~500MB).
    print(f"  Loading item keys from: {args.item_file}")
    valid_offer_ids = set()
    try:
        import ijson
        with open(args.item_file, "rb") as f:
            for key, _ in ijson.kvitems(f, ""):
                valid_offer_ids.add(key)
    except ImportError:
        # Fallback: load full dict, extract keys, free immediately
        with open(args.item_file, "r", encoding="utf-8") as f:
            item_data = json.load(f)
        valid_offer_ids = set(item_data.keys())
        del item_data
        gc.collect()
    print(f"  Loaded {len(valid_offer_ids):,} item keys")

    # =========================================================================
    # Step 2: Read TSV & deduplicate (only needed columns)
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Reading journey prediction file (selective columns)")
    print("=" * 70)

    require_profile = (args.task == "profile2journey")
    if require_profile:
        print(f"  Task: profile2journey (Profile column required)")
    else:
        print(f"  Task: event2journey")

    if args.prompt_results_dir:
        row_iter = _load_journey_rows_from_dir(args.prompt_results_dir,
                                              require_profile=require_profile)
    else:
        row_iter = _load_journey_rows_from_file(args.journey_file)

    # Pre-deduplicate by UserId (sequential, fast — avoids wasting workers)
    seen_uids = set()
    deduped_tuples = []
    total_rows_processed = 0
    duplicate_users = 0
    no_userid = 0

    for uid, events, journey, profile in row_iter:
        total_rows_processed += 1
        uid_s = uid.strip()
        if not uid_s:
            no_userid += 1
            continue
        if uid_s in seen_uids:
            duplicate_users += 1
            continue
        seen_uids.add(uid_s)
        deduped_tuples.append((uid_s, events.strip(), journey.strip(),
                               profile.strip()))

    del seen_uids
    print(f"  Total rows read:           {total_rows_processed:>12,}")
    print(f"  Rows with empty UserId:    {no_userid:>12,}")
    print(f"  Duplicate UserIds removed: {duplicate_users:>12,}")
    print(f"  Unique rows to process:    {len(deduped_tuples):>12,}")

    # =========================================================================
    # Step 3: Parallel parsing with multiprocessing
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Parsing rows with multiprocessing")
    print("=" * 70)

    num_workers = (args.num_workers if args.num_workers > 0
                   else max(1, multiprocessing.cpu_count() - 1))
    batch_size = args.batch_size
    print(f"  Workers: {num_workers}, chunksize: {batch_size}")

    # Set global before Pool so forked workers inherit via copy-on-write
    global _worker_valid_offer_ids
    _worker_valid_offer_ids = valid_offer_ids

    results = {}
    no_events = 0
    no_final_journey = 0
    filtered_no_events = 0
    entries_with_empty_journeys = 0
    total_raw_events = 0
    total_deduped_events = 0
    agg_total_offer_ids = 0
    agg_found_offer_ids = 0
    agg_missing_offer_ids = 0
    agg_total_journeys = 0
    agg_kept_journeys = 0
    agg_empty_product_journeys = 0
    missing_offer_id_set = set()
    processed_count = 0
    n_total = len(deduped_tuples)

    with multiprocessing.Pool(num_workers) as pool:
        for r in pool.imap_unordered(_process_row, deduped_tuples,
                                     chunksize=batch_size):
            processed_count += 1
            if processed_count % 50_000 == 0:
                print(f"    Processed {processed_count:>12,} / {n_total:,}")

            total_raw_events += r["raw_event_count"]
            total_deduped_events += r["deduped_event_count"]

            if not r["has_events"]:
                no_events += 1

            if r["no_final_journey"]:
                no_final_journey += 1
                continue

            stats = r["stats"]
            if stats:
                agg_total_offer_ids += stats["total_offer_ids"]
                agg_found_offer_ids += stats["found_offer_ids"]
                agg_missing_offer_ids += stats["missing_offer_ids"]
                agg_total_journeys += stats["total_journeys"]
                agg_kept_journeys += stats["kept_journeys"]
                agg_empty_product_journeys += stats["empty_product_journeys"]
            if len(missing_offer_id_set) < _MAX_MISSING_IDS:
                missing_offer_id_set.update(r["missing_ids"])

            # Filter: must have non-empty events
            if not r["has_events"]:
                filtered_no_events += 1
                continue

            result = r["result"]
            if result is not None:
                if not result["journeys"]:
                    entries_with_empty_journeys += 1
                results[r["user_id"]] = result

    del deduped_tuples
    gc.collect()

    print(f"  Successfully parsed entries: {len(results):>12,}")
    print(f"  Rows with empty UserId: {no_userid:>12,}")
    print(f"  Duplicate UserIds (skipped): {duplicate_users:>12,}")
    print(f"  Entries without events (raw): {no_events:>12,}")
    print(f"  Entries without FinalJourney: {no_final_journey:>12,}")
    print(f"  Filtered out (no events): {filtered_no_events:>12,}")
    print(f"  Entries with empty journeys:      {entries_with_empty_journeys:>12,}")

    print(f"\n  --- Event Dedup ---")
    print(f"  Total raw events (before dedup): {total_raw_events:>12,}")
    print(f"  Total events (after dedup): {total_deduped_events:>12,}")
    print(f"  Duplicate events removed: "
          f"{total_raw_events - total_deduped_events:>12,}")

    print(f"\n  --- OfferId Validation ---")
    print(f"  Total OfferIds in journeys: {agg_total_offer_ids:>12,}")
    print(f"  OfferIds found in item.json: {agg_found_offer_ids:>12,} "
          f"({agg_found_offer_ids / max(agg_total_offer_ids, 1) * 100:.2f}%)")
    print(f"  OfferIds NOT found: {agg_missing_offer_ids:>12,} "
          f"({agg_missing_offer_ids / max(agg_total_offer_ids, 1) * 100:.2f}%)")
    print(f"  Distinct missing OfferIds: {len(missing_offer_id_set):>12,}")

    if missing_offer_id_set:
        sample = sorted(missing_offer_id_set)[:5]
        print(f"\n  Sample missing OfferIds:")
        for oid in sample:
            print(f"    -> {oid}")

    print(f"\n  --- Journey Mapping ---")
    print(f"  Total journeys extracted: {agg_total_journeys:>12,}")
    print(f"  Journeys with empty product_ids: {agg_empty_product_journeys:>12,}")
    print(f"  Journeys kept (with products): {agg_kept_journeys:>12,}")

    # =========================================================================
    # Step 4: Statistics
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Statistics")
    print("=" * 70)

    entries_with_journeys = sum(
        1 for v in results.values() if v["journeys"]
    )
    entries_without_journeys = len(results) - entries_with_journeys

    print(f"  Total entries: {len(results):>12,}")
    print(f"  Entries with >= 1 journey: {entries_with_journeys:>12,}")
    print(f"  Entries without journeys: {entries_without_journeys:>12,}")

    # Event count distribution
    event_counts = [
        len(v["user_shopping_events"]) for v in results.values()
    ]
    if event_counts:
        ec_sorted = sorted(event_counts)

        def percentile(arr, p):
            idx = int(len(arr) * p)
            return arr[min(idx, len(arr) - 1)]

        print(f"\n  User events per entry:")
        print(f"    Min:  {min(event_counts):>6}")
        print(f"    P25:  {percentile(ec_sorted, 0.25):>6}")
        print(f"    P50:  {percentile(ec_sorted, 0.5):>6}")
        print(f"    P75:  {percentile(ec_sorted, 0.75):>6}")
        print(f"    P90:  {percentile(ec_sorted, 0.9):>6}")
        print(f"    Max:  {max(event_counts):>6}")
        print(f"    Avg:  {sum(event_counts) / len(event_counts):>6.1f}")

        # Event count histogram
        ec_dist = defaultdict(int)
        for c in event_counts:
            if c == 0:
                bucket = "0"
            elif c <= 5:
                bucket = "1-5"
            elif c <= 10:
                bucket = "6-10"
            elif c <= 20:
                bucket = "11-20"
            elif c <= 50:
                bucket = "21-50"
            else:
                bucket = "51+"
            ec_dist[bucket] += 1
        print(f"\n  Event count distribution:")
        for bucket in ["0", "1-5", "6-10", "11-20", "21-50", "51+"]:
            if bucket in ec_dist:
                print(f"    {bucket:>8s} events: {ec_dist[bucket]:>10,} entries")

    # Journey count distribution
    journey_counts = [len(v["journeys"]) for v in results.values()]
    if journey_counts:
        jc_dist = defaultdict(int)
        for c in journey_counts:
            jc_dist[c] += 1
        print(f"\n  Journey count distribution:")
        for cnt in sorted(jc_dist.keys()):
            label = f"{cnt} journey(s)" if cnt != 1 else f"{cnt} journey"
            print(f"    {label:20s} {jc_dist[cnt]:>10,} entries")

    # Products per journey
    product_counts = [
        len(j["product_ids"])
        for v in results.values()
        for j in v["journeys"]
    ]
    if product_counts:
        pc_sorted = sorted(product_counts)
        print(f"\n  Products per journey:")
        print(f"    Min:  {min(product_counts):>6}")
        print(f"    P50:  {percentile(pc_sorted, 0.5):>6}")
        print(f"    P90:  {percentile(pc_sorted, 0.9):>6}")
        print(f"    Max:  {max(product_counts):>6}")
        print(f"    Avg:  {sum(product_counts) / len(product_counts):>6.1f}")

    # Show sample entries
    sample_entries = list(results.items())[:2]
    if sample_entries:
        print(f"\n  Sample entries:")
        for user_id, data in sample_entries:
            n_events = len(data["user_shopping_events"])
            n_journeys = len(data["journeys"])
            print(f"    UserId: {user_id}")
            print(f"      user_shopping_events: {n_events} events")
            if data["user_shopping_events"]:
                print(f"        [0]: {data['user_shopping_events'][0][:100]}")
                if n_events > 1:
                    print(f"        [1]: {data['user_shopping_events'][1][:100]}")
            print(f"      journeys: {n_journeys}")
            for ji, j in enumerate(data["journeys"][:2]):
                print(f"        [{ji}]: title={j['title'][:60]}")
                print(f"              reason={j['reason'][:60]}")
                print(f"              products={len(j['product_ids'])} "
                      f"ids={j['product_ids'][:3]}")
            print()

    # =========================================================================
    # Step 5: Write output
    # =========================================================================
    print("=" * 70)
    print("Step 5: Writing output")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"{args.task}.json"
    output_path = os.path.join(args.output_dir, output_name)

    # Stream-write JSON to avoid building the entire serialized string in memory
    if _HAS_ORJSON:
        with open(output_path, "wb") as f:
            f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            # Stream key-by-key to reduce peak memory vs json.dump on full dict
            f.write("{\n")
            for i, (uid, data) in enumerate(results.items()):
                if i > 0:
                    f.write(",\n")
                f.write(f"  {json.dumps(uid)}: {json.dumps(data, ensure_ascii=False)}")
            f.write("\n}\n")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output written to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Total entries: {len(results):,}")
    print(f"  Total journeys (with products): {agg_kept_journeys:,}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"  Input:")
    if args.prompt_results_dir:
        print(f"    Journey dir:                      {args.prompt_results_dir}")
    else:
        print(f"    Journey file:                     {args.journey_file}")
    print(f"    Journey TSV rows:                 {total_rows_processed:>12,}")
    print(f"    Items in item.json:               {len(valid_offer_ids):>12,}")
    print(f"  Processing:")
    print(f"    Duplicate UserIds (skipped):       {duplicate_users:>12,}")
    print(f"    OfferIds found in item.json:       {agg_found_offer_ids:>12,} "
          f"({agg_found_offer_ids / max(agg_total_offer_ids, 1) * 100:.1f}%)")
    print(f"    OfferIds NOT found:                {agg_missing_offer_ids:>12,} "
          f"({agg_missing_offer_ids / max(agg_total_offer_ids, 1) * 100:.1f}%)")
    print(f"  Output:")
    print(f"    Entries:                          {len(results):>12,}")
    print(f"    Entries with journeys:            {entries_with_journeys:>12,}")
    print(f"    Journeys (kept with products):    {agg_kept_journeys:>12,}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
