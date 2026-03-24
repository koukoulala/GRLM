"""
Construct shopping journey data from journey prediction TSV and item metadata.

Reads:
  1. Shopping journey TSV file with columns:
     UserId, ReadableUserEvents, UserHistory, ShoppingJourney,
     JourneyWithProducts, OUTPUT, FinalJourney
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
import json
import os
import re
import sys
from collections import defaultdict

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)


# =============================================================================
# Utility Functions
# =============================================================================

def read_tsv(filepath, expected_columns=None):
    """Read a TSV file and return rows as list of dicts."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        if expected_columns:
            next(reader, None)  # skip header
            columns = expected_columns
        else:
            header = next(reader, None)
            if header is None:
                return rows
            columns = header

        for row in reader:
            if len(row) < len(columns):
                row.extend([""] * (len(columns) - len(row)))
            elif len(row) > len(columns):
                row = row[:len(columns)]
            rows.append(dict(zip(columns, row)))

    return rows


def _normalize_event_key(event):
    """Normalize an event string for deduplication."""
    key = event.lower()
    key = re.sub(r"[^a-z0-9\s|]", " ", key)
    key = re.sub(r"\s+", " ", key)
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
        match = re.match(r"^\d+\s*\|\s*(.*)", line)
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
            data = json.loads(attempt_text)
            break
        except (json.JSONDecodeError, TypeError):
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

        journeys.append({
            "title": title,
            "reason": reason,
            "product_ids": product_ids,
        })
        stats["kept_journeys"] += 1

    return journeys, stats, missing_ids


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct shopping journey data from journey prediction "
                    "TSV and item metadata"
    )
    parser.add_argument(
        "--journey_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/0128_0301/ShoppingJourney_Input_500K_His50_Final_Training_clean.tsv",
        help="Path to the shopping journey TSV file",
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="./raw_data/item.json",
        help="Path to item metadata JSON file (from pre_s0_combine_item_data.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw_data",
        help="Path to the output directory (default: ./raw_data)",
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

    with open(args.item_file, "r", encoding="utf-8") as f:
        item_data = json.load(f)
    valid_offer_ids = set(item_data.keys())
    print(f"  Loaded {len(valid_offer_ids):,} items from: {args.item_file}")

    # Free memory - we only need the keys
    del item_data

    # =========================================================================
    # Step 2: Read journey prediction TSV
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Reading journey prediction file")
    print("=" * 70)

    tsv_columns = [
        "UserId", "ReadableUserEvents", "UserHistory",
        "JourneyWithProducts", "FinalJourney"
    ]
    rows = read_tsv(args.journey_file, expected_columns=tsv_columns)
    print(f"  Total rows: {len(rows):,}")

    # =========================================================================
    # Step 3: Parse and construct journey data
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Parsing rows and constructing journeys")
    print("=" * 70)

    results = {}
    duplicate_users = 0
    no_userid = 0
    no_events = 0
    no_final_journey = 0
    parse_failures = 0
    filtered_no_events = 0
    entries_with_empty_journeys = 0

    # Event statistics
    total_raw_events = 0
    total_deduped_events = 0

    # OfferId statistics
    agg_total_offer_ids = 0
    agg_found_offer_ids = 0
    agg_missing_offer_ids = 0
    agg_total_journeys = 0
    agg_kept_journeys = 0
    agg_empty_product_journeys = 0
    missing_offer_id_set = set()

    for row in rows:
        user_id = row.get("UserId", "").strip()
        if not user_id:
            no_userid += 1
            continue

        if user_id in results:
            duplicate_users += 1
            continue

        # --- Parse ReadableUserEvents ---
        events_text = row.get("ReadableUserEvents", "").strip()
        user_events, raw_event_count = parse_readable_user_events(events_text)
        total_raw_events += raw_event_count
        total_deduped_events += len(user_events)

        if not user_events:
            no_events += 1

        # --- Parse FinalJourney ---
        journey_text = row.get("FinalJourney", "").strip()
        if not journey_text:
            no_final_journey += 1
            continue

        journeys, stats, missing_ids = parse_final_journey(
            journey_text, valid_offer_ids
        )

        agg_total_offer_ids += stats["total_offer_ids"]
        agg_found_offer_ids += stats["found_offer_ids"]
        agg_missing_offer_ids += stats["missing_offer_ids"]
        agg_total_journeys += stats["total_journeys"]
        agg_kept_journeys += stats["kept_journeys"]
        agg_empty_product_journeys += stats["empty_product_journeys"]
        missing_offer_id_set.update(missing_ids)

        # Filter: must have non-empty events
        if not user_events:
            filtered_no_events += 1
            continue

        if not journeys:
            entries_with_empty_journeys += 1

        results[user_id] = {
            "user_shopping_events": user_events,
            "journeys": journeys,
        }

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
    output_path = os.path.join(args.output_dir, "shopping_journeys.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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
    print(f"    Journey TSV rows:                 {len(rows):>12,}")
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
