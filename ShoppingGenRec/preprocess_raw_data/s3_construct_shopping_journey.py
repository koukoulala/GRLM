"""
Construct shopping journey data from raw model prediction outputs.

Reads:
  1. Shopping journey prediction TSV file (columns: source, uuid, messages,
     metadata). The "messages" column contains a JSON array of chat messages
     (system / user / assistant). User messages embed <USER-EVENTS> and
     <SYSTEM-TIME> blocks; assistant messages embed <OUTPUT> blocks with
     ContinuedJourneys JSON.
  2. Query products TSV file (columns: query_id, SID, GlobalOfferId, Title,
     SimilarityScore). Used to map ProductSIDs to GlobalOfferIds.

Produces:
  shopping_journey.json - keyed by uuid, each entry containing:
    - user_shopping_events: list of event strings
        (format: "time_ago | action | description")
    - system_time: date string (e.g., "9/21/2025")
    - journeys: list of journey dicts, each with:
        - title: journey title string
        - query: shopping query string
        - product_ids: list of GlobalOfferId strings

Pipeline:
  1. Read query products TSV and build SID -> GlobalOfferId mapping
  2. Read shopping journey prediction TSV
  3. Parse messages JSON: extract USER-EVENTS, SYSTEM-TIME, and OUTPUT
  4. Map ProductSIDs (e.g., <|sid_begin|><a_811><b_1134><c_153><|sid_end|>)
     to GlobalOfferIds using query products data
  5. Deduplicate by uuid (keep first occurrence, report duplicates)
  6. Compute statistics and write output
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
    """Read a TSV file and return rows as list of dicts.

    Args:
        filepath: Path to the TSV file.
        expected_columns: Optional list of column names to use instead of
            the file header.

    Returns:
        A list of dicts, one per row.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        if expected_columns:
            # Skip the header row if present
            next(reader, None)
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
                row = row[: len(columns)]
            rows.append(dict(zip(columns, row)))

    return rows


def build_sid_to_gid_mapping(query_product_rows):
    """Build a mapping from SID to GlobalOfferId.

    Args:
        query_product_rows: List of dicts from query products TSV, each
            containing "SID" and "GlobalOfferId" keys.

    Returns:
        Tuple of (mapping dict, number of SID conflicts).
    """
    mapping = {}
    conflicts = 0
    for row in query_product_rows:
        sid = row.get("SID", "").strip()
        gid = row.get("GlobalOfferId", "").strip()
        if sid and gid:
            if sid in mapping and mapping[sid] != gid:
                conflicts += 1
            mapping[sid] = gid
    return mapping, conflicts


def extract_sids(product_sids_list):
    """Extract SID strings from a list of ProductSID entries.

    Each entry has format: <|sid_begin|><a_X><b_Y><c_Z><|sid_end|>
    Extracts the inner SID part: <a_X><b_Y><c_Z>

    Args:
        product_sids_list: List of ProductSID strings.

    Returns:
        List of extracted SID strings.
    """
    sids = []
    for sid_str in product_sids_list:
        match = re.search(r"<\|sid_begin\|>(.*?)<\|sid_end\|>", sid_str)
        if match:
            sids.append(match.group(1).strip())
    return sids


def _normalize_event_key(event):
    """Normalize an event string for deduplication.

    Lowercases, strips non-alphanumeric characters (keeping spaces and pipes),
    and collapses whitespace.

    Args:
        event: Event description string.

    Returns:
        Normalized key string for dedup comparison.
    """
    key = event.lower()
    key = re.sub(r"[^a-z0-9\s|]", " ", key)
    key = re.sub(r"\s+", " ", key)
    return key.strip()


def parse_user_events(events_text):
    """Parse USER-EVENTS text block into a list of event strings.

    Strips leading event numbers (e.g., "1 | ") from each event line.
    Deduplicates events after normalization (lowercase + strip symbols),
    preserving original text and first-occurrence order.

    Input format:  "1 | 1 week ago | Searched | shoes\\n2 | 3 days ago | ..."
    Output format: ["1 week ago | Searched | shoes", "3 days ago | ..."]

    Args:
        events_text: Raw text between <USER-EVENTS> and </USER-EVENTS>.

    Returns:
        Tuple of (deduplicated event list, total raw events before dedup).
    """
    # Normalize: literal backslash-n to actual newline
    text = events_text.replace("\\n", "\n")
    lines = text.strip().split("\n")

    raw_events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Only accept lines matching the numbered event format:
        # "1 | time_ago | action | description"
        # Lines not starting with a number+pipe (e.g., instruction text,
        # stray tags) are silently discarded.
        match = re.match(r"^\d+\s*\|\s*(.*)", line)
        if match:
            event = match.group(1).strip()
            if event:
                raw_events.append(event)

    # Deduplicate while preserving order and original text
    seen_keys = set()
    deduped = []
    for event in raw_events:
        key = _normalize_event_key(event)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(event)

    # Reverse to chronological order (oldest first, newest last)
    # so the most recent events appear at the end of the prompt,
    # leveraging LLM recency bias for better attention.
    deduped.reverse()

    return deduped, len(raw_events)


def parse_messages(messages_str):
    """Parse the messages JSON and extract structured data.

    Extracts:
      - User shopping events from <USER-EVENTS> block
      - System time from <SYSTEM-TIME> block
      - Journey data from <OUTPUT> block in assistant response

    Args:
        messages_str: JSON string containing the messages array.

    Returns:
        Tuple of (user_events, raw_event_count, system_time, journeys_raw):
          - user_events: list of event strings (deduplicated)
          - raw_event_count: int, number of events before dedup
          - system_time: date string
          - journeys_raw: list of raw journey dicts (with Title, Query,
            ProductSIDs keys)
        Returns (None, None, None, None) if parsing fails entirely.
    """
    try:
        messages = json.loads(messages_str)
    except (json.JSONDecodeError, TypeError):
        return None, None, None, None

    # Collect text by role
    user_text = ""
    assistant_text = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", [])
        # Content can be a list of {type, text} objects or a plain string
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if role == "user":
                        user_text += text
                    elif role == "assistant":
                        assistant_text += text
        elif isinstance(content, str):
            if role == "user":
                user_text += content
            elif role == "assistant":
                assistant_text += content

    # --- Extract USER-EVENTS ---
    # The user text contains multiple bare <USER-EVENTS> mentions in the
    # instruction portion (e.g., "based on given <USER-EVENTS>",
    # "event in <USER-EVENTS>."), but only ONE closing </USER-EVENTS> tag
    # wrapping the real event data.
    # Use rfind to locate the LAST <USER-EVENTS> opening tag, then find
    # the </USER-EVENTS> closing tag after it. This is more robust than
    # regex-based approaches which can mis-anchor on earlier occurrences.
    user_events = []
    raw_event_count = 0
    _OPEN_TAG = "<USER-EVENTS>"
    _CLOSE_TAG = "</USER-EVENTS>"
    last_open = user_text.rfind(_OPEN_TAG)
    if last_open != -1:
        close_pos = user_text.find(_CLOSE_TAG, last_open)
        if close_pos != -1:
            events_block = user_text[
                last_open + len(_OPEN_TAG):close_pos
            ].strip()
            user_events, raw_event_count = parse_user_events(events_block)

    # --- Extract SYSTEM-TIME ---
    # Format: <SYSTEM-TIME>\nCurrent system time (UTC): 9/21/2025\n</SYSTEM-TIME>
    system_time = ""
    time_match = re.search(
        r"<SYSTEM-TIME>\s*.*?:\s*([\d/]+)\s*</SYSTEM-TIME>",
        user_text, re.DOTALL,
    )
    if time_match:
        system_time = time_match.group(1).strip()

    # --- Extract OUTPUT -> ContinuedJourneys ---
    journeys_raw = []
    output_match = re.search(
        r"<OUTPUT>\s*(.*?)\s*</OUTPUT>", assistant_text, re.DOTALL,
    )
    if output_match:
        output_str = output_match.group(1).strip()
        # Try parsing as JSON; fall back to fixing common escaping issues
        for attempt in [output_str, output_str.replace('\\"', '"')]:
            try:
                data = json.loads(attempt)
                journeys_raw = data.get("ContinuedJourneys", [])
                break
            except (json.JSONDecodeError, AttributeError):
                continue

    return user_events, raw_event_count, system_time, journeys_raw


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct shopping journey data from raw model "
                    "prediction outputs"
    )
    parser.add_argument(
        "--journey_prediction_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/ShoppingJourney/train_shopping_journey_prediction_SID_output_simplified.tsv",
        help="Path to the shopping journey prediction TSV file "
             "(columns: source, uuid, messages, metadata)",
    )
    parser.add_argument(
        "--query_products_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/ShoppingJourney_Query_Products_Resolved.tsv",
        help="Path to the query products TSV file "
             "(columns: query_id, SID, GlobalOfferId, Title, SimilarityScore)",
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
    # Step 1: Read query products TSV and build SID -> GlobalOfferId mapping
    # =========================================================================
    print("=" * 70)
    print("Step 1: Building SID -> GlobalOfferId mapping")
    print("=" * 70)

    query_columns = [
        "query_id", "SID", "GlobalOfferId", "Title", "SimilarityScore",
    ]
    query_rows = read_tsv(
        args.query_products_file, expected_columns=query_columns,
    )

    sid_to_gid, sid_conflicts = build_sid_to_gid_mapping(query_rows)

    print(f"  Rows in query products file: {len(query_rows):>12,}")
    print(f"  Distinct SIDs mapped: {len(sid_to_gid):>12,}")
    print(f"  SID -> GID conflicts: {sid_conflicts:>12,}")

    # Show sample SID -> GID mappings
    if sid_to_gid:
        sample_items = list(sid_to_gid.items())[:3]
        print(f"\n  Sample SID -> GlobalOfferId mappings:")
        for sid, gid in sample_items:
            print(f"    {sid} -> {gid}")

    # =========================================================================
    # Step 2: Read shopping journey prediction TSV
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Reading shopping journey prediction file")
    print("=" * 70)

    prediction_columns = ["source", "uuid", "messages", "metadata"]
    prediction_rows = read_tsv(
        args.journey_prediction_file, expected_columns=prediction_columns,
    )
    print(f"  Total rows: {len(prediction_rows):>12,}")

    # Source distribution
    source_counts = defaultdict(int)
    for row in prediction_rows:
        source_counts[row.get("source", "").strip()] += 1
    if source_counts:
        print(f"\n  Source distribution:")
        for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"    {src:50s} {cnt:>10,}")

    # =========================================================================
    # Step 3: Parse messages and extract journey data
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Parsing messages and extracting journey data")
    print("=" * 70)

    results = {}  # uuid -> {user_shopping_events, system_time, journeys}
    duplicate_uuids = 0
    parse_failures = 0
    no_uuid = 0
    no_events = 0
    no_system_time = 0
    no_journeys = 0

    # Event dedup statistics
    total_raw_events = 0
    total_deduped_events = 0

    # SID mapping statistics
    total_journeys = 0
    total_journeys_kept = 0
    total_sids_found = 0
    total_sids_mapped = 0
    total_sids_unmapped = 0
    unmapped_sid_set = set()
    empty_product_ids_journeys = 0

    for row in prediction_rows:
        uuid = row.get("uuid", "").strip()
        if not uuid:
            no_uuid += 1
            continue

        # Handle duplicate uuids: keep the first occurrence
        if uuid in results:
            duplicate_uuids += 1
            continue

        messages_str = row.get("messages", "").strip()
        if not messages_str:
            parse_failures += 1
            continue

        user_events, raw_event_count, system_time, journeys_raw = (
            parse_messages(messages_str)
        )

        if user_events is None:
            parse_failures += 1
            continue

        # Track event dedup statistics
        total_raw_events += raw_event_count
        total_deduped_events += len(user_events)

        if not user_events:
            no_events += 1
        if not system_time:
            no_system_time += 1
        if not journeys_raw:
            no_journeys += 1

        # Process journeys: map ProductSIDs to GlobalOfferIds
        journeys = []
        for j_raw in journeys_raw:
            title = j_raw.get("Title", "").strip()
            query = j_raw.get("Query", "").strip()
            product_sids_raw = j_raw.get("ProductSIDs", [])

            # Extract SIDs from <|sid_begin|>...<|sid_end|> format
            sids = extract_sids(product_sids_raw)
            total_sids_found += len(sids)

            # Map each SID to its GlobalOfferId
            product_ids = []
            for sid in sids:
                if sid in sid_to_gid:
                    product_ids.append(sid_to_gid[sid])
                    total_sids_mapped += 1
                else:
                    total_sids_unmapped += 1
                    unmapped_sid_set.add(sid)

            total_journeys += 1

            # Skip journeys with no mapped products
            if not product_ids:
                empty_product_ids_journeys += 1
                continue

            journeys.append({
                "title": title,
                "query": query,
                "product_ids": product_ids,
            })
            total_journeys_kept += 1

        # Only store entries that have at least one journey with products
        if not journeys:
            continue

        results[uuid] = {
            "user_shopping_events": user_events,
            "system_time": system_time,
            "journeys": journeys,
        }

    print(f"  Successfully parsed entries: {len(results):>12,}")
    print(f"  Rows with empty uuid: {no_uuid:>12,}")
    print(f"  Duplicate uuids (skipped): {duplicate_uuids:>12,}")
    print(f"  Parse failures (bad JSON): {parse_failures:>12,}")
    print(f"  Entries without user events: {no_events:>12,}")
    print(f"  Entries without system time: {no_system_time:>12,}")
    print(f"  Entries without journeys: {no_journeys:>12,}")

    print(f"\n  --- Event Dedup ---")
    print(f"  Total raw events (before dedup): {total_raw_events:>12,}")
    print(f"  Total events (after dedup): {total_deduped_events:>12,}")
    print(f"  Duplicate events removed: "
          f"{total_raw_events - total_deduped_events:>12,}")

    print(f"\n  --- SID Mapping ---")
    print(f"  Total journeys extracted: {total_journeys:>12,}")
    print(f"  Journeys with empty product_ids: {empty_product_ids_journeys:>12,}")
    print(f"  Journeys kept (with products): {total_journeys_kept:>12,}")
    print(f"  Total ProductSIDs found: {total_sids_found:>12,}")
    print(f"  SIDs mapped to GlobalOfferId: {total_sids_mapped:>12,}")
    print(f"  SIDs unmapped: {total_sids_unmapped:>12,}")
    print(f"  Distinct unmapped SIDs: {len(unmapped_sid_set):>12,}")

    if unmapped_sid_set:
        sample = sorted(unmapped_sid_set)[:5]
        print(f"\n  Sample unmapped SIDs:")
        for sid in sample:
            print(f"    -> {sid}")

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

    print(f"  Total entries: {len(results):>12,}")
    print(f"  Entries with >= 1 journey: {entries_with_journeys:>12,}")
    print(f"  Entries without journeys: "
          f"{len(results) - entries_with_journeys:>12,}")

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

    # User events per entry
    event_counts = [
        len(v["user_shopping_events"]) for v in results.values()
    ]
    if event_counts:
        ec_sorted = sorted(event_counts)
        print(f"\n  User events per entry:")
        print(f"    Min:  {min(event_counts):>6}")
        print(f"    P50:  {ec_sorted[len(ec_sorted) // 2]:>6}")
        print(f"    Max:  {max(event_counts):>6}")
        print(f"    Avg:  {sum(event_counts) / len(event_counts):>6.1f}")

    # Products per journey
    product_counts = [
        len(j["product_ids"])
        for v in results.values()
        for j in v["journeys"]
    ]
    if product_counts:
        pc_sorted = sorted(product_counts)

        def percentile(arr, p):
            idx = int(len(arr) * p)
            return arr[min(idx, len(arr) - 1)]

        print(f"\n  Products per journey:")
        print(f"    Min:  {min(product_counts):>6}")
        print(f"    P50:  {percentile(pc_sorted, 0.5):>6}")
        print(f"    P90:  {percentile(pc_sorted, 0.9):>6}")
        print(f"    Max:  {max(product_counts):>6}")
        print(f"    Avg:  {sum(product_counts) / len(product_counts):>6.1f}")

    # Show a few sample entries
    sample_entries = list(results.items())[:2]
    if sample_entries:
        print(f"\n  Sample entries:")
        for uuid, data in sample_entries:
            n_events = len(data["user_shopping_events"])
            n_journeys = len(data["journeys"])
            print(f"    uuid: {uuid}")
            print(f"      system_time: {data['system_time']}")
            print(f"      user_shopping_events: {n_events} events")
            if data["user_shopping_events"]:
                print(f"        [0]: {data['user_shopping_events'][0][:80]}")
            print(f"      journeys: {n_journeys}")
            for ji, j in enumerate(data["journeys"][:2]):
                print(f"        [{ji}]: title={j['title'][:50]}, "
                      f"query={j['query'][:40]}, "
                      f"products={len(j['product_ids'])}")
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
    print(f"  Total journeys (with products): {total_journeys_kept:,}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"  Input:")
    print(f"    Prediction rows:                  {len(prediction_rows):>12,}")
    print(f"    Query product rows:               {len(query_rows):>12,}")
    print(f"  Processing:")
    print(f"    Rows with empty uuid:             {no_uuid:>12,}")
    print(f"    Duplicate uuids (skipped):        {duplicate_uuids:>12,}")
    print(f"    Parse failures:                   {parse_failures:>12,}")
    print(f"    SIDs mapped successfully:         {total_sids_mapped:>12,}")
    print(f"    SIDs unmapped:                    {total_sids_unmapped:>12,}")
    print(f"  Output:")
    print(f"    Entries:                          {len(results):>12,}")
    print(f"    Journeys (total extracted):       {total_journeys:>12,}")
    print(f"    Journeys (kept with products):    {total_journeys_kept:>12,}")
    print(f"    Entries with journeys:            {entries_with_journeys:>12,}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
