"""Step 2b: Build Journey-to-Product SFT Data

Creates SFT training data for a user-event-to-product prediction task.
Given a chronological sequence of user shopping events (browsing, purchasing,
searching), the model predicts the next product the user will interact with.

This differs from the standard recommendation SFT (s2_build_rec_sft_data.py)
in several ways:
  - Input is a heterogeneous event sequence (Browse/Purchase/Search) with
    natural-language descriptions and relative timestamps, rather than a
    homogeneous sequence of product text IDs.
  - The prediction target is a single next product (represented by its text
    ID), rather than a sequence of future products.
  - Events carry richer context: page titles for browsing, product info for
    purchases, and query text for searches.

Data source:
  full_sequential_data.json (from preprocess_raw_data/s2_construct_sequential_data.py)
    Format: { UserId: [ {Timestamp, Source, GlobalOfferId, PageTitle, Query}, ... ] }
  id2meta.json (from s1_generate_tid.py)
    Format: { ItemId: {title, description, categories, summary_words, ...} }

Pipeline:
  1. Load full_sequential_data.json and id2meta.json
  2. For each user, walk backwards from the end of their action sequence to
     find a valid prediction target — an action with a non-empty GlobalOfferId
     or PageTitle (item ID).
  3. Enforce a 2-hour continuity constraint: the target action must be within
     2 hours of the preceding action. If the gap exceeds 2 hours, the target
     is considered an isolated action unrelated to the prior event history;
     continue searching backwards for the next valid target.
  4. Require at least 2 preceding events to form a meaningful input context.
  5. Format preceding actions as a shopping event sequence in the style:
       "time_ago | action_type | description"
     where action_type is one of:
       - Browsed  (actions with a valid PageTitle or GlobalOfferId)
       - Clicked  (actions with a Source containing "Click")
       - Searched (actions with a non-empty Query)
  6. Construct instruction/input/output for SFT training.

Usage:
    python s2_build_journey_sft_data.py \
        --full_sequential_file ./raw_data/full_sequential_data.json \
        --id2meta_file ./processed/id2meta.json \
        --output_dir ./sft_data \
        --max_events 50 \
        --gap_threshold 7200
"""

import os
import json
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from tqdm import tqdm
import numpy as np

# =============================================================================
# Constants
# =============================================================================

# Maximum allowed time gap (in seconds) between the target action and its
# preceding action. If exceeded, the target is considered an isolated action.
DEFAULT_GAP_THRESHOLD = 7200  # 2 hours

# Minimum number of preceding events required to form a valid input sequence.
MIN_PRECEDING_EVENTS = 2


# =============================================================================
# Utility Functions
# =============================================================================

def parse_timestamp(ts_str):
    """Parse a timestamp string to float (seconds since epoch).

    Auto-detects millisecond timestamps (value > 10^12) and converts
    to seconds.

    Args:
        ts_str: Timestamp string.

    Returns:
        Float timestamp in seconds, or None if parsing fails.
    """
    try:
        ts = float(ts_str)
        if ts > 1e12:
            ts = ts / 1000.0
        return ts
    except (ValueError, TypeError):
        return None


def format_time_ago(seconds_diff):
    """Format a time difference in seconds into a human-readable string.

    Uses simple units: "X days ago" or "X hours ago" (if < 1 day).

    Args:
        seconds_diff: Non-negative float, time difference in seconds.

    Returns:
        Human-readable relative time string (e.g., "3 days ago", "5 hours ago").
    """
    if seconds_diff < 60:
        return "just now"

    hours = int(seconds_diff // 3600)
    days = hours // 24

    if days >= 1:
        plural = "s" if days > 1 else ""
        return f"{days} day{plural} ago"
    else:
        if hours < 1:
            minutes = int(seconds_diff // 60)
            plural = "s" if minutes > 1 else ""
            return f"{minutes} minute{plural} ago"
        plural = "s" if hours > 1 else ""
        return f"{hours} hour{plural} ago"


def classify_action(action):
    """Classify an action into one of three event types based on its fields.

    Classification logic:
      1. Clicked  — has a non-empty GlobalOfferId (user clicked on a product)
      2. Browsed  — has a non-empty PageTitle (user browsed a page)
      3. Searched — has a non-empty Query (user performed a search)

    If an action has both a GlobalOfferId and a PageTitle, the GlobalOfferId
    (Clicked) takes precedence.

    An action that doesn't match any category returns None.

    Args:
        action: Dict with keys Source, GlobalOfferId, PageTitle, Query.

    Returns:
        Tuple of (event_type: str, description: str) or None if unclassifiable.
    """
    gid = action.get("GlobalOfferId", "")
    pt = action.get("PageTitle", "")
    query = action.get("Query", "")

    # Priority 1: Has GlobalOfferId -> Clicked (product click)
    if gid:
        return "Clicked", gid

    # Priority 2: Has PageTitle -> Browsed (page view)
    if pt:
        return "Browsed", pt

    # Priority 3: Has search query -> Searched
    if query:
        return "Searched", query

    return None


def get_item_description(item_id, id2meta, page_title_items=None, item_data=None):
    """Get a human-readable description for an item ID.

    Lookup priority:
      1. id2meta (items with valid TIDs)
      2. item_data (full item metadata, broader coverage)
      3. page_title_items (for P-prefixed PageTitle indices)
      4. Raw item_id as fallback

    Args:
        item_id: Item identifier string (GlobalOfferId or "P123").
        id2meta: Dict mapping item IDs to metadata.
        page_title_items: Optional dict mapping P-prefixed indices to
            page title data.
        item_data: Optional dict of all items (broader than id2meta).

    Returns:
        Description string (product title or page title or the raw ID).
    """
    if item_id in id2meta:
        title = id2meta[item_id].get("title", "")
        if title:
            return title
    if item_data and item_id in item_data:
        title = item_data[item_id].get("title", "")
        if title:
            return title
    if item_id.startswith("P") and page_title_items:
        pt_data = page_title_items.get(item_id, {})
        title = pt_data.get("title", "")
        if title:
            return title
    return item_id


def get_item_tid(item_id, id2meta):
    """Get the text ID (7 summary words) for an item.

    Args:
        item_id: Item identifier string.
        id2meta: Dict mapping item IDs to metadata with summary_words.

    Returns:
        List of 7 summary words, or None if not found or invalid.
    """
    if item_id not in id2meta:
        return None
    meta = id2meta[item_id]
    summary_words = meta.get("summary_words", [])
    if not summary_words or "" in summary_words:
        return None
    valid_words = [
        word.replace("[", "").replace("]", "")
        for word in summary_words
        if word and word.strip()
    ]
    if len(valid_words) < 7:
        return None
    return valid_words[:7]


def has_valid_item_id(action, use_page_title=True):
    """Check if an action has a non-empty item identifier.

    Args:
        action: Dict with keys GlobalOfferId, PageTitle.
        use_page_title: If True, also accept PageTitle as a valid item ID.
            If False, only accept GlobalOfferId.

    Returns:
        True if the action has a valid item identifier.
    """
    if bool(action.get("GlobalOfferId", "")):
        return True
    if use_page_title and bool(action.get("PageTitle", "")):
        return True
    return False


def get_action_item_id(action, use_page_title=True):
    """Get the item ID from an action, preferring GlobalOfferId over PageTitle.

    Args:
        action: Dict with keys GlobalOfferId, PageTitle.
        use_page_title: If True, fall back to PageTitle when GlobalOfferId
            is empty. If False, only return GlobalOfferId.

    Returns:
        Item ID string, or empty string if not found.
    """
    gid = action.get("GlobalOfferId", "")
    if gid:
        return gid
    if use_page_title:
        return action.get("PageTitle", "")
    return ""


# =============================================================================
# Core Logic
# =============================================================================

def find_target_and_context(
    actions, gap_threshold, id2meta,
    use_page_title=True, min_events=MIN_PRECEDING_EVENTS,
):
    """Find a valid prediction target and its preceding event context.

    Walks backwards from the end of the action list to find the last action
    that satisfies all of:
      1. Has a valid item ID (GlobalOfferId; or PageTitle if use_page_title).
      2. The item has a valid TID in id2meta.
      3. The target is within gap_threshold seconds of the preceding action.
      4. There are at least min_events preceding actions.

    If any constraint is violated, continues searching backwards.

    Args:
        actions: List of action dicts sorted by timestamp (ascending).
        gap_threshold: Maximum allowed time gap in seconds between the target
            and the preceding action.
        id2meta: Dict mapping item IDs to metadata (used for TID validation).
        use_page_title: If True, also consider PageTitle as a valid item ID
            for the target. If False, only GlobalOfferId is accepted.
        min_events: Minimum number of preceding events required.

    Returns:
        Tuple of (target_action, target_item_id, target_tid, preceding_actions,
                  skip_reason) where:
          - target_action: the action dict chosen, or None
          - target_item_id: item ID of the target, or None
          - target_tid: list of 7 summary words, or None
          - preceding_actions: list of action dicts before the target
          - skip_reason: string describing why skipped, or None if successful
    """
    n = len(actions)
    if n < min_events + 1:
        return None, None, None, None, "too_few_actions"

    # Walk backwards to find a valid target
    for target_idx in range(n - 1, min_events - 1, -1):
        candidate = actions[target_idx]

        # Must have a valid item ID
        if not has_valid_item_id(candidate, use_page_title=use_page_title):
            continue

        # Must have at least min_events actions before it
        if target_idx < min_events:
            continue

        # Get item ID and check TID
        item_id = get_action_item_id(candidate, use_page_title=use_page_title)
        if not item_id:
            continue
        tid = get_item_tid(item_id, id2meta)
        if tid is None:
            # No valid TID for this item; keep searching backwards
            continue

        # Parse timestamps for gap check
        candidate_ts = parse_timestamp(candidate.get("Timestamp", ""))
        if candidate_ts is None:
            continue

        # Check gap with the immediately preceding action
        prev_action = actions[target_idx - 1]
        prev_ts = parse_timestamp(prev_action.get("Timestamp", ""))
        if prev_ts is None:
            continue

        time_gap = abs(candidate_ts - prev_ts)
        if time_gap > gap_threshold:
            # This target is isolated from prior events; skip and keep looking
            continue

        # Valid target found
        preceding = actions[:target_idx]
        return candidate, item_id, tid, preceding, None

    return None, None, None, None, "no_valid_target"


def build_event_sequence(
    preceding_actions,
    target_timestamp,
    id2meta,
    page_title_items=None,
    item_data=None,
    max_events=20,
):
    """Build a formatted event sequence from preceding actions.

    Converts raw actions into human-readable event strings in the format:
      "time_ago | action_type | description"

    Events are ordered chronologically (oldest first, newest last) to leverage
    LLM recency bias — the most recent events appear closest to the prediction
    target in the prompt.

    Args:
        preceding_actions: List of action dicts (chronologically sorted).
        target_timestamp: Float timestamp of the target action (used to
            compute relative time).
        id2meta: Item ID to metadata mapping.
        page_title_items: Optional PageTitle index to data mapping.
        max_events: Maximum number of events to include (most recent kept).

    Returns:
        Tuple of (event_strings, event_type_counts) where:
          - event_strings: list of formatted event strings
          - event_type_counts: dict of event type -> count
    """
    events = []
    type_counts = defaultdict(int)

    for action in preceding_actions:
        ts = parse_timestamp(action.get("Timestamp", ""))
        if ts is None:
            continue

        result = classify_action(action)
        if result is None:
            continue

        event_type, raw_desc = result

        # Build human-readable description
        if event_type in ("Clicked", "Browsed"):
            desc = get_item_description(raw_desc, id2meta, page_title_items, item_data)
        else:
            # Searched: use the query text directly
            desc = raw_desc
            
        if len(desc) > 150:
            desc = desc[:150] + "..."

        time_diff = target_timestamp - ts
        time_ago = format_time_ago(max(0, time_diff))

        event_str = f"{time_ago} | {event_type} | {desc}"
        events.append(event_str)
        type_counts[event_type] += 1

    # Keep only the most recent max_events
    if len(events) > max_events:
        events = events[-max_events:]

    return events, dict(type_counts)


def create_instruction():
    """Create the instruction text for the journey-to-product prediction task.

    Returns:
        Instruction string.
    """
    return (
        "Given the user's search queries, browsing history, and click history, "
        "predict the next product the user will interact with. Output strictly in the format: "
        "Item text ID: [s1, s2, s3, s4, s5, s6, s7].\n"
    )


def create_sft_sample(
    user_id,
    event_strings,
    target_action,
    target_item_id,
    target_tid,
    target_meta,
    event_type_counts,
    system_time_str="",
):
    """Create a single SFT training sample.

    Args:
        user_id: User identifier string.
        event_strings: List of formatted event strings.
        target_action: The target action dict.
        target_item_id: Item ID of the target product.
        target_tid: List of 7 summary words for the target product.
        target_meta: Metadata dict for the target product.
        event_type_counts: Dict of event type -> count.
        system_time_str: Optional system time string for context.

    Returns:
        SFT sample dict with instruction, input, output, and metadata.
    """
    instruction = create_instruction()

    # Build input: numbered event list
    input_lines = []
    if system_time_str:
        input_lines.append(f"Current time: {system_time_str}")
        input_lines.append("")

    input_lines.append("User Event History:")
    for idx, event in enumerate(event_strings, 1):
        input_lines.append(f"{idx} | {event}")

    input_lines.append("")
    input_lines.append("Predict the next product's text ID:")

    input_text = "\n".join(input_lines)

    # Build output: text ID of the target product
    output_text = "Item text ID: [" + ", ".join(target_tid) + "]"

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": {
            "user_id": user_id,
            "target_item_id": target_item_id,
            "target_title": target_meta.get("title", ""),
            "num_events": len(event_strings),
            "event_type_counts": event_type_counts,
        },
        "target_tid": target_tid,
        "target_meta": target_meta,
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def create_event2product_sft_data(
    full_sequential_data,
    id2meta,
    output_dir,
    page_title_items=None,
    item_data=None,
    max_events=20,
    gap_threshold=DEFAULT_GAP_THRESHOLD,
):
    """Create SFT data from full sequential interaction data.

    When page_title_items is None, only GlobalOfferId is used for target
    selection; PageTitle text appears as raw browsing descriptions in the
    event sequence. When page_title_items is provided, both GlobalOfferId
    and PageTitle IDs are considered as potential targets.

    Args:
        full_sequential_data: Dict of UserId -> list of action dicts.
        id2meta: Item ID to metadata mapping.
        output_dir: Directory for output files.
        page_title_items: Optional PageTitle index to data mapping.
            If None, targets are selected by GlobalOfferId only.
        max_events: Maximum events per input sequence.
        gap_threshold: Maximum time gap (seconds) between target and
            preceding action.

    Returns:
        List of SFT sample dicts.
    """
    sft_data = []
    skip_reasons = defaultdict(int)
    total_users = len(full_sequential_data)
    use_page_title = page_title_items is not None

    # Statistics
    event_counts = []
    type_dist = defaultdict(int)

    for user_id, actions in tqdm(
        full_sequential_data.items(), desc="Building event2product SFT data"
    ):
        if not actions:
            skip_reasons["empty_actions"] += 1
            continue

        # Find target with TID validation integrated into the search
        (target_action, target_item_id, target_tid,
         preceding_actions, skip_reason) = find_target_and_context(
            actions, gap_threshold, id2meta,
            use_page_title=use_page_title,
        )

        if skip_reason:
            skip_reasons[skip_reason] += 1
            continue

        target_meta = id2meta.get(target_item_id, {})

        # Parse target timestamp for relative time computation
        target_ts = parse_timestamp(target_action.get("Timestamp", ""))
        if target_ts is None:
            skip_reasons["bad_target_timestamp"] += 1
            continue

        # Optionally compute a system time string from the target timestamp
        try:
            dt = datetime.fromtimestamp(target_ts, tz=timezone.utc)
            system_time_str = dt.strftime("%-m/%-d/%Y")
        except (OSError, ValueError):
            system_time_str = ""

        # Build the event sequence
        event_strings, event_type_counts = build_event_sequence(
            preceding_actions,
            target_ts,
            id2meta,
            page_title_items=page_title_items,
            item_data=item_data,
            max_events=max_events,
        )

        # Require at least MIN_PRECEDING_EVENTS classifiable events
        if len(event_strings) < MIN_PRECEDING_EVENTS:
            skip_reasons["too_few_classifiable_events"] += 1
            continue

        # Create the SFT sample
        sample = create_sft_sample(
            user_id=user_id,
            event_strings=event_strings,
            target_action=target_action,
            target_item_id=target_item_id,
            target_tid=target_tid,
            target_meta=target_meta,
            event_type_counts=event_type_counts,
            system_time_str=system_time_str,
        )
        sft_data.append(sample)

        # Track statistics
        event_counts.append(len(event_strings))
        for etype, cnt in event_type_counts.items():
            type_dist[etype] += cnt

    # Print statistics
    print(f"\nData statistics:")
    print(f"  Total users:                  {total_users:>10,}")
    print(f"  Generated samples:            {len(sft_data):>10,}")
    print(f"  Skipped users:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"    - {reason:30s} {count:>10,}")

    if event_counts:
        arr = np.array(event_counts)
        print(f"\n  Event count per sample:")
        print(f"    Min:    {arr.min()}")
        print(f"    Max:    {arr.max()}")
        print(f"    Mean:   {arr.mean():.2f}")
        print(f"    Median: {np.median(arr):.1f}")
        print(f"    P25:    {np.percentile(arr, 25):.1f}")
        print(f"    P75:    {np.percentile(arr, 75):.1f}")
        print(f"    P95:    {np.percentile(arr, 95):.1f}")

    if type_dist:
        total_events = sum(type_dist.values())
        print(f"\n  Event type distribution:")
        for etype, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
            pct = cnt / total_events * 100 if total_events > 0 else 0
            print(f"    {etype:20s} {cnt:>10,} ({pct:5.1f}%)")

    return sft_data


def save_sft_data(sft_data, output_file):
    """Save SFT data (full and training versions).

    Full version (with metadata): <name>_full.json
    Training version (instruction/input/output only): <name>.json

    Args:
        sft_data: List of SFT sample dicts.
        output_file: Path to the training JSON file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Full version with all metadata
    full_file = output_file.replace(".json", "_full.json")
    with open(full_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"Full data saved: {full_file}")

    # Training version (instruction, input, output only)
    training_data = [
        {
            "instruction": s["instruction"],
            "input": s["input"],
            "output": s["output"],
        }
        for s in sft_data
    ]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    print(f"Training data saved: {output_file}")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build event-to-product SFT data from user event sequences"
    )
    parser.add_argument(
        "--full_sequential_file",
        type=str,
        default="./raw_data/full_sequential_data_sample.json",
        help="Path to full_sequential_data.json from s2_construct_sequential_data "
             "(default: ./raw_data/full_sequential_data.json)",
    )
    parser.add_argument(
        "--id2meta_file",
        type=str,
        default="./processed/id2meta.json",
        help="Path to id2meta JSON from s1_generate_tid "
             "(default: ./processed/id2meta.json)",
    )
    parser.add_argument(
        "--page_title_items_file",
        type=str,
        default="",
        help="Path to page_title_item.json from s1_construct_page_title_data. "
             "If empty (default), only GlobalOfferId is used for targets "
             "and PageTitle appears as raw text in events.",
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="./raw_data/merged_clean_item.json",
        help="Path to full item metadata JSON for description fallback "
             "(default: ./raw_data/merged_clean_item.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_data",
        help="Output directory. SFT data saved to <output_dir>/event2product_sft.json "
             "(default: ./sft_data)",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=50,
        help="Maximum number of events per input sequence",
    )
    parser.add_argument(
        "--gap_threshold",
        type=float,
        default=DEFAULT_GAP_THRESHOLD,
        help="Maximum time gap in seconds between target action and preceding "
             "action. Actions with larger gaps are considered isolated and "
             "skipped (default: 7200 = 2 hours)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # =========================================================================
    # Step 1: Load input files
    # =========================================================================
    print("=" * 70)
    print("Step 1: Loading input files")
    print("=" * 70)

    print(f"  Loading full sequential data: {args.full_sequential_file}")
    with open(args.full_sequential_file, "r", encoding="utf-8") as f:
        full_sequential_data = json.load(f)
    print(f"    Users: {len(full_sequential_data):,}")
    total_actions = sum(len(v) for v in full_sequential_data.values())
    print(f"    Total actions: {total_actions:,}")

    print(f"  Loading id2meta: {args.id2meta_file}")
    with open(args.id2meta_file, "r", encoding="utf-8") as f:
        id2meta = json.load(f)
    print(f"    Items: {len(id2meta):,}")

    # Load page title items (optional — empty path disables PageTitle ID mode)
    page_title_items = None
    if args.page_title_items_file:
        if os.path.exists(args.page_title_items_file):
            print(f"  Loading page title items: {args.page_title_items_file}")
            with open(args.page_title_items_file, "r", encoding="utf-8") as f:
                page_title_items = json.load(f)
            print(f"    Page title items: {len(page_title_items):,}")
        else:
            print(f"  WARNING: page_title_items_file not found: "
                  f"{args.page_title_items_file}")
            print(f"    Falling back to GlobalOfferId-only mode.")
    else:
        print(f"  Page title items: disabled (GlobalOfferId-only mode)")
        print(f"    Targets use GlobalOfferId only; PageTitle used as raw text.")

    # Load full item data for description fallback
    item_data = None
    if os.path.exists(args.item_file):
        print(f"  Loading item data (fallback): {args.item_file}")
        with open(args.item_file, "r", encoding="utf-8") as f:
            item_data = json.load(f)
        print(f"    Items: {len(item_data):,}")
    else:
        print(f"  Item data file not found: {args.item_file}")

    # =========================================================================
    # Step 2: Build SFT data
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Building event2product SFT data")
    print(f"  mode = {'GID+PageTitle' if page_title_items else 'GlobalOfferId-only'}")
    print(f"  gap_threshold = {args.gap_threshold:.0f}s "
          f"({args.gap_threshold / 3600:.1f}h)")
    print(f"  max_events = {args.max_events}")
    print("=" * 70)

    output_file = os.path.join(args.output_dir, "event2product_sft.json")

    sft_data = create_event2product_sft_data(
        full_sequential_data,
        id2meta,
        args.output_dir,
        page_title_items=page_title_items,
        item_data=item_data,
        max_events=args.max_events,
        gap_threshold=args.gap_threshold,
    )

    # =========================================================================
    # Step 3: Save output
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Saving output")
    print("=" * 70)

    save_sft_data(sft_data, output_file)

    # =========================================================================
    # Step 4: Show example cases
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("Example cases (first 3):")
    print(f"{'=' * 70}")
    for idx, sample in enumerate(sft_data[:3]):
        meta = sample["metadata"]
        print(f"\n--- Example {idx + 1} ---")
        print(f"  User ID:          {meta['user_id']}")
        print(f"  Num events:       {meta['num_events']}")
        print(f"  Event types:      {meta['event_type_counts']}")
        print(f"  Target item:      {meta['target_item_id']}")
        print(f"  Target title:     {meta['target_title'][:80]}")
        print(f"  Target TID:       {sample['target_tid']}")
        print(f"  Instruction:      {sample['instruction'][:100]}...")
        print(f"  Input (first 5 lines):")
        for line in sample["input"].split("\n")[:5]:
            print(f"    {line}")
        print(f"  Output:           {sample['output']}")
    print(f"\n{'=' * 70}")

    print(f"\nDone! Generated {len(sft_data)} training samples")


if __name__ == "__main__":
    main()
