"""Step 2d: Build Shopping Journey Prediction SFT Data

Creates SFT training data for predicting multiple shopping journeys from
a user's browsing/search/purchase history.  Given a user's event history
and the current time, the model generates a structured JSON containing
one or more shopping journeys, each with a title, search query, and a
list of recommended products represented as text IDs (TIDs).

This differs from journey-to-product SFT (s2_build_journey2product_sft_data.py)
in several key ways:
  - Output is MULTIPLE journeys (not a single next product).
  - Each journey is a high-level shopping intent with a title + query +
    multiple product recommendations.
  - Products are represented as 7-word text IDs (TIDs), not raw item IDs.
  - The output is structured JSON, not a single text ID line.

Data source:
  shopping_journey.json (from preprocess_raw_data/s3_construct_shopping_journey.py)
    Format: { uuid: {user_shopping_events, system_time, journeys: [{title, query, product_ids}]} }
  id2meta.json (from s1_generate_tid.py)
    Format: { ItemId: {title, description, categories, summary_words, ...} }

Pipeline:
  1. Load shopping_journey.json and id2meta.json.
  2. For each user entry, resolve product_ids to TIDs via id2meta.
  3. Skip journeys where no product has a valid TID; skip users with
     no valid journeys or no events.
  4. Build instruction/input/output for SFT training.  Output is a JSON
     string with the ContinuedJourneys structure.
  5. Save full and simplified versions.

Usage:
    python s2_build_journey_prediction_sft_data.py \
        --shopping_journey_file ./raw_data/shopping_journey.json \
        --id2meta_file ./processed/id2meta.json \
        --output_dir ./sft_data \
        --max_events 20 \
        --max_products_per_journey 6
"""

import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Maximum number of user events to include in the input prompt.
DEFAULT_MAX_EVENTS = 50

# Maximum number of product TIDs to include per journey in the output.
DEFAULT_MAX_PRODUCTS = 10


# =============================================================================
# Utility Functions
# =============================================================================

def get_item_tid(item_id, id2meta):
    """Get the text ID (7 summary words) for an item.

    Args:
        item_id: Item identifier string (GlobalOfferId).
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


def resolve_journey_tids(journey, id2meta, max_products):
    """Resolve a journey's product_ids to text IDs.

    Args:
        journey: Dict with keys title, reason, product_ids.
        id2meta: Item ID to metadata mapping.
        max_products: Maximum number of products to include.

    Returns:
        Dict with title, reason, product_tids (list of 7-word lists),
        or None if no products could be resolved.
    """
    product_tids = []
    for pid in journey.get("product_ids", []):
        tid = get_item_tid(pid, id2meta)
        if tid is not None:
            product_tids.append(tid)
            if len(product_tids) >= max_products:
                break

    if not product_tids:
        return None

    return {
        "title": journey.get("title", ""),
        "reason": journey.get("reason", ""),
        "product_tids": product_tids,
    }


def build_output_json(resolved_journeys):
    """Build the structured JSON output string for SFT training.

    Output format:
    {"ContinuedJourneys":[{"Title":"...","Reason":"...","ProductTIDs":[["a","b",...],...]},...]}"

    Args:
        resolved_journeys: List of dicts with title, reason, product_tids.

    Returns:
        JSON string.
    """
    continued = []
    for j in resolved_journeys:
        continued.append({
            "Title": j["title"],
            "Reason": j["reason"],
            "ProductTIDs": j["product_tids"],
        })
    return json.dumps({"ContinuedJourneys": continued}, ensure_ascii=False)


def create_instruction(num_journeys):
    """Create the instruction text for the journey prediction task.

    Args:
        num_journeys: Number of journeys to predict.

    Returns:
        Instruction string.
    """
    return (
        f"Based on the user's shopping event history, predict {num_journeys} shopping "
        "journey(s) the user is likely to pursue. Each journey includes a "
        "title, a reason, and recommended products as text IDs (7 slots each). "
        '{"ContinuedJourneys":[{"Title":"...","Reason":"...",'
        '"ProductTIDs":[["s1","s2","s3","s4","s5","s6","s7"],...]},...]}.'
    )


def create_sft_sample(
    user_id,
    user_events,
    resolved_journeys,
    max_events,
):
    """Create a single SFT training sample.

    Args:
        user_id: User identifier string.
        user_events: List of event strings.
        resolved_journeys: List of resolved journey dicts.
        max_events: Maximum events to include.

    Returns:
        SFT sample dict.
    """
    instruction = create_instruction(len(resolved_journeys))

    # Build input: numbered event list
    input_lines = []

    # Truncate to most recent events (events are ordered newest-first)
    events = user_events[:max_events]

    input_lines.append("User Event History:")
    for idx, event in enumerate(events, 1):
        if len(event) > 150:
            event = event[:150] + "..."
        input_lines.append(f"{idx} | {event}")

    input_lines.append("")
    input_lines.append("Predict the user's shopping journeys:")

    input_text = "\n".join(input_lines)

    # Build output: structured JSON
    output_text = build_output_json(resolved_journeys)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": {
            "user_id": user_id,
            "num_events": len(events),
            "num_journeys": len(resolved_journeys),
            "num_products_per_journey": [
                len(j["product_tids"]) for j in resolved_journeys
            ],
        },
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def create_event2journey_sft_data(
    shopping_journey_data,
    id2meta,
    max_events=DEFAULT_MAX_EVENTS,
    max_products=DEFAULT_MAX_PRODUCTS,
):
    """Create SFT data from event-to-journey predictions.

    Args:
        shopping_journey_data: Dict of user_id -> journey entry.
        id2meta: Item ID to metadata mapping.
        max_events: Maximum events per input sequence.
        max_products: Maximum products per journey.

    Returns:
        List of SFT sample dicts.
    """
    sft_data = []
    skip_reasons = defaultdict(int)
    total_entries = len(shopping_journey_data)

    # Statistics
    event_counts = []
    journey_counts = []
    product_counts = []

    for user_id, entry in tqdm(
        shopping_journey_data.items(), desc="Building event2journey SFT data"
    ):
        user_events = entry.get("user_shopping_events", [])
        journeys = entry.get("journeys", [])

        # Validate
        if not user_events:
            skip_reasons["no_user_events"] += 1
            continue

        # Resolve journeys' product IDs to TIDs (may be empty)
        resolved_journeys = []
        for journey in journeys:
            resolved = resolve_journey_tids(journey, id2meta, max_products)
            if resolved is not None:
                resolved_journeys.append(resolved)

        is_empty_journey = (len(resolved_journeys) == 0)
        if is_empty_journey:
            skip_reasons["empty_journeys"] += 1

        # Create the SFT sample
        sample = create_sft_sample(
            user_id=user_id,
            user_events=user_events,
            resolved_journeys=resolved_journeys,
            max_events=max_events,
        )
        sft_data.append(sample)

        # Track statistics
        event_counts.append(sample["metadata"]["num_events"])
        journey_counts.append(len(resolved_journeys))
        for j in resolved_journeys:
            product_counts.append(len(j["product_tids"]))

    # Print statistics
    print(f"\nData statistics:")
    print(f"  Total entries:                {total_entries:>10,}")
    print(f"  Generated samples:            {len(sft_data):>10,}")
    print(f"  Skipped entries:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"    - {reason:30s} {count:>10,}")

    if event_counts:
        arr = np.array(event_counts)
        print(f"\n  Events per sample:")
        print(f"    Min: {arr.min()}, Max: {arr.max()}, "
              f"Mean: {arr.mean():.1f}, Median: {np.median(arr):.1f}")

    if journey_counts:
        arr = np.array(journey_counts)
        print(f"\n  Journeys per sample:")
        print(f"    Min: {arr.min()}, Max: {arr.max()}, "
              f"Mean: {arr.mean():.1f}, Median: {np.median(arr):.1f}")
        jc_dist = defaultdict(int)
        for c in journey_counts:
            jc_dist[c] += 1
        for cnt in sorted(jc_dist.keys()):
            print(f"    {cnt} journey(s): {jc_dist[cnt]:>6,} samples")

    if product_counts:
        arr = np.array(product_counts)
        print(f"\n  Products per journey:")
        print(f"    Min: {arr.min()}, Max: {arr.max()}, "
              f"Mean: {arr.mean():.1f}, Median: {np.median(arr):.1f}")

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
        description="Build shopping journey prediction SFT data"
    )
    parser.add_argument(
        "--shopping_journey_file",
        type=str,
        default="./raw_data/shopping_journeys.json",
        help="Path to shopping_journeys.json from pre_s3_construct_shopping_journey "
             "(default: ./raw_data/shopping_journeys.json)",
    )
    parser.add_argument(
        "--id2meta_file",
        type=str,
        default="./processed/id2meta.json",
        help="Path to id2meta JSON from s1_generate_tid "
             "(default: ./processed/id2meta.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_data",
        help="Output directory (default: ./sft_data)",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help=f"Maximum events per input sequence (default: {DEFAULT_MAX_EVENTS})",
    )
    parser.add_argument(
        "--max_products_per_journey",
        type=int,
        default=DEFAULT_MAX_PRODUCTS,
        help=f"Maximum products per journey in output "
             f"(default: {DEFAULT_MAX_PRODUCTS})",
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

    print(f"  Loading shopping journeys: {args.shopping_journey_file}")
    with open(args.shopping_journey_file, "r", encoding="utf-8") as f:
        shopping_data = json.load(f)
    print(f"    Entries: {len(shopping_data):,}")

    print(f"  Loading id2meta: {args.id2meta_file}")
    with open(args.id2meta_file, "r", encoding="utf-8") as f:
        id2meta = json.load(f)
    print(f"    Items: {len(id2meta):,}")

    # Quick coverage check: how many product_ids in journeys exist in id2meta?
    all_pids = set()
    for entry in shopping_data.values():
        for j in entry.get("journeys", []):
            all_pids.update(j.get("product_ids", []))
    found = sum(1 for pid in all_pids if pid in id2meta)
    has_tid = sum(1 for pid in all_pids if get_item_tid(pid, id2meta) is not None)
    print(f"    Distinct product IDs in journeys: {len(all_pids):,}")
    print(f"    Found in id2meta: {found:,} ({found/len(all_pids)*100:.1f}%)")
    print(f"    With valid TID: {has_tid:,} ({has_tid/len(all_pids)*100:.1f}%)")

    # =========================================================================
    # Step 2: Build SFT data
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Building event2journey SFT data")
    print(f"  max_events = {args.max_events}")
    print(f"  max_products_per_journey = {args.max_products_per_journey}")
    print("=" * 70)

    sft_data = create_event2journey_sft_data(
        shopping_data,
        id2meta,
        max_events=args.max_events,
        max_products=args.max_products_per_journey,
    )

    # =========================================================================
    # Step 3: Save output
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Saving output")
    print("=" * 70)

    output_file = os.path.join(args.output_dir, "event2journey_sft.json")
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
        print(f"  UUID:           {meta['uuid']}")
        print(f"  Num events:     {meta['num_events']}")
        print(f"  Num journeys:   {meta['num_journeys']}")
        print(f"  Products/j:     {meta['num_products_per_journey']}")
        print(f"  Instruction:    {sample['instruction'][:100]}...")
        print(f"  Input (first 5 lines):")
        for line in sample["input"].split("\n")[:5]:
            print(f"    {line}")
        # Pretty-print the output JSON
        try:
            out_obj = json.loads(sample["output"])
            for ji, j in enumerate(out_obj["ContinuedJourneys"][:2]):
                print(f"  Journey {ji+1}: {j['Title']}")
                print(f"    Reason: {j['Reason']}")
                print(f"    Products: {len(j['ProductTIDs'])} TIDs")
                if j["ProductTIDs"]:
                    print(f"      [0]: {j['ProductTIDs'][0]}")
        except (json.JSONDecodeError, KeyError):
            print(f"  Output: {sample['output'][:200]}...")
    print(f"\n{'=' * 70}")

    print(f"\nDone! Generated {len(sft_data)} training samples")


if __name__ == "__main__":
    main()
