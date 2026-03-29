"""Step 2.6: Unified Journey SFT Data Builder (Event2Journey + Profile2Journey)

Unified pipeline for building SFT training data from user events/profiles
to shopping journey predictions.  Supports two tasks via --task:

  event2journey:
    Input: user event history -> Output: predicted shopping journeys

  profile2journey:
    Input: shopping profile + recent events -> Output: predicted shopping journeys

Compared to s2_4 (event2journey) and s2_5 (profile2journey), this unified
version adds:
  - --min_products_per_journey: drop journeys below this product count
  - --max_journeys: cap journey count per sample (random subsample if exceeded)
  - --keep_empty_ratio: retain a fraction of zero-journey samples so the model
    learns that some users have no meaningful upcoming journeys
  - Comprehensive unified statistics

Data sources:
  shopping_journeys.json
    { UserId: {user_shopping_events, journeys: [{title, reason, product_ids}]} }
  id2meta.json
    { ItemId: {title, description, categories, summary_words, ...} }
  shopping_profiles.tsv (profile2journey only)
    UserId \\t ShoppingProfile (JSON string)

Usage:
  # Event to journey
  python s2_6_build_journey_sft_data.py --task event2journey \\
      --shopping_journey_file ./raw_data/shopping_journeys.json \\
      --id2meta_file ./processed/id2meta.json \\
      --output_dir ./sft_data

  # Profile to journey
  python s2_6_build_journey_sft_data.py --task profile2journey \\
      --shopping_journey_file ./raw_data/shopping_journeys.json \\
      --shopping_profile_file ./raw_data/shopping_profiles.tsv \\
      --id2meta_file ./processed/id2meta.json \\
      --output_dir ./sft_data
"""

import os
import csv
import json
import sys
import random
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# Increase CSV field size limit to handle very large fields
csv.field_size_limit(sys.maxsize)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_EVENTS = 100
DEFAULT_MAX_RECENT_EVENTS = 100
DEFAULT_MAX_PRODUCTS = 20
DEFAULT_MIN_PRODUCTS = 5
DEFAULT_MIN_JOURNEYS = 1
DEFAULT_MAX_JOURNEYS = 10
DEFAULT_KEEP_EMPTY_RATIO = 0.8
DEFAULT_HALF_FILTERED_DROP = True


# =============================================================================
# Data Loading
# =============================================================================

def load_shopping_profiles(profile_file):
    """Load shopping profiles from TSV file.

    Args:
        profile_file: Path to shopping_profiles.tsv with columns
            UserId and ShoppingProfile (JSON string).

    Returns:
        Dict mapping UserId -> profile JSON string.
    """
    profiles = {}
    with open(profile_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return profiles

        col_map = {name.strip(): idx for idx, name in enumerate(header)}
        uid_idx = col_map.get("UserId", 0)
        profile_idx = col_map.get("ShoppingProfile", 1)

        for row in reader:
            if len(row) <= max(uid_idx, profile_idx):
                continue
            user_id = row[uid_idx].strip()
            profile_text = row[profile_idx].strip()
            if user_id and profile_text:
                profiles[user_id] = profile_text

    return profiles


# =============================================================================
# Shared Utility Functions
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
        product_ids (list of resolved item IDs, parallel to product_tids),
        or None if no products could be resolved.
    """
    product_tids = []
    resolved_pids = []
    for pid in journey.get("product_ids", []):
        tid = get_item_tid(pid, id2meta)
        if tid is not None:
            product_tids.append(tid)
            resolved_pids.append(pid)
            if len(product_tids) >= max_products:
                break

    if not product_tids:
        return None

    return {
        "title": journey.get("title", ""),
        "reason": journey.get("reason", ""),
        "product_tids": product_tids,
        "product_ids": resolved_pids,
    }


def _get_brand_seller(pid, id2meta):
    """Extract (brand, seller) from id2meta for a product id."""
    meta = id2meta.get(pid)
    if meta is None:
        return ("", "")
    attrs = meta.get("attributes", {})
    brand = attrs.get("Brand", "").strip().lower() if isinstance(attrs.get("Brand"), str) else ""
    seller = attrs.get("Seller", "").strip().lower() if isinstance(attrs.get("Seller"), str) else ""
    return (brand, seller)


def diversify_journey_products(product_tids, product_ids, id2meta,
                               dup_threshold=6):
    """Diversify products within a single journey via dedup + greedy reranking.

    Stage 1 — Hard dedup: remove products whose summary words overlap
    >= dup_threshold/7 with any already-selected product.

    Stage 2 — Greedy diversity reranking: iteratively pick the candidate
    with the lowest max-overlap against the already-selected set.
    Ties broken by brand/seller: prefer a product whose brand AND seller
    are both different from all selected products (more diverse).

    Args:
        product_tids: List of 7-word lists (parallel with product_ids).
        product_ids: List of item IDs (parallel with product_tids).
        id2meta: Item metadata dict for brand/seller lookup.
        dup_threshold: Min word overlap to consider near-duplicate (default 6).

    Returns:
        Tuple (deduped_tids, deduped_pids, num_removed_dedup) where
        deduped_tids/pids are reordered for diversity.
    """
    n = len(product_tids)
    if n <= 1:
        return product_tids, product_ids, 0

    # Convert tids to word sets for fast overlap computation
    word_sets = [set(tid) for tid in product_tids]

    # Stage 1: Hard dedup — remove near-duplicates
    keep_mask = [True] * n
    num_removed = 0
    for i in range(1, n):
        if not keep_mask[i]:
            continue
        for j in range(i):
            if not keep_mask[j]:
                continue
            overlap = len(word_sets[i] & word_sets[j])
            if overlap >= dup_threshold:
                keep_mask[i] = False
                num_removed += 1
                break

    # Build candidate pool after dedup
    cand_tids = [product_tids[i] for i in range(n) if keep_mask[i]]
    cand_pids = [product_ids[i] for i in range(n) if keep_mask[i]]
    cand_sets = [word_sets[i] for i in range(n) if keep_mask[i]]

    if len(cand_tids) <= 1:
        return cand_tids, cand_pids, num_removed

    # Pre-fetch brand/seller for candidates
    cand_bs = [_get_brand_seller(pid, id2meta) for pid in cand_pids]

    # Stage 2: Greedy diversity reranking
    # Start with the first candidate (highest ANN relevance)
    selected_idx = [0]
    remaining = set(range(1, len(cand_tids)))

    while remaining:
        best_i = None
        best_max_overlap = 8  # worse than any real overlap (max is 7)
        best_is_diverse = False

        for i in remaining:
            # max overlap with any selected product
            max_ov = max(len(cand_sets[i] & cand_sets[s]) for s in selected_idx)

            # brand/seller diversity bonus: True if brand AND seller
            # are both different from ALL selected products
            b_i, s_i = cand_bs[i]
            is_diverse = True
            if b_i or s_i:  # only check if this product has brand/seller info
                for s in selected_idx:
                    b_s, s_s = cand_bs[s]
                    if b_i and b_s and b_i == b_s:
                        is_diverse = False
                        break
                    if s_i and s_s and s_i == s_s:
                        is_diverse = False
                        break

            # Prefer: lower max_overlap, then diverse brand/seller
            if (max_ov < best_max_overlap or
                    (max_ov == best_max_overlap and is_diverse and not best_is_diverse)):
                best_max_overlap = max_ov
                best_is_diverse = is_diverse
                best_i = i

        selected_idx.append(best_i)
        remaining.remove(best_i)

    reranked_tids = [cand_tids[i] for i in selected_idx]
    reranked_pids = [cand_pids[i] for i in selected_idx]
    return reranked_tids, reranked_pids, num_removed


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


# =============================================================================
# Task-Specific Builders
# =============================================================================

def create_instruction(task, num_journeys, min_products_in_user):
    """Create instruction text based on task and journey count.

    50% of the time (when num_journeys > 0), the specific journey count is
    omitted so the model learns to decide how many journeys to generate.
    The "at least N products" is set to the minimum product count across
    this user's journeys, so the instruction matches the actual output.

    Returns:
        Tuple of (instruction_text, has_count) where has_count indicates
        whether the specific journey count was included.
    """
    if task == "event2journey":
        base = "Based on the user's shopping event history, "
    else:
        base = "Based on the user's shopping profile and recent shopping events, "

    # 50% chance to specify the count (only when non-zero)
    has_count = num_journeys > 0 and random.random() < 0.5
    if has_count:
        jword = "journey" if num_journeys == 1 else "journeys"
        base += f"predict {num_journeys} distinct shopping {jword} "
    else:
        base += "predict the shopping journey(s) "
    base += "the user is likely to pursue. "

    base += (
        "Each journey represents a different product category. "
        "Each journey has a short, engaging title, "
        "a brief user-centric reason referencing the user's history, "
        f"and at least {min_products_in_user} recommended products as text IDs (7 slots each). "
        "Products within each journey should cover different brands, styles, use cases "
        "and subcategories — avoid recommending near-identical items. "
        "If no meaningful journeys can be inferred, output an empty list. "
        'Output JSON: {"ContinuedJourneys":[{"Title":"...","Reason":"...",'
        '"ProductTIDs":[["s1","s2","s3","s4","s5","s6","s7"],...]},...]}.'
    )
    return base, has_count


def build_input_text(task, user_events, max_events,
                     profile_text=None, max_recent_events=None):
    """Build input text based on task type.

    Args:
        task: "event2journey" or "profile2journey".
        user_events: Full list of user event strings (newest first).
        max_events: Max events for event2journey.
        profile_text: Profile JSON string (profile2journey only).
        max_recent_events: Max recent events for profile2journey.

    Returns:
        Tuple of (input_text, num_events_used).
    """
    if task == "event2journey":
        events = user_events[:max_events]
        lines = ["User Event History:"]
        for idx, event in enumerate(events, 1):
            if len(event) > 150:
                event = event[:150] + "..."
            lines.append(f"{idx} | {event}")
        lines.append("")
        lines.append("Predict the user's shopping journeys:")
        return "\n".join(lines), len(events)
    else:  # profile2journey
        n = max_recent_events or DEFAULT_MAX_RECENT_EVENTS
        recent = user_events[:n]
        lines = [
            "User Shopping Profile:",
            profile_text or "",
            "",
            "Recent Shopping Events:",
        ]
        for idx, event in enumerate(recent, 1):
            if len(event) > 150:
                event = event[:150] + "..."
            lines.append(f"{idx} | {event}")
        lines.append("")
        lines.append("Predict the user's shopping journeys:")
        return "\n".join(lines), len(recent)


# =============================================================================
# Save
# =============================================================================

def save_sft_data(sft_data, output_file):
    """Save SFT data (full and training versions).

    Full version (with metadata): <name>_full.json
    Training version (instruction/input/output only): <name>.json
    """
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Full version with all metadata
    full_file = output_file.replace(".json", "_full.json")
    with open(full_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    full_mb = os.path.getsize(full_file) / (1024 * 1024)
    print(f"Full data saved: {full_file} ({full_mb:.1f} MB)")

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
    train_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Training data saved: {output_file} ({train_mb:.1f} MB)")


# =============================================================================
# Main Pipeline
# =============================================================================

def create_sft_data(
    task,
    shopping_journey_data,
    id2meta,
    profiles=None,
    max_events=DEFAULT_MAX_EVENTS,
    max_recent_events=DEFAULT_MAX_RECENT_EVENTS,
    max_products=DEFAULT_MAX_PRODUCTS,
    min_products=DEFAULT_MIN_PRODUCTS,
    min_journeys=DEFAULT_MIN_JOURNEYS,
    max_journeys=DEFAULT_MAX_JOURNEYS,
    keep_empty_ratio=DEFAULT_KEEP_EMPTY_RATIO,
):
    """Create SFT data for the specified task.

    Args:
        task: "event2journey" or "profile2journey".
        shopping_journey_data: Dict of UserId -> journey entry.
        id2meta: Item ID to metadata mapping.
        profiles: Dict of UserId -> profile JSON string (profile2journey).
        max_events: Max events for event2journey input.
        max_recent_events: Max recent events for profile2journey input.
        max_products: Max products per journey in output.
        min_products: Min products per journey; journeys below are dropped.
        min_journeys: Min journeys per user after filtering (default 1).
        max_journeys: Max journeys per sample; excess are subsampled (default 10).
        keep_empty_ratio: Fraction of zero-journey users to retain.

    Returns:
        List of SFT sample dicts.
    """
    sft_data = []
    skip_reasons = defaultdict(int)
    total_entries = len(shopping_journey_data)

    # Statistics
    event_counts = []
    journey_counts = []       # per-user journey count (after all filtering)
    product_counts = []       # per-journey product count (after all filtering)
    user_total_products = []  # per-user total product count
    original_products_per_journey = []
    resolved_products_per_journey = []
    skipped_journeys_no_tid = 0
    skipped_journeys_min_products = 0
    empty_journey_total = 0
    empty_journey_kept = 0
    subsampled_users = 0
    original_journey_counts_before_subsample = []
    instruction_with_count = 0
    instruction_without_count = 0
    users_half_filtered = 0        # users dropped because >= 50% journeys filtered
    users_below_min_journeys = 0   # users dropped because < min_journeys after filtering
    min_product_values_in_instruction = []  # track the dynamic "at least N" values
    # Diversity statistics
    total_dedup_removed = 0
    products_before_diversity = []
    products_after_diversity = []

    for user_id, entry in tqdm(
        shopping_journey_data.items(),
        desc=f"Building {task} SFT data",
    ):
        user_events = entry.get("user_shopping_events", [])
        journeys = entry.get("journeys", [])

        if not user_events:
            skip_reasons["no_user_events"] += 1
            continue

        # Task-specific: profile2journey requires a profile
        if task == "profile2journey":
            if profiles is None or user_id not in profiles:
                skip_reasons["no_profile"] += 1
                continue

        # Resolve all journeys' product IDs to TIDs, then diversify
        resolved_journeys = []
        journeys_before_filter = 0
        for journey in journeys:
            orig_count = len(journey.get("product_ids", []))
            resolved = resolve_journey_tids(journey, id2meta, max_products)
            if resolved is None:
                skipped_journeys_no_tid += 1
                journeys_before_filter += 1
                continue
            journeys_before_filter += 1

            # Apply diversity: hard dedup + greedy reranking
            pre_div_count = len(resolved["product_tids"])
            div_tids, div_pids, n_removed = diversify_journey_products(
                resolved["product_tids"], resolved["product_ids"], id2meta,
            )
            resolved["product_tids"] = div_tids
            resolved["product_ids"] = div_pids
            total_dedup_removed += n_removed
            products_before_diversity.append(pre_div_count)
            products_after_diversity.append(len(div_tids))

            # Apply min_products filter AFTER diversity
            if len(resolved["product_tids"]) < min_products:
                skipped_journeys_min_products += 1
                continue
            original_products_per_journey.append(orig_count)
            resolved_products_per_journey.append(len(resolved["product_tids"]))
            resolved_journeys.append(resolved)

        # Check if >= 50% of this user's journeys were filtered out
        if journeys_before_filter > 0 and resolved_journeys:
            filtered_count = journeys_before_filter - len(resolved_journeys)
            if filtered_count >= journeys_before_filter / 2:
                users_half_filtered += 1
                skip_reasons["half_journeys_filtered"] += 1
                continue

        # Handle empty journeys: only keep users who originally had NO journeys
        # (not users whose journeys were all filtered out)
        if not resolved_journeys:
            if not journeys:
                # Originally had no journeys — candidate for empty sample
                empty_journey_total += 1
                if random.random() >= keep_empty_ratio:
                    skip_reasons["empty_journeys_sampled_out"] += 1
                    continue
                empty_journey_kept += 1
            else:
                # Had journeys but all were filtered out — skip entirely
                skip_reasons["all_journeys_filtered"] += 1
                continue

        # Journey subsampling if exceeding max_journeys
        if max_journeys is not None and len(resolved_journeys) > max_journeys:
            subsampled_users += 1
            original_journey_counts_before_subsample.append(len(resolved_journeys))
            resolved_journeys = random.sample(resolved_journeys, max_journeys)

        # Check min_journeys (skip if below, but allow empty-journey samples)
        if resolved_journeys and len(resolved_journeys) < min_journeys:
            users_below_min_journeys += 1
            skip_reasons["below_min_journeys"] += 1
            continue

        # Compute num_journeys and min_products_in_user from FINAL output
        final_num_journeys = len(resolved_journeys)
        if resolved_journeys:
            min_products_in_user = min(
                len(j["product_tids"]) for j in resolved_journeys
            )
        else:
            min_products_in_user = min_products

        # Build input text
        if task == "event2journey":
            input_text, num_events_used = build_input_text(
                task, user_events, max_events,
            )
        else:  # profile2journey
            input_text, num_events_used = build_input_text(
                task, user_events, max_events,
                profile_text=profiles[user_id],
                max_recent_events=max_recent_events,
            )

        # Instruction uses FINAL counts (after diversity + subsampling)
        instruction, has_count = create_instruction(
            task, final_num_journeys, min_products_in_user,
        )
        if has_count:
            instruction_with_count += 1
        else:
            instruction_without_count += 1
        min_product_values_in_instruction.append(min_products_in_user)
        output_text = build_output_json(resolved_journeys)

        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "user_id": user_id,
                "task": task,
                "num_events": num_events_used,
                "num_journeys": final_num_journeys,
                "num_products_per_journey": [
                    len(j["product_tids"]) for j in resolved_journeys
                ],
            },
        }
        sft_data.append(sample)

        event_counts.append(num_events_used)
        journey_counts.append(final_num_journeys)
        user_prod_total = 0
        for j in resolved_journeys:
            n = len(j["product_tids"])
            product_counts.append(n)
            user_prod_total += n
        user_total_products.append(user_prod_total)

    # =========================================================================
    # Comprehensive Statistics
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"Data Statistics ({task})")
    print(f"{'=' * 70}")
    print(f"  Total entries in data:        {total_entries:>10,}")
    if task == "profile2journey" and profiles is not None:
        print(f"  Total profiles available:     {len(profiles):>10,}")
    print(f"  Generated samples:            {len(sft_data):>10,}")

    # Skip reasons
    print(f"\n  --- Skip Reasons ---")
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:35s} {count:>10,}")
    else:
        print(f"    (none)")

    # Journey resolution
    total_input_journeys = (len(original_products_per_journey)
                            + skipped_journeys_no_tid
                            + skipped_journeys_min_products)
    print(f"\n  --- Journey Resolution ---")
    print(f"  Total journeys in data:       {total_input_journeys:>10,}")
    print(f"  Kept (>= {min_products} product(s)):     {len(original_products_per_journey):>10,}")
    print(f"  Dropped (0 TIDs resolved):    {skipped_journeys_no_tid:>10,}")
    print(f"  Dropped (< {min_products} products):     {skipped_journeys_min_products:>10,}")

    # User-level filtering
    print(f"\n  --- User-Level Filtering ---")
    print(f"  Users dropped (>= 50% journeys filtered): {users_half_filtered:>10,}")
    print(f"  Users dropped (< {min_journeys} journey after filter): {users_below_min_journeys:>10,}")

    # Empty journey handling
    print(f"\n  --- Empty Journey Handling ---")
    print(f"  Users with 0 valid journeys:  {empty_journey_total:>10,}")
    print(f"  Kept as empty samples:        {empty_journey_kept:>10,}")
    print(f"  Sampled out:                  {empty_journey_total - empty_journey_kept:>10,}")
    print(f"  Keep ratio (config):          {keep_empty_ratio:>10.1%}")
    if empty_journey_total > 0:
        actual_ratio = empty_journey_kept / empty_journey_total
        print(f"  Actual kept ratio:            {actual_ratio:>10.1%}")

    # Journey subsampling
    if max_journeys is not None:
        print(f"\n  --- Journey Subsampling ---")
        print(f"  max_journeys:                 {max_journeys:>10}")
        print(f"  Users subsampled:             {subsampled_users:>10,}")
        if original_journey_counts_before_subsample:
            arr = np.array(original_journey_counts_before_subsample)
            print(f"  Orig journeys (subsampled users): "
                  f"Mean={arr.mean():.1f}, Max={arr.max()}")

    # Instruction count inclusion
    total_instructions = instruction_with_count + instruction_without_count
    print(f"\n  --- Instruction Variants ---")
    print(f"  With journey count:           {instruction_with_count:>10,} "
          f"({instruction_with_count / max(total_instructions, 1) * 100:.1f}%)")
    print(f"  Without journey count:        {instruction_without_count:>10,} "
          f"({instruction_without_count / max(total_instructions, 1) * 100:.1f}%)")
    if min_product_values_in_instruction:
        arr = np.array(min_product_values_in_instruction)
        print(f"  'at least N' in instruction:  "
              f"Min={arr.min()}, Max={arr.max()}, Mean={arr.mean():.1f}, "
              f"Median={int(np.median(arr))}")

    # Event distribution
    if event_counts:
        arr = np.array(event_counts)
        print(f"\n  --- Events per Sample ---")
        print(f"    Min: {arr.min():>6}  P25: {int(np.percentile(arr, 25)):>6}  "
              f"P50: {int(np.percentile(arr, 50)):>6}  P75: {int(np.percentile(arr, 75)):>6}  "
              f"P90: {int(np.percentile(arr, 90)):>6}  Max: {arr.max():>6}  "
              f"Mean: {arr.mean():.1f}")

    # Journey count distribution (bucket view)
    if journey_counts:
        arr = np.array(journey_counts)
        print(f"\n  --- Journeys per User (after filtering) ---")
        print(f"    Min: {arr.min():>6}  Max: {arr.max():>6}  "
              f"Mean: {arr.mean():.1f}  Median: {np.median(arr):.1f}")
        jc_dist = defaultdict(int)
        for c in journey_counts:
            jc_dist[c] += 1
        print(f"    Bucket distribution:")
        for cnt in sorted(jc_dist.keys()):
            label = f"{cnt} journey" if cnt == 1 else f"{cnt} journeys"
            pct = jc_dist[cnt] / len(journey_counts) * 100
            bar = "#" * int(pct / 2)
            print(f"      {label:>12s}: {jc_dist[cnt]:>8,} users ({pct:5.1f}%) {bar}")

    # Products per journey
    if product_counts:
        arr = np.array(product_counts)
        print(f"\n  --- Products per Journey (resolved TIDs) ---")
        print(f"    Min: {arr.min():>6}  P25: {int(np.percentile(arr, 25)):>6}  "
              f"P50: {int(np.percentile(arr, 50)):>6}  P75: {int(np.percentile(arr, 75)):>6}  "
              f"P90: {int(np.percentile(arr, 90)):>6}  Max: {arr.max():>6}  "
              f"Mean: {arr.mean():.1f}")
        pc_dist = defaultdict(int)
        for c in product_counts:
            pc_dist[c] += 1
        print(f"    Distribution:")
        for cnt in sorted(pc_dist.keys()):
            pct = pc_dist[cnt] / len(product_counts) * 100
            print(f"      {cnt:>3} products: {pc_dist[cnt]:>10,} journeys ({pct:5.1f}%)")

    # Total products per user
    if user_total_products:
        arr = np.array(user_total_products)
        print(f"\n  --- Total Products per User ---")
        print(f"    Min: {arr.min():>6}  P25: {int(np.percentile(arr, 25)):>6}  "
              f"P50: {int(np.percentile(arr, 50)):>6}  P75: {int(np.percentile(arr, 75)):>6}  "
              f"P90: {int(np.percentile(arr, 90)):>6}  Max: {arr.max():>6}  "
              f"Mean: {arr.mean():.1f}")

    # Product TID resolution statistics
    if original_products_per_journey:
        orig_arr = np.array(original_products_per_journey)
        res_arr = np.array(resolved_products_per_journey)
        rates = res_arr / np.maximum(orig_arr, 1)
        print(f"\n  --- Product TID Resolution (per kept journey) ---")
        print(f"    Original products/journey:  "
              f"Mean={orig_arr.mean():.1f}, Median={np.median(orig_arr):.1f}, "
              f"Min={orig_arr.min()}, Max={orig_arr.max()}")
        print(f"    Resolved products/journey:  "
              f"Mean={res_arr.mean():.1f}, Median={np.median(res_arr):.1f}, "
              f"Min={res_arr.min()}, Max={res_arr.max()}")
        print(f"    Resolution rate:            "
              f"Mean={rates.mean():.1%}, Median={np.median(rates):.1%}")

    # Product diversity statistics
    if products_before_diversity:
        before_arr = np.array(products_before_diversity)
        after_arr = np.array(products_after_diversity)
        print(f"\n  --- Product Diversity (hard dedup + greedy reranking) ---")
        print(f"    Total near-duplicate products removed: {total_dedup_removed:>10,}")
        print(f"    Products/journey before diversity: "
              f"Mean={before_arr.mean():.1f}, Min={before_arr.min()}, Max={before_arr.max()}")
        print(f"    Products/journey after diversity:  "
              f"Mean={after_arr.mean():.1f}, Min={after_arr.min()}, Max={after_arr.max()}")
        reduced = before_arr - after_arr
        if reduced.sum() > 0:
            print(f"    Reduction per journey:            "
                  f"Mean={reduced.mean():.1f}, Max={reduced.max()}")

    return sft_data


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Journey SFT Data Builder "
                    "(event2journey / profile2journey)"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["event2journey", "profile2journey"],
        help="Task type: event2journey or profile2journey",
    )

    # Input files
    parser.add_argument(
        "--shopping_journey_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260324/raw_data/shopping_journeys.json",
        help="Path to shopping_journeys.json",
    )
    parser.add_argument(
        "--shopping_profile_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260324/raw_data/shopping_profiles_merged.tsv",
        help="Path to shopping_profiles.tsv (required for profile2journey)",
    )
    parser.add_argument(
        "--id2meta_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260324/processed/id2meta.json",
        help="Path to id2meta JSON from s1_generate_tid",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
                "Data/LLMTrainingData/20260324/sft_data",
        help="Output directory",
    )

    # Event/input controls
    parser.add_argument(
        "--max_events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help=f"Max events for event2journey input (default: {DEFAULT_MAX_EVENTS})",
    )
    parser.add_argument(
        "--max_recent_events",
        type=int,
        default=DEFAULT_MAX_RECENT_EVENTS,
        help=f"Max recent events for profile2journey input "
             f"(default: {DEFAULT_MAX_RECENT_EVENTS})",
    )

    # Journey/product controls
    parser.add_argument(
        "--max_products_per_journey",
        type=int,
        default=DEFAULT_MAX_PRODUCTS,
        help=f"Max products per journey (default: {DEFAULT_MAX_PRODUCTS})",
    )
    parser.add_argument(
        "--min_products_per_journey",
        type=int,
        default=DEFAULT_MIN_PRODUCTS,
        help=f"Min products per journey; journeys with fewer are dropped "
             f"(default: {DEFAULT_MIN_PRODUCTS})",
    )
    parser.add_argument(
        "--min_journeys",
        type=int,
        default=DEFAULT_MIN_JOURNEYS,
        help=f"Min journeys per user after filtering; users below are dropped "
             f"(default: {DEFAULT_MIN_JOURNEYS})",
    )
    parser.add_argument(
        "--max_journeys",
        type=int,
        default=DEFAULT_MAX_JOURNEYS,
        help=f"Max journeys per sample; excess are randomly subsampled "
             f"(default: {DEFAULT_MAX_JOURNEYS})",
    )
    parser.add_argument(
        "--keep_empty_ratio",
        type=float,
        default=DEFAULT_KEEP_EMPTY_RATIO,
        help=f"Fraction of zero-journey users to keep as training samples "
             f"(default: {DEFAULT_KEEP_EMPTY_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    task = args.task

    # Validate profile2journey requirements
    if task == "profile2journey" and not args.shopping_profile_file:
        print("ERROR: --shopping_profile_file is required for profile2journey task",
              file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # Step 1: Load input files
    # =========================================================================
    print("=" * 70)
    print(f"Step 1: Loading input files (task={task})")
    print("=" * 70)

    print(f"  Loading shopping journeys: {args.shopping_journey_file}")
    with open(args.shopping_journey_file, "r", encoding="utf-8") as f:
        shopping_data = json.load(f)
    print(f"    Entries: {len(shopping_data):,}")

    profiles = None
    if task == "profile2journey":
        print(f"  Loading shopping profiles: {args.shopping_profile_file}")
        profiles = load_shopping_profiles(args.shopping_profile_file)
        print(f"    Profiles: {len(profiles):,}")
        journey_uids = set(shopping_data.keys())
        profile_uids = set(profiles.keys())
        overlap = journey_uids & profile_uids
        print(f"    Journey users with profile: {len(overlap):,} / "
              f"{len(journey_uids):,} "
              f"({len(overlap) / max(len(journey_uids), 1) * 100:.1f}%)")

    print(f"  Loading id2meta: {args.id2meta_file}")
    with open(args.id2meta_file, "r", encoding="utf-8") as f:
        id2meta = json.load(f)
    print(f"    Items: {len(id2meta):,}")

    # Quick coverage check
    all_pids = set()
    for entry in shopping_data.values():
        for j in entry.get("journeys", []):
            all_pids.update(j.get("product_ids", []))
    found = sum(1 for pid in all_pids if pid in id2meta)
    has_tid = sum(1 for pid in all_pids if get_item_tid(pid, id2meta) is not None)
    print(f"    Distinct product IDs in journeys: {len(all_pids):,}")
    print(f"    Found in id2meta: {found:,} "
          f"({found / max(len(all_pids), 1) * 100:.1f}%)")
    print(f"    With valid TID: {has_tid:,} "
          f"({has_tid / max(len(all_pids), 1) * 100:.1f}%)")

    # =========================================================================
    # Step 2: Build SFT data
    # =========================================================================
    print()
    print("=" * 70)
    print(f"Step 2: Building {task} SFT data")
    print(f"  max_events = {args.max_events}")
    if task == "profile2journey":
        print(f"  max_recent_events = {args.max_recent_events}")
    print(f"  max_products_per_journey = {args.max_products_per_journey}")
    print(f"  min_products_per_journey = {args.min_products_per_journey}")
    print(f"  min_journeys = {args.min_journeys}")
    print(f"  max_journeys = {args.max_journeys}")
    print(f"  keep_empty_ratio = {args.keep_empty_ratio}")
    print(f"  seed = {args.seed}")
    print("=" * 70)

    sft_data = create_sft_data(
        task=task,
        shopping_journey_data=shopping_data,
        id2meta=id2meta,
        profiles=profiles,
        max_events=args.max_events,
        max_recent_events=args.max_recent_events,
        max_products=args.max_products_per_journey,
        min_products=args.min_products_per_journey,
        min_journeys=args.min_journeys,
        max_journeys=args.max_journeys,
        keep_empty_ratio=args.keep_empty_ratio,
    )

    # =========================================================================
    # Step 3: Save output
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Saving output")
    print("=" * 70)

    output_file = os.path.join(args.output_dir, f"{task}_sft.json")
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
        print(f"  User ID:        {meta['user_id']}")
        print(f"  Task:           {meta['task']}")
        print(f"  Num events:     {meta['num_events']}")
        print(f"  Num journeys:   {meta['num_journeys']}")
        print(f"  Products/j:     {meta['num_products_per_journey']}")
        print(f"  Instruction:    {sample['instruction'][:200]}...")
        input_lines = sample["input"].split("\n")
        max_show = 12 if task == "profile2journey" else 12
        print(f"  Input (first {max_show} lines):")
        for line in input_lines[:max_show]:
            print(f"    {line[:150]}")
        if len(input_lines) > max_show:
            print(f"    ... ({len(input_lines) - max_show} more lines)")
        # Pretty-print the output JSON
        try:
            out_obj = json.loads(sample["output"])
            cj = out_obj.get("ContinuedJourneys", [])
            if not cj:
                print(f"  Output: (empty - no journeys)")
            else:
                for ji, j in enumerate(cj[:3]):
                    print(f"  Journey {ji+1}: {j['Title']}")
                    print(f"    Reason: {j['Reason'][:120]}")
                    print(f"    Products: {len(j['ProductTIDs'])} TIDs")
                    for pi, tid in enumerate(j["ProductTIDs"]):
                        print(f"      [{pi}]: {tid}")
        except (json.JSONDecodeError, KeyError):
            print(f"  Output: {sample['output'][:200]}...")
    print(f"\n{'=' * 70}")

    print(f"\nDone! Generated {len(sft_data)} training samples -> {output_file}")


if __name__ == "__main__":
    main()
