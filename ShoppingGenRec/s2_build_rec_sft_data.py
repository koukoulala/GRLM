"""Step 4a: Build Recommendation SFT Data

Creates SFT training data from user interaction sequences. Supports both
in-domain (single category) and cross-domain (multi-category) scenarios.

The sequence is split as:
  [item1, ..., itemN-2] -> training input/output
  itemN-1               -> validation ground truth
  itemN                 -> test ground truth

Usage:
    python s4_build_rec_sft_data.py \
        --id2meta_file ./processed/id2meta.json \
        --sequential_file ./raw_data/item_sequential_data.txt \
        --output_dir ./sft_data \
        --max_seq_len 20
"""

import os
import json
import random
import argparse
from tqdm import tqdm


def load_mapping_data(id2meta_file: str, sequential_file: str):
    """Load metadata and user interaction sequences."""
    with open(id2meta_file, "r", encoding="utf-8") as f:
        parent_asin2meta = json.load(f)

    with open(sequential_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Loaded {len(parent_asin2meta)} items, {len(lines)} user sequences")

    return parent_asin2meta, lines


def create_sft_data(
    parent_asin2meta: dict,
    user_interactions: list,
    output_dir: str,
    max_seq_len: int = 20,
    min_items: int = 3,
    multi_input_prob: float = 0.0,
):
    """Create SFT training data from user interaction sequences.

    Args:
        parent_asin2meta: Item ID to metadata mapping.
        user_interactions: Raw lines of user interaction data.
        output_dir: Directory for all outputs.
        max_seq_len: Maximum sequence length to use (last N items).
        min_items: Minimum items in sequence (need >= 3 for train/valid/test split).
        multi_input_prob: Probability of using multiple items as input (0=disabled).
            When triggered, randomly picks a split point in the first half of
            training items so that more items go into input, reducing output length.
    """
    sft_data = []
    skipped_users = 0
    skip_reasons = {"id_not_found": 0, "empty_summary": 0, "too_short": 0}
    total_sequences = 0
    item_id2tid = {}
    seq_len_list = []  # track sequence lengths for statistics

    for line in tqdm(user_interactions, desc="Building rec SFT data"):
        line = line.strip()
        if not line:
            continue

        elements = line.split()
        if len(elements) <= 1:
            continue

        user_id = elements[0]
        item_ids = elements[1:]
        item_ids = item_ids[-max_seq_len:]  # Truncate to last N items

        # Validate all items in sequence
        all_summary_words = []
        item_id_list = []
        meta_msg_list = []
        valid_sequence = True

        for item_id in item_ids:
            if item_id not in parent_asin2meta:
                valid_sequence = False
                skip_reasons["id_not_found"] += 1
                break
            meta = parent_asin2meta[item_id]
            summary_words = meta.get("summary_words", [])
            if "" in summary_words:
                valid_sequence = False
                skip_reasons["empty_summary"] += 1
                break
            valid_words = [
                word.replace("[", "").replace("]", "")
                for word in summary_words
                if word and word.strip()
            ]
            if len(valid_words) < 7:
                valid_sequence = False
                skip_reasons["empty_summary"] += 1
                break
            all_summary_words.extend(valid_words[:7])
            item_id_list.append(item_id)
            meta_msg_list.append(meta)
            item_id2tid[item_id] = valid_words[:7]

        # Need at least 3 items (train items + valid + test)
        if not valid_sequence or len(item_id_list) < min_items:
            if valid_sequence and len(item_id_list) < min_items:
                skip_reasons["too_short"] += 1
            skipped_users += 1
            continue

        # Need at least 21 summary words (3 items * 7 words)
        if len(all_summary_words) < 21:
            skipped_users += 1
            skip_reasons["too_short"] += 1
            continue

        # Split: last 2 items are valid/test, rest are train (input + output)
        num_total = len(item_id_list)
        num_train_items = num_total - 2  # exclude valid and test

        # Decide how many items go into input
        num_input_items = 1  # default: only first item
        if (
            multi_input_prob > 0
            and num_train_items > 5
            and random.random() < multi_input_prob
        ):
            # Pick a random split point in the first half of training items
            half = num_train_items // 2
            num_input_items = random.randint(2, half)

        input_words = all_summary_words[: num_input_items * 7]
        test_ground_truth = all_summary_words[-7:]
        valid_ground_truth = all_summary_words[-14:-7]
        train_output_words = all_summary_words[num_input_items * 7 : -14]

        sft_sample = _create_instruction_sample(
            input_words,
            train_output_words,
            user_id,
            num_total,
            item_id_list,
            meta_msg_list,
            num_input_items=num_input_items,
        )
        sft_sample["item_id_list"] = item_id_list
        sft_sample["item_id_len"] = len(item_id_list)
        sft_sample["all_summary_words"] = all_summary_words
        sft_sample["valid_ground_truth_id"] = item_id_list[-2]
        sft_sample["test_ground_truth_id"] = item_id_list[-1]
        sft_sample["valid_ground_truth_tid"] = valid_ground_truth
        sft_sample["test_ground_truth_tid"] = test_ground_truth
        sft_sample["valid_ground_truth_msg"] = meta_msg_list[-2]
        sft_sample["test_ground_truth_msg"] = meta_msg_list[-1]
        sft_data.append(sft_sample)
        total_sequences += 1
        seq_len_list.append(len(item_id_list))

    print(f"\nData statistics:")
    print(f"  Total users:          {len(user_interactions):>10,}")
    print(f"  Skipped:              {skipped_users:>10,}")
    print(f"    - ID not found:     {skip_reasons['id_not_found']:>10,}")
    print(f"    - Empty/bad summary:{skip_reasons['empty_summary']:>10,}")
    print(f"    - Too short seq:    {skip_reasons['too_short']:>10,}")
    print(f"  Generated samples:    {total_sequences:>10,}")

    if seq_len_list:
        import numpy as np
        arr = np.array(seq_len_list)
        print(f"\n  Sequence length stats:")
        print(f"    Min:    {arr.min()}")
        print(f"    Max:    {arr.max()}")
        print(f"    Mean:   {arr.mean():.2f}")
        print(f"    Median: {np.median(arr):.1f}")
        print(f"    P25:    {np.percentile(arr, 25):.1f}")
        print(f"    P75:    {np.percentile(arr, 75):.1f}")
        print(f"    P95:    {np.percentile(arr, 95):.1f}")

    # Save id2tid mappings
    tid_dir = os.path.join(output_dir, "item_id2tid")
    os.makedirs(tid_dir, exist_ok=True)

    with open(os.path.join(tid_dir, "item_id2tid.json"), "w", encoding="utf-8") as f:
        json.dump(item_id2tid, f, ensure_ascii=False, indent=2)

    # Build reverse mapping: tid -> item_ids
    value2keys = {}
    for key, value in item_id2tid.items():
        tid_str = ",".join(value)
        if tid_str not in value2keys:
            value2keys[tid_str] = []
        value2keys[tid_str].append(key)

    with open(os.path.join(tid_dir, "tid2item_id.json"), "w", encoding="utf-8") as f:
        json.dump(value2keys, f, ensure_ascii=False, indent=2)

    print(f"  item_id2tid: {len(item_id2tid)} items")
    print(f"  tid2item_id: {len(value2keys)} unique TIDs")

    return sft_data


def _create_instruction_sample(
    input_words, output_words, user_id, total_items, item_id_list, meta_msg_list,
    num_input_items=1,
):
    """Create a single instruction-tuning sample."""
    instruction = (
        "Given the user's product interaction sequence, predict the next products "
        "in the sequence. Each product is represented by a text ID of exactly 7 "
        "characteristic words.\n"
    )

    # Input: first num_input_items items
    input_text = ""
    for idx in range(num_input_items):
        words = input_words[idx * 7 : (idx + 1) * 7]
        input_text += "Item text ID: [" + ", ".join(words) + "]"
        title = meta_msg_list[idx].get("title", "")
        input_text += f" Title: {title}.\n" if title else " Title: None.\n"

    # Output: items from num_input_items to total_items - 2 (exclusive of valid/test)
    output_lines = []
    num_output_items = total_items - 2 - num_input_items
    assert len(output_words) % 7 == 0
    for i in range(num_output_items):
        meta = meta_msg_list[num_input_items + i]
        output_lines.append("Item text ID: [" + ", ".join(meta["summary_words"]) + "]")
    output_text = "\n".join(output_lines)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": {
            "user_id": user_id,
            "total_items": total_items,
            "num_input_items": num_input_items,
            "total_words": len(input_words) + len(output_words),
            "input_word_count": len(input_words),
            "output_word_count": len(output_words),
        },
    }


def save_sft_data(sft_data: list, output_file: str):
    """Save SFT data (full and training versions).

    Full version (with metadata): <name>_full.json
    Training version (instruction/input/output only): <name>.json
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build recommendation SFT data from user sequences"
    )
    parser.add_argument(
        "--id2meta_file",
        type=str,
        default="./processed/id2meta.json",
        help="Path to id2meta JSON from step 1 (default: ./processed/id2meta.json)",
    )
    parser.add_argument(
        "--sequential_file",
        type=str,
        default="./raw_data/item_sequential_data.txt",
        help="Path to sequential interaction data file "
             "(default: ./raw_data/item_sequential_data.txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_data",
        help="Output directory. SFT data saved to <output_dir>/rec_sft.json, "
             "auxiliary mappings to <output_dir>/item_id2tid/ "
             "(default: ./sft_data)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=20,
        help="Max items per user sequence (default: 20)",
    )
    parser.add_argument(
        "--multi_input_prob",
        type=float,
        default=0.0,
        help="Probability of using multiple items as input (0=disabled, 0~1). "
        "When triggered and train items > 5, randomly picks a split point in "
        "the first half of training items to use as input.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    parent_asin2meta, user_interactions = load_mapping_data(
        args.id2meta_file, args.sequential_file
    )

    output_dir = args.output_dir
    output_file = os.path.join(output_dir, "rec_sft.json")

    sft_data = create_sft_data(
        parent_asin2meta,
        user_interactions,
        output_dir,
        max_seq_len=args.max_seq_len,
        multi_input_prob=args.multi_input_prob,
    )

    save_sft_data(sft_data, output_file)

    # ---- Show example cases ----
    print(f"\n{'='*70}")
    print("Example cases (first 3):")
    print(f"{'='*70}")
    for idx, sample in enumerate(sft_data[:3]):
        print(f"\n--- Example {idx + 1} ---")
        print(f"  User ID:        {sample['metadata']['user_id']}")
        print(f"  Seq length:     {sample['item_id_len']} items")
        print(f"  Input items:    {sample['metadata']['num_input_items']}")
        print(f"  Output items:   {sample['metadata']['total_items'] - 2 - sample['metadata']['num_input_items']}")
        print(f"  Valid GT ID:    {sample['valid_ground_truth_id']}")
        print(f"  Valid GT TID:   {sample['valid_ground_truth_tid']}")
        print(f"  Test GT ID:     {sample['test_ground_truth_id']}")
        print(f"  Test GT TID:    {sample['test_ground_truth_tid']}")
        print(f"  Instruction:    {sample['instruction']}")
        print(f"  Input (trunc):  {sample['input']}")
        print(f"  Output (trunc): {sample['output']}")
    print(f"\n{'='*70}")

    print(f"\nDone! Generated {len(sft_data)} training samples")


if __name__ == "__main__":
    main()
