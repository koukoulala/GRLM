"""Step 5: Unified Evaluation with Beam Search

Evaluates a trained model on recommendation data using beam search to generate
candidate text IDs (TIDs), then maps them back to item IDs (IIDs) using
exact + fuzzy matching. Supports any dataset/domain.

This replaces the separate s5_beauty_eval.py, s5_sports_eval.py, s5_toys_eval.py.

Usage:
    python s5_eval.py \
        --model_path /path/to/checkpoint \
        --test_file ./processed/sft_data/rec_sft.json \
        --tid2item_id_file ./processed/sum_data/item_id2tid/shopping_tid2item_id.json \
        --output_dir ./processed/eval_results \
        --num_beams 20 --batch_size 1
"""

import os
import re
import json
import random
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# TID <-> IID Mapping Utilities
# ---------------------------------------------------------------------------

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


def get_iid_by_tid(content, tid2item_id, reverse_mapping, word_to_keys):
    """Map a TID string to item IDs (exact match first, then fuzzy)."""
    tids = content.replace("[", "").replace("]", "").split(", ")
    tid_key = ",".join(tids)

    if tid_key in tid2item_id:
        return list(tid2item_id[tid_key])

    # Fuzzy matching
    candidate_scores = defaultdict(float)
    for i, query_word in enumerate(tids):
        position_weight = 1.0 / (i + 1)
        for candidate_word, candidate_keys in word_to_keys.items():
            similarity = 0.0
            if query_word == candidate_word:
                similarity = 1.0
            elif query_word in candidate_word or candidate_word in query_word:
                similarity = 0.8
            if similarity > 0:
                for candidate_key in candidate_keys:
                    candidate_scores[candidate_key] += similarity * position_weight

    sorted_candidates = sorted(
        candidate_scores.items(), key=lambda x: x[1], reverse=True
    )
    iids = []
    for candidate_key, _ in sorted_candidates:
        iids.extend(reverse_mapping[candidate_key]["ids"])
    return iids[:1]


def extend_iid_by_tid(content, reverse_mapping, word_to_keys):
    """Extended fuzzy matching for filling up candidate list."""
    tids = content.replace("[", "").replace("]", "").split(", ")
    candidate_scores = defaultdict(float)

    for i, query_word in enumerate(tids):
        position_weight = 1.0 / (i + 1)
        for candidate_word, candidate_keys in word_to_keys.items():
            similarity = 0.0
            if query_word == candidate_word:
                similarity = 1.0
            elif query_word in candidate_word or candidate_word in query_word:
                similarity = 0.8
            if similarity > 0:
                for candidate_key in candidate_keys:
                    candidate_scores[candidate_key] += similarity * position_weight

    sorted_candidates = sorted(
        candidate_scores.items(), key=lambda x: x[1], reverse=True
    )
    iids = []
    for candidate_key, _ in sorted_candidates:
        iids.extend(reverse_mapping[candidate_key]["ids"])
    return iids[:1]


# ---------------------------------------------------------------------------
# Prompt & Result Processing
# ---------------------------------------------------------------------------

def prepare_batch_prompts(batch_data):
    """Prepare prompts for batch evaluation."""
    batch_prompts = []
    batch_metadata = []

    for d in batch_data:
        # Build prompt: input + output + valid_ground_truth (model predicts test)
        l = d["input"] + d["output"] + "Item text ID: [" + ", ".join(d["valid_ground_truth_tid"]) + "]"
        if "title" in d.get("valid_ground_truth_msg", {}):
            l += f" Title: {d['valid_ground_truth_msg']['title']}.\n"
        else:
            l += " Title: None.\n"

        prompt = (
            "Based on the user's historical product interaction sequence, predict the "
            "next product's characteristic words. \n"
            "Each product is represented by exactly 5 characteristic words enclosed in "
            "square brackets []. The historical sequence shows the user's interaction pattern.\n"
        )
        prompt += l
        prompt += "Item text ID: "

        messages = [{"role": "user", "content": prompt}]
        batch_prompts.append(messages)
        batch_metadata.append(
            {
                "original_data": d,
                "iid_gt": d["test_ground_truth_id"],
                "tid_gt": d["test_ground_truth_tid"],
                "prompt": prompt,
            }
        )

    return batch_prompts, batch_metadata


def process_batch_results(
    generated_ids,
    model_inputs,
    batch_metadata,
    tokenizer,
    tid2item_id,
    reverse_mapping,
    word_to_keys,
    top_k=20,
):
    """Process batch generation results, mapping TIDs to IIDs."""
    batch_results = []
    num_sequences_per_sample = generated_ids.shape[0] // len(batch_metadata)

    for batch_idx, metadata in enumerate(batch_metadata):
        dic = metadata["original_data"].copy()
        iid_gt = metadata["iid_gt"]

        all_results = []
        contents = []
        raw_contents = []

        start_idx = batch_idx * num_sequences_per_sample
        end_idx = (batch_idx + 1) * num_sequences_per_sample

        for seq_idx in range(start_idx, end_idx):
            input_len = model_inputs.input_ids[batch_idx].shape[0]
            output_ids = generated_ids[seq_idx][input_len:].tolist()

            # Skip thinking tokens (Qwen-specific token 151668)
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            content = tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
            raw_contents.append(content)

            # Extract all [word1, word2, ...] patterns
            for c in re.findall(r"\[(.*?)\]", content):
                content_str = "[" + c + "]"
                if content_str not in contents:
                    contents.append(content_str)

        dic["contents_len"] = len(contents)

        # Map TIDs to IIDs
        iids = []
        for i, content in enumerate(contents):
            iid = get_iid_by_tid(content, tid2item_id, reverse_mapping, word_to_keys)
            all_results.append(
                {"sequence_id": i, "content": content, "iid": iid}
            )
            iids.extend(iid)

        # Deduplicate while preserving order
        seen = set()
        unique_iids = []
        for iid in iids:
            if iid not in seen:
                seen.add(iid)
                unique_iids.append(iid)
        iids = unique_iids[:top_k]

        # Fill up to top_k using extended fuzzy matching
        attempt = 30
        while len(iids) < top_k and attempt <= 100:
            for content in contents:
                extended = extend_iid_by_tid(content, reverse_mapping, word_to_keys)
                for single_iid in extended:
                    if single_iid not in iids:
                        iids.append(single_iid)
                    if len(iids) >= top_k:
                        break
                if len(iids) >= top_k:
                    break
            attempt += 20

        iids = iids[:top_k]

        dic["prompt"] = metadata["prompt"]
        dic["raw_contents"] = raw_contents
        dic["all_results"] = all_results
        dic["iids"] = iids
        dic["iids_len"] = len(iids)
        dic["iid_gt"] = iid_gt
        batch_results.append(dic)

    return batch_results


# ---------------------------------------------------------------------------
# Multi-GPU Evaluation
# ---------------------------------------------------------------------------

def process_single_gpu(
    rank,
    data_slice,
    output_queue,
    model_name,
    tid2item_id,
    reverse_mapping,
    word_to_keys,
    num_beams=20,
    batch_size=1,
    top_k=20,
):
    """Process evaluation on a single GPU."""
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{rank}",
    )
    model.eval()

    print(f"Rank {rank}: Processing {len(data_slice)} samples, batch_size={batch_size}")

    local_score = [0] * top_k
    local_results = []

    for batch_start in tqdm(
        range(0, len(data_slice), batch_size), desc=f"GPU {rank}"
    ):
        batch_data = data_slice[batch_start : batch_start + batch_size]
        batch_prompts, batch_metadata = prepare_batch_prompts(batch_data)

        batch_texts = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for msgs in batch_prompts
        ]

        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32768,
            return_attention_mask=True,
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=30,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=False,
            )

        batch_results = process_batch_results(
            generated_ids,
            model_inputs,
            batch_metadata,
            tokenizer,
            tid2item_id,
            reverse_mapping,
            word_to_keys,
            top_k=top_k,
        )

        local_results.extend(batch_results)

        for result in batch_results:
            for i, iid in enumerate(result["iids"]):
                if i < len(local_score) and result["iid_gt"] == iid:
                    local_score[i] += 1
                    break

    output_queue.put((rank, local_score, local_results))
    print(f"Rank {rank}: Done, processed {len(local_results)} samples")


def calculate_recall(scores, total_samples):
    """Calculate recall@K metrics."""
    return {
        f"recall@{k}": sum(scores[:k]) / total_samples
        for k in [1, 5, 10, 20]
        if k <= len(scores)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified evaluation with beam search and TID-to-IID mapping"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint for evaluation",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to SFT rec data JSON (from step 4a)",
    )
    parser.add_argument(
        "--tid2item_id_file",
        type=str,
        required=True,
        help="Path to tid2item_id JSON (from step 4a)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="eval",
        help="Prefix for output files (default: eval)",
    )
    parser.add_argument(
        "--num_beams", type=int, default=20, help="Number of beams (default: 20)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size per GPU (default: 1)"
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Top-K candidates to evaluate (default: 20)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs (default: all available)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    num_gpus = args.num_gpus or torch.cuda.device_count()

    # Load TID-to-IID mapping
    print(f"Loading TID mapping: {args.tid2item_id_file}")
    with open(args.tid2item_id_file, "r", encoding="utf-8") as f:
        tid2item_id = json.load(f)
    reverse_mapping, word_to_keys = create_reverse_mapping(tid2item_id)

    # Load test data
    print(f"Loading test data: {args.test_file}")
    with open(args.test_file, "r", encoding="utf-8") as f:
        sft_data = json.load(f)
    print(f"Loaded {len(sft_data)} test samples")

    # Split data across GPUs
    chunk_size = len(sft_data) // num_gpus
    data_chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = len(sft_data) if i == num_gpus - 1 else start + chunk_size
        data_chunks.append(sft_data[start:end])

    processes = []
    output_queue = mp.Queue()
    start_time = time.time()

    for rank in range(num_gpus):
        p = mp.Process(
            target=process_single_gpu,
            args=(
                rank,
                data_chunks[rank],
                output_queue,
                args.model_path,
                tid2item_id,
                reverse_mapping,
                word_to_keys,
                args.num_beams,
                args.batch_size,
                args.top_k,
            ),
        )
        processes.append(p)
        p.start()

    all_results = []
    all_scores = [0] * args.top_k

    for _ in range(num_gpus):
        rank, local_score, local_results = output_queue.get()
        print(f"Received {len(local_results)} results from GPU {rank}")
        for i in range(min(len(local_score), len(all_scores))):
            all_scores[i] += local_score[i]
        all_results.extend(local_results)

    for p in processes:
        p.join()

    print(f"Total time: {time.time() - start_time:.2f}s")

    # Calculate metrics
    recall_metrics = calculate_recall(all_scores, len(sft_data))

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for metric, value in recall_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    eval_info_file = os.path.join(args.output_dir, f"{args.output_prefix}_info.json")
    with open(eval_info_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Detailed results: {eval_info_file}")

    recall_file = os.path.join(args.output_dir, f"{args.output_prefix}_recall.json")
    with open(recall_file, "w", encoding="utf-8") as f:
        json.dump(recall_metrics, f, ensure_ascii=False, indent=2)
    print(f"Recall results: {recall_file}")

    print(f"\nDone! Processed {len(all_results)} samples")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
