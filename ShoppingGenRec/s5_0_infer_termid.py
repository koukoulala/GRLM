"""
Batch inference script for TermId generation on MI300 (vLLM + ROCm).

Reads the TSV output from cosmos_index_join.py, builds chat-format prompts via
apply_chat_template(), and runs batch inference through a local vLLM model to
produce 7-slot text IDs for each product.

Usage:
    python infer_termid.py \
        --input_path /data/index_joined.tsv \
        --output_path /data/termid_results.tsv \
        --model_path /models/termid-sft \
        --tensor_parallel_size 8
"""

import argparse
import csv
import itertools
import json
import os
import re
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data preparation (from SFT training pipeline)
# =============================================================================

INSTRUCTION = (
    "Summarize the product into a text ID of exactly 7 distinct slots. "
    "Each slot is one base-form word; use a multi-word phrase only for "
    "brand/seller names or fixed proper nouns. "
    "Priority: category, function, feature, attribute, brand, seller, audience/style. "
    "Output strictly in the format: Item text ID: [s1, s2, s3, s4, s5, s6, s7]."
)


def prepare_data(item: dict) -> dict:
    """Prepare a single SFT training sample."""
    info_lines = ["Product Information:"]

    title = item.get("title", "")
    if title:
        if len(title) > 150:
            title = title[:150] + "..."
        info_lines.append(f"Title: {title}")

    description = item.get("description", "")
    if description:
        if len(description) > 150:
            description = description[:150] + "..."
        info_lines.append(f"Description: {description}")

    categories = item.get("categories", "")
    if categories:
        if len(categories) > 150:
            categories = categories[:150] + "..."
        info_lines.append(f"Categories: {categories}")

    # Append structured attributes (from s6 enrichment)
    attributes = item.get("attributes", {})
    brand = attributes.get("Brand", "")
    if isinstance(brand, str):
        brand = " ".join(brand.split())
    seller = attributes.get("Seller", "")
    if isinstance(seller, str):
        seller = " ".join(seller.split())
    if brand and seller and brand.lower() == seller.lower():
        info_lines.append(f"Brand/Seller: {brand}")
    else:
        if brand:
            info_lines.append(f"Brand: {brand}")
        if seller:
            info_lines.append(f"Seller: {seller}")
    for attr_name in ["Color", "Size"]:
        attr_val = attributes.get(attr_name, "")
        if isinstance(attr_val, str):
            attr_val = attr_val.strip()
        if attr_val:
            info_lines.append(f"{attr_name}: {attr_val}")
    gender = attributes.get("Gender", "").strip()
    if gender and gender.lower() != "unisex":
        info_lines.append(f"Gender: {gender}")
    age_group = attributes.get("AgeGroup", "").strip()
    if age_group and age_group.lower() != "adult":
        info_lines.append(f"AgeGroup: {age_group}")

    input_str = "\n".join(info_lines) + "\n"

    return {
        "instruction": INSTRUCTION,
        "input": input_str,
    }


def tsv_row_to_item(row: dict) -> dict:
    """Convert a TSV row from cosmos_index_join.py output to the dict format
    expected by prepare_data().

    TSV columns: CategoryId, GlobalOfferId, Title, Description, Brand, Seller,
                 Color, Size, Gender, AgeGroup, ProductPopularityScore,
                 CategoryHierarchy, CategoryIdHierarchy
    """
    return {
        "global_offer_id": row.get("GlobalOfferId", ""),
        "title": row.get("Title", "") or "",
        "description": row.get("Description", "") or "",
        "categories": row.get("CategoryHierarchy", "") or "",
        "attributes": {
            "Brand": row.get("Brand", "") or "",
            "Seller": row.get("Seller", "") or "",
            "Color": row.get("Color", "") or "",
            "Size": row.get("Size", "") or "",
            "Gender": row.get("Gender", "") or "",
            "AgeGroup": row.get("AgeGroup", "") or "",
        },
    }


def build_chat_prompts(items: list[dict], tokenizer) -> tuple[list[dict], list[str]]:
    """Build chat-format prompts via tokenizer.apply_chat_template().

    Returns (valid_items, prompts). Items that fail tokenization are skipped
    and counted separately; the two returned lists are always the same length.
    """
    valid_items = []
    prompts = []
    skip_count = 0
    for item in items:
        try:
            sample = prepare_data(item)
            content = sample["instruction"] + "\n" + sample["input"]
            messages = [{"role": "user", "content": content}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            valid_items.append(item)
            prompts.append(formatted)
        except Exception as e:
            skip_count += 1
            logger.warning(f"Tokenization error for {item.get('global_offer_id', '?')}: {e}")
    if skip_count:
        logger.warning(f"Skipped {skip_count} items in this batch due to tokenization errors")
    logger.info(f"Built {len(prompts)} chat prompts")
    return valid_items, prompts


# =============================================================================
# Output parsing
# =============================================================================

TERMID_PATTERN = re.compile(
    r"Item text ID:\s*\[([^\]]+)\]",
    re.IGNORECASE,
)


def parse_termid_output(text: str) -> list[str]:
    """Parse the model output to extract the 7-slot term ID.

    Returns a list of slot strings, or empty list if parsing fails.
    """
    match = TERMID_PATTERN.search(text)
    if not match:
        return []
    slots = [s.strip() for s in match.group(1).split(",")]
    return slots


# =============================================================================
# Inference backends
# =============================================================================

def init_vllm(
    model_path: str,
    tensor_parallel_size: int = 8,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    seed: int = 42,
):
    """Initialize vLLM LLM instance and SamplingParams once for reuse across batches."""
    from vllm import LLM, SamplingParams

    logger.info(f"Loading model from {model_path} with TP={tensor_parallel_size}")
    logger.info(f"  max_model_len={max_model_len}  gpu_mem={gpu_memory_utilization}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        seed=seed,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.8,
        top_k=20,
    )
    return llm, sampling_params


def batch_inference(llm, sampling_params, prompts: list[str]) -> list[str]:
    """Run inference on a single batch of prompts."""
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    logger.info(f"  Batch done: {len(outputs)} prompts in {elapsed:.1f}s "
                f"({len(outputs)/elapsed:.1f} prompts/sec)")
    return [output.outputs[0].text.strip() for output in outputs]


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TermId batch inference on MI300")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to TSV file from cosmos_index_join.py")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSON path (GlobalOfferId -> [slot1..slot7])")

    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local model path for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
                        help="Tensor parallel size (default: 8 for MI300)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="GPU memory utilization (default: 0.90)")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Maximum model context length for vLLM (default: 8192)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for vLLM (default: 42)")

    # Inference params
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max tokens to generate (default: 8192)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")

    # I/O control
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Max rows to process (for testing; default: all)")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Number of prompts per vLLM generate() call (default: 10000)")
    parser.add_argument("--checkpoint_interval", type=int, default=500000,
                        help="Write checkpoint every N successfully processed items (default: 500000)")

    return parser.parse_args()


def iter_input_tsv(input_path: str, max_rows: int = None):
    """Generator: yield item dicts one by one without loading the full file."""
    csv.field_size_limit(sys.maxsize)
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", escapechar="\\")
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            yield tsv_row_to_item(row)


# =============================================================================
# Checkpoint helpers
# =============================================================================

def load_checkpoint(output_path: str) -> dict:
    """Load completed results from the output JSONL + any checkpoint files.

    Returns dict: global_offer_id -> list of 7 slots.
    """
    completed = {}

    # Load from main output file
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    for goid, slots in obj.items():
                        completed[goid] = slots
                except json.JSONDecodeError:
                    continue
        logger.info(f"[RESUME] Loaded {len(completed):,} completed items from {output_path}")

    # Also load from checkpoint directory
    ckpt_dir = os.path.splitext(output_path)[0] + "_checkpoint"
    if os.path.isdir(ckpt_dir):
        ckpt_count = 0
        for fname in sorted(os.listdir(ckpt_dir)):
            if fname.endswith(".jsonl"):
                fpath = os.path.join(ckpt_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            for goid, slots in obj.items():
                                if goid not in completed:
                                    completed[goid] = slots
                                    ckpt_count += 1
                        except json.JSONDecodeError:
                            continue
        if ckpt_count:
            logger.info(f"[RESUME] Loaded {ckpt_count:,} additional items from checkpoint dir {ckpt_dir}")

    return completed


def save_checkpoint(results_buffer: list, output_path: str, chunk_idx: int):
    """Save a chunk of results to the checkpoint directory.

    Args:
        results_buffer: list of (global_offer_id, slots) tuples
        output_path: main output path (used to derive checkpoint dir)
        chunk_idx: checkpoint chunk number
    """
    ckpt_dir = os.path.splitext(output_path)[0] + "_checkpoint"
    os.makedirs(ckpt_dir, exist_ok=True)
    fpath = os.path.join(ckpt_dir, f"ckpt_{chunk_idx:05d}.jsonl")
    with open(fpath, "w", encoding="utf-8") as f:
        for goid, slots in results_buffer:
            f.write(json.dumps({goid: slots}, ensure_ascii=False) + "\n")
    logger.info(f"[CHECKPOINT] Saved {len(results_buffer):,} items to {fpath}")


def merge_checkpoint_to_output(output_path: str):
    """Merge all checkpoint files into the main output file and clean up."""
    ckpt_dir = os.path.splitext(output_path)[0] + "_checkpoint"
    if not os.path.isdir(ckpt_dir):
        return

    # Load everything (output + checkpoints), deduplicate
    all_results = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    for goid, slots in obj.items():
                        all_results[goid] = slots
                except json.JSONDecodeError:
                    continue

    ckpt_files = sorted([
        os.path.join(ckpt_dir, fname)
        for fname in os.listdir(ckpt_dir)
        if fname.endswith(".jsonl")
    ])
    for fpath in ckpt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    for goid, slots in obj.items():
                        all_results[goid] = slots
                except json.JSONDecodeError:
                    continue

    # Rewrite main output
    with open(output_path, "w", encoding="utf-8") as f:
        for goid, slots in all_results.items():
            f.write(json.dumps({goid: slots}, ensure_ascii=False) + "\n")
    logger.info(f"[MERGE] Merged {len(all_results):,} total items into {output_path}")

    # Clean up checkpoint dir
    import shutil
    shutil.rmtree(ckpt_dir)
    logger.info(f"[MERGE] Removed checkpoint directory {ckpt_dir}")

    return len(all_results)


def main():
    args = parse_args()

    # 1. Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_path} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 2. Load checkpoint / resume from previous run
    completed = load_checkpoint(args.output_path)
    if completed:
        logger.info(f"[RESUME] {len(completed):,} items already done, will skip them")

    # 3. Init vLLM once
    llm, sampling_params = init_vllm(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    total = len(completed)
    success = len(completed)
    fail_tokenize = 0
    fail_parse = 0
    chunk: list[dict] = []
    first_batch = True
    ckpt_buffer = []  # buffer for checkpoint writing
    ckpt_chunk_idx = 0
    items_since_last_ckpt = 0

    row_iter = iter_input_tsv(args.input_path, max_rows=args.max_rows)
    first_item = next(row_iter, None)
    if first_item is None:
        logger.info("No data to process.")
        sys.exit(0)

    full_iter = itertools.chain([first_item], row_iter)
    skipped = 0

    # Open output in append mode so we don't overwrite existing results
    # (checkpoint merge at the end will deduplicate)
    with open(args.output_path, "a", encoding="utf-8") as out_f:

        def flush_chunk():
            nonlocal total, success, fail_tokenize, fail_parse, first_batch
            nonlocal ckpt_buffer, ckpt_chunk_idx, items_since_last_ckpt
            valid_items, prompts = build_chat_prompts(chunk, tokenizer)
            tokenize_fail = len(chunk) - len(valid_items)
            fail_tokenize += tokenize_fail
            total += tokenize_fail
            if not prompts:
                logger.info(f"Progress: {total:,} rows processed ({success:,} success, "
                            f"{fail_tokenize:,} tokenize_fail, {fail_parse:,} parse_fail)")
                return
            if first_batch:
                logger.info(f"Sample prompt:\n{prompts[0]}\n--- end sample ---")
                first_batch = False
            raw_outputs = batch_inference(llm, sampling_params, prompts)
            for item, raw in zip(valid_items, raw_outputs):
                slots = parse_termid_output(raw)
                total += 1
                if len(slots) == 7:
                    out_f.write(json.dumps({item["global_offer_id"]: slots}, ensure_ascii=False) + "\n")
                    ckpt_buffer.append((item["global_offer_id"], slots))
                    success += 1
                    items_since_last_ckpt += 1
                else:
                    fail_parse += 1

            # Flush file buffer to disk
            out_f.flush()

            # Save checkpoint if enough new items accumulated
            if items_since_last_ckpt >= args.checkpoint_interval:
                save_checkpoint(ckpt_buffer, args.output_path, ckpt_chunk_idx)
                ckpt_chunk_idx += 1
                ckpt_buffer = []
                items_since_last_ckpt = 0

            logger.info(f"Progress: {total:,} rows processed ({success:,} success, "
                        f"{fail_tokenize:,} tokenize_fail, {fail_parse:,} parse_fail, "
                        f"{skipped:,} resumed)")

        # 4. Stream TSV and process in batches, skipping already-completed items
        for item in full_iter:
            goid = item.get("global_offer_id", "")
            if goid in completed:
                skipped += 1
                continue
            chunk.append(item)
            if len(chunk) >= args.batch_size:
                flush_chunk()
                chunk = []

        if chunk:
            flush_chunk()

    # Save any remaining checkpoint buffer
    if ckpt_buffer:
        save_checkpoint(ckpt_buffer, args.output_path, ckpt_chunk_idx)

    # 5. Merge all checkpoint files into main output (deduplicated)
    logger.info("Merging checkpoints into final output ...")
    final_count = merge_checkpoint_to_output(args.output_path)

    # 6. Build tid2item_id.json from the merged output JSONL
    tid2item_id_path = os.path.splitext(args.output_path)[0] + "_tid2item_id.json"
    logger.info(f"Building tid2item_id.json from {args.output_path} ...")
    from collections import defaultdict as _dd
    tid2ids = _dd(list)
    with open(args.output_path, "r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for goid, slots in obj.items():
                tid_key = ",".join(slots)
                tid2ids[tid_key].append(str(goid))
    with open(tid2item_id_path, "w", encoding="utf-8") as wf:
        json.dump(dict(tid2ids), wf, ensure_ascii=False, indent=2)
    tid2item_id_mb = os.path.getsize(tid2item_id_path) / (1024 * 1024)
    logger.info(f"  Saved tid2item_id.json: {tid2item_id_path} "
                f"({len(tid2ids):,} unique TIDs, {tid2item_id_mb:.1f} MB)")

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"  Total items:          {total:,}")
    logger.info(f"  Valid term IDs:       {success:,} ({success/max(total,1)*100:.1f}%)")
    logger.info(f"  Tokenization errors:  {fail_tokenize:,}")
    logger.info(f"  Parse failures:       {fail_parse:,}")
    logger.info(f"  Resumed from ckpt:    {len(completed):,}")
    logger.info(f"  Output file:          {args.output_path}")
    logger.info(f"  tid2item_id file:     {tid2item_id_path}")
    logger.info(f"  Unique TIDs:          {len(tid2ids):,}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
