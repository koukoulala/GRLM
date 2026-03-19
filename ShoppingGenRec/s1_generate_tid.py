"""Step 1: Generate Product Summaries & Build ID-to-Metadata Mapping

Supports two inference backends:
1. vLLM offline inference (when --summary_model points to a valid local model)
2. GitHub Copilot API (when --summary_model is empty or path doesn't exist)

Both backends support checkpoint/resume: intermediate results are saved to
a checkpoint directory so that interrupted runs can continue from where they
left off. Completed checkpoint files are loaded on startup, and only
remaining items are processed. After all items are done, results are merged
and checkpoint files are cleaned up.

Inputs (from s0_init_emb.py):
    - merged_clean_item.json   : unified item metadata (title, description,
                                 categories)
    - similarities.json        : top-k similar item IDs per item

Outputs:
    - summaries_with_similarity.jsonl : per-item results with LLM output
    - id2meta.json                    : item ID -> metadata mapping
    - statistics.json                 : word frequency & conflict stats
    - failed_items.json               : items that did not produce 7 words

Usage (vLLM - local model):
    python s1_generate_tid.py \\
        --item_file ./raw_data/merged_clean_item.json \\
        --similarity_file ./processed/similarities.json \\
        --output_dir ./processed/ \\
        --summary_model /path/to/Qwen3-8B \\
        --num_gpus 2

Usage (Copilot API):
    python s1_generate_tid.py \\
        --item_file ./raw_data/merged_clean_item.json \\
        --similarity_file ./processed/similarities.json \\
        --output_dir ./processed/ \\
        --summary_model "" \\
        --token_file ./resources/tokens.txt \\
        --copilot_model gpt-5.4 \\
        --copilot_workers 20
"""

import os
import json
import re
import random
import argparse
from collections import Counter
import time
import sys

from tqdm import tqdm

SEED = 42
random.seed(SEED)

# Add resources directory to path for llm_utils import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "resources")
sys.path.insert(0, RESOURCES_DIR)
from llm_utils import (load_prompts, run_llm_parallel,
                      run_llm_parallel_with_checkpoint,
                      load_checkpoint as _load_checkpoint_raw,
                      save_checkpoint as _save_checkpoint_raw,
                      cleanup_checkpoint)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(file_path):
    """Load item JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result_list = []
    for key, value in data.items():
        new_item = {"id": key}
        new_item.update(value)
        result_list.append(new_item)
    return result_list


def load_similarities(similarity_file):
    """Load precomputed similarities."""
    with open(similarity_file, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Prompt Construction
# =============================================================================

def build_product_info_text(item):
    """Build product information text block from item metadata."""
    info_lines = []

    title = item.get("title", "")
    if title:
        info_lines.append(f"Title: {title}")

    description = item.get("description", "")
    if description:
        if len(description) > 150:
            description = description[:150] + "..."
        info_lines.append(f"Description: {description}")

    categories = item.get("categories", "")
    if categories:
        info_lines.append(f"Categories: {categories}")

    # Append structured attributes
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

    return "\n".join(info_lines) if info_lines else "(no information)"


def build_similar_items_text(top_similar_items, all_items_dict):
    """Build similar items text block."""
    similar_items_info = []
    for similar_item in top_similar_items:
        similar_item_id = similar_item["item_id"]
        similarity_score = similar_item["similarity"]

        if similar_item_id in all_items_dict:
            similar_item_data = all_items_dict[similar_item_id]
            similar_title = similar_item_data.get("title", "")
            similar_desc = similar_item_data.get("description", "")

            similar_info = (
                f"Similar Item {similar_item_id} (similarity: {similarity_score:.3f}):"
            )
            if similar_title:
                similar_info += f" Title: {similar_title}"
            if similar_desc:
                if len(similar_desc) > 150:
                    similar_info += f" Description: {similar_desc[:150]}..."
                else:
                    similar_info += f" Description: {similar_desc}"
            similar_items_info.append(similar_info)

    return "\n".join(similar_items_info) if similar_items_info else "(none)"


def prepare_prompt(item, top_similar_items, all_items_dict, prompt_template):
    """Prepare prompt for a single item using the template from prompts.yaml.

    Args:
        item: Item dict with product metadata.
        top_similar_items: List of similar item dicts.
        all_items_dict: Dict mapping item_id -> item dict.
        prompt_template: Prompt template string with {product_info_text}
                         and {similar_items_text} placeholders.

    Returns:
        Formatted prompt string.
    """
    product_info_text = build_product_info_text(item)
    similar_items_text = build_similar_items_text(top_similar_items, all_items_dict)
    return prompt_template.format(
        product_info_text=product_info_text,
        similar_items_text=similar_items_text,
    )


# =============================================================================
# Parsing
# =============================================================================

def parse_summary_words(content: str) -> list:
    """Parse 7-word summary or empty list from LLM output.

    Handles thinking output (<think>...</think>) by stripping it.
    Extracts the final answer from <Output>...</Output> tags.

    Returns:
        A list of 7 word strings for valid products, or an empty list []
        for non-product entries.  If parsing fails, returns a list with
        fewer than 7 non-empty words.
    """
    if not content:
        return []

    # Strip <think>...</think> blocks if present
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Extract content from <Output>...</Output> tags (use the last match)
    output_matches = re.findall(r"<Output>(.*?)</Output>", content, flags=re.DOTALL | re.IGNORECASE)
    if output_matches:
        content = output_matches[-1].strip()

    content_stripped = content.strip()

    # Check for empty list output: [] (non-product)
    if re.search(r"^\s*\[\s*\]\s*$", content_stripped):
        return []

    # Also handle [] appearing anywhere in output with nothing else meaningful
    if "[]" in content_stripped:
        non_empty_bracket = re.search(r"\[[^\]]+\]", content_stripped)
        if not non_empty_bracket:
            return []

    # Try [word1, word2, ...] format
    pattern = r"\[([^\]]+)\]"
    match = re.search(pattern, content)
    if match:
        inner_content = match.group(1)
        # Handle full-width commas and Japanese ideographic commas
        inner_content = re.sub(r"[，、]", ",", inner_content)
        if "," in inner_content:
            words = [
                word.strip().strip("\"'[]") for word in inner_content.split(",")
            ]
        else:
            words = [
                word.strip().strip("\"'[]") for word in inner_content.split()
            ]
    else:
        # Fallback: try line-by-line
        words = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                inner = line[1:-1].strip()
                if not inner:
                    return []
                inner = re.sub(r"[，、]", ",", inner)
                if "," in inner:
                    words = [
                        word.strip().strip("\"'") for word in inner.split(",")
                    ]
                else:
                    words = [
                        word.strip().strip("\"'") for word in inner.split()
                    ]
                break
        if not words:
            content = re.sub(r"[，、]", ",", content)
            if "," in content:
                words = [
                    word.strip().strip("\"'[]") for word in content.split(",")
                ]
            else:
                words = [
                    word.strip().strip("\"'[]") for word in content.split()
                ]

    words = [w for w in words if w]
    # Deduplicate words (case-insensitive), preserving order
    seen = set()
    deduped = []
    for w in words:
        w_lower = w.lower()
        if w_lower not in seen:
            seen.add(w_lower)
            deduped.append(w)
    return deduped[:7]


# =============================================================================
# Statistics
# =============================================================================

def analyze_statistics(all_items):
    """Analyze summary statistics, separating products from non-products."""
    print("\n" + "=" * 50)
    print("Statistical Analysis Results")
    print("=" * 50)

    product_items = []
    non_product_items = []
    failed_items = []

    for item in all_items:
        words = item.get("summary_words", [])
        non_empty_words = [w for w in words if w]
        if len(words) == 0:
            non_product_items.append(item)
        elif len(non_empty_words) == 7:
            product_items.append(item)
        else:
            failed_items.append(item)

    total = len(all_items)
    print(f"\n0. Item Classification:")
    print(f"   Total items:              {total:>10,}")
    print(f"   Valid products (7 words): {len(product_items):>10,} "
          f"({len(product_items) / total * 100:.2f}%)")
    print(f"   Non-products (empty []):  {len(non_product_items):>10,} "
          f"({len(non_product_items) / total * 100:.2f}%)")
    print(f"   Failed parsing:           {len(failed_items):>10,} "
          f"({len(failed_items) / total * 100:.2f}%)")

    all_words = []
    word_freq = Counter()
    word_by_position = [Counter() for _ in range(7)]

    for item in product_items:
        words = item.get("summary_words", [])
        all_words.extend([word for word in words if word])
        for i, word in enumerate(words):
            if i < 7 and word:
                word_by_position[i][word] += 1

    word_freq.update(all_words)

    print(f"\n1. Vocabulary (products only): {len(all_words)} total, "
          f"{len(word_freq)} unique")
    print("   Top 20 words:")
    for word, count in word_freq.most_common(20):
        print(f"     {word}: {count}")

    summary_tuples = [tuple(item.get("summary_words", [])) for item in product_items]
    tuple_counter = Counter(summary_tuples)
    duplicate_tuples = [
        (tup, count) for tup, count in tuple_counter.items() if count > 1
    ]
    total_conflicts = sum(count - 1 for _, count in duplicate_tuples)
    conflict_rate = total_conflicts / len(product_items) if product_items else 0

    print(f"\n2. Conflicts (products): {len(duplicate_tuples)} duplicates, "
          f"rate={conflict_rate:.4f}")

    if non_product_items:
        print(f"\n3. Non-product examples (first 5):")
        for item in non_product_items[:5]:
            print(f"     ID={item['id']}, Title={item.get('title', 'N/A')}")

    if failed_items:
        print(f"\n4. Failed parsing examples (first 5):")
        for item in failed_items[:5]:
            print(f"     ID={item['id']}, Title={item.get('title', 'N/A')}")
            print(f"       LLM output: {item.get('llm_output', 'N/A')[:100]}")

    return {
        "word_frequency": dict(word_freq.most_common()),
        "position_frequency": [
            dict(counter.most_common()) for counter in word_by_position
        ],
        "conflict_rate": conflict_rate,
        "valid_product_count": len(product_items),
        "non_product_count": len(non_product_items),
        "failed_count": len(failed_items),
        "total_items": total,
    }, failed_items, non_product_items


# =============================================================================
# vLLM Checkpoint Helpers (stores full result dicts, not just text)
# =============================================================================

def _load_vllm_checkpoint(checkpoint_dir):
    """Load vLLM checkpoint results (full item dicts keyed by item id).

    Only loads records that have 'llm_output' key, which distinguishes
    vLLM format (full item dicts) from Copilot format ({"id", "result"}).
    """
    completed = {}
    if not os.path.exists(checkpoint_dir):
        return completed
    for fname in sorted(os.listdir(checkpoint_dir)):
        if fname.endswith('.jsonl'):
            fpath = os.path.join(checkpoint_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        result = json.loads(line)
                        # Only accept vLLM format (has llm_output);
                        # skip Copilot format (has 'result' instead)
                        if 'llm_output' in result and 'id' in result:
                            completed[result['id']] = result
    if completed:
        print(f"  [CHECKPOINT] Loaded {len(completed)} completed items "
              f"from {checkpoint_dir}")
    return completed


def _save_vllm_checkpoint(results, checkpoint_dir, chunk_idx):
    """Save vLLM results (list of full item dicts) to checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    fpath = os.path.join(checkpoint_dir, f"chunk_{chunk_idx:05d}.jsonl")
    with open(fpath, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"  [CHECKPOINT] Saved chunk {chunk_idx}: {fpath} ({len(results)} items)")


# =============================================================================
# Inference Mode Detection
# =============================================================================

def should_use_copilot(summary_model):
    """Determine whether to use Copilot API or vLLM.

    Returns True if summary_model is empty or path doesn't exist.
    """
    if not summary_model or not summary_model.strip():
        print("  [INFO] summary_model is empty, using Copilot API")
        return True
    if not os.path.exists(summary_model):
        print(f"  [INFO] Model path not found: {summary_model}, using Copilot API")
        return True
    return False


# =============================================================================
# vLLM Inference
# =============================================================================

def build_all_prompts_vllm(data, similarities_dict, all_items_dict, prompt_template,
                           model_name, enable_thinking=False):
    """Build formatted prompts for vLLM using tokenizer's chat template.

    Args:
        data: List of item dicts.
        similarities_dict: Dict of item_id -> similar items list.
        all_items_dict: Dict of item_id -> item dict.
        prompt_template: Prompt template string from prompts.yaml.
        model_name: Path to the model (for tokenizer loading).
        enable_thinking: Whether to enable thinking/reasoning in template.

    Returns:
        List of formatted prompt strings ready for vLLM generate().
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompts = []
    for item in tqdm(data, desc="Building prompts"):
        item_id = item["id"]
        top_similar_items = similarities_dict.get(item_id, [])[:5]
        raw_prompt = prepare_prompt(item, top_similar_items, all_items_dict,
                                    prompt_template)
        messages = [{"role": "user", "content": raw_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(formatted)
    return prompts


def run_vllm_inference(
    prompts,
    model_name,
    num_gpus,
    gpu_memory_utilization,
    max_model_len,
    max_tokens=80,
    chunk_size=100000,
    data=None,
    similarities_dict=None,
    checkpoint_dir=None,
):
    """Run vLLM offline inference on all prompts with checkpoint support.

    Args:
        prompts: List of formatted prompt strings.
        model_name: Path to the model.
        num_gpus: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use (0-1).
        max_model_len: Maximum context length for the model.
        max_tokens: Maximum number of output tokens per prompt.
        chunk_size: Number of prompts per generate() call.
        data: List of item dicts (for checkpoint saving).
        similarities_dict: Similarities dict (for checkpoint saving).
        checkpoint_dir: Directory for checkpoint files (optional).

    Returns:
        List of generated text strings, aligned with input prompts.
    """
    from vllm import LLM, SamplingParams

    print(f"\nInitializing vLLM engine ...")
    print(f"  Model: {model_name}")
    print(f"  Tensor parallel size: {num_gpus}")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    print(f"  Max model length: {max_model_len}")
    print(f"  Max output tokens: {max_tokens}")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        seed=SEED,
    )

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0, top_p=1.0)

    all_outputs = []
    total = len(prompts)
    num_chunks = (total + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total)
        chunk_prompts = prompts[start:end]
        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} "
              f"(items {start:,}-{end - 1:,}, size={len(chunk_prompts):,}) ...")

        chunk_start_time = time.time()
        outputs = llm.generate(chunk_prompts, sampling_params)
        chunk_elapsed = time.time() - chunk_start_time

        throughput = len(chunk_prompts) / chunk_elapsed if chunk_elapsed > 0 else 0
        print(f"  Chunk done in {chunk_elapsed:.1f}s "
              f"({throughput:.1f} items/s)")

        chunk_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            chunk_texts.append(generated_text)

        all_outputs.extend(chunk_texts)

        # Save checkpoint after each vLLM chunk
        if checkpoint_dir and data and similarities_dict:
            chunk_data = data[start:end]
            chunk_results = []
            for item, gen_text in zip(chunk_data, chunk_texts):
                words = parse_summary_words(gen_text)
                item_id = item["id"]
                similar_item_ids = [
                    sim["item_id"]
                    for sim in similarities_dict.get(item_id, [])[:5]
                ]
                result = item.copy()
                result["llm_output"] = gen_text
                result["summary_words"] = words
                result["similar_item_ids"] = similar_item_ids
                chunk_results.append(result)
            _save_vllm_checkpoint(chunk_results, checkpoint_dir, chunk_idx)

    return all_outputs


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 7-word product summaries using vLLM or Copilot API"
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="./raw_data/merged_clean_item.json",
        help="Path to item metadata JSON file",
    )
    parser.add_argument(
        "--similarity_file",
        type=str,
        default="./processed/similarities.json",
        help="Path to similarities JSON from step 0",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed/",
        help="Directory to save summaries and statistics",
    )
    parser.add_argument(
        "--summary_model",
        type=str,
        default="",
        #default="/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3.5-9B",
        help="Path to local LLM for vLLM inference. Set to empty string "
             "or non-existent path to use Copilot API instead.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="./resources/prompts.yaml",
        help="Path to prompts.yaml file with prompt templates",
    )
    parser.add_argument(
        "--prompt_template_name",
        type=str,
        default="term_generation",
        help="Name of the prompt template to use from prompts.yaml (default: term_generation)",
    )
    # --- vLLM-specific args ---
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (default: all available)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM KV-cache (default: 0.90)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200000,
        help="Number of prompts per vLLM generate() call (default: 200000)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning mode in chat template",
    )
    # --- Copilot API-specific args ---
    parser.add_argument(
        "--token_file",
        type=str,
        default="./resources/tokens.txt",
        help="Path to tokens.txt file for Copilot API authentication",
    )
    parser.add_argument(
        "--copilot_model",
        type=str,
        default="gpt-5.2",
        #default="gemini-3-flash-preview",
        help="Copilot model name",
    )
    parser.add_argument(
        "--copilot_workers",
        type=int,
        default=80,
        help="Number of parallel workers for Copilot API calls",
    )
    parser.add_argument(
        "--copilot_chunk_size",
        type=int,
        default=20000,
        help="Number of items per Copilot processing chunk for checkpoint "
             "saving",
    )
    # --- Common args ---
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of output tokens per prompt",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode: only process a subset of items",
    )
    parser.add_argument(
        "--debug_sample_size",
        type=int,
        default=100,
        help="Number of items to sample in debug mode (default: 100)",
    )
    parser.add_argument(
        "--id2meta_file",
        type=str,
        default=None,
        help="Output path for id2meta JSON (default: <output_dir>/id2meta.json)",
    )
    parser.add_argument(
        "--export_prompts_only",
        action="store_true",
        default=False,
        help="Build prompts for all remaining items (after resume/checkpoint), "
             "save to <output_dir>/prompts/<item_file_stem>_prompts.tsv, "
             "then exit without running inference.",
    )
    parser.add_argument(
        "--save_intermediate_only",
        action="store_true",
        default=False,
        help="Load resume + checkpoint results, save all output files "
             "(summaries, id2meta, statistics, etc.), then exit without "
             "running inference. Useful for testing downstream code while "
             "s1 is still running in another process.",
    )
    parser.add_argument(
        "--global_offer_only",
        action="store_true",
        default=False,
        help="Only process GlobalOfferID items (IDs not starting with 'P')",
    )
    parser.add_argument(
        "--resume_from_multi_path",
        type=str,
        nargs="*",
        default=["./processed/bak/summaries_with_similarity.jsonl","./processed/summaries_with_similarity.jsonl","./processed/s1_split_2/summaries_with_similarity.jsonl"],
        help="One or more paths to .jsonl files from previous runs to resume "
             "from. Each file should be a summaries_with_similarity.jsonl. "
             "Results are merged (later files overwrite earlier ones for "
             "duplicate IDs). Set to empty to skip resume.",
    )
    return parser.parse_args()


# =============================================================================
# Output Saving
# =============================================================================

def save_all_outputs(all_results, output_dir, id2meta_file=None):
    """Save all output files from a list of result dicts.

    Saves: summaries_with_similarity.jsonl, id2meta.json, id2words.tsv,
           statistics.json, failed_items.json.

    Args:
        all_results: List of result dicts (each has 'id', 'summary_words', etc.)
        output_dir: Directory for output files.
        id2meta_file: Optional custom path for id2meta.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- summaries_with_similarity.jsonl ----
    output_file = os.path.join(output_dir, "summaries_with_similarity.jsonl")
    print(f"\nSaving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- Statistics ----
    stats, failed_items, non_product_items = analyze_statistics(all_results)

    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_file}")

    # ---- Failed items ----
    all_problematic = []
    for item in failed_items:
        all_problematic.append({
            "id": item["id"],
            "title": item.get("title", ""),
            "reason": "failed_parse",
            "llm_output": item.get("llm_output", ""),
            "summary_words": item.get("summary_words", []),
        })
    for item in non_product_items:
        all_problematic.append({
            "id": item["id"],
            "title": item.get("title", ""),
            "reason": "non_product",
            "llm_output": item.get("llm_output", ""),
        })

    failed_file = os.path.join(output_dir, "failed_items.json")
    with open(failed_file, "w", encoding="utf-8") as f:
        json.dump(all_problematic, f, ensure_ascii=False, indent=2)
    print(f"Failed/non-product items saved to: {failed_file} "
          f"({len(all_problematic)} items)")

    # ---- Build id2meta mapping ----
    print("\nBuilding id2meta mapping ...")
    id2meta = {}
    skipped_count = 0
    for item in all_results:
        words = item.get("summary_words", [])
        non_empty_words = [w for w in words if w]
        if len(non_empty_words) != 7:
            skipped_count += 1
            continue
        item_id = item.get("id")
        if item_id:
            id2meta[item_id] = item

    id2meta_path = id2meta_file or os.path.join(output_dir, "id2meta.json")
    with open(id2meta_path, "w", encoding="utf-8") as f:
        json.dump(id2meta, f, ensure_ascii=False, indent=2)
    print(f"id2meta saved to: {id2meta_path}")
    print(f"  Mapped items: {len(id2meta):,} "
          f"(skipped {skipped_count:,} without valid 7-word summary)")

    # ---- Build id2words mapping ----
    id2words_file = os.path.join(output_dir, "id2words.tsv")
    with open(id2words_file, "w", encoding="utf-8") as f:
        for item_id, meta in id2meta.items():
            f.write(json.dumps({item_id: meta["summary_words"]},
                               ensure_ascii=False) + "\n")
    print(f"id2words saved to: {id2words_file} ({len(id2meta):,} items)")

    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    use_copilot = should_use_copilot(args.summary_model)

    if use_copilot:
        print("=" * 60)
        print("Inference mode: GitHub Copilot API")
        print(f"  Model: {args.copilot_model}")
        print(f"  Workers: {args.copilot_workers}")
        print(f"  Token file: {args.token_file}")
        print("=" * 60)
    else:
        import torch
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        num_gpus = args.num_gpus if args.num_gpus is not None else max(available_gpus, 1)
        print("=" * 60)
        print("Inference mode: vLLM (local model)")
        print(f"  PyTorch: {torch.__version__}, "
              f"CUDA: {torch.cuda.is_available()}, "
              f"HIP: {getattr(torch.version, 'hip', 'N/A')}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Using {num_gpus} GPU(s) for tensor parallelism")
        print("=" * 60)

    debug = args.debug
    if debug:
        print("\n*** DEBUG MODE: processing only 100 items ***\n")

    # ---- Load prompt template from prompts.yaml ----
    print(f"\nLoading prompt template from: {args.prompts_file}")
    prompts_config = load_prompts(args.prompts_file)
    prompt_template = prompts_config[args.prompt_template_name]['user']
    print(f"  Prompt template '{args.prompt_template_name}' loaded successfully")

    # ---- Load data ----
    print(f"\nLoading data: {args.item_file}")
    full_data = load_data(args.item_file)
    print(f"Loaded {len(full_data)} items")

    # Build full items dict BEFORE trimming, so similarity lookups work
    all_items_dict = {item["id"]: item for item in full_data}

    # Filter to GlobalOfferID items only (IDs not starting with 'P')
    if args.global_offer_only:
        full_data = [item for item in full_data if not item["id"].startswith("P")]
        print(f"Filtered to GlobalOfferID items: {len(full_data)} items")

    if debug:
        data = random.sample(full_data, min(args.debug_sample_size, len(full_data)))
        print(f"DEBUG: randomly sampled {len(data)} items for processing")
    else:
        data = full_data

    print(f"Loading similarities: {args.similarity_file}")
    similarities_dict = load_similarities(args.similarity_file)
    print(f"Loaded similarities for {len(similarities_dict)} items")

    # ---- Checkpoint setup ----
    checkpoint_dir = os.path.join(args.output_dir, "_s1_checkpoint")

    # ---- Resume: load previous results if available ----
    previous_results = {}  # item_id -> result dict
    resume_paths = args.resume_from_multi_path or []
    # Filter to existing .jsonl files
    valid_resume_files = [p for p in resume_paths if p and os.path.exists(p)]

    if valid_resume_files:
        repaired_count = 0
        for prev_file in valid_resume_files:
            print(f"\n[RESUME] Loading previous results from: {prev_file}")
            file_count = 0
            with open(prev_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        result = json.loads(line)
                        rid = result["id"]

                        # Repair records from Copilot checkpoint that were
                        # saved with 'result' instead of 'llm_output'
                        if "llm_output" not in result and "result" in result:
                            result["llm_output"] = result.pop("result")
                            result["summary_words"] = parse_summary_words(
                                result["llm_output"]
                            )
                            repaired_count += 1
                        # Also repair records with missing summary_words
                        elif "summary_words" not in result and "llm_output" in result:
                            result["summary_words"] = parse_summary_words(
                                result["llm_output"]
                            )
                            repaired_count += 1

                        previous_results[rid] = result
                        file_count += 1
            print(f"  Loaded {file_count:,} records from this file")

        if repaired_count > 0:
            print(f"  [RESUME] Repaired {repaired_count:,} records "
                  f"(re-parsed llm_output)")

        # Only keep results that are in the current data set
        current_ids = {item["id"] for item in data}
        prev_total = len(previous_results)
        previous_results = {
            k: v for k, v in previous_results.items() if k in current_ids
        }
        prev_kept = len(previous_results)
        prev_discarded = prev_total - prev_kept

        # Backfill item metadata for records missing title/description
        backfilled = 0
        for rid, result in previous_results.items():
            if "title" not in result and rid in all_items_dict:
                item_meta = all_items_dict[rid]
                for key, val in item_meta.items():
                    if key not in result:
                        result[key] = val
                backfilled += 1
        if backfilled > 0:
            print(f"  [RESUME] Backfilled metadata for {backfilled:,} records")

        # Filter data to only items not yet processed
        remaining_data_for_run = [
            item for item in data if item["id"] not in previous_results
        ]

        print(f"\n  Resume summary ({len(valid_resume_files)} files):")
        print(f"    Total loaded:             {prev_total:>10,}")
        print(f"    Kept (in current input):  {prev_kept:>10,}")
        print(f"    Discarded (not in input): {prev_discarded:>10,}")
        print(f"    Total items in run:       {len(data):>10,}")
        print(f"    Already completed:        {prev_kept:>10,}")
        print(f"    Remaining to process:     {len(remaining_data_for_run):>10,}")

        if not remaining_data_for_run:
            print(f"\n[RESUME] All {len(data)} items already completed!")
            all_results = [previous_results[item["id"]] for item in data]
            inference_time = 0.001
            _skip_inference = True
        else:
            data = remaining_data_for_run
            _skip_inference = False
    else:
        if resume_paths:
            print(f"\n[RESUME] No valid resume files found in: {resume_paths}")
        _skip_inference = False

    # ---- Load checkpoint results from interrupted previous runs ----
    # Handles both vLLM checkpoint format (full item dicts with 'id' key)
    # and Copilot checkpoint format ({"id": item_id, "result": response_text})
    if not _skip_inference:
        # Build a quick lookup for item data by id
        data_by_id = {item["id"]: item for item in data}

        # Try vLLM format first: each line is a full result dict
        ckpt_loaded = _load_vllm_checkpoint(checkpoint_dir)
        if ckpt_loaded:
            remaining_ids = {item["id"] for item in data}
            new_from_ckpt = 0
            for cid, cresult in ckpt_loaded.items():
                if cid in remaining_ids and cid not in previous_results:
                    previous_results[cid] = cresult
                    new_from_ckpt += 1
            if new_from_ckpt > 0:
                print(f"  [CHECKPOINT-vLLM] Recovered {new_from_ckpt:,} items")

        # Also try Copilot format: {"id": item_id, "result": response_text}
        copilot_ckpt = _load_checkpoint_raw(checkpoint_dir)
        if copilot_ckpt:
            new_from_copilot_ckpt = 0
            for cid, response_text in copilot_ckpt.items():
                if cid in data_by_id and cid not in previous_results:
                    item = data_by_id[cid]
                    words = parse_summary_words(response_text)
                    similar_item_ids = [
                        sim["item_id"]
                        for sim in similarities_dict.get(cid, [])[:5]
                    ]
                    result = item.copy()
                    result["llm_output"] = response_text
                    result["summary_words"] = words
                    result["similar_item_ids"] = similar_item_ids
                    previous_results[cid] = result
                    new_from_copilot_ckpt += 1
            if new_from_copilot_ckpt > 0:
                print(f"  [CHECKPOINT-Copilot] Recovered "
                      f"{new_from_copilot_ckpt:,} items")

        # Update remaining data after all checkpoint loading
        total_recovered = len(previous_results)
        if total_recovered > 0:
            data = [item for item in data
                    if item["id"] not in previous_results]
            print(f"\n  Total recovered from all sources: {total_recovered:,}")
            print(f"  Remaining to process: {len(data):,}")
            if not data:
                all_results = [
                    previous_results[item["id"]]
                    for item in full_data
                    if item["id"] in previous_results
                ]
                inference_time = 0.001
                _skip_inference = True

    # ---- Pre-save: save all output files before processing new data ----
    if previous_results:
        print(f"\n[PRE-SAVE] Saving {len(previous_results):,} completed "
              f"results and all derived files ...")
        save_all_outputs(
            list(previous_results.values()), args.output_dir,
            id2meta_file=args.id2meta_file,
        )
        # Only clean checkpoint if we're about to run inference
        # (not in export_prompts_only or save_intermediate_only mode)
        if not args.export_prompts_only and not args.save_intermediate_only:
            cleanup_checkpoint(checkpoint_dir)

    # ---- Save intermediate only (if requested) ----
    if args.save_intermediate_only:
        if previous_results:
            print(f"\nDone! (--save_intermediate_only mode, inference skipped)")
        else:
            print(f"\n[SAVE-INTERMEDIATE] No results to save "
                  f"(no resume or checkpoint data found).")
        return

    # ---- Export prompts only (if requested) ----
    if args.export_prompts_only:
        # Build prompts for remaining items (after resume/checkpoint)
        print(f"\n[EXPORT-PROMPTS] Building prompts for {len(data):,} "
              f"remaining items ...")

        # Derive output filename from item_file stem
        item_stem = os.path.splitext(os.path.basename(args.item_file))[0]
        prompts_dir = os.path.join(args.output_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        prompts_file = os.path.join(prompts_dir, f"{item_stem}_prompts.tsv")

        with open(prompts_file, "w", encoding="utf-8") as f:
            f.write("GlobalOfferId\tPrompt\n")
            for item in tqdm(data, desc="Building prompts"):
                item_id = item["id"]
                top_similar_items = similarities_dict.get(item_id, [])[:5]
                raw_prompt = prepare_prompt(
                    item, top_similar_items, all_items_dict, prompt_template
                )
                escaped_prompt = raw_prompt.replace("\t", " ").replace("\n", "\\n")
                f.write(f"{item_id}\t{escaped_prompt}\n")

        file_size_mb = os.path.getsize(prompts_file) / (1024 * 1024)
        print(f"  Saved {len(data):,} prompts to: {prompts_file} "
              f"({file_size_mb:.1f} MB)")
        print(f"\nDone! (--export_prompts_only mode, inference skipped)")
        return

    # ---- Run inference ----
    start_time = time.time()

    if _skip_inference:
        pass  # all_results already set above
    elif use_copilot:
        # --- Copilot API path ---
        # Build (item_id, prompt) inputs for all items
        print("\nBuilding prompts for Copilot API ...")
        copilot_inputs = []
        for item in tqdm(data, desc="Building prompts"):
            item_id = item["id"]
            top_similar_items = similarities_dict.get(item_id, [])[:5]
            raw_prompt = prepare_prompt(
                item, top_similar_items, all_items_dict, prompt_template
            )
            copilot_inputs.append((item_id, raw_prompt))

        # Use shared run_llm_parallel_with_checkpoint
        copilot_results = run_llm_parallel_with_checkpoint(
            inputs=copilot_inputs,
            token_file=args.token_file,
            checkpoint_dir=checkpoint_dir,
            num_workers=args.copilot_workers,
            model=args.copilot_model,
            temperature=0,
            max_tokens=args.max_tokens,
            chunk_size=args.copilot_chunk_size,
        )

        # Convert (item_id, llm_output) results to full result dicts
        copilot_map = {item_id: gen_text for item_id, gen_text in copilot_results}
        all_results = []
        for item in data:
            item_id = item["id"]
            gen_text = copilot_map.get(item_id, "")
            words = parse_summary_words(gen_text)
            similar_item_ids = [
                sim["item_id"]
                for sim in similarities_dict.get(item_id, [])[:5]
            ]
            result = item.copy()
            result["llm_output"] = gen_text
            result["summary_words"] = words
            result["similar_item_ids"] = similar_item_ids
            all_results.append(result)
    else:
        # --- vLLM path (with checkpoint/resume) ---
        # Load checkpoint to skip already-processed items
        completed = _load_vllm_checkpoint(checkpoint_dir)
        remaining_data = [item for item in data if item["id"] not in completed]

        if not remaining_data:
            print(f"  All {len(data)} items already completed from checkpoint")
            all_results = [completed[item["id"]] for item in data]
        else:
            if completed:
                print(f"  Items to process: {len(remaining_data)} "
                      f"(skipped {len(completed)} from checkpoint)")

            # Build prompts for remaining items
            print(f"\nBuilding prompts (enable_thinking={args.enable_thinking}) ...")
            prompts = build_all_prompts_vllm(
                remaining_data, similarities_dict, all_items_dict,
                prompt_template, args.summary_model, args.enable_thinking
            )
            print(f"Built {len(prompts)} prompts")

            if debug:
                print(f"\n{'='*60}")
                print("[DEBUG] Full sample prompt (first item):")
                print(prompts[0])
                print(f"{'='*60}")
                for idx in range(1, min(5, len(prompts))):
                    print(f"\n[DEBUG] Prompt #{idx+1} "
                          f"(item ID: {remaining_data[idx]['id']}):")
                    print(prompts[idx])
                    print(f"{'='*60}")
                print()

            # Run vLLM inference with checkpoint saving
            generated_texts = run_vllm_inference(
                prompts,
                model_name=args.summary_model,
                num_gpus=num_gpus,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_tokens=args.max_tokens,
                chunk_size=args.chunk_size,
                data=remaining_data,
                similarities_dict=similarities_dict,
                checkpoint_dir=checkpoint_dir,
            )

            # Parse results for remaining items
            print("\nParsing LLM outputs ...")
            for item, gen_text in zip(remaining_data, generated_texts):
                words = parse_summary_words(gen_text)
                item_id = item["id"]
                similar_item_ids = [
                    sim["item_id"]
                    for sim in similarities_dict.get(item_id, [])[:5]
                ]
                result = item.copy()
                result["llm_output"] = gen_text
                result["summary_words"] = words
                result["similar_item_ids"] = similar_item_ids
                completed[item_id] = result

            # Merge: build all_results in original data order
            all_results = [completed[item["id"]] for item in data]

    if not _skip_inference:
        inference_time = time.time() - start_time
        print(f"\nInference time: {inference_time:.1f}s "
              f"({len(data) / inference_time:.1f} items/s)")

        # Merge with previous results if resuming
        if valid_resume_files and previous_results:
            print(f"\n[RESUME] Merging {len(previous_results):,} previous + "
                  f"{len(all_results):,} new results")
            merged = dict(previous_results)  # start with previous
            for result in all_results:
                merged[result["id"]] = result  # new results overwrite
            # Rebuild all_results from full_data order (pre-filter data)
            # We need to use the original full data order
            original_data = full_data
            all_results = [
                merged[item["id"]] for item in original_data
                if item["id"] in merged
            ]
            print(f"  Merged total: {len(all_results):,} items")

    # Print first few examples
    num_debug_show = 10 if debug else 3
    print(f"\n{'='*60}")
    print(f"First {num_debug_show} LLM outputs:")
    for idx, res in enumerate(all_results[:num_debug_show]):
        print(f"\n  [{idx+1}] ID={res['id']}")
        print(f"      Title: {res.get('title', 'N/A')[:120]}")
        print(f"      LLM raw output: {res.get('llm_output', 'N/A')[:200]}")
        print(f"      Parsed words:   {res.get('summary_words', [])}")
    print(f"\n{'='*60}")

    # ---- Save all output files ----
    stats = save_all_outputs(all_results, args.output_dir,
                             id2meta_file=args.id2meta_file)

    # ---- Clean up checkpoint ----
    cleanup_checkpoint(checkpoint_dir)

    # ---- Final summary ----
    print(f"\nCompleted! Total: {len(all_results)}, "
          f"Products: {stats['valid_product_count']}, "
          f"Non-products: {stats['non_product_count']}, "
          f"Failed: {stats['failed_count']}")
    if inference_time > 0.01:
        print(f"Throughput: {len(data) / inference_time:.1f} items/s")


if __name__ == "__main__":
    main()
