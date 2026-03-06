"""Step 1: Generate Product Summaries & Build ID-to-Metadata Mapping

Uses vLLM offline inference to generate 7-word summaries for each product,
leveraging similar items context from step 0.  The model first determines
whether the input describes a real product or is just a general marketing
sentence.  Non-product entries receive an empty summary ([]).

After inference, builds an id2meta.json mapping (previously s2_build_id2meta.py)
that maps each item ID to its full metadata including summary words.

vLLM handles batching, KV-cache management, and continuous batching
automatically, achieving 10-30x throughput vs HuggingFace generate().

Inputs (from s0_init_emb.py):
    - merged_clean_item.json   : unified item metadata (title, description,
                                 categories, related_queries)
    - similarities.json        : top-k similar item IDs per item

Outputs:
    - summaries_with_similarity.jsonl : per-item results with LLM output
    - id2meta.json                    : item ID -> metadata mapping
    - statistics.json                 : word frequency & conflict stats
    - failed_items.json               : items that did not produce 7 words

Usage:
    python s1_init_sum_meta.py \
        --item_file ./raw_data/merged_clean_item.json \
        --similarity_file ./processed/similarities.json \
        --output_dir ./processed/ \
        --summary_model /path/to/Qwen3-8B \
        --num_gpus 2
"""

import os
import json
import re
import random
import argparse
from collections import Counter
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch

SEED = 42
random.seed(SEED)


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

def prepare_prompt(item, top_similar_items, all_items_dict):
    """Prepare prompt including information from the 5 most similar items.

    The prompt asks the model to first determine whether the product information
    actually describes a concrete product.  If it is a general marketing phrase
    or non-product text, the model should output an empty list [].
    """
    # Build product information block (skip empty fields to avoid blank lines)
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

    related_queries = item.get("related_queries", "")
    if related_queries:
        if len(related_queries) > 150:
            related_queries = related_queries[:150] + "..."
        info_lines.append(f"Related Queries: {related_queries}")

    # Append structured attributes (from s6 enrichment)
    # Only include Brand, Seller, Model, and non-default Gender/AgeGroup.
    # Exclude Color, Size, Material (too granular; distracts from core product identity).
    # Exclude Price, Market (not relevant to product summarization).
    attributes = item.get("attributes", {})
    brand = attributes.get("Brand", "")
    if isinstance(brand, str):
        brand = " ".join(brand.split())  # strip + collapse whitespace
    seller = attributes.get("Seller", "")
    if isinstance(seller, str):
        seller = " ".join(seller.split())  # strip + collapse whitespace
    # If brand and seller are the same (case-insensitive), merge into one line
    # to prevent LLM from outputting duplicate entries
    if brand and seller and brand.lower() == seller.lower():
        info_lines.append(f"Brand/Seller: {brand}")
    else:
        if brand:
            info_lines.append(f"Brand: {brand}")
        if seller:
            info_lines.append(f"Seller: {seller}")
    for attr_name in ["Gender", "Color", "Size"]:
        attr_val = attributes.get(attr_name, "")
        if isinstance(attr_val, str):
            attr_val = attr_val.strip()
        if attr_val:
            info_lines.append(f"{attr_name}: {attr_val}")
    # Gender: only if specific (skip "Unisex" — it's the default)
    gender = attributes.get("Gender", "").strip()
    if gender and gender.lower() != "unisex":
        info_lines.append(f"Gender: {gender}")
    # AgeGroup: only if specific (skip "Adult" — it's the default)
    age_group = attributes.get("AgeGroup", "").strip()
    if age_group and age_group.lower() != "adult":
        info_lines.append(f"AgeGroup: {age_group}")

    product_info_text = "\n".join(info_lines) if info_lines else "(no information)"

    # Prepare similar items information
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

    similar_items_text = "\n".join(similar_items_info) if similar_items_info else "(none)"

    prompt = f"""You are an expert product summarizer. Your task has TWO steps:

STEP 1 - PRODUCT VALIDATION:
First, carefully analyze the PRODUCT INFORMATION below. Determine whether it describes a REAL, SPECIFIC product (e.g., a physical item, a digital product, a tool, a food item, etc.).
If the information is just a general marketing slogan, a brand tagline, a vague browsing/search phrase, or does NOT describe a concrete product, output ONLY an empty list: []

STEP 2 - SUMMARIZATION (only if Step 1 confirms a real product):
Generate exactly SEVEN words to summarize this product. The 7 words should cover as many of the following aspects as possible, ordered by priority. Not every product has all aspects — include only what is available and informative:

PRIORITY ORDER
   1. Main product category or type (e.g., "headphone", "sneaker", "cream")
   2. Key function or primary capability (e.g., "noise-canceling", "dimmable", "anti-aging")
   3. Distinctive feature, form factor, or sub-type (e.g., "over-ear", "sphere", "spf-15")
   4. Brand or platform ecosystem (e.g., "sony", "nike", "apple-compatible", "joss & main") — MUST include if provided
   5. Seller or retailer (e.g., "amazon", "walmart", "home depot") — MUST include if provided
   6. Target audience — include ONLY if specific (e.g., "kids", "women", "professional", "cat"); skip generic values like "unisex" or "adult"
   7. Style, occasion, or unique selling point (e.g., "holiday", "outdoor", "waterproof", "budget-friendly")

RULES:
1. WORD FORM: All words must be in their base form (nouns or adjectives, no -ed, -ing, -s endings). Use hyphens to combine closely related concepts into a single word (e.g., "anti-aging", "spf-15", "noise-canceling").
2. DO NOT SPLIT PROPER NOUNS: Brand names and seller names are each ONE comma-separated entry, even if they contain multiple English words. Keep them intact — do NOT split them across entries.
   - Brand "Home Depot" → one entry "home depot", NOT two entries "home", "depot"
   - Brand "Joss & Main" → one entry "joss & main", NOT two entries "joss", "main"
   - Seller "楽天市場" → one entry "楽天市場"
3. NO DUPLICATES: Every entry in the output list must be distinct. If brand and seller are the same or very similar (e.g., Brand="Hammitt", Seller="Hammitt"), output it only ONCE and use the freed slot for another informative aspect.
   - Example: Brand="Hammitt", Seller="Hammitt" → include "hammitt" once, NOT twice.
4. COVERAGE: MUST cover as many available aspects as possible within 7 words. If brand is present, it MUST appear. If seller is present and different from brand, it MUST appear. If a specific audience (not "unisex" or "adult") is present, it MUST appear. Only omit an aspect if the product genuinely lacks that information.
5. PRODUCT-FIRST: The first 2-3 words should describe the product itself (category, function, distinctive feature). A reader should be able to picture what the product is from these words alone.
6. CONSISTENCY WITH SIMILAR ITEMS: Consider the similar items provided. If they share common characteristics, use consistent terminology for those aspects.
7. UNIQUENESS: Include at least 1-2 words that distinguish this product from the similar items.
8. LANGUAGE: Use the same language as the product information. If the product is in Japanese, Chinese, etc., the output words should also be in that language. Do NOT translate to English.

OUTPUT FORMAT:
- If NOT a real product: <Output>[]</Output>
- If a real product: <Output>[word1, word2, word3, word4, word5, word6, word7]</Output>
- Wrap your final answer in <Output></Output> tags. No text outside these tags except thinking.

PRODUCT INFORMATION:
{product_info_text}

TOP 5 SIMILAR PRODUCTS (for reference):
{similar_items_text}

ANALYSIS GUIDANCE:
1. Validate: is this a real product? If not, output []
2. Pick category + function + feature (first 3 words)
3. Add brand (keep intact, no splitting). If brand = seller, include once only
4. Add seller if different from brand (keep intact, no splitting)
5. Add audience / style / unique trait to fill remaining slots
6. Verify: no duplicates, exactly 7 entries, same language as input

Output:
"""

    return prompt


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
                word.strip().lower().strip("\"'[]") for word in inner_content.split(",")
            ]
        else:
            words = [
                word.strip().lower().strip("\"'[]") for word in inner_content.split()
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
                        word.strip().lower().strip("\"'") for word in inner.split(",")
                    ]
                else:
                    words = [
                        word.strip().lower().strip("\"'") for word in inner.split()
                    ]
                break
        if not words:
            content = re.sub(r"[，、]", ",", content)
            if "," in content:
                words = [
                    word.strip().lower().strip("\"'[]") for word in content.split(",")
                ]
            else:
                words = [
                    word.strip().lower().strip("\"'[]") for word in content.split()
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
    words = deduped[:7]

    return words


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
# vLLM Inference
# =============================================================================

def build_all_prompts(data, similarities_dict, all_items_dict, tokenizer,
                      enable_thinking=False):
    """Build formatted prompts for all items using the tokenizer's chat template.

    Args:
        enable_thinking: Whether to enable thinking/reasoning in the chat template.

    Returns:
        List of formatted prompt strings ready for vLLM generate().
    """
    prompts = []
    for item in tqdm(data, desc="Building prompts"):
        item_id = item["id"]
        top_similar_items = similarities_dict.get(item_id, [])[:5]
        raw_prompt = prepare_prompt(item, top_similar_items, all_items_dict)
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
):
    """Run vLLM offline inference on all prompts.

    vLLM internally handles continuous batching, PagedAttention,
    and scheduling.  We pass prompts in chunks (default 100K) to allow
    progress tracking.

    Args:
        prompts: List of formatted prompt strings.
        model_name: Path to the model.
        num_gpus: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use (0-1).
        max_model_len: Maximum context length for the model.
        max_tokens: Maximum number of output tokens per prompt.
        chunk_size: Number of prompts per generate() call.

    Returns:
        List of generated text strings, aligned with input prompts.
    """
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

        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            all_outputs.append(generated_text)

    return all_outputs


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 7-word product summaries using vLLM"
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="./raw_data/merged_clean_item_with_attr.json",
        help="Path to item metadata JSON file (merged_clean_item_with_attr.json from s6)",
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
        default="./processed_tid/",
        help="Directory to save summaries and statistics",
    )
    parser.add_argument(
        "--summary_model",
        type=str,
        # default="/data/xiaoyukou/ckpts/Qwen3-8B",
        default="/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3.5-9B",
        help="Path to summary LLM (must be vLLM-supported, e.g. Qwen3.5-9B)",
    )
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
        help="Maximum model context length. Our prompts are ~500-800 tokens + "
             "50 output tokens, so 4096 is sufficient. Lower saves GPU memory "
             "and increases throughput. (default: 4096)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Number of prompts per vLLM generate() call for progress "
             "tracking (default: 100000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode: only process 100 items, print sample prompt "
             "and first few LLM outputs (default: False)",
    )
    parser.add_argument(
        "--id2meta_file",
        type=str,
        default=None,
        help="Output path for id2meta JSON. If not set, defaults to "
             "<output_dir>/id2meta.json",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=80,
        help="Maximum number of output tokens per prompt. Increase when "
             "using --enable_thinking (e.g., 512-1024). (default: 80)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning mode in chat template "
             "(default: False)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_gpus = args.num_gpus if args.num_gpus is not None else max(available_gpus, 1)

    print(f"PyTorch: {torch.__version__}, "
          f"CUDA: {torch.cuda.is_available()}, "
          f"HIP: {getattr(torch.version, 'hip', 'N/A')}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Using {num_gpus} GPU(s) for tensor parallelism")

    debug = args.debug
    if debug:
        print("\n*** DEBUG MODE: processing only 100 items ***\n")

    # ---- Load data ----
    print(f"\nLoading data: {args.item_file}")
    full_data = load_data(args.item_file)
    print(f"Loaded {len(full_data)} items")

    # Build full items dict BEFORE trimming, so similarity lookups work
    all_items_dict = {item["id"]: item for item in full_data}

    if debug:
        data = random.sample(full_data, min(100, len(full_data)))
        print(f"DEBUG: randomly sampled {len(data)} items for processing")
    else:
        data = full_data

    print(f"Loading similarities: {args.similarity_file}")
    similarities_dict = load_similarities(args.similarity_file)
    print(f"Loaded similarities for {len(similarities_dict)} items")

    # ---- Build prompts ----
    print(f"\nBuilding prompts (enable_thinking={args.enable_thinking}) ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.summary_model, trust_remote_code=True
    )
    prompts = build_all_prompts(data, similarities_dict, all_items_dict, tokenizer,
                                enable_thinking=args.enable_thinking)
    print(f"Built {len(prompts)} prompts")

    if debug:
        print(f"\n{'='*60}")
        print("[DEBUG] Full sample prompt (first item):")
        print(prompts[0])
        print(f"{'='*60}")
        # Print a few more prompts (truncated) so we can see if similarities work
        for idx in range(1, min(5, len(prompts))):
            print(f"\n[DEBUG] Prompt #{idx+1} (item ID: {data[idx]['id']}):")
            print(prompts[idx])
            print(f"{'='*60}")
        print()

    # ---- vLLM inference ----
    start_time = time.time()
    generated_texts = run_vllm_inference(
        prompts,
        model_name=args.summary_model,
        num_gpus=num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
    )
    inference_time = time.time() - start_time
    print(f"\nTotal vLLM inference time: {inference_time:.1f}s "
          f"({len(data) / inference_time:.1f} items/s)")

    # ---- Parse results ----
    print("\nParsing LLM outputs ...")
    all_results = []
    for item, gen_text in zip(data, generated_texts):
        words = parse_summary_words(gen_text)
        item_id = item["id"]
        similar_item_ids = [
            sim["item_id"] for sim in similarities_dict.get(item_id, [])[:5]
        ]
        result = item.copy()
        result["llm_output"] = gen_text
        result["summary_words"] = words
        result["similar_item_ids"] = similar_item_ids
        all_results.append(result)

    # Print first few debug examples
    num_debug_show = 10 if debug else 3
    print(f"\n{'='*60}")
    print(f"First {num_debug_show} LLM outputs:")
    for idx, res in enumerate(all_results[:num_debug_show]):
        print(f"\n  [{idx+1}] ID={res['id']}")
        print(f"      Title: {res.get('title', 'N/A')[:120]}")
        print(f"      LLM raw output: {res['llm_output']}")
        print(f"      Parsed words:   {res['summary_words']}")
    print(f"\n{'='*60}")

    # ---- Save results ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "summaries_with_similarity.jsonl")
    print(f"\nSaving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- Statistics ----
    stats, failed_items, non_product_items = analyze_statistics(all_results)

    stats_file = os.path.join(args.output_dir, "statistics.json")
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

    failed_file = os.path.join(args.output_dir, "failed_items.json")
    with open(failed_file, "w", encoding="utf-8") as f:
        json.dump(all_problematic, f, ensure_ascii=False, indent=2)
    print(f"Failed/non-product items saved to: {failed_file} "
          f"({len(all_problematic)} items)")

    # ---- Build id2meta mapping (previously s2_build_id2meta.py) ----
    print("\nBuilding id2meta mapping ...")
    id2meta = {}
    skipped_count = 0
    for item in all_results:
        words = item.get("summary_words", [])
        non_empty_words = [w for w in words if w]
        if len(non_empty_words) != 7:
            skipped_count += 1
            continue
        # Normalize multi-word summaries (join with hyphen)
        item["summary_words"] = [
            "-".join(word.split()) for word in words
        ]
        item_id = item.get("id")
        if item_id:
            id2meta[item_id] = item

    id2meta_file = args.id2meta_file or os.path.join(args.output_dir, "id2meta.json")
    with open(id2meta_file, "w", encoding="utf-8") as f:
        json.dump(id2meta, f, ensure_ascii=False, indent=2)
    print(f"id2meta saved to: {id2meta_file}")
    print(f"  Mapped items: {len(id2meta):,} "
          f"(skipped {skipped_count:,} without valid 7-word summary)")

    # ---- Final summary ----
    print(f"\nCompleted! Total: {len(all_results)}, "
          f"Products: {stats['valid_product_count']}, "
          f"Non-products: {stats['non_product_count']}, "
          f"Failed: {stats['failed_count']}")
    print(f"Throughput: {len(data) / inference_time:.1f} items/s")


if __name__ == "__main__":
    main()
