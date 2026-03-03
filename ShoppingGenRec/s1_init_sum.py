"""Step 1: Generate Product Summaries using LLM

Uses a causal LLM (e.g., Qwen3-4B) to generate 5-word summaries for each
product, leveraging similar items context from step 0.  The model first
determines whether the input describes a real product or is just a general
marketing sentence.  Non-product entries receive an empty summary ([]).

Inputs (from s0_init_emb.py):
    - merged_clean_item.json   : unified item metadata (title, description,
                                 categories, related_queries)
    - similarities.json        : top-k similar item IDs per item

Outputs:
    - summaries_with_similarity.jsonl : per-item results with LLM output
    - statistics.json                 : word frequency & conflict stats
    - failed_items.json               : items that did not produce 5 words

Usage:
    python s1_init_sum.py \
        --item_file ./raw_data/merged_clean_item.json \
        --similarity_file ./processed/similarities.json \
        --output_dir ./processed/sum_data \
        --summary_model /path/to/Qwen3-4B \
        --num_gpus 2
"""

import os
import json
import re
import random
import argparse
from collections import Counter
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


def prepare_prompt(item, top_similar_items, all_items_dict):
    """Prepare prompt including information from the 5 most similar items.

    The prompt asks the model to first determine whether the product information
    actually describes a concrete product.  If it is a general marketing phrase
    or non-product text, the model should output an empty list [].
    """
    title = item.get("title", "")
    title_text = f"Title: {title}" if title else ""

    description = item.get("description", "")
    if description:
        if len(description) > 150:
            description_text = f"Description: {description[:150]}..."
        else:
            description_text = f"Description: {description}"
    else:
        description_text = ""

    categories = item.get("categories", "")
    categories_text = f"Categories: {categories}" if categories else ""

    related_queries = item.get("related_queries", "")
    related_queries_text = f"Related Queries: {related_queries}" if related_queries else ""

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
Generate exactly FIVE words to summarize this product following these guidelines:

GUIDELINES:
1. WORD FORM: All words must be in their base form (nouns or adjectives, no -ed, -ing, -s endings)
2. WORD ORDER: Order words by importance (most important aspect first)
3. CONTENT FOCUS: Focus on these aspects in order:
   a) Main product category/type (e.g., "doll", "puzzle", "car")
   b) Key function or purpose (e.g., "educational", "remote-control")
   c) Distinctive features (e.g., "wooden", "electronic", "collectible")
   d) Target audience (e.g., "toddler", "boys", "family")
   e) Unique selling point (e.g., "glow-in-dark", "interactive")
4. CONSISTENCY WITH SIMILAR ITEMS: Consider the similar items provided. If they share common characteristics, use consistent terminology for those aspects.
5. UNIQUENESS: Include at least 1-2 words that distinguish this product from the similar items. Each product should have some unique aspects.

OUTPUT FORMAT:
- If NOT a real product: []
- If a real product: [word1, word2, word3, word4, word5]
- NO ADDITIONAL TEXT. Do not include any explanations, thoughts, or other content.

PRODUCT INFORMATION:
{title_text}
{description_text}
{categories_text}
{related_queries_text}

TOP 5 SIMILAR PRODUCTS (for reference):
{similar_items_text}

ANALYSIS GUIDANCE:
1. First, check whether the PRODUCT INFORMATION describes a specific, concrete product. 
2. If it IS a product, identify what it has in common with similar products (shared category, features, audience)
3. Then, identify what makes this product unique or different
4. Use consistent vocabulary for shared characteristics
5. Include distinctive vocabulary for unique aspects
6. Ensure words cover the five required aspects in order
7. Finally, output exactly five words in this exact format: [word1, word2, word3, word4, word5], or [] if not a product.

Output:
"""

    return prompt


def process_batch_on_gpu(
    rank, data_slice, output_queue, model_name, similarities_dict, all_items_dict
):
    """Process data slice on specific GPU."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}: Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{rank}",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Rank {rank}: Processing {len(data_slice)} items")
    results = []

    for i in tqdm(range(len(data_slice)), desc=f"Rank {rank}"):
        batch_items = [data_slice[i]]
        batch_results = process_single_batch(
            batch_items, model, tokenizer, device, similarities_dict, all_items_dict
        )
        results.extend(batch_results)

    output_queue.put((rank, results))
    print(f"Rank {rank}: Completed {len(results)} items")


def process_single_batch(
    items, model, tokenizer, device, similarities_dict, all_items_dict
):
    """Process a single batch of items."""
    prompts = []
    for item in items:
        item_id = item["id"]
        top_similar_items = similarities_dict.get(item_id, [])[:5]
        prompt = prepare_prompt(item, top_similar_items, all_items_dict)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)

    tokenizer.padding_side = "left"
    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
        max_length=32768,
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

    results = []
    for item, input_ids, output_ids in zip(
        items, model_inputs.input_ids, generated_ids
    ):
        generated_output_ids = output_ids[len(input_ids) :].tolist()
        content = tokenizer.decode(generated_output_ids, skip_special_tokens=True).strip("\n")

        item_id = item["id"]
        similar_item_ids = [
            sim["item_id"] for sim in similarities_dict.get(item_id, [])[:5]
        ]

        words = parse_summary_words(content)

        item_copy = item.copy()
        item_copy["llm_output"] = content
        item_copy["summary_words"] = words
        item_copy["similar_item_ids"] = similar_item_ids
        results.append(item_copy)

    return results


def parse_summary_words(content: str) -> list:
    """Parse 5-word summary or empty list from LLM output.

    Returns:
        A list of 5 word strings for valid products, or an empty list []
        for non-product entries.  If parsing fails, returns a list with
        fewer than 5 non-empty words (some may be "").
    """
    if not content:
        return []

    content_stripped = content.strip()

    # Check for empty list output: [] (non-product)
    if re.search(r"^\s*\[\s*\]\s*$", content_stripped):
        return []

    # Also handle [] appearing anywhere in  output with nothing else meaningful
    # e.g. model outputs "This is not a product.\n[]"
    if "[]" in content_stripped:
        # Check if there's also a non-empty bracket — if not, treat as empty
        non_empty_bracket = re.search(r"\[[^\]]+\]", content_stripped)
        if not non_empty_bracket:
            return []

    # Try [word1, word2, ...] format
    pattern = r"\[([^\]]+)\]"
    match = re.search(pattern, content)
    if match:
        inner_content = match.group(1)
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
                    return []  # empty list on its own line
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
            if "," in content:
                words = [
                    word.strip().lower().strip("\"'[]") for word in content.split(",")
                ]
            else:
                words = [
                    word.strip().lower().strip("\"'[]") for word in content.split()
                ]

    # Filter out empty strings produced by splitting
    words = [w for w in words if w]
    words = words[:5]
    while len(words) < 5:
        words.append("")

    return words


def analyze_statistics(all_items, similarities_dict):
    """Analyze summary statistics, separating products from non-products."""
    print("\n" + "=" * 50)
    print("Statistical Analysis Results")
    print("=" * 50)

    # Separate product items (5 words) from non-product items (empty list)
    product_items = []
    non_product_items = []
    failed_items = []  # items that are neither valid 5-word nor empty-list

    for item in all_items:
        words = item.get("summary_words", [])
        non_empty_words = [w for w in words if w]
        if len(words) == 0:
            # Model determined this is not a product
            non_product_items.append(item)
        elif len(non_empty_words) == 5:
            product_items.append(item)
        else:
            # Has some words but not exactly 5 non-empty ones
            failed_items.append(item)

    total = len(all_items)
    print(f"\n0. Item Classification:")
    print(f"   Total items:              {total:>10,}")
    print(f"   Valid products (5 words): {len(product_items):>10,} "
          f"({len(product_items) / total * 100:.2f}%)")
    print(f"   Non-products (empty []):  {len(non_product_items):>10,} "
          f"({len(non_product_items) / total * 100:.2f}%)")
    print(f"   Failed parsing:           {len(failed_items):>10,} "
          f"({len(failed_items) / total * 100:.2f}%)")

    # Word frequency analysis (only on valid product items)
    all_words = []
    word_freq = Counter()
    word_by_position = [Counter() for _ in range(5)]

    for item in product_items:
        words = item.get("summary_words", [])
        all_words.extend([word for word in words if word])
        for i, word in enumerate(words):
            if i < 5 and word:
                word_by_position[i][word] += 1

    word_freq.update(all_words)

    print(f"\n1. Vocabulary (products only): {len(all_words)} total, {len(word_freq)} unique")
    print("   Top 20 words:")
    for word, count in word_freq.most_common(20):
        print(f"     {word}: {count}")

    positions = [
        "Product Category",
        "Function/Purpose",
        "Features",
        "Audience",
        "Unique Point",
    ]
    for i, (pos, counter) in enumerate(zip(positions, word_by_position)):
        print(f"\n   Position {i + 1} ({pos}) top 10:")
        for word, count in counter.most_common(10):
            print(f"     {word}: {count}")

    # Conflict analysis (only among valid products)
    summary_tuples = [tuple(item.get("summary_words", [])) for item in product_items]
    tuple_counter = Counter(summary_tuples)
    duplicate_tuples = [
        (tup, count) for tup, count in tuple_counter.items() if count > 1
    ]
    total_conflicts = sum(count - 1 for _, count in duplicate_tuples)
    conflict_rate = total_conflicts / len(product_items) if product_items else 0

    print(f"\n2. Conflicts (products): {len(duplicate_tuples)} duplicates, "
          f"rate={conflict_rate:.4f}")

    # Show some non-product examples
    if non_product_items:
        print(f"\n3. Non-product examples (first 5):")
        for item in non_product_items[:5]:
            print(f"     ID={item['id']}, Title={item.get('title', 'N/A')}")

    # Show some failed examples
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 5-word product summaries using LLM"
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="./raw_data/merged_clean_item.json",
        help="Path to item metadata JSON file (merged_clean_item.json from s4)",
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
        default="./processed/sum_data",
        help="Directory to save summaries and statistics",
    )
    parser.add_argument(
        "--summary_model",
        type=str,
        default="/data/xiaoyukou/ckpts/Qwen3-4B-Instruct-2507",
        help="Path to summary LLM",
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

    print(f"Loading data: {args.item_file}")
    data = load_data(args.item_file)
    print(f"Loaded {len(data)} items")

    print(f"Loading similarities: {args.similarity_file}")
    similarities_dict = load_similarities(args.similarity_file)
    print(f"Loaded similarities for {len(similarities_dict)} items")

    all_items_dict = {item["id"]: item for item in data}

    # Split data across GPUs
    chunk_size = len(data) // num_gpus
    data_chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = len(data) if i == num_gpus - 1 else start + chunk_size
        data_chunks.append(data[start:end])

    processes = []
    output_queue = mp.Queue()
    start_time = time.time()

    for rank in range(num_gpus):
        p = mp.Process(
            target=process_batch_on_gpu,
            args=(
                rank,
                data_chunks[rank],
                output_queue,
                args.summary_model,
                similarities_dict,
                all_items_dict,
            ),
        )
        processes.append(p)
        p.start()

    all_results = []
    for _ in range(num_gpus):
        rank, results = output_queue.get()
        print(f"Received {len(results)} results from Rank {rank}")
        all_results.extend(results)

    for p in processes:
        p.join()

    print(f"Total time: {time.time() - start_time:.2f}s")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "summaries_with_similarity.jsonl")
    print(f"Saving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats, failed_items, non_product_items = analyze_statistics(
        all_results, similarities_dict
    )

    stats_file = os.path.join(args.output_dir, "statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_file}")

    # Save failed items (items that did not produce exactly 5 words)
    # Includes both failed-parse items and non-product items
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

    # Show examples
    print("\nExamples (valid products):")
    shown = 0
    for item in all_results:
        if shown >= 3:
            break
        words = item.get("summary_words", [])
        if len([w for w in words if w]) == 5:
            print(f"  ID={item['id']}, Title={item.get('title', 'N/A')}")
            print(f"  Summary: {words}")
            shown += 1

    print(f"\nCompleted! Total: {len(all_results)}, "
          f"Products: {stats['valid_product_count']}, "
          f"Non-products: {stats['non_product_count']}, "
          f"Failed: {stats['failed_count']}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
