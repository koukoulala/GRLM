"""Step 1: Generate Product Summaries using LLM

Uses a causal LLM (e.g., Qwen3-4B) to generate 5-word summaries for each
product, leveraging similar items context from step 0.

Usage:
    python s1_init_sum.py \
        --item_file ./raw_data/item.json \
        --similarity_file ./processed/sum_data/similarities.json \
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
    """Prepare prompt including information from the 5 most similar items."""
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

    similar_items_text = "\n".join(similar_items_info)

    prompt = f"""You are an expert product summarizer. Your task is to generate exactly FIVE words to summarize this product. Please follow ALL guidelines carefully:

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
6. OUTPUT FORMAT: Provide ONLY the five words in this exact format: [word1, word2, word3, word4, word5]
7. NO ADDITIONAL TEXT: Do not include any explanations, thoughts, or other content.

PRODUCT INFORMATION:
{title_text}
{description_text}

TOP 5 SIMILAR PRODUCTS (for reference):
{similar_items_text}

ANALYSIS GUIDANCE:
1. First, identify what this product has in common with similar products (shared category, features, audience)
2. Then, identify what makes this product unique or different
3. Use consistent vocabulary for shared characteristics
4. Include distinctive vocabulary for unique aspects
5. Ensure words cover the five required aspects in order

Please provide exactly five words in this exact format: [word1, word2, word3, word4, word5]:"""

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
    """Parse 5-word summary from LLM output."""
    if not content:
        return [""] * 5

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
                inner = line[1:-1]
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

    words = words[:5]
    while len(words) < 5:
        words.append("")

    return words


def analyze_statistics(all_items, similarities_dict):
    """Analyze summary statistics."""
    print("\n" + "=" * 50)
    print("Statistical Analysis Results")
    print("=" * 50)

    all_words = []
    word_freq = Counter()
    word_by_position = [Counter() for _ in range(5)]

    for item in all_items:
        words = item.get("summary_words", [])
        all_words.extend([word for word in words if word])
        for i, word in enumerate(words):
            if i < 5 and word:
                word_by_position[i][word] += 1

    word_freq.update(all_words)

    print(f"\n1. Vocabulary: {len(all_words)} total, {len(word_freq)} unique")
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

    # Conflict analysis
    summary_tuples = [tuple(item.get("summary_words", [])) for item in all_items]
    tuple_counter = Counter(summary_tuples)
    duplicate_tuples = [
        (tup, count) for tup, count in tuple_counter.items() if count > 1
    ]
    total_conflicts = sum(count - 1 for _, count in duplicate_tuples)
    conflict_rate = total_conflicts / len(all_items) if all_items else 0

    print(f"\n3. Conflicts: {len(duplicate_tuples)} duplicates, rate={conflict_rate:.4f}")

    # Validity check
    valid_items = sum(
        1
        for item in all_items
        if len([w for w in item.get("summary_words", []) if w]) == 5
    )
    print(
        f"\n4. Validity: {valid_items}/{len(all_items)} ({valid_items / len(all_items) * 100:.2f}%)"
    )

    return {
        "word_frequency": dict(word_freq.most_common()),
        "position_frequency": [
            dict(counter.most_common()) for counter in word_by_position
        ],
        "conflict_rate": conflict_rate,
        "valid_items": valid_items,
        "total_items": len(all_items),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 5-word product summaries using LLM"
    )
    parser.add_argument(
        "--item_file",
        type=str,
        required=True,
        help="Path to item metadata JSON file",
    )
    parser.add_argument(
        "--similarity_file",
        type=str,
        required=True,
        help="Path to similarities JSON from step 0",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
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

    stats = analyze_statistics(all_results, similarities_dict)

    stats_file = os.path.join(args.output_dir, "statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_file}")

    # Show examples
    for item in all_results[:3]:
        print(f"\n  ID={item['id']}, Title={item.get('title', 'N/A')}")
        print(f"  Summary: {item.get('summary_words', [])}")

    print(f"\nCompleted! Total items: {len(all_results)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
