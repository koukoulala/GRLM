"""Step 1: Generate Product Summaries & Build ID-to-Metadata Mapping

Supports two inference backends:
1. GitHub Copilot API
2. Papyrus API

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

Usage (Copilot API):
    python s1_generate_tid.py \\
        --item_file ./raw_data/merged_clean_item.json \\
        --similarity_file ./processed/similarities.json \\
        --output_dir ./processed/ \\
        --token_file ./resources/tokens.txt \\
        --copilot_model gpt-5.4 \\
        --copilot_workers 20

Usage (Papyrus API):
    python s1_generate_tid.py \\
        --item_file ./raw_data/merged_clean_item.json \\
        --similarity_file ./processed/similarities.json \\
        --output_dir ./processed/ \\
        --inference_backend papyrus \\
        --papyrus_model gpt-5-chat-shortco-2025-08-07-Bing \\
        --papyrus_workers 40
"""

import os
import json
import re
import random
import argparse
from collections import Counter
import time
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import partial

from tqdm import tqdm

SEED = 42
random.seed(SEED)

# Add resources directory to path for llm_utils import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "resources")
sys.path.insert(0, RESOURCES_DIR)
from llm_utils import (run_llm_parallel,
                      run_llm_parallel_with_checkpoint,
                      load_checkpoint as _load_checkpoint_raw,
                      save_checkpoint as _save_checkpoint_raw,
                      cleanup_checkpoint)
# Lazy import: Infer_by_papyrus requires httpx which may not be installed.
# Only imported when papyrus backend is actually used.
from term_normalizer import normalize_term


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
        if len(description) > 500:
            description = description[:500] + "..."
        info_lines.append(f"Description: {description}")

    categories = item.get("categories", "")
    if categories:
        info_lines.append(f"Categories: {categories}")

    # Append structured attributes
    attributes = item.get("attributes", {})
    brand = attributes.get("Brand", "")
    if isinstance(brand, str):
        brand = " ".join(brand.split())
        if brand:
            info_lines.append(f"Brand: {brand}")
    seller = attributes.get("Seller", "")
    if isinstance(seller, str):
        seller = " ".join(seller.split())
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
# Seller Validation
# =============================================================================

def _normalize_for_comparison(text):
    """Normalize text for seller comparison: normalize_term + lowercase."""
    if not text:
        return ""
    return normalize_term(text).lower().strip()


def validate_seller_in_terms(item, summary_words):
    """Check whether the seller appears in the last 2 terms of summary_words.

    Args:
        item: Item dict with product metadata (has 'attributes.Seller').
        summary_words: List of 7 summary word strings.

    Returns:
        (is_valid, seller_raw):
            is_valid = True if seller is empty/missing OR seller matches
                       one of the last 2 terms (after normalization).
            seller_raw = the raw seller string (for logging).
    """
    attributes = item.get("attributes", {})
    seller = attributes.get("Seller", "")
    if isinstance(seller, str):
        seller = " ".join(seller.split())
    if not seller:
        return True, ""

    seller_norm = _normalize_for_comparison(seller)
    if not seller_norm:
        return True, seller

    # Check last 2 terms
    if len(summary_words) < 7:
        return False, seller

    for term in summary_words[-2:]:
        term_norm = _normalize_for_comparison(term)
        if term_norm == seller_norm:
            return True, seller

    return False, seller


def is_result_valid(item, result, check_seller=True):
    """Check whether a result is valid (7 non-empty words + seller check).

    Args:
        item: Item dict with product metadata.
        result: Result dict with 'summary_words'.
        check_seller: Whether to also validate seller presence.

    Returns:
        (is_valid, reason):
            is_valid: True if result passes all checks.
            reason: None if valid, else 'not_7_words', 'non_product',
                    or 'seller_mismatch'.
    """
    words = result.get("summary_words", [])
    non_empty = [w for w in words if w]

    if len(words) == 0:
        return False, "non_product"
    if len(non_empty) != 7:
        return False, "not_7_words"

    if check_seller:
        seller_ok, _ = validate_seller_in_terms(item, words)
        if not seller_ok:
            return False, "seller_mismatch"

    return True, None


# =============================================================================
# Statistics
# =============================================================================

def analyze_statistics(all_items, check_seller=True):
    """Analyze summary statistics, separating products from non-products."""
    print("\n" + "=" * 50)
    print("Statistical Analysis Results")
    print("=" * 50)

    product_items = []
    non_product_items = []
    failed_items = []
    seller_mismatch_items = []

    for item in all_items:
        words = item.get("summary_words", [])
        non_empty_words = [w for w in words if w]
        if len(words) == 0:
            non_product_items.append(item)
        elif len(non_empty_words) == 7:
            if check_seller:
                seller_ok, seller_raw = validate_seller_in_terms(item, words)
                if not seller_ok:
                    seller_mismatch_items.append(item)
                    continue
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
    if check_seller:
        print(f"   Seller mismatch:          {len(seller_mismatch_items):>10,} "
              f"({len(seller_mismatch_items) / total * 100:.2f}%)")

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

    if seller_mismatch_items:
        print(f"\n5. Seller mismatch examples (first 5):")
        for item in seller_mismatch_items[:5]:
            _, seller_raw = validate_seller_in_terms(
                item, item.get("summary_words", []))
            print(f"     ID={item['id']}, Seller={seller_raw}, "
                  f"Terms={item.get('summary_words', [])[-2:]}")

    return {
        "word_frequency": dict(word_freq.most_common()),
        "position_frequency": [
            dict(counter.most_common()) for counter in word_by_position
        ],
        "conflict_rate": conflict_rate,
        "valid_product_count": len(product_items),
        "non_product_count": len(non_product_items),
        "failed_count": len(failed_items),
        "seller_mismatch_count": len(seller_mismatch_items),
        "total_items": total,
    }, failed_items, non_product_items, seller_mismatch_items


# =============================================================================
# Checkpoint Helpers
# =============================================================================

def _load_jsonl_checkpoint(checkpoint_dir):
    """Load checkpoint results stored as JSONL (full item dicts keyed by item id).

    Loads records that have 'llm_output' and 'id' keys.
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
                        if 'llm_output' in result and 'id' in result:
                            completed[result['id']] = result
    if completed:
        print(f"  [CHECKPOINT] Loaded {len(completed)} completed items "
              f"from {checkpoint_dir}")
    return completed


def determine_backend(args):
    """Determine which inference backend to use.

    Returns:
        str: One of 'copilot', 'papyrus'.
    """
    return args.inference_backend


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 7-word product summaries using Copilot or Papyrus API"
    )
    parser.add_argument(
        "--item_file",
        type=str,
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/item.json",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/item.json",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_PG/item.json",
        help="Path to item metadata JSON file",
    )
    parser.add_argument(
        "--similarity_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/MatadorEmb_Index/similarities.json",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_PG/MatadorEmb_Index/similarities.json",
        help="Path to similarities JSON from step 0",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/processed_IDB_v4/",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/processed_PG/",
        help="Directory to save summaries and statistics",
    )
    parser.add_argument(
        "--inference_backend",
        type=str,
        default="copilot",
        choices=["copilot", "papyrus"],
        help="Inference backend: 'copilot' or 'papyrus'.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=os.path.join(SCRIPT_DIR, "prompts", "term_generationV4.md"),
        help="Path to prompt template file (.md/.txt) with {placeholders}",
    )
    # --- Copilot API-specific args ---
    parser.add_argument(
        "--token_file",
        type=str,
        default="./resources/tokens_full.txt",
        help="Path to tokens.txt file for Copilot API authentication",
    )
    parser.add_argument(
        "--copilot_model",
        type=str,
        default="gpt-5.4",
        #default="gemini-3-flash-preview",
        help="Copilot model name",
    )
    parser.add_argument(
        "--copilot_workers",
        type=int,
        default=40,
        help="Number of parallel workers for Copilot API calls",
    )
    parser.add_argument(
        "--copilot_chunk_size",
        type=int,
        default=10000,
        help="Number of items per Copilot processing chunk for checkpoint "
             "saving",
    )
    # --- Papyrus API-specific args ---
    parser.add_argument(
        "--papyrus_endpoint",
        type=str,
        default="https://westus2batch.papyrus.binginternal.com",
        help="Papyrus API base endpoint (without /chat/completions)",
    )
    parser.add_argument(
        "--papyrus_model",
        type=str,
        #default="gpt-54-2026-03-05-Eval",
        default="gpt-5-chat-shortco-2025-08-07-Bing",
        help="Papyrus model name",
    )
    parser.add_argument(
        "--papyrus_quota_id",
        type=str,
        default="",
        help="Papyrus quota ID (default: empty)",
    )
    parser.add_argument(
        "--papyrus_timeout_ms",
        type=int,
        default=120000,
        help="Papyrus request timeout in ms (default: 120000)",
    )
    parser.add_argument(
        "--papyrus_workers",
        type=int,
        default=40,
        help="Number of parallel async workers for Papyrus API (default: 5)",
    )
    parser.add_argument(
        "--papyrus_chunk_size",
        type=int,
        default=40000,
        help="Number of items per Papyrus processing chunk for checkpoint "
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
        "--check_seller",
        action="store_true",
        dest="check_seller",
        default=True,
        help="Validate that seller appears in the last 2 terms of the "
             "summary (default: enabled).",
    )
    parser.add_argument(
        "--no-check_seller",
        action="store_false",
        dest="check_seller",
        help="Disable seller validation.",
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
        default=2000,
        help="Number of items to sample in debug mode",
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
             "split into multiple files by --prompts_chunk_size, "
             "save to <output_dir>/prompts/<stem>_prompts_1.tsv, etc., "
             "then exit without running inference.",
    )
    parser.add_argument(
        "--prompts_chunk_size",
        type=int,
        default=900000,
        help="Number of prompts per file when using --export_prompts_only "
             "(default: 900000)",
    )
    parser.add_argument(
        "--prompts_input_file",
        type=str,
        default=None,
        help="Path to a pre-built prompts TSV file (GlobalOfferId<tab>Prompt). "
             "If provided, runs LLM inference and saves a simple 2-column "
             "results TSV (GlobalOfferId<tab>Output) as <prefix>_results.tsv "
             "in the same directory, then exits.",
    )
    parser.add_argument(
        "--prompt_results_dir",
        type=str,
        default=None,
        help="Directory containing *_results.tsv files from --prompts_input_file "
             "runs. If set to a valid directory, merges all results files, "
             "loads item metadata from --item_file, and produces final output "
             "files (summaries, id2meta, statistics, etc.) to --output_dir.",
    )
    parser.add_argument(
        "--retry_invalid",
        action="store_true",
        default=False,
        help="When used with --prompt_results_dir, re-run inference on items "
             "that have no results (missing), failed parsing, or seller "
             "mismatch. The new results are merged back before saving. "
             "Without this flag, merge mode just saves whatever it has.",
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
        "--skip_presave",
        action="store_true",
        default=False,
        help="Skip the PRE-SAVE step that writes summaries/id2meta/statistics "
             "before inference or prompt export. Saves ~30GB RAM by not "
             "keeping previous_results alive during prompt building. "
             "Use when running --export_prompts_only on low-memory machines.",
    )
    parser.add_argument(
        "--resume_from_multi_path",
        type=str,
        nargs="*",
        default=[],
        #default=["/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/processed_v2/summaries_with_similarity.jsonl", "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/processed/summaries_with_similarity.jsonl"],
        #default=["/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/processed/summaries_with_similarity.jsonl","/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/processed/s1_split_1/summaries_with_similarity.jsonl","/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/processed/s1_split_2/summaries_with_similarity.jsonl","/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/processed/s1_split_3/summaries_with_similarity.jsonl"],
        help="One or more paths to .jsonl or .json files from previous runs "
             "to resume from. Supports both JSONL (one record per line) and "
             "JSON dict format (e.g., id2meta.json: {id: item_dict}). "
             "Results are merged (later files overwrite earlier ones for "
             "duplicate IDs). Set to empty to skip resume.",
    )
    parser.add_argument(
        "--checkpoint_dirs",
        type=str,
        nargs="*",
        #default=[],
        default=["/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/processed_IDB_v4/prompts/_item_prompts_16_checkpoint/",],
        help="Additional checkpoint directories to load results from. "
             "Each should be a _s1_checkpoint folder. Results are merged "
             "with resume files and the default checkpoint_dir.",
    )
    parser.add_argument(
        "--filter_items_file",
        type=str,
        default=None,
        help="Path to a text file with one OfferId per line. "
             "If provided, only process items whose ID appears in this file. "
             "The full item_file and similarity_file are still loaded "
             "(so similar-item context is available), but only filtered "
             "items get inference and output.",
    )
    return parser.parse_args()


# =============================================================================
# Output Saving
# =============================================================================

def save_all_outputs(all_results, output_dir, id2meta_file=None,
                     output_prefix=None, check_seller=True):
    """Save all output files from a list of result dicts.

    Saves: summaries_with_similarity.jsonl, id2meta.json, id2words.tsv,
           statistics.json, failed_items.json.

    Args:
        all_results: List of result dicts (each has 'id', 'summary_words', etc.)
        output_dir: Directory for output files.
        id2meta_file: Optional custom path for id2meta.json.
        output_prefix: Optional prefix for output filenames. If provided,
                       files are named <prefix>_summaries.jsonl, etc.
        check_seller: Whether to validate seller presence in last 2 terms.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- summaries_with_similarity.jsonl ----
    if output_prefix:
        output_file = os.path.join(output_dir, f"{output_prefix}_summaries.jsonl")
    else:
        output_file = os.path.join(output_dir, "summaries_with_similarity.jsonl")
    print(f"\nSaving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- Statistics ----
    stats, failed_items, non_product_items, seller_mismatch_items = \
        analyze_statistics(all_results, check_seller=check_seller)

    if output_prefix:
        stats_file = os.path.join(output_dir, f"{output_prefix}_statistics.json")
    else:
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
    for item in seller_mismatch_items:
        _, seller_raw = validate_seller_in_terms(
            item, item.get("summary_words", []))
        all_problematic.append({
            "id": item["id"],
            "title": item.get("title", ""),
            "reason": "seller_mismatch",
            "seller": seller_raw,
            "summary_words": item.get("summary_words", []),
            "llm_output": item.get("llm_output", ""),
        })

    if output_prefix:
        failed_file = os.path.join(output_dir, f"{output_prefix}_failed_items.json")
    else:
        failed_file = os.path.join(output_dir, "failed_items.json")
    with open(failed_file, "w", encoding="utf-8") as f:
        json.dump(all_problematic, f, ensure_ascii=False, indent=2)
    print(f"Failed/non-product items saved to: {failed_file} "
          f"({len(all_problematic)} items)")

    # ---- Build id2meta mapping (only valid items) ----
    print("\nBuilding id2meta mapping ...")
    # Collect IDs of seller-mismatch items for exclusion
    seller_mismatch_ids = {item["id"] for item in seller_mismatch_items}
    id2meta = {}
    skip_bad_summary = 0
    skip_seller = 0
    for item in all_results:
        words = item.get("summary_words", [])
        non_empty_words = [w for w in words if w]
        if len(non_empty_words) != 7:
            skip_bad_summary += 1
            continue
        item_id = item.get("id")
        if item_id and item_id not in seller_mismatch_ids:
            id2meta[item_id] = item
        elif item_id in seller_mismatch_ids:
            skip_seller += 1
        else:
            skip_bad_summary += 1

    if id2meta_file:
        id2meta_path = id2meta_file
    elif output_prefix:
        id2meta_path = os.path.join(output_dir, f"{output_prefix}_id2meta.json")
    else:
        id2meta_path = os.path.join(output_dir, "id2meta.json")
    with open(id2meta_path, "w", encoding="utf-8") as f:
        json.dump(id2meta, f, ensure_ascii=False, indent=2)
    skipped_total = skip_bad_summary + skip_seller
    print(f"id2meta saved to: {id2meta_path}")
    print(f"  Mapped items: {len(id2meta):,} "
          f"(skipped {skipped_total:,}: "
          f"{skip_bad_summary:,} invalid summary, "
          f"{skip_seller:,} seller mismatch)")

    # ---- Build id2meta_with_norm (parallel normalization) ----
    print("\nBuilding id2meta_with_norm (normalizing summary_words) ...")
    # Only pass (item_id, words) to workers — NOT full item dicts —
    # to avoid pickle/memory overhead that causes OOM with large datasets.
    items_words_list = [(item_id, item.get("summary_words", []))
                        for item_id, item in id2meta.items()]
    num_norm_workers = min(8, os.cpu_count() or 4,
                           max(1, len(items_words_list) // 1000))
    norm_map = {}  # item_id -> normalized_words
    if num_norm_workers > 1 and len(items_words_list) >= 1000:
        # Use smaller batches (50k each) to reduce per-process memory
        norm_batch_size = min(50000,
                              max(1, len(items_words_list) // num_norm_workers))
        norm_batches = [
            items_words_list[i:i + norm_batch_size]
            for i in range(0, len(items_words_list), norm_batch_size)
        ]
        print(f"  Normalizing {len(items_words_list):,} items with "
              f"{num_norm_workers} workers ({len(norm_batches)} batches) ...")
        try:
            with ProcessPoolExecutor(max_workers=num_norm_workers) as executor:
                for batch_result in executor.map(
                        _normalize_words_batch, norm_batches):
                    for item_id, norm_words in batch_result:
                        norm_map[item_id] = norm_words
        except (BrokenProcessPool, Exception) as e:
            print(f"  [WARN] Parallel normalization failed ({e}), "
                  f"falling back to sequential ...")
            norm_map = {}
            for item_id, words in items_words_list:
                norm_map[item_id] = [normalize_term(w) for w in words]
    else:
        for item_id, words in items_words_list:
            norm_map[item_id] = [normalize_term(w) for w in words]
    # Reconstruct id2meta_norm from id2meta + normalized words
    id2meta_norm = {}
    for item_id, item in id2meta.items():
        item_copy = dict(item)
        item_copy["summary_words_norm"] = norm_map.get(item_id, [])
        id2meta_norm[item_id] = item_copy
    del norm_map

    if output_prefix:
        norm_path = os.path.join(output_dir,
                                 f"{output_prefix}_id2meta_with_norm.json")
    else:
        norm_path = os.path.join(output_dir, "id2meta_with_norm.json")
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(id2meta_norm, f, ensure_ascii=False, indent=2)
    print(f"id2meta_with_norm saved to: {norm_path} ({len(id2meta_norm):,} items)")

    # ---- Build id2words mapping ----
    if output_prefix:
        id2words_file = os.path.join(output_dir, f"{output_prefix}_id2words.tsv")
    else:
        id2words_file = os.path.join(output_dir, "id2words.tsv")
    with open(id2words_file, "w", encoding="utf-8") as f:
        for item_id, meta in id2meta.items():
            f.write(json.dumps({item_id: meta["summary_words"]},
                               ensure_ascii=False) + "\n")
    print(f"id2words saved to: {id2words_file} ({len(id2meta):,} items)")

    return stats


# =============================================================================
# Parallel Helpers (module-level for pickling with ProcessPoolExecutor)
# =============================================================================

def _build_result_batch(batch):
    """Process a batch of (idx, item, gen_text) tuples.
    Returns list of (idx, item_id, result_dict)."""
    results = []
    for idx, item, gen_text in batch:
        words = parse_summary_words(gen_text)
        item_id = item["id"]
        result = item.copy()
        result["llm_output"] = gen_text
        result["summary_words"] = words
        results.append((idx, item_id, result))
    return results


def _normalize_words_batch(batch):
    """Normalize summary_words for a batch of (item_id, words_list) pairs.
    Returns list of (item_id, normalized_words_list)."""
    return [(item_id, [normalize_term(w) for w in words])
            for item_id, words in batch]


def _parallel_build_results(items_with_text, similarities_dict):
    """Build result dicts with parallel parse_summary_words.

    Args:
        items_with_text: List of (idx, item_dict, gen_text) tuples.
        similarities_dict: Dict mapping item_id -> list of similar items.

    Returns:
        List of result dicts in original order.
    """
    if not items_with_text:
        return []

    num_workers = min(os.cpu_count() or 4,
                      max(1, len(items_with_text) // 500))
    batch_size = max(1, len(items_with_text) // max(1, num_workers))
    batches = [
        items_with_text[i:i + batch_size]
        for i in range(0, len(items_with_text), batch_size)
    ]

    results_indexed = {}
    if num_workers > 1 and len(items_with_text) >= 500:
        print(f"  Building results: {len(items_with_text):,} items, "
              f"{num_workers} workers, {len(batches)} batches ...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for batch_result in executor.map(_build_result_batch, batches):
                for idx, item_id, result in batch_result:
                    result["similar_item_ids"] = [
                        sim["item_id"]
                        for sim in similarities_dict.get(item_id, [])[:5]
                    ]
                    results_indexed[idx] = result
    else:
        for idx, item, gen_text in items_with_text:
            words = parse_summary_words(gen_text)
            item_id = item["id"]
            result = item.copy()
            result["llm_output"] = gen_text
            result["summary_words"] = words
            result["similar_item_ids"] = [
                sim["item_id"]
                for sim in similarities_dict.get(item_id, [])[:5]
            ]
            results_indexed[idx] = result

    return [results_indexed[idx] for idx in sorted(results_indexed.keys())]


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    backend = determine_backend(args)

    if backend == "copilot":
        print("=" * 60)
        print("Inference mode: GitHub Copilot API")
        print(f"  Model: {args.copilot_model}")
        print(f"  Workers: {args.copilot_workers}")
        print(f"  Token file: {args.token_file}")
        print("=" * 60)
    elif backend == "papyrus":
        print("=" * 60)
        print("Inference mode: Papyrus API")
        print(f"  Endpoint: {args.papyrus_endpoint}")
        print(f"  Model: {args.papyrus_model}")
        print(f"  Workers: {args.papyrus_workers}")
        print(f"  Quota ID: {args.papyrus_quota_id or '(default)'}")
        print("=" * 60)

    debug = args.debug
    if debug:
        print("\n*** DEBUG MODE: processing only sampled items, NO files written ***\n")

    # ---- Merge prompt results mode (--prompt_results_dir) ----
    if args.prompt_results_dir and os.path.isdir(args.prompt_results_dir):
        print("=" * 60)
        print("Mode: Merge prompt results from directory")
        print(f"  Results dir: {args.prompt_results_dir}")
        print(f"  Output dir:  {args.output_dir}")
        print("=" * 60)

        # Load item metadata
        print(f"\nLoading data: {args.item_file}")
        full_data = load_data(args.item_file)
        print(f"Loaded {len(full_data)} items")
        all_items_dict = {item["id"]: item for item in full_data}

        print(f"Loading similarities: {args.similarity_file}")
        similarities_dict = load_similarities(args.similarity_file)
        print(f"Loaded similarities for {len(similarities_dict)} items")

        # Find and merge all *_results.tsv files (PARALLEL)
        import csv as csv_mod
        results_files = sorted([
            os.path.join(args.prompt_results_dir, f)
            for f in os.listdir(args.prompt_results_dir)
            if f.endswith("_results.tsv") and os.path.isfile(
                os.path.join(args.prompt_results_dir, f))
        ])
        print(f"\nFound {len(results_files)} results files, reading in parallel ...")

        # Column name candidates for item ID and LLM output
        id_col_names = {"GlobalOfferId", "globalofferid", "item_id", "ItemId", "ID", "id"}
        output_col_names = {"Output", "output", "OUTPUT", "LLMOutput", "llm_output"}

        def _read_one_results_tsv(rf):
            """Read a single results TSV file. Returns (basename, dict, count, msg)."""
            local_dict = {}
            count = 0
            basename = os.path.basename(rf)
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    reader = csv_mod.reader(f, delimiter="\t")
                    header = next(reader, None)
                    if header is None:
                        return (basename, local_dict, 0, "SKIP (empty)")
                    id_idx = None
                    output_idx = None
                    for i, col in enumerate(header):
                        col_stripped = col.strip()
                        if col_stripped in id_col_names:
                            id_idx = i
                        elif col_stripped in output_col_names:
                            output_idx = i
                    if id_idx is None or output_idx is None:
                        msg = (f"SKIP (missing columns: "
                               f"id={'FOUND' if id_idx is not None else 'MISSING'}, "
                               f"output={'FOUND' if output_idx is not None else 'MISSING'}) "
                               f"header={header}")
                        return (basename, local_dict, 0, msg)
                    min_cols = max(id_idx, output_idx) + 1
                    for row in reader:
                        if len(row) >= min_cols:
                            item_id = row[id_idx].strip()
                            gen_text = row[output_idx].replace("\\n", "\n")
                            local_dict[item_id] = gen_text
                            count += 1
                    msg = (f"{count:,} items "
                           f"(id_col={header[id_idx].strip()}, "
                           f"output_col={header[output_idx].strip()})")
                    return (basename, local_dict, count, msg)
            except Exception as e:
                return (basename, local_dict, 0, f"ERROR: {e}")

        merged_outputs = {}  # item_id -> llm_output
        read_start = time.time()
        num_read_workers = min(32, len(results_files))
        with ThreadPoolExecutor(max_workers=num_read_workers) as executor:
            futures = {executor.submit(_read_one_results_tsv, rf): rf
                       for rf in results_files}
            for future in as_completed(futures):
                basename, local_dict, count, msg = future.result()
                merged_outputs.update(local_dict)
                print(f"  {basename}: {msg}")
        read_elapsed = time.time() - read_start
        print(f"Total merged: {len(merged_outputs):,} unique items "
              f"(read {len(results_files)} files in {read_elapsed:.1f}s "
              f"with {num_read_workers} workers)")

        # Also load resume files if specified (PARALLEL)
        previous_results = {}
        resume_paths = args.resume_from_multi_path or []
        valid_resume_files = [p for p in resume_paths if p and os.path.exists(p)]

        def _load_one_resume_file(prev_file):
            """Load a single resume file. Returns (path, dict, count)."""
            local_results = {}
            is_json_dict = (prev_file.endswith('.json')
                            and not prev_file.endswith('.jsonl'))
            if is_json_dict:
                with open(prev_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                for rid, record in json_data.items():
                    if "id" not in record:
                        record["id"] = rid
                    local_results[rid] = record
            else:
                with open(prev_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            rid = record.get("id")
                            if rid:
                                local_results[rid] = record
                        except json.JSONDecodeError:
                            continue
            return (prev_file, local_results, len(local_results))

        if valid_resume_files:
            resume_start = time.time()
            with ThreadPoolExecutor(max_workers=len(valid_resume_files)) as executor:
                resume_futures = [executor.submit(_load_one_resume_file, pf)
                                  for pf in valid_resume_files]
                for future in as_completed(resume_futures):
                    pf, local_res, cnt = future.result()
                    previous_results.update(local_res)
                    print(f"  Loaded resume file: {pf} ({cnt:,} items)")
            resume_elapsed = time.time() - resume_start
            print(f"  Loaded {len(previous_results):,} items from "
                  f"{len(valid_resume_files)} resume files in {resume_elapsed:.1f}s")

        # Also load checkpoint directories if specified
        extra_ckpt_dirs = args.checkpoint_dirs or []
        for ckpt_dir_path in extra_ckpt_dirs:
            if not ckpt_dir_path or not os.path.exists(ckpt_dir_path):
                continue
            # Try JSONL checkpoint format
            ckpt_loaded = _load_jsonl_checkpoint(ckpt_dir_path)
            if ckpt_loaded:
                new_from_ckpt = 0
                for cid, cresult in ckpt_loaded.items():
                    if cid not in previous_results and cid not in merged_outputs:
                        previous_results[cid] = cresult
                        new_from_ckpt += 1
                if new_from_ckpt > 0:
                    print(f"  [CHECKPOINT-JSONL] Loaded {new_from_ckpt:,} items "
                          f"from {ckpt_dir_path}")
            # Try Copilot checkpoint format
            copilot_ckpt = _load_checkpoint_raw(ckpt_dir_path)
            if copilot_ckpt:
                new_from_copilot = 0
                for cid, response_text in copilot_ckpt.items():
                    if cid not in previous_results and cid not in merged_outputs:
                        if cid in all_items_dict:
                            item = all_items_dict[cid]
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
                            new_from_copilot += 1
                if new_from_copilot > 0:
                    print(f"  [CHECKPOINT-Copilot] Loaded {new_from_copilot:,} items "
                          f"from {ckpt_dir_path}")
        if extra_ckpt_dirs:
            print(f"  Total from resume + checkpoints: {len(previous_results):,} items")

        # Build full result dicts (PARALLEL parse_summary_words)
        build_start = time.time()
        # Separate items into three buckets
        items_to_parse = []   # (index, item, gen_text) - need parse
        items_from_resume = []  # (index, item_id) - from resume
        for idx, item in enumerate(full_data):
            item_id = item["id"]
            if item_id in merged_outputs:
                items_to_parse.append((idx, item, merged_outputs[item_id]))
            elif item_id in previous_results:
                items_from_resume.append((idx, item_id))

        # Split items_to_parse into batches for parallel processing
        parse_batch_size = max(1, len(items_to_parse) // (os.cpu_count() or 4))
        parse_batches = [
            items_to_parse[i:i + parse_batch_size]
            for i in range(0, len(items_to_parse), parse_batch_size)
        ]

        all_results_indexed = {}  # idx -> result
        matched = 0
        from_resume = 0

        if parse_batches:
            num_parse_workers = min(os.cpu_count() or 4, len(parse_batches))
            print(f"\nParsing {len(items_to_parse):,} items with "
                  f"{num_parse_workers} workers ({len(parse_batches)} batches) ...")
            with ProcessPoolExecutor(max_workers=num_parse_workers) as executor:
                parse_futures = [executor.submit(_build_result_batch, batch)
                                 for batch in parse_batches]
                for future in as_completed(parse_futures):
                    for idx, item_id, result in future.result():
                        # Add similar_item_ids
                        result["similar_item_ids"] = [
                            sim["item_id"]
                            for sim in similarities_dict.get(item_id, [])[:5]
                        ]
                        all_results_indexed[idx] = result
                        matched += 1

        # Add resume items
        for idx, item_id in items_from_resume:
            all_results_indexed[idx] = previous_results[item_id]
            from_resume += 1

        # Rebuild ordered list
        all_results = [all_results_indexed[idx]
                       for idx in sorted(all_results_indexed.keys())]
        build_elapsed = time.time() - build_start

        print(f"\nBuilt {len(all_results):,} results in {build_elapsed:.1f}s "
              f"(from results dir: {matched:,}, from resume: {from_resume:,}, "
              f"missing: {len(full_data) - matched - from_resume:,})")

        # ---- Retry invalid items (--retry_invalid) ----
        if args.retry_invalid:
            # Load prompt template
            print(f"\n[RETRY] Loading prompt template from: {args.prompt_file}")
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # Identify items to retry: missing + failed + seller_mismatch
            results_by_id = {r["id"]: r for r in all_results}
            retry_items = []
            missing_count = 0
            failed_count = 0
            seller_mismatch_retry = 0
            for item in full_data:
                item_id = item["id"]
                if item_id not in results_by_id:
                    retry_items.append(item)
                    missing_count += 1
                else:
                    result = results_by_id[item_id]
                    valid, reason = is_result_valid(
                        item, result, check_seller=args.check_seller)
                    if not valid and reason != "non_product":
                        retry_items.append(item)
                        if reason == "seller_mismatch":
                            seller_mismatch_retry += 1
                        else:
                            failed_count += 1

            print(f"[RETRY] Items to retry: {len(retry_items):,} "
                  f"(missing: {missing_count:,}, failed: {failed_count:,}, "
                  f"seller_mismatch: {seller_mismatch_retry:,})")

            if retry_items:
                # Build prompts
                print(f"[RETRY] Building prompts for {len(retry_items):,} items ...")
                retry_inputs = []
                for item in retry_items:
                    item_id = item["id"]
                    top_similar_items = similarities_dict.get(item_id, [])[:5]
                    raw_prompt = prepare_prompt(
                        item, top_similar_items, all_items_dict, prompt_template
                    )
                    retry_inputs.append((item_id, raw_prompt))

                # Run inference
                retry_checkpoint_dir = os.path.join(
                    args.output_dir, "_retry_checkpoint")
                retry_start = time.time()

                if backend == "papyrus":
                    from Infer_by_papyrus import run_papyrus_parallel_with_checkpoint
                    retry_api_results = run_papyrus_parallel_with_checkpoint(
                        inputs=retry_inputs,
                        checkpoint_dir=retry_checkpoint_dir,
                        papyrus_endpoint=args.papyrus_endpoint,
                        model_name=args.papyrus_model,
                        quota_id=args.papyrus_quota_id,
                        timeout_ms=args.papyrus_timeout_ms,
                        num_workers=args.papyrus_workers,
                        max_tokens=args.max_tokens,
                        max_retries=3,
                        chunk_size=args.papyrus_chunk_size,
                    )
                else:  # copilot
                    retry_api_results = run_llm_parallel_with_checkpoint(
                        inputs=retry_inputs,
                        token_file=args.token_file,
                        checkpoint_dir=retry_checkpoint_dir,
                        num_workers=args.copilot_workers,
                        model=args.copilot_model,
                        temperature=0,
                        max_tokens=args.max_tokens,
                        chunk_size=args.copilot_chunk_size,
                    )

                retry_elapsed = time.time() - retry_start
                retry_map = {item_id: gen_text
                             for item_id, gen_text in retry_api_results}
                print(f"[RETRY] Inference done: {len(retry_map):,} results "
                      f"in {retry_elapsed:.1f}s")

                # Build result dicts for retried items
                retry_items_with_text = [
                    (i, item, retry_map.get(item["id"], ""))
                    for i, item in enumerate(retry_items)
                ]
                retry_results = _parallel_build_results(
                    retry_items_with_text, similarities_dict)

                # Merge retry results into all_results
                retry_results_by_id = {r["id"]: r for r in retry_results}
                updated = 0
                added = 0
                for i, result in enumerate(all_results):
                    if result["id"] in retry_results_by_id:
                        new_result = retry_results_by_id.pop(result["id"])
                        all_results[i] = new_result
                        updated += 1
                # Append items that were missing entirely (not in all_results)
                for item_id, new_result in retry_results_by_id.items():
                    all_results.append(new_result)
                    added += 1

                print(f"[RETRY] Merged: {updated:,} updated, "
                      f"{added:,} newly added")
                cleanup_checkpoint(retry_checkpoint_dir)
            else:
                print("[RETRY] No items to retry, skipping inference.")

        # Save all output files
        stats = save_all_outputs(all_results, args.output_dir,
                                 id2meta_file=args.id2meta_file,
                                 check_seller=args.check_seller)
        print(f"\nCompleted! Total: {len(all_results)}, "
              f"Products: {stats['valid_product_count']}, "
              f"Non-products: {stats['non_product_count']}, "
              f"Failed: {stats['failed_count']}, "
              f"Seller mismatch: {stats['seller_mismatch_count']}")
        return

    # ---- Prompts input file mode (--prompts_input_file) ----
    # Early exit: load prompts, run LLM, save 2-column _results.tsv, done.
    # No need to load item.json, similarities, resume, etc.
    if args.prompts_input_file:
        prompts_input_file = args.prompts_input_file
        print(f"\n[PROMPTS-INPUT] Loading prompts from: {prompts_input_file}")
        if not os.path.exists(prompts_input_file):
            raise FileNotFoundError(
                f"Prompts input file not found: {prompts_input_file}"
            )
        import csv as csv_mod
        prompts_from_file = []
        with open(prompts_input_file, "r", encoding="utf-8") as f:
            reader = csv_mod.reader(f, delimiter="\t")
            header = next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    item_id = row[0].strip()
                    prompt_text = row[1].replace("\\n", "\n")
                    prompts_from_file.append((item_id, prompt_text))
        print(f"  Loaded {len(prompts_from_file):,} prompts")

        # Derive output paths
        output_prefix = os.path.splitext(
            os.path.basename(prompts_input_file)
        )[0]
        results_output_dir = os.path.dirname(os.path.abspath(prompts_input_file))
        results_output_file = os.path.join(
            results_output_dir, f"{output_prefix}_results.tsv"
        )
        checkpoint_dir = os.path.join(results_output_dir, f"_{output_prefix}_checkpoint")
        print(f"  Output prefix: {output_prefix}")
        print(f"  Results file:  {results_output_file}")

        # Run LLM inference
        print(f"\nRunning {backend} inference on {len(prompts_from_file):,} "
              f"pre-built prompts ...")
        start_time = time.time()

        if backend == "papyrus":
            from Infer_by_papyrus import (run_papyrus_parallel,
                                          run_papyrus_parallel_with_checkpoint)
            if debug:
                api_results = run_papyrus_parallel(
                    inputs=prompts_from_file,
                    papyrus_endpoint=args.papyrus_endpoint,
                    model_name=args.papyrus_model,
                    quota_id=args.papyrus_quota_id,
                    timeout_ms=args.papyrus_timeout_ms,
                    num_workers=args.papyrus_workers,
                    max_tokens=args.max_tokens,
                    max_retries=3,
                )
            else:
                api_results = run_papyrus_parallel_with_checkpoint(
                    inputs=prompts_from_file,
                    checkpoint_dir=checkpoint_dir,
                    papyrus_endpoint=args.papyrus_endpoint,
                    model_name=args.papyrus_model,
                    quota_id=args.papyrus_quota_id,
                    timeout_ms=args.papyrus_timeout_ms,
                    num_workers=args.papyrus_workers,
                    max_tokens=args.max_tokens,
                    max_retries=3,
                    chunk_size=args.papyrus_chunk_size,
                )
        else:  # copilot
            api_results = run_llm_parallel_with_checkpoint(
                inputs=prompts_from_file,
                token_file=args.token_file,
                checkpoint_dir=checkpoint_dir,
                num_workers=args.copilot_workers,
                model=args.copilot_model,
                temperature=0,
                max_tokens=args.max_tokens,
                chunk_size=args.copilot_chunk_size,
            )

        # Save simple 2-column results TSV and exit
        result_map = {item_id: gen_text for item_id, gen_text in api_results}
        inference_time = time.time() - start_time
        print(f"\nInference time: {inference_time:.1f}s")

        with open(results_output_file, "w", encoding="utf-8") as f:
            f.write("GlobalOfferId\tOutput\n")
            for item_id, _ in prompts_from_file:
                gen_text = result_map.get(item_id, "")
                safe_text = gen_text.replace("\n", "\\n").replace("\t", " ")
                f.write(f"{item_id}\t{safe_text}\n")
        print(f"Results saved to: {results_output_file} "
              f"({len(prompts_from_file):,} items)")

        cleanup_checkpoint(checkpoint_dir)
        return

    # ---- Load prompt template from file ----
    print(f"\nLoading prompt template from: {args.prompt_file}")
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    print(f"  Prompt template loaded successfully")

    # ---- Load data ----
    print(f"\nLoading data: {args.item_file}")
    full_data = load_data(args.item_file)
    print(f"Loaded {len(full_data)} items")

    # Build full items dict BEFORE trimming, so similarity lookups work
    all_items_dict = {item["id"]: item for item in full_data}

    # ---- Filter to subset if --filter_items_file is provided ----
    if args.filter_items_file and os.path.isfile(args.filter_items_file):
        print(f"\n[FILTER] Loading filter IDs from: {args.filter_items_file}")
        with open(args.filter_items_file, "r") as f:
            filter_ids = {line.strip() for line in f if line.strip()}
        print(f"  Filter IDs loaded: {len(filter_ids):,}")
        full_data = [item for item in full_data if item["id"] in filter_ids]
        print(f"  Items after filter: {len(full_data):,} "
              f"(full item dict kept for similarity lookups)")

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
    _skip_inference = False
    valid_resume_files = []

    if not debug:
        resume_paths = args.resume_from_multi_path or []
        # Filter to existing files (.jsonl or .json)
        valid_resume_files = [p for p in resume_paths if p and os.path.exists(p)]

    if valid_resume_files:
        repaired_count = 0
        for prev_file in valid_resume_files:
            print(f"\n[RESUME] Loading previous results from: {prev_file}")
            file_count = 0

            # Detect format: .json (dict) vs .jsonl (line-by-line)
            is_json_dict = (prev_file.endswith('.json')
                           and not prev_file.endswith('.jsonl'))

            if is_json_dict:
                # JSON dict format (e.g., id2meta.json: {item_id: item_dict})
                with open(prev_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                records_iter = (
                    (rid, result) for rid, result in json_data.items()
                )
            else:
                # JSONL format (one JSON object per line)
                def _iter_jsonl(path):
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                result = json.loads(line)
                                yield result["id"], result
                records_iter = _iter_jsonl(prev_file)

            for rid, result in records_iter:
                if "id" not in result:
                    result["id"] = rid

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

        # Filter out failed items (no valid 7-word summary or seller mismatch)
        # so they get re-processed
        prev_total_raw = len(previous_results)
        failed_ids = set()
        seller_mismatch_count = 0
        not_7_words_count = 0
        for rid, result in previous_results.items():
            if rid in all_items_dict:
                item_for_check = all_items_dict[rid]
            else:
                item_for_check = result
            valid, reason = is_result_valid(
                item_for_check, result, check_seller=args.check_seller)
            if not valid:
                failed_ids.add(rid)
                if reason == "seller_mismatch":
                    seller_mismatch_count += 1
                else:
                    not_7_words_count += 1
        for rid in failed_ids:
            del previous_results[rid]
        print(f"  [RESUME] Filtered out {len(failed_ids):,} invalid items "
              f"for re-processing")
        print(f"    Not valid 7 words: {not_7_words_count:,}")
        if args.check_seller:
            print(f"    Seller mismatch:   {seller_mismatch_count:,}")

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
            data = []  # nothing left to process / export
        else:
            data = remaining_data_for_run
            _skip_inference = False
    else:
        print("  [DEBUG] Skipping resume loading")

    # ---- Load checkpoint results from interrupted previous runs ----
    # Handles both JSONL checkpoint format (full item dicts with 'id' key)
    # and Copilot checkpoint format ({"id": item_id, "result": response_text})
    if not _skip_inference and not debug:
        # Build a quick lookup for item data by id
        data_by_id = {item["id"]: item for item in data}

        # Load from checkpoint directories
        all_ckpt_dirs = [checkpoint_dir]
        extra_ckpt_dirs = args.checkpoint_dirs or []
        for d in extra_ckpt_dirs:
            if d and os.path.exists(d) and d != checkpoint_dir:
                all_ckpt_dirs.append(d)

        for ckpt_dir_path in all_ckpt_dirs:
            ckpt_loaded = _load_jsonl_checkpoint(ckpt_dir_path)
            if ckpt_loaded:
                remaining_ids = {item["id"] for item in data}
                new_from_ckpt = 0
                for cid, cresult in ckpt_loaded.items():
                    if cid in remaining_ids and cid not in previous_results:
                        previous_results[cid] = cresult
                        new_from_ckpt += 1
                if new_from_ckpt > 0:
                    print(f"  [CHECKPOINT-JSONL] Recovered {new_from_ckpt:,} items "
                          f"from {ckpt_dir_path}")

            # Also try Copilot format: {"id": item_id, "result": response_text}
            copilot_ckpt = _load_checkpoint_raw(ckpt_dir_path)
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
                          f"{new_from_copilot_ckpt:,} items "
                          f"from {ckpt_dir_path}")

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
    if previous_results and not debug and not args.skip_presave:
        print(f"\n[PRE-SAVE] Saving {len(previous_results):,} completed "
              f"results and all derived files ...")
        save_all_outputs(
            list(previous_results.values()), args.output_dir,
            id2meta_file=args.id2meta_file,
            check_seller=args.check_seller,
        )
        # Only clean checkpoint if we're about to run inference
        # (not in export_prompts_only or save_intermediate_only mode)
        if not args.export_prompts_only and not args.save_intermediate_only:
            cleanup_checkpoint(checkpoint_dir)
    elif args.skip_presave and previous_results:
        print(f"\n[PRE-SAVE] Skipped (--skip_presave). "
              f"Freeing {len(previous_results):,} previous_results ...")
        del previous_results
        previous_results = {}
        import gc; gc.collect()

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

        chunk_size = args.prompts_chunk_size
        saved_files = []
        total = 0

        # Build all prompts in memory
        all_prompt_rows = []  # list of (item_id, escaped_prompt)
        for item in tqdm(data, desc="Building prompts", mininterval=30):
            item_id = item["id"]
            top_similar_items = similarities_dict.get(item_id, [])[:5]
            raw_prompt = prepare_prompt(
                item, top_similar_items, all_items_dict, prompt_template
            )
            escaped_prompt = raw_prompt.replace("\t", " ").replace("\n", "\\n")
            all_prompt_rows.append((item_id, escaped_prompt))

            # Flush to disk every chunk_size items to bound memory
            if len(all_prompt_rows) >= chunk_size:
                file_num = len(saved_files) + 1
                prompts_file = os.path.join(
                    prompts_dir, f"{item_stem}_prompts_{file_num}.tsv")
                with open(prompts_file, "w", encoding="utf-8") as f:
                    f.write("GlobalOfferId\tPrompt\n")
                    for pid, ep in all_prompt_rows:
                        f.write(f"{pid}\t{ep}\n")
                file_size_mb = os.path.getsize(prompts_file) / (1024 * 1024)
                saved_files.append(prompts_file)
                print(f"  Chunk {file_num}: {prompts_file} "
                      f"({len(all_prompt_rows):,} prompts, {file_size_mb:.1f} MB)")
                total += len(all_prompt_rows)
                all_prompt_rows = []  # free memory

        # Write remaining prompts
        if all_prompt_rows:
            file_num = len(saved_files) + 1
            prompts_file = os.path.join(
                prompts_dir, f"{item_stem}_prompts_{file_num}.tsv")
            with open(prompts_file, "w", encoding="utf-8") as f:
                f.write("GlobalOfferId\tPrompt\n")
                for pid, ep in all_prompt_rows:
                    f.write(f"{pid}\t{ep}\n")
            file_size_mb = os.path.getsize(prompts_file) / (1024 * 1024)
            saved_files.append(prompts_file)
            total += len(all_prompt_rows)
            print(f"  Chunk {file_num}: {prompts_file} "
                  f"({len(all_prompt_rows):,} prompts, {file_size_mb:.1f} MB)")
            del all_prompt_rows

        print(f"\n  Total: {total:,} prompts split into {len(saved_files)} files")
        for f in saved_files:
            print(f"    {f}")
        print(f"\nDone! (--export_prompts_only mode, inference skipped)")
        return

    # ---- Run inference ----
    start_time = time.time()

    if _skip_inference:
        pass  # all_results already set above
    elif backend == "copilot":
        # --- Copilot API path ---
        # Build (item_id, prompt) inputs for all items
        print("\nBuilding prompts for Copilot API ...")
        copilot_inputs = []
        for item in tqdm(data, desc="Building prompts", mininterval=30):
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
        items_to_build = [
            (idx, item, copilot_map.get(item["id"], ""))
            for idx, item in enumerate(data)
        ]
        all_results = _parallel_build_results(items_to_build, similarities_dict)
    else:  # papyrus
        # --- Papyrus API path ---
        print("\nBuilding prompts for Papyrus API ...")
        papyrus_inputs = []
        for item in tqdm(data, desc="Building prompts", mininterval=30):
            item_id = item["id"]
            top_similar_items = similarities_dict.get(item_id, [])[:5]
            raw_prompt = prepare_prompt(
                item, top_similar_items, all_items_dict, prompt_template
            )
            papyrus_inputs.append((item_id, raw_prompt))

        if debug:
            from Infer_by_papyrus import run_papyrus_parallel
            papyrus_results = run_papyrus_parallel(
                inputs=papyrus_inputs,
                papyrus_endpoint=args.papyrus_endpoint,
                model_name=args.papyrus_model,
                quota_id=args.papyrus_quota_id,
                timeout_ms=args.papyrus_timeout_ms,
                num_workers=args.papyrus_workers,
                max_tokens=args.max_tokens,
                max_retries=3,
            )
        else:
            from Infer_by_papyrus import run_papyrus_parallel_with_checkpoint
            papyrus_results = run_papyrus_parallel_with_checkpoint(
                inputs=papyrus_inputs,
                checkpoint_dir=checkpoint_dir,
                papyrus_endpoint=args.papyrus_endpoint,
                model_name=args.papyrus_model,
                quota_id=args.papyrus_quota_id,
                timeout_ms=args.papyrus_timeout_ms,
                num_workers=args.papyrus_workers,
                max_tokens=args.max_tokens,
                max_retries=3,
                chunk_size=args.papyrus_chunk_size,
            )

        papyrus_map = {item_id: gen_text for item_id, gen_text in papyrus_results}
        items_to_build = [
            (idx, item, papyrus_map.get(item["id"], ""))
            for idx, item in enumerate(data)
        ]
        all_results = _parallel_build_results(items_to_build, similarities_dict)

    if not _skip_inference:
        inference_time = time.time() - start_time
        print(f"\nInference time: {inference_time:.1f}s "
              f"({len(data) / inference_time:.1f} items/s)")

        # Merge with previous results if resuming
        if previous_results:
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

    # ---- Final save: write all merged results to disk ----
    if not debug:
        print(f"\n[FINAL-SAVE] Saving {len(all_results):,} merged results ...")
        stats = save_all_outputs(all_results, args.output_dir,
                                 id2meta_file=args.id2meta_file,
                                 check_seller=args.check_seller)
        cleanup_checkpoint(checkpoint_dir)
        print(f"\nCompleted! Total: {len(all_results)}, "
              f"Products: {stats['valid_product_count']}, "
              f"Non-products: {stats['non_product_count']}, "
              f"Failed: {stats['failed_count']}, "
              f"Seller mismatch: {stats['seller_mismatch_count']}")

    # Print first few examples
    num_debug_show = len(all_results) if debug else 3
    print(f"\n{'='*60}")
    print(f"First {num_debug_show} LLM outputs:")
    for idx, res in enumerate(all_results[:num_debug_show]):
        print(f"\n  [{idx+1}] ID={res['id']}")


if __name__ == "__main__":
    main()
     