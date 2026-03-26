#!/usr/bin/env python3
"""
Extract products whose summary_words contain low-information words (Other, Generic, etc.)
from summaries_with_similarity.jsonl, rebuild prompts using the same logic as s1_generate_tid.py,
and write to a single TSV file for re-running.
"""

import json
import os
import sys
import yaml
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
SUMMARIES_FILE = (
    "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/"
    "LLMTrainingData/20260324/processed/summaries_with_similarity.jsonl"
)
PROMPTS_YAML = os.path.join(os.path.dirname(__file__), "resources", "prompts.yaml")
OUTPUT_FILE = (
    "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/"
    "LLMTrainingData/20260324/processed/prompts/item_other_rerun_prompts.tsv"
)

# Words to filter – same list as Rule 9 in prompts.yaml
BANNED_WORDS = {
    "other", "generic", "unbranded", "n/a", "unknown",
    "general", "standard", "default", "basic", "regular",
    "normal", "various", "no brand",
}


# ── Prompt construction (mirrors s1_generate_tid.py) ──────────────────────

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


def build_similar_items_text(similar_item_ids, all_items_dict):
    """Build similar items text block (without similarity scores)."""
    similar_items_info = []
    for sim_id in similar_item_ids[:5]:
        if sim_id in all_items_dict:
            sim_item = all_items_dict[sim_id]
            sim_title = sim_item.get("title", "")
            sim_desc = sim_item.get("description", "")

            info = f"Similar Item {sim_id}:"
            if sim_title:
                info += f" Title: {sim_title}"
            if sim_desc:
                if len(sim_desc) > 150:
                    info += f" Description: {sim_desc[:150]}..."
                else:
                    info += f" Description: {sim_desc}"
            similar_items_info.append(info)

    return "\n".join(similar_items_info) if similar_items_info else "(none)"


def has_banned_word(summary_words):
    """Check if any summary word is a banned low-information word."""
    for w in summary_words:
        if w.lower() in BANNED_WORDS:
            return True
    return False


def main():
    # ── Load prompt template ──────────────────────────────────────────────
    print(f"[1/4] Loading prompt template from {PROMPTS_YAML} ...")
    with open(PROMPTS_YAML, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    prompt_template = prompts["prompts"]["term_generation"]["user"]

    # ── Pass 1: Load all items into dict ──────────────────────────────────
    print(f"[2/4] Loading all items from {SUMMARIES_FILE} ...")
    all_items_dict = {}       # id -> item (lightweight: only meta fields)
    target_ids = []           # ids that need re-prompting
    target_similar_ids = {}   # id -> similar_item_ids list

    line_count = 0
    with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading items", total=3745923):
            item = json.loads(line)
            item_id = item["id"]
            line_count += 1

            # Store lightweight meta for all items (needed for similar-item lookups)
            all_items_dict[item_id] = {
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "categories": item.get("categories", ""),
                "attributes": item.get("attributes", {}),
            }

            # Check if this item has banned words
            summary_words = item.get("summary_words", [])
            if summary_words and has_banned_word(summary_words):
                target_ids.append(item_id)
                target_similar_ids[item_id] = item.get("similar_item_ids", [])

    print(f"    Total items loaded: {line_count:,}")
    print(f"    Items with banned words: {len(target_ids):,}")

    # ── Show breakdown by banned word ─────────────────────────────────────
    # Quick re-scan of target items for stats
    word_counts = {}
    # We need to re-read summary_words for target items - store them during loading
    # Actually let's just re-scan the file for the stats since we didn't store summary_words
    # Better approach: check if target_ids is nonzero and provide basic info
    print(f"\n[3/4] Building prompts for {len(target_ids):,} items ...")

    # ── Build prompts ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        out_f.write("GlobalOfferId\tPrompt\n")

        for item_id in tqdm(target_ids, desc="Building prompts"):
            item_meta = all_items_dict[item_id]
            sim_ids = target_similar_ids[item_id]

            product_info_text = build_product_info_text(item_meta)
            similar_items_text = build_similar_items_text(sim_ids, all_items_dict)

            raw_prompt = prompt_template.format(
                product_info_text=product_info_text,
                similar_items_text=similar_items_text,
            )
            escaped_prompt = raw_prompt.replace("\t", " ").replace("\n", "\\n")
            out_f.write(f"{item_id}\t{escaped_prompt}\n")
            written += 1

    print(f"\n[4/4] Done! Wrote {written:,} prompts to:\n    {OUTPUT_FILE}")

    # ── File size ─────────────────────────────────────────────────────────
    file_size = os.path.getsize(OUTPUT_FILE)
    if file_size > 1_073_741_824:
        print(f"    File size: {file_size / 1_073_741_824:.2f} GB")
    else:
        print(f"    File size: {file_size / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
