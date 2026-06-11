#!/usr/bin/env python3
"""Step 2.0: Evaluate Term ID Quality via LLM.

Reads id2meta_with_norm.json (which already has title, description,
categories, attributes, summary_words), builds evaluation prompts,
calls LLM to score each Term ID across 6 dimensions.

Usage:
    python s2_0_evaluate_tid.py --id2meta_file .../processed_v2/id2meta_with_norm.json
    python s2_0_evaluate_tid.py --id2meta_file .../processed_v3/id2meta_with_norm.json --output_dir .../eval_v3/
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "resources")
sys.path.insert(0, RESOURCES_DIR)
from llm_utils import run_llm_parallel_with_checkpoint, cleanup_checkpoint


# =============================================================================
# Prompt Construction
# =============================================================================

def build_product_info_text(item):
    """Build product info text from id2meta record."""
    lines = []
    title = item.get("title", "")
    if title:
        lines.append(f"Title: {title}")
    desc = item.get("description", "")
    if desc:
        lines.append(f"Description: {desc[:500]}{'...' if len(desc) > 500 else ''}")
    cats = item.get("categories", "")
    if cats:
        lines.append(f"Categories: {cats}")
    attrs = item.get("attributes", {})
    for key in ["Brand", "Seller", "Color", "Size", "Gender", "AgeGroup"]:
        val = attrs.get(key, "")
        if isinstance(val, str):
            val = val.strip()
        if val and val.lower() not in ("unisex", "adult"):
            lines.append(f"{key}: {val}")
    return "\n".join(lines) if lines else "(no information)"


def build_eval_prompt(item, summary_words, prompt_template):
    """Build evaluation prompt for a single item."""
    product_info_text = build_product_info_text(item)
    term_id_text = json.dumps(summary_words, ensure_ascii=False)
    return (prompt_template
            .replace("{product_info_text}", product_info_text)
            .replace("{term_id_text}", term_id_text))


# =============================================================================
# Result Parsing
# =============================================================================

def parse_eval_result(llm_output):
    """Parse LLM evaluation output into structured result."""
    if not llm_output:
        return None
    text = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL).strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# =============================================================================
# Statistics
# =============================================================================

def compute_statistics(eval_results):
    """Compute aggregate statistics from evaluation results."""
    dimensions = ["D1_searchability", "D2_model_name", "D3_info_density",
                   "D4_attribute_coverage", "D5_brand_seller",
                   "D6_category_precision", "D7_factual_accuracy"]
    stats = {}
    for dim in dimensions:
        scores = []
        for r in eval_results:
            s = r.get("scores", {}).get(dim)
            if s is not None and s != "N/A":
                try:
                    scores.append(int(s))
                except (ValueError, TypeError):
                    pass
        if scores:
            stats[dim] = {
                "mean": round(sum(scores) / len(scores), 2),
                "count": len(scores),
                "dist": {str(i): scores.count(i) for i in range(3)},
            }

    overalls = [r.get("overall", 0) for r in eval_results if "overall" in r]
    if overalls:
        stats["overall"] = {
            "mean": round(sum(overalls) / len(overalls), 1),
            "min": min(overalls),
            "max": max(overalls),
            "count": len(overalls),
        }

    all_issues = []
    for r in eval_results:
        all_issues.extend(r.get("issues", []))
    stats["top_issues"] = Counter(all_issues).most_common(20)
    return stats


# =============================================================================
# Consistency Check (multi-run comparison)
# =============================================================================

def compare_runs(id2meta_files):
    """Compare TID consistency across multiple generation runs."""
    print(f"\n{'=' * 60}")
    print(f"  TID Consistency Check ({len(id2meta_files)} runs)")
    print(f"{'=' * 60}")

    all_data = []
    for fpath in id2meta_files:
        print(f"  Loading: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read().rstrip('\x00')
            all_data.append(json.loads(raw))

    common_ids = set(all_data[0].keys())
    for d in all_data[1:]:
        common_ids &= set(d.keys())
    print(f"  Common items across all runs: {len(common_ids):,}")

    n_runs = len(all_data)
    identical = 0
    different = 0
    slot_same = [0] * 7
    diff_examples = []

    for item_id in sorted(common_ids):
        tids = []
        for d in all_data:
            sw = d[item_id].get("summary_words", [])
            tids.append(tuple(sw) if sw else ())

        if any(len(t) != 7 for t in tids):
            continue

        if len(set(tids)) == 1:
            identical += 1
        else:
            different += 1
            if len(diff_examples) < 20:
                diff_examples.append({
                    "item_id": item_id,
                    "title": all_data[0][item_id].get("title", "")[:80],
                    "tids": [list(t) for t in tids],
                })

        for s in range(7):
            vals = set(t[s] for t in tids)
            if len(vals) == 1:
                slot_same[s] += 1

    total = identical + different
    if total == 0:
        print("  No valid items to compare.")
        return

    slot_names = ["Category", "Attr2", "Attr3", "Attr4", "Attr5", "Brand", "Seller"]

    print(f"\n  Overall Consistency:")
    print(f"    Identical: {identical:,}/{total:,} ({identical/total*100:.1f}%)")
    print(f"    Different: {different:,}/{total:,} ({different/total*100:.1f}%)")

    print(f"\n  Per-Slot Consistency:")
    for s in range(7):
        pct = slot_same[s] / total * 100
        print(f"    Slot {s+1} ({slot_names[s]:>8}): {slot_same[s]:,}/{total:,} ({pct:.1f}%)")

    if diff_examples:
        print(f"\n  Examples of Differing TIDs (showing up to 10):")
        for ex in diff_examples[:10]:
            print(f"    {ex['title']}")
            for i, tid in enumerate(ex["tids"]):
                print(f"      Run {i+1}: {tid}")
            print()

    report = {
        "n_runs": n_runs,
        "n_items": total,
        "identical": identical,
        "identical_pct": round(identical / total * 100, 1),
        "different": different,
        "different_pct": round(different / total * 100, 1),
        "per_slot_consistency": {
            slot_names[s]: round(slot_same[s] / total * 100, 1)
            for s in range(7)
        },
        "diff_examples": diff_examples,
    }

    out_dir = os.path.dirname(id2meta_files[0])
    out_path = os.path.join(out_dir, "consistency_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    VIP_DIR = ("/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/"
               "Data/LLMTrainingData/20260528/vip_case_study_IDB_new")
    parser = argparse.ArgumentParser(
        description="Evaluate Term ID quality via LLM scoring"
    )
    parser.add_argument(
        "--id2meta_file", type=str,
        default=f"{VIP_DIR}/processed_v2/id2meta_with_norm.json",
        help="Path to id2meta_with_norm.json (from s1 output)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save results (default: <id2meta_dir>/eval/)",
    )
    parser.add_argument(
        "--eval_prompt_file", type=str,
        default=os.path.join(SCRIPT_DIR, "prompts", "term_evaluation.md"),
    )
    parser.add_argument("--sample_size", type=int, default=0,
                        help="Items to evaluate (0 = all)")
    parser.add_argument("--token_file", type=str,
                        default="./resources/tokens_full.txt")
    parser.add_argument("--copilot_model", type=str, default="gpt-5.4")
    parser.add_argument("--copilot_workers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare_runs", nargs="+", default=None,
                        help="Compare TID consistency across multiple id2meta files")
    args = parser.parse_args()

    # Default output_dir next to id2meta
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.id2meta_file), "eval")
    return args


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Consistency comparison mode
    if args.compare_runs:
        compare_runs(args.compare_runs)
        return

    random.seed(args.seed)

    print("=" * 60)
    print("  Step 2.0: Term ID Evaluation")
    print("=" * 60)
    print(f"  id2meta:     {args.id2meta_file}")
    print(f"  output_dir:  {args.output_dir}")
    print(f"  sample_size: {args.sample_size or 'ALL'}")
    print(f"  model:       {args.copilot_model}")
    print()

    # Load id2meta (has all product info + summary_words)
    print("Loading id2meta...")
    with open(args.id2meta_file, "r", encoding="utf-8") as f:
        raw = f.read().rstrip('\x00')
        id2meta = json.loads(raw)
    print(f"  Loaded {len(id2meta):,} items")

    # Load eval prompt
    print(f"Loading eval prompt: {args.eval_prompt_file}")
    with open(args.eval_prompt_file, "r", encoding="utf-8") as f:
        eval_prompt_template = f.read()

    # Sample
    item_ids = list(id2meta.keys())
    if args.sample_size > 0 and args.sample_size < len(item_ids):
        item_ids = random.sample(item_ids, args.sample_size)
        print(f"  Sampled {len(item_ids):,} items")
    else:
        print(f"  Evaluating all {len(item_ids):,} items")

    # Build prompts
    print("\nBuilding evaluation prompts...")
    eval_inputs = []
    for item_id in tqdm(item_ids, desc="Building prompts", mininterval=5):
        meta = id2meta[item_id]
        sw = meta.get("summary_words", [])
        if not sw or len(sw) != 7:
            continue
        prompt = build_eval_prompt(meta, sw, eval_prompt_template)
        eval_inputs.append((item_id, prompt))
    print(f"  Built {len(eval_inputs):,} prompts")

    # Run inference
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "_eval_checkpoint")

    print(f"\nRunning evaluation ({len(eval_inputs):,} items)...")
    t0 = time.time()
    api_results = run_llm_parallel_with_checkpoint(
        inputs=eval_inputs,
        token_file=args.token_file,
        checkpoint_dir=ckpt_dir,
        num_workers=args.copilot_workers,
        model=args.copilot_model,
        temperature=0,
        max_tokens=500,
        chunk_size=5000,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Parse
    print("\nParsing results...")
    eval_results = []
    parse_fail = 0
    for item_id, llm_output in api_results:
        parsed = parse_eval_result(llm_output)
        if parsed is None:
            parse_fail += 1
            continue
        parsed["item_id"] = item_id
        parsed["summary_words"] = id2meta[item_id].get("summary_words", [])
        parsed["title"] = id2meta[item_id].get("title", "")
        eval_results.append(parsed)
    print(f"  Parsed: {len(eval_results):,}, Failed: {parse_fail:,}")

    # Stats
    stats = compute_statistics(eval_results)
    print(f"\n{'=' * 60}")
    print(f"  Evaluation Summary ({len(eval_results):,} items)")
    print(f"{'=' * 60}")
    for dim in ["D1_searchability", "D2_model_name", "D3_info_density",
                "D4_attribute_coverage", "D5_brand_seller",
                "D6_category_precision", "D7_factual_accuracy"]:
        if dim in stats:
            s = stats[dim]
            d = s["dist"]
            print(f"  {dim:25s}: mean={s['mean']:.2f}  "
                  f"(0:{d.get('0',0)} 1:{d.get('1',0)} 2:{d.get('2',0)})")
    if "overall" in stats:
        o = stats["overall"]
        print(f"\n  Overall: mean={o['mean']:.1f}  min={o['min']} max={o['max']}")
    if stats.get("top_issues"):
        print(f"\n  Top Issues:")
        for issue, cnt in stats["top_issues"][:15]:
            print(f"    {cnt:4d}x  {issue}")

    # Print lowest-scoring items as top issues
    sorted_by_score = sorted(eval_results, key=lambda x: x.get("overall", 12))
    print(f"\n  Lowest Scoring Items:")
    for r in sorted_by_score[:10]:
        print(f"    score={r.get('overall', '?'):>2}  {r.get('title', '')[:70]}")
        print(f"           TID: {r.get('summary_words', [])}")
        for iss in r.get("issues", []):
            print(f"           - {iss}")
        if r.get("suggested_fix"):
            print(f"           fix: {r.get('suggested_fix')}")
        print()

    # Save
    for name, data in [("eval_results.json", eval_results),
                        ("eval_statistics.json", stats)]:
        path = os.path.join(args.output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  Saved: {path}")

    low = [r for r in eval_results if r.get("overall", 12) <= 8]
    low.sort(key=lambda x: x.get("overall", 0))
    low_path = os.path.join(args.output_dir, "eval_low_scores.json")
    with open(low_path, "w", encoding="utf-8") as f:
        json.dump(low, f, ensure_ascii=False, indent=2)
    print(f"  Low-score items ({len(low):,}): {low_path}")

    cleanup_checkpoint(ckpt_dir)
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
