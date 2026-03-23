"""Step 6: Post-Evaluation Metrics

Computes Hit@K and NDCG@K metrics from beam search evaluation results (s5_eval.py).
Supports optional per-domain filtering for cross-domain evaluation.

Usage:
    # Basic (all samples):
    python s6_post_eval.py --results_file ./processed/eval_results/eval_info.json

    # Per-domain filtering (cross-domain):
    python s6_post_eval.py \
        --results_file ./processed/eval_results/eval_info.json \
        --eval_domains "Beauty" "Electronics"
"""

import json
import argparse
import numpy as np
from typing import List, Dict, Any


def hit_k(topk_results, k):
    """Calculate hit@k metric."""
    hit = 0.0
    for row in topk_results:
        if len(row) >= k and max(row[:k]) == 1:
            hit += 1
    return hit / len(topk_results)


def ndcg_k(topk_results, k):
    """Calculate ndcg@k metric."""
    ndcg = 0.0
    for row in topk_results:
        dcg = 0.0
        for i in range(min(k, len(row))):
            if row[i] == 1:
                dcg += 1.0 / np.log2(i + 2)
        idcg = 1.0 / np.log2(2)  # Best case: hit at position 1
        ndcg += dcg / idcg
    return ndcg / len(topk_results)


def get_metrics_results(topk_results, metrics):
    """Calculate evaluation metrics."""
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError(f"Metric {m} not implemented")
    return res


def load_results(file_path: str, eval_domain: str = None) -> List[Dict[str, Any]]:
    """Load inference results, optionally filtering by domain."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} sample results")

    if eval_domain:
        data = [
            d
            for d in data
            if d.get("test_ground_truth_msg", {}).get("domain") == eval_domain
        ]
        print(f"After domain filter '{eval_domain}': {len(data)} samples")

    return data


def create_topk_results(
    data: List[Dict[str, Any]], max_k: int = 200
) -> List[List[int]]:
    """Create topk hit/miss matrix from evaluation results."""
    topk_results = []
    for sample in data:
        predicted_iids = sample.get("iids", [])
        ground_truth_iid = sample.get("iid_gt", "")
        hit_sequence = [
            1 if pred == ground_truth_iid else 0
            for pred in predicted_iids[:max_k]
        ]
        hit_sequence.extend([0] * (max_k - len(hit_sequence)))
        topk_results.append(hit_sequence)
    return topk_results


def analyze_hit_positions(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze hit position distribution."""
    hit_positions = []
    missed_samples = 0

    for sample in data:
        predicted_iids = sample.get("iids", [])
        ground_truth_iid = sample.get("iid_gt", "")
        hit_pos = -1
        for pos, pred_iid in enumerate(predicted_iids):
            if pred_iid == ground_truth_iid:
                hit_pos = pos + 1
                break
        if hit_pos != -1:
            hit_positions.append(hit_pos)
        else:
            missed_samples += 1

    if hit_positions:
        return {
            "total_samples": len(data),
            "hit_samples": len(hit_positions),
            "missed_samples": missed_samples,
            "hit_rate": len(hit_positions) / len(data),
            "mean_hit_position": float(np.mean(hit_positions)),
            "median_hit_position": float(np.median(hit_positions)),
            "min_hit_position": int(np.min(hit_positions)),
            "max_hit_position": int(np.max(hit_positions)),
            "hit_at_1": sum(1 for p in hit_positions if p == 1),
            "hit_at_5": sum(1 for p in hit_positions if p <= 5),
            "hit_at_10": sum(1 for p in hit_positions if p <= 10),
        }
    else:
        return {
            "total_samples": len(data),
            "hit_samples": 0,
            "missed_samples": missed_samples,
            "hit_rate": 0.0,
        }


def evaluate_and_print(data, label="Overall"):
    """Run full evaluation pipeline and print results."""
    metrics = [
        "hit@1", "hit@3", "hit@5", "hit@10", "hit@20",
        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20",
    ]
    topk_results = create_topk_results(data, max_k=20)
    evaluation_results = get_metrics_results(topk_results, metrics)
    hit_stats = analyze_hit_positions(data)

    print(f"\n{'=' * 60}")
    print(f"  {label}  ({len(data)} samples)")
    print(f"{'=' * 60}")

    print("\n--- Metrics ---")
    for metric, value in evaluation_results.items():
        print(f"  {metric}: {value:.4f}")

    print("\n--- Hit Position Statistics ---")
    for stat, value in hit_stats.items():
        if isinstance(value, float):
            print(f"  {stat}: {value:.4f}")
        else:
            print(f"  {stat}: {value}")

    return {"evaluation_metrics": evaluation_results, "hit_statistics": hit_stats}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-evaluation metrics (Hit@K / NDCG@K)"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to eval_info.json from s5_eval.py",
    )
    parser.add_argument(
        "--eval_domains",
        type=str,
        nargs="*",
        default=None,
        help="Optional domain names to evaluate separately (cross-domain mode). "
        'E.g. --eval_domains "Beauty" "Electronics"',
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save metrics JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_data = load_results(args.results_file)
    all_domain_results = {}

    if args.eval_domains:
        # Per-domain evaluation
        for domain in args.eval_domains:
            domain_data = [
                d
                for d in all_data
                if d.get("test_ground_truth_msg", {}).get("domain") == domain
            ]
            if not domain_data:
                print(f"\nWarning: No samples found for domain '{domain}'")
                continue
            result = evaluate_and_print(domain_data, label=f"Domain: {domain}")
            all_domain_results[domain] = result

        # Also show overall
        result = evaluate_and_print(all_data, label="Overall (all domains)")
        all_domain_results["overall"] = result
    else:
        # Single evaluation (no domain filtering)
        result = evaluate_and_print(all_data, label="Overall")
        all_domain_results["overall"] = result

    # Save results
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(all_domain_results, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to: {args.output_file}")


if __name__ == "__main__":
    main()
