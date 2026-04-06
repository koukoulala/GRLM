"""
Analyze the trade-off between number of journeys and products per journey
in the SFT training data.

Usage:
    python analyze_journey_product_tradeoff.py
    python analyze_journey_product_tradeoff.py --data_dir /path/to/sft_data_v3_new
"""

import json
import argparse
from collections import defaultdict


DATA_DIR = (
    "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
    "/Data/LLMTrainingData/20260324/sft_data_v3_new"
)


def load_metadata(path):
    """Load num_journeys and num_products_per_journey from a JSON file."""
    print(f"  Loading {path} ...")
    with open(path) as f:
        data = json.load(f)
    records = []
    for item in data:
        meta = item.get("metadata", {})
        nj = meta.get("num_journeys")
        npj = meta.get("num_products_per_journey", [])
        if nj is None or not npj:
            continue
        records.append({
            "num_journeys": nj,
            "avg_products": sum(npj) / len(npj),
            "min_products": min(npj),
            "max_products": max(npj),
            "total_products": sum(npj),
        })
    print(f"    -> {len(records):,} valid records")
    return records


def print_table(title, rows, headers):
    """Print a simple fixed-width table."""
    print(f"\n{title}")
    print("-" * 70)
    col_w = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2
             for i, h in enumerate(headers)]
    fmt = "".join(f"{{:>{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("-" * 70)
    for row in rows:
        print(fmt.format(*row))
    print("-" * 70)


def analyze(records, task_name):
    print(f"\n{'=' * 70}")
    print(f"Task: {task_name}  ({len(records):,} records)")
    print("=" * 70)

    # --- Group by num_journeys ---
    by_nj = defaultdict(list)
    for r in records:
        by_nj[r["num_journeys"]].append(r["avg_products"])

    rows = []
    for nj in sorted(by_nj):
        vals = by_nj[nj]
        rows.append((
            nj,
            len(vals),
            f"{len(vals)/len(records)*100:.1f}%",
            f"{sum(vals)/len(vals):.2f}",
            f"{min(vals):.0f}",
            f"{max(vals):.0f}",
        ))
    print_table(
        "Distribution: #journeys  →  avg products per journey",
        rows,
        ["#journey", "count", "pct", "avg_prod/j", "min_avg", "max_avg"],
    )

    # --- Group by avg products per journey (buckets) ---
    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 999)]
    bucket_data = defaultdict(list)
    for r in records:
        ap = r["avg_products"]
        for lo, hi in buckets:
            if lo <= ap <= hi:
                bucket_data[(lo, hi)].append(r["num_journeys"])
                break

    rows2 = []
    for lo, hi in buckets:
        vals = bucket_data[(lo, hi)]
        if not vals:
            continue
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        rows2.append((
            label,
            len(vals),
            f"{len(vals)/len(records)*100:.1f}%",
            f"{sum(vals)/len(vals):.2f}",
            f"{min(vals)}",
            f"{max(vals)}",
        ))
    print_table(
        "Distribution: avg products/journey bucket  →  #journeys",
        rows2,
        ["prod_bucket", "count", "pct", "avg_#j", "min_j", "max_j"],
    )

    # --- Overall correlation ---
    nj_list = [r["num_journeys"] for r in records]
    ap_list = [r["avg_products"] for r in records]
    n = len(records)
    mean_nj = sum(nj_list) / n
    mean_ap = sum(ap_list) / n
    cov = sum((nj_list[i] - mean_nj) * (ap_list[i] - mean_ap) for i in range(n)) / n
    std_nj = (sum((x - mean_nj) ** 2 for x in nj_list) / n) ** 0.5
    std_ap = (sum((x - mean_ap) ** 2 for x in ap_list) / n) ** 0.5
    corr = cov / (std_nj * std_ap) if std_nj and std_ap else 0

    print(f"\n  Overall correlation (#journeys vs avg_products/journey): {corr:+.4f}")
    print(f"  Mean #journeys: {mean_nj:.2f},  Mean avg_products/journey: {mean_ap:.2f}")
    if corr < -0.1:
        print("  -> Negative correlation: more journeys = fewer products per journey (as expected)")
    elif corr > 0.1:
        print("  -> Positive correlation: more journeys = more products per journey")
    else:
        print("  -> Near-zero correlation: no clear trade-off in the data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    args = parser.parse_args()

    import os
    e2j_path = os.path.join(args.data_dir, "event2journey_sft_full.json")
    p2j_path = os.path.join(args.data_dir, "profile2journey_sft_full.json")

    for path, task in [(e2j_path, "event2journey"), (p2j_path, "profile2journey")]:
        if os.path.exists(path):
            records = load_metadata(path)
            analyze(records, task)
        else:
            print(f"  Skipping {task}: file not found at {path}")


if __name__ == "__main__":
    main()
