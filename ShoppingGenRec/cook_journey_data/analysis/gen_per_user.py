"""
Generate per-user markdown analysis files from paired_data.json.
Phase 2 automated analysis: product overlap, seller distribution, ranking diffs.

Usage: python3 gen_per_user.py --run-name <name> [--output-dir <path>]
"""
import json, os, sys, argparse, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-dir", default=None,
                    help="Additional output directory; per-user md files are copied there too")
    return ap.parse_args()


def get_gids(products):
    """Extract global_offer_id → product info dict from a journey's products."""
    gids = {}
    for p in products:
        if not isinstance(p, dict):
            continue
        if "matched_products" in p:
            for mp in p["matched_products"]:
                gid = mp.get("global_offer_id", "")
                if gid:
                    gids[gid] = {
                        "Title": mp.get("Title", ""),
                        "Seller": mp.get("Seller", ""),
                        "Price": mp.get("OriginalPrice", ""),
                        "DisplayPosition": mp.get("DisplayPosition", ""),
                        "OriginalSLMRank": mp.get("OriginalSLMRank", ""),
                    }
        elif "global_offer_id" in p:
            gid = p["global_offer_id"]
            if gid:
                gids[gid] = {
                    "Title": p.get("Title", ""),
                    "Seller": p.get("Seller", ""),
                    "Price": p.get("OriginalPrice", ""),
                    "DisplayPosition": p.get("DisplayPosition", ""),
                    "OriginalSLMRank": p.get("OriginalSLMRank", ""),
                }
    return gids


def count_sellers_all(journeys):
    sellers = {}
    for j in journeys:
        for p in j.get("products", []):
            if not isinstance(p, dict):
                continue
            if "matched_products" in p:
                for mp in p["matched_products"]:
                    s = mp.get("Seller", "")
                    if s:
                        sellers[s] = sellers.get(s, 0) + 1
            elif "Seller" in p:
                s = p["Seller"]
                if s:
                    sellers[s] = sellers.get(s, 0) + 1
    return sellers


def gen_user_md(u):
    uid = u["stableid"][:8].upper()
    p1 = u["p1"]
    p2 = u["p2"]
    p1_name = u.get("p1_name", "P1")
    p2_name = u.get("p2_name", "P2")

    lines = []
    lines.append(f"# 用户 {uid} 对比分析")
    lines.append("")
    lines.append(f"> **分析视角：** {p1_name} vs {p2_name} 对称对比，仅 step6 ranker 不同")
    lines.append("")

    # 零、数据校验
    lines.append("## 零、数据校验")
    lines.append("")
    lines.append(f"- P1 Journey 数：{p1['journey_count']} 个")
    lines.append(f"- P2 Journey 数：{p2['journey_count']} 个")
    lines.append(f"- 浏览记录数：{p2['recent_events_count']} 条")
    rp = p1.get("profile_retailer_preferences", [])
    lines.append(f"- Profile retailerPreferences：{rp if rp else '空'}")
    lines.append("")

    # 一、Journey 与产品选择对比
    lines.append("## 一、Journey 与产品选择对比")
    lines.append("")

    total_overlap = 0
    total_p1_only = 0
    total_p2_only = 0

    j_pairs = list(zip(p1["journeys"], p2["journeys"]))
    for ji, (j1, j2) in enumerate(j_pairs):
        title = j1.get("title", f"Journey {ji+1}")
        lines.append(f'### Journey {ji+1}: "{title}"')
        lines.append("")

        g1 = get_gids(j1.get("products", []))
        g2 = get_gids(j2.get("products", []))

        overlap = set(g1.keys()) & set(g2.keys())
        p1_only = set(g1.keys()) - overlap
        p2_only = set(g2.keys()) - overlap

        total_overlap += len(overlap)
        total_p1_only += len(p1_only)
        total_p2_only += len(p2_only)

        stats1 = j1.get("stats", {})
        stats2 = j2.get("stats", {})

        lines.append("| 指标 | P1 | P2 |")
        lines.append("|------|----|----|")
        lines.append(f"| 最终产品数 | {len(g1)} | {len(g2)} |")
        lines.append(f"| 重叠产品 | {len(overlap)} | {len(overlap)} |")
        lines.append(f"| P独有产品 | {len(p1_only)} | {len(p2_only)} |")

        tc1 = stats1.get("totalCandidates", "N/A")
        tc2 = stats2.get("totalCandidates", "N/A")
        sc1 = stats1.get("selectedCount", "N/A")
        sc2 = stats2.get("selectedCount", "N/A")
        lines.append(f"| 候选/入选 | {tc1}/{sc1} | {tc2}/{sc2} |")
        lines.append("")

        # P1-only products (top 3)
        if p1_only:
            items = [g1[gid] for gid in list(p1_only)[:3]]
            lines.append("**P1独有产品（前3）：**")
            for item in items:
                lines.append(f"- {item['Title'][:60]} | {item['Seller']} | ${item['Price']}")
            lines.append("")

        if p2_only:
            items = [g2[gid] for gid in list(p2_only)[:3]]
            lines.append("**P2独有产品（前3）：**")
            for item in items:
                lines.append(f"- {item['Title'][:60]} | {item['Seller']} | ${item['Price']}")
            lines.append("")

        # Position changes for overlapping products
        pos_changes = []
        for gid in overlap:
            dp1 = g1[gid].get("DisplayPosition", "")
            dp2 = g2[gid].get("DisplayPosition", "")
            if dp1 and dp2 and str(dp1) != str(dp2):
                pos_changes.append(
                    (g1[gid]["Title"][:40], dp1, dp2, g1[gid]["Seller"])
                )

        if pos_changes:
            lines.append("**排序变化（重叠产品）：**")
            lines.append("| 产品 | P1位 | P2位 | Seller |")
            lines.append("|------|------|------|--------|")
            for t, dp1, dp2, s in sorted(pos_changes, key=lambda x: abs(int(x[1]) - int(x[2])), reverse=True)[:5]:
                lines.append(f"| {t} | {dp1} | {dp2} | {s} |")
            lines.append("")

    # Summary
    lines.append("### 产品选择汇总")
    lines.append("")
    lines.append(f"- 总重叠产品: {total_overlap}")
    lines.append(f"- P1独有: {total_p1_only}")
    lines.append(f"- P2独有: {total_p2_only}")
    overlap_rate = total_overlap / max(total_overlap + total_p1_only, 1) * 100
    lines.append(f"- 产品重叠率: {overlap_rate:.1f}%")
    lines.append("")

    # 二、Seller/Brand 分布
    lines.append("## 二、Seller/Brand 分布对比")
    lines.append("")

    s1 = count_sellers_all(p1["journeys"])
    s2 = count_sellers_all(p2["journeys"])

    top1 = sorted(s1.items(), key=lambda x: -x[1])[:8]
    top2 = sorted(s2.items(), key=lambda x: -x[1])[:8]

    lines.append("**P1 Top Sellers：**")
    for s, c in top1:
        lines.append(f"- {s}: {c}")
    lines.append("")
    lines.append("**P2 Top Sellers：**")
    for s, c in top2:
        lines.append(f"- {s}: {c}")
    lines.append("")
    lines.append(f"P1 独立卖家数: {len(s1)} | P2 独立卖家数: {len(s2)}")
    lines.append("")

    return uid, "\n".join(lines)


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(script_dir, args.run_name)
    paired_file = os.path.join(run_dir, "paired_data.json")

    with open(paired_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_user_dir = os.path.join(run_dir, "per_user")
    os.makedirs(per_user_dir, exist_ok=True)

    count = 0
    for u in data:
        if u["triage"] != "deep":
            continue
        uid, md_content = gen_user_md(u)
        out_path = os.path.join(per_user_dir, f"user_{uid}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  wrote user_{uid}.md")
        count += 1

    print(f"\nGenerated {count} per-user analysis files → {per_user_dir}")

    # ── Mirror to --output-dir ──
    if args.output_dir:
        import shutil
        mirror_per_user = os.path.join(args.output_dir, args.run_name, "per_user")
        os.makedirs(mirror_per_user, exist_ok=True)
        for fname in os.listdir(per_user_dir):
            shutil.copy2(os.path.join(per_user_dir, fname), mirror_per_user)
        print(f"Mirrored {count} files → {mirror_per_user}")


if __name__ == "__main__":
    main()
