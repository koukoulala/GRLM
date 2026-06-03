"""
Build a single visualized HTML dashboard from a completed run directory.

Reads from the run directory:
  paired_data.json
  seller_analysis.json    (7-tier; from analyze_sellers.py)
  comprehensive_summary.md
  gap_analysis.md
  per_user/user_*.md

Writes:
  report.html

Output structure (7 tabs):
  概览        — KPI cards + tier bars + user cards
  ⭐ 最终结论  — Executive summary + Journey/Product axes + total scoring + scenario recommendation
  差距分析     — gap_analysis.md
  综合总结     — comprehensive_summary.md
  逐用户深入分析 — all per_user/*.md (switchable)
  Seller / 价格分析 — full tier tables + top sellers
  📋 评估方法  — methodology notes (history-union view, tier classification, scoring rules)

Usage:
  py build_html_report.py <run_dir> [--dimensions <dims.json>]

If --dimensions is omitted, the script uses the default 17-dimension scoring
template defined inline. To customize for a different pipeline pair, copy
the DEFAULT_DIMENSIONS dict, edit, and pass via --dimensions <path>.

Place sub-agent-generated dimensions JSON at:
  <run_dir>/dimensions.json

The dimensions JSON shape:
  {
    "executive_summary": "<HTML paragraph(s) for the top-of-verdict summary>",
    "journey_axes": [ {"name": "...", "p1": "...", "p2": "...", "winner": "...", "evidence": "..."}, ... ],
    "product_axes": [ ... ],
    "dimensions": [ ... ],   # 17-dim full table
    "scenarios": [ {"scenario": "...", "winner": "...", "rationale": "..."}, ... ],
    "p2_advantages": [ {"label": "...", "value": "...", "sub": "..."}, ... ],   # 4 KPI cards
    "p1_advantages": [ ... ]  # 4 KPI cards
  }
"""
import os, re, json, sys, io, glob, argparse
from html import escape

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ---------- Markdown → HTML converter ----------
def convert_md(md: str) -> str:
    lines = md.split("\n")
    out = []
    i = 0
    in_code = False
    code_buf = []
    code_lang = ""
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^```(\w*)\s*$", line)
        if m:
            if in_code:
                out.append(f'<pre class="code"><code class="lang-{escape(code_lang)}">{escape(chr(10).join(code_buf))}</code></pre>')
                in_code = False
                code_buf = []
                code_lang = ""
            else:
                in_code = True
                code_lang = m.group(1) or ""
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|\-]+\|[\s:|\-]+\s*$", lines[i+1]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(row)
                i += 1
            t = ['<div class="table-wrap"><table>']
            t.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr></thead>")
            t.append("<tbody>")
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("\n".join(t))
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            text = m.group(2).strip()
            anchor = re.sub(r"[^\w一-鿿]+", "-", text).strip("-").lower()[:60]
            out.append(f'<h{lvl} id="{anchor}">{inline(text)}</h{lvl}>')
            i += 1
            continue

        if re.match(r"^\s*>\s?", line):
            buf = []
            while i < len(lines) and re.match(r"^\s*>\s?", lines[i]):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            inner = convert_md("\n".join(buf))
            out.append(f'<blockquote>{inner}</blockquote>')
            continue

        if re.match(r"^\s*[-*]\s+", line):
            buf = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                buf.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(b)}</li>" for b in buf) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            buf = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                buf.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(b)}</li>" for b in buf) + "</ol>")
            continue

        if re.match(r"^\s*---+\s*$", line):
            out.append("<hr/>")
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6}\s|\s*[-*]\s|\s*\d+\.\s|\s*>\s|```|\|)", lines[i]):
            para.append(lines[i])
            i += 1
        out.append(f'<p>{inline(" ".join(para))}</p>')

    return "\n".join(out)


def inline(s: str) -> str:
    s = escape(s, quote=False)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"`([^`]+)`", lambda m: f'<code>{m.group(1)}</code>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\*])\*([^*\n]+)\*(?![\*])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    s = s.replace("✅", '<span class="ok">✅</span>').replace("❌", '<span class="bad">❌</span>')
    s = re.sub(r"\bP0\b", '<span class="badge p0">P0</span>', s)
    s = re.sub(r"\bP1\b(?!-)", '<span class="badge p1">P1</span>', s)
    s = re.sub(r"\bP2\b(?!-)", '<span class="badge p2">P2</span>', s)
    return s


# ---------- Default dimension scoring template ----------
# Override by writing a custom dimensions.json into the run dir.
DEFAULT_DIMENSIONS = {
    "executive_summary": (
        "<p>填写：用 <strong>同一套维度（Journey + Product）</strong>对两端 pipeline 做<strong>对称对比</strong>。"
        "至少覆盖：(1) P1 和 P2 各自的核心特点 + 关键短板（不预设哪一端是基线）；"
        "(2) 两端是互补还是替代关系；(3) 给业务方的一句话建议。</p>"
        "<p style='color:#9a6700'><strong>提示：</strong>这是默认占位文本。"
        "请在 run_dir/dimensions.json 中提供 executive_summary 字段以替换。</p>"
    ),
    "journey_axes": [
        {"name": "Journey Quality<br/>主题准确性", "p1": "—", "p2": "—", "winner": "—", "evidence": "请在 dimensions.json 中填充"},
        {"name": "Type Diversity<br/>类型多样性", "p1": "—", "p2": "—", "winner": "—", "evidence": "—"},
        {"name": "Coverage Diversity<br/>品类覆盖", "p1": "—", "p2": "—", "winner": "—", "evidence": "—"},
        {"name": "Relevance<br/>用户意图吻合", "p1": "—", "p2": "—", "winner": "—", "evidence": "—"},
    ],
    "product_axes": [
        {"name": "Product Quality<br/>产品-query 匹配", "p1": "—", "p2": "—", "winner": "—", "evidence": "请在 dimensions.json 中填充"},
        {"name": "Seller Authority — 垂直专业", "p1": "—", "p2": "—", "winner": "—", "evidence": "—"},
        {"name": "Seller Authority — 探索/百货", "p1": "—", "p2": "—", "winner": "—", "evidence": "—"},
        {"name": "Diversity<br/>seller/brand 多样性", "p1": "—", "p2": "—", "winner": "—", "evidence": "—"},
        {"name": "Relevance<br/>价格/偏好执行", "p1": "—", "p2": "—", "winner": "—", "evidence": "—"},
    ],
    "dimensions": [],   # 17-dim full table — populate via dimensions.json
    "scenarios": [],
    "p2_advantages": [],
    "p1_advantages": [],
}


# ---------- Build dashboard ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to analysis/<run_name> directory")
    ap.add_argument("--dimensions", help="Path to dimensions.json (else uses run_dir/dimensions.json or defaults)")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_html = os.path.join(run_dir, "report.html")
    seller_path = os.path.join(run_dir, "seller_analysis.json")
    paired_path = os.path.join(run_dir, "paired_data.json")
    summary_path = os.path.join(run_dir, "comprehensive_summary.md")
    gap_path = os.path.join(run_dir, "gap_analysis.md")
    per_user_glob = os.path.join(run_dir, "per_user", "user_*.md")

    dim_path = args.dimensions or os.path.join(run_dir, "dimensions.json")
    if os.path.exists(dim_path):
        with open(dim_path, encoding="utf-8") as f:
            dims = json.load(f)
        # merge over defaults
        merged = dict(DEFAULT_DIMENSIONS)
        merged.update(dims)
        dims = merged
    else:
        dims = DEFAULT_DIMENSIONS

    seller = json.load(open(seller_path, encoding="utf-8"))
    paired = json.load(open(paired_path, encoding="utf-8"))
    deep_users = [u for u in paired if u.get("triage") == "deep"]

    # Extract pipeline labels from paired_data (set by parse_pair.py --p1-name / --p2-name)
    p1_label = paired[0].get("p1_name", "P1") if paired else "P1"
    p2_label = paired[0].get("p2_name", "P2") if paired else "P2"

    # User cards
    user_cards = []
    for u in deep_users:
        sid = u["stableid"][:8]
        p1c = u["p1"]["journey_count"]
        p2c = u["p2"]["journey_count"]
        rec = u["p2"].get("recent_events_count", 0)
        delta = p2c - p1c
        delta_class = "neutral" if abs(delta) <= 2 else ("p2-more" if delta > 0 else "p1-more")
        retailer_pref = ", ".join(u["p2"].get("profile", {}).get("retailerPreferences") or []) or "—"
        user_cards.append(dict(sid=sid, p1=p1c, p2=p2c, delta=delta, delta_class=delta_class, events=rec, retailer_pref=retailer_pref))

    # Tier
    p1_tiers = seller["p1_tiers"]
    p2_tiers = seller["p2_tiers"]
    p1_total = sum(p1_tiers.values()) or 1
    p2_total = sum(p2_tiers.values()) or 1
    TIERS = ["luxury", "department", "specialty", "brand_dtc", "mass", "marketplace", "other"]
    tier_labels = {
        "luxury": "luxury 奢华",
        "department": "department 高端百货",
        "specialty": "specialty 垂直专家",
        "brand_dtc": "brand_dtc 品牌官店",
        "mass": "mass 大众综合",
        "marketplace": "marketplace P2P聚合",
        "other": "other 未分类",
    }
    tier_examples = {
        "luxury": "Bloomingdale's / Saks / Neiman Marcus / FARFETCH / Bergdorf Goodman",
        "department": "Macy's / Nordstrom / Nordstrom Rack / Dillard's / Anthropologie",
        "specialty": "REI / Williams Sonoma / B&H / Albee Baby / Clive Coffee / CDW / Sephora / Chewy / Home Depot / Hobby Lobby",
        "brand_dtc": "Nike.com / Apple / Anker / The North Face / Hoka / Ann Taylor / Quince",
        "mass": "Amazon / Walmart / Target / Kohl's / Wayfair / JCPenney",
        "marketplace": "Etsy / eBay / Poshmark",
        "other": "未匹配规则的零散站点",
    }
    tier_rows = []
    for t in TIERS:
        p1n = p1_tiers.get(t, 0)
        p2n = p2_tiers.get(t, 0)
        tier_rows.append(dict(name=t, label=tier_labels[t], examples=tier_examples[t],
                              p1_pct=round(p1n/p1_total*100, 1), p2_pct=round(p2n/p2_total*100, 1),
                              p1_n=p1n, p2_n=p2n))

    p1_top = seller["p1_top_sellers"][:15]
    p2_top = seller["p2_top_sellers"][:15]
    ps1 = seller.get("p1_price_stats", {})
    ps2 = seller.get("p2_price_stats", {})

    # Markdown tabs
    summary_html = convert_md(open(summary_path, encoding="utf-8").read()) if os.path.exists(summary_path) else "<p><em>comprehensive_summary.md not found</em></p>"
    gap_html = convert_md(open(gap_path, encoding="utf-8").read()) if os.path.exists(gap_path) else "<p><em>gap_analysis.md not found</em></p>"

    per_user_files = sorted(glob.glob(per_user_glob))
    per_user_sections = []
    for f in per_user_files:
        sid = os.path.basename(f).replace("user_", "").replace(".md", "")
        per_user_sections.append((sid, convert_md(open(f, encoding="utf-8").read())))

    # Scoring counts
    n_dim = len(dims["dimensions"])
    p1_wins = sum(1 for d in dims["dimensions"] if d.get("winner", "").startswith("P1") and "P2" not in d["winner"])
    p2_wins = sum(1 for d in dims["dimensions"] if d.get("winner", "").startswith("P2") and "P1" not in d["winner"])

    # Aggregates
    total_users = len(deep_users)
    total_p1 = sum(u["p1"]["journey_count"] for u in deep_users)
    total_p2 = sum(u["p2"]["journey_count"] for u in deep_users)
    total_events = sum(u["p2"].get("recent_events_count", 0) for u in deep_users)
    p2_more = sum(1 for u in deep_users if u["p2"]["journey_count"] > u["p1"]["journey_count"])
    p1_more = sum(1 for u in deep_users if u["p1"]["journey_count"] > u["p2"]["journey_count"])

    # ---------- Render fragments ----------
    def render_axes_table(axes):
        rows = []
        for a in axes:
            w = a.get("winner", "—")
            cls = "p1-win" if "P1" in w and "P2" not in w else ("p2-win" if "P2" in w and "P1" not in w else "neutral-win")
            rows.append(f'<tr><td><strong>{a.get("name","")}</strong></td><td>{a.get("p1","")}</td><td>{a.get("p2","")}</td><td class="{cls}"><strong>{escape(w)}</strong></td><td style="font-size:12.5px;color:#57606a">{a.get("evidence","")}</td></tr>')
        return f'<div class="table-wrap"><table><thead><tr><th>子维度</th><th>P1 {escape(p1_label)}</th><th>P2 {escape(p2_label)}</th><th>胜者</th><th>关键证据</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

    def render_dim_table(dimensions):
        rows = []
        for d in dimensions:
            w = d.get("winner", "—")
            cls = "p1-win" if "P1" in w and "P2" not in w else ("p2-win" if "P2" in w and "P1" not in w else "neutral-win")
            rows.append(f'<tr><td><strong>{escape(d.get("name",""))}</strong></td><td>{escape(d.get("p1",""))}</td><td>{escape(d.get("p2",""))}</td><td class="{cls}"><strong>{escape(w)}</strong></td><td style="font-size:12.5px;color:#57606a">{escape(d.get("evidence",""))}</td></tr>')
        joined = "".join(rows)
        return f'<div class="table-wrap"><table><thead><tr><th>维度</th><th>P1 {escape(p1_label)}</th><th>P2 {escape(p2_label)}</th><th>胜者</th><th>证据</th></tr></thead><tbody>{joined}</tbody></table></div>'

    def render_kpi_grid(items, color):
        cards = []
        for it in items:
            cards.append(f'<div class="kpi {color}"><div class="label">{escape(it.get("label",""))}</div><div class="value">{escape(it.get("value",""))}</div><div class="sub">{it.get("sub","")}</div></div>')
        return f'<div class="kpi-grid">{"".join(cards)}</div>'

    def render_scenarios(scenarios):
        rows = []
        for s in scenarios:
            w = s.get("winner", "")
            wcls = "ok" if w else ""
            rows.append(f'<tr><td>{escape(s.get("scenario",""))}</td><td><span class="{wcls}">{escape(w)}</span></td><td>{escape(s.get("rationale",""))}</td></tr>')
        return f'<div class="table-wrap"><table><thead><tr><th>场景</th><th>推荐 Pipeline</th><th>数据依据</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

    # User journey table (rows for verdict tab)
    user_rows = []
    for u in deep_users:
        sid = u["stableid"][:8]
        p1c = u["p1"]["journey_count"]
        p2c = u["p2"]["journey_count"]
        rec = u["p2"].get("recent_events_count", 0)
        diff = p2c - p1c
        if diff > 0:
            winner = f'<span class="ok">P2 +{diff}</span>'
        elif diff < 0:
            winner = f'<span class="bad">P1 +{-diff}</span>'
        else:
            winner = '<span style="color:#57606a">持平</span>'
        user_rows.append(f'<tr><td><code>{sid}</code></td><td>{p1c}</td><td>{p2c}</td><td>{rec}</td><td>{winner}</td></tr>')

    # Tier bars + table
    tier_bars_p1 = ['<div class="tier-block"><h4>P1</h4>']
    tier_bars_p2 = ['<div class="tier-block"><h4>P2</h4>']
    for t in tier_rows:
        tier_bars_p1.append(f'<div class="bar-row"><div class="label" title="{escape(t["examples"])}">{t["label"]}</div><div class="bar-bg"><div class="bar p1" style="width:{min(t["p1_pct"]*1.5, 100)}%"></div></div><div class="pct">{t["p1_pct"]}%</div></div>')
        tier_bars_p2.append(f'<div class="bar-row"><div class="label" title="{escape(t["examples"])}">{t["label"]}</div><div class="bar-bg"><div class="bar p2" style="width:{min(t["p2_pct"]*1.5, 100)}%"></div></div><div class="pct">{t["p2_pct"]}%</div></div>')
    tier_bars_p1.append("</div>"); tier_bars_p2.append("</div>")
    tier_bars = "".join(tier_bars_p1) + "".join(tier_bars_p2)

    tier_table_rows = []
    for t in tier_rows:
        delta = round(t["p2_pct"] - t["p1_pct"], 1)
        sign = "+" if delta >= 0 else ""
        if t["name"] in ("specialty", "brand_dtc"):
            cls = "ok" if delta > 0 else "bad"
        elif t["name"] in ("luxury", "department", "marketplace"):
            cls = "ok" if delta >= 0 else "bad"
        elif t["name"] == "mass":
            cls = "ok" if delta < 0 else "bad"
        else:
            cls = ""
        tier_table_rows.append(f'<tr><td><strong>{t["label"]}</strong></td><td style="font-size:12px;color:#57606a">{escape(t["examples"])}</td><td>{t["p1_pct"]}% ({t["p1_n"]})</td><td>{t["p2_pct"]}% ({t["p2_n"]})</td><td class="{cls}"><strong>{sign}{delta} pp</strong></td></tr>')
    tier_table = f'<div class="table-wrap"><table><thead><tr><th>Tier</th><th>代表 seller</th><th>P1</th><th>P2</th><th>差异</th></tr></thead><tbody>{"".join(tier_table_rows)}</tbody></table></div>'

    def render_top(seller_list, label):
        rows = ['<div class="table-wrap"><table><thead><tr><th>#</th><th>Seller</th><th>Count</th></tr></thead><tbody>']
        for idx, (s, c) in enumerate(seller_list, 1):
            rows.append(f'<tr><td>{idx}</td><td>{escape(s)}</td><td>{c}</td></tr>')
        rows.append('</tbody></table></div>')
        return f'<div class="tier-block"><h4>{label}</h4>' + "".join(rows) + '</div>'

    top_sellers_html = '<div class="tier-grid">' + render_top(p1_top, "P1 Top 15 Sellers") + render_top(p2_top, "P2 Top 15 Sellers") + '</div>'

    # User tabs
    user_tabs_html = '<div class="user-tabs">' + "".join(f'<button onclick="showUser(\'{sid}\', this)">{sid}</button>' for sid, _ in per_user_sections) + '</div>'
    user_details_html = "".join(f'<div class="user-detail" id="user-{sid}">{body}</div>' for sid, body in per_user_sections)

    # User cards
    cards_html_parts = ['<div class="user-grid">']
    for uc in user_cards:
        cards_html_parts.append(f'''
<div class="user-card" onclick="jumpToUser('{uc['sid']}')">
  <div class="sid">{uc['sid']}</div>
  <div class="stats">
    <div class="stat-pill">P1: {uc['p1']}j</div>
    <div class="stat-pill">P2: {uc['p2']}j</div>
    <div class="delta {uc['delta_class']}">Δ {uc['delta']:+d}</div>
  </div>
  <div class="pref">事件 {uc['events']} 条 · 偏好: {escape(uc['retailer_pref'])}</div>
  <a class="user-link" onclick="event.stopPropagation();jumpToUser('{uc['sid']}')">查看详细分析 →</a>
</div>''')
    cards_html_parts.append('</div>')
    cards_html = "".join(cards_html_parts)

    # Verdict body
    verdict_html = f'''
<h2>✍️ 执行摘要</h2>
<blockquote style="background:#fff8c5;border-left-color:#d4a72c">
{dims["executive_summary"]}
</blockquote>

<h2>1. Journey 维度评估</h2>
{render_axes_table(dims["journey_axes"])}

<h2>2. Product 维度评估</h2>
{render_axes_table(dims["product_axes"])}

<h2>3. 总评分</h2>
<div class="kpi-grid" style="margin-bottom:20px">
  <div class="kpi good"><div class="label">P2 胜出</div><div class="value" style="color:#1a7f37">{p2_wins}</div><div class="sub">/ {n_dim} 维度</div></div>
  <div class="kpi good"><div class="label">P1 胜出</div><div class="value" style="color:#0969da">{p1_wins}</div><div class="sub">/ {n_dim} 维度</div></div>
  <div class="kpi info"><div class="label">互补 / 中性</div><div class="value">{n_dim - p1_wins - p2_wins}</div><div class="sub">/ {n_dim} 维度</div></div>
</div>

<h2>4. {n_dim} 维度逐项数据</h2>
{render_dim_table(dims["dimensions"]) if dims["dimensions"] else "<p><em>未提供 dimensions — 在 run_dir/dimensions.json 中填充</em></p>"}

<h2>5. 关键数字</h2>
<h3>5.1 Journey 数量（{total_users} Deep 用户）</h3>
<div class="kpi-grid">
  <div class="kpi info"><div class="label">P1 Journey 总数</div><div class="value">{total_p1}</div><div class="sub">人均 {total_p1/max(total_users,1):.1f} 个</div></div>
  <div class="kpi info"><div class="label">P2 Journey 总数</div><div class="value">{total_p2}</div><div class="sub">人均 {total_p2/max(total_users,1):.1f} 个</div></div>
  <div class="kpi good"><div class="label">P2 多于 P1 的用户</div><div class="value">{p2_more}</div><div class="sub">/{total_users}</div></div>
  <div class="kpi warn"><div class="label">P1 多于 P2 的用户</div><div class="value">{p1_more}</div><div class="sub">/{total_users}</div></div>
</div>
<div class="table-wrap"><table>
<thead><tr><th>用户</th><th>P1</th><th>P2</th><th>P2 浏览事件</th><th>差异</th></tr></thead>
<tbody>{"".join(user_rows)}</tbody></table></div>

<h3>5.2 Seller Tier 分布（7 档分类）</h3>
{tier_table}

<h3>5.3 价格分布</h3>
<div class="table-wrap"><table>
<thead><tr><th>Pipeline</th><th>min</th><th>p25</th><th>median</th><th>p75</th><th>max</th><th>mean</th><th>n</th></tr></thead>
<tbody>
<tr><td><strong>P1</strong></td><td>${ps1.get('min',0):.0f}</td><td>${ps1.get('p25',0):.0f}</td><td>${ps1.get('median',0):.0f}</td><td>${ps1.get('p75',0):.0f}</td><td>${ps1.get('max',0):.0f}</td><td>${ps1.get('mean',0):.0f}</td><td>{ps1.get('count',0)}</td></tr>
<tr><td><strong>P2</strong></td><td>${ps2.get('min',0):.0f}</td><td>${ps2.get('p25',0):.0f}</td><td>${ps2.get('median',0):.0f}</td><td>${ps2.get('p75',0):.0f}</td><td>${ps2.get('max',0):.0f}</td><td>${ps2.get('mean',0):.0f}</td><td>{ps2.get('count',0)}</td></tr>
</tbody></table></div>

<h2>6. P1 优势（KPI 卡片）</h2>
{render_kpi_grid(dims["p1_advantages"], "info") if dims["p1_advantages"] else "<p><em>未提供 — 在 dimensions.json 的 p1_advantages 字段填充 4 个 KPI 卡片</em></p>"}

<h2>7. P2 优势（KPI 卡片）</h2>
{render_kpi_grid(dims["p2_advantages"], "good") if dims["p2_advantages"] else "<p><em>未提供 — 在 dimensions.json 的 p2_advantages 字段填充 4 个 KPI 卡片</em></p>"}

<h2>8. 场景推荐</h2>
{render_scenarios(dims["scenarios"]) if dims["scenarios"] else "<p><em>未提供 — 在 dimensions.json 的 scenarios 字段填充</em></p>"}
'''

    method_html = '''
<h2>评估方法说明</h2>

<h3>1. 数据来源与配对</h3>
<p>P1（baseline）和 P2（target）通过 user_map.tsv 配对到同一用户。仅分析两端均有 journey 的 Deep 用户。</p>

<h3>2. 同一用户的两个 history 切片：联合视角</h3>
<blockquote>
<p>P1 ({escape(p1_label)}) 和 P2 ({escape(p2_label)}) 看的是<strong>同一个用户</strong>。两端的 history 是<strong>同一用户的两个互补切片</strong>，不是两组不同的数据源。</p>
<p>因此 "P1/P2 journey 重叠率低" 并不直接是 bug，而是反映了对话信号和浏览信号各自抓住了用户不同侧面的意图。两端联合后才是用户的完整画像。</p>
<p><strong>评估问题：</strong></p>
<ol>
<li>对各自的 history 切片，pipeline 是否充分捕获了用户意图？</li>
<li>给定同一用户的合并意图集合，哪一端的<strong>召回质量</strong>（query 准确性、seller authority、长尾品牌覆盖、价格区间）更优？</li>
<li>能否用 P2 的强项 + P1 的强项混合召回？</li>
</ol>
</blockquote>

<h3>3. Seller Tier 分类（7 档）</h3>
<p>初版分类只用 4 档（luxury/premium/mass/other），把所有垂直专家（Albee Baby / Clive Coffee / Williams Sonoma / B&H / CDW 等）归入 "other"，会得出错误的 "tier 退化" 结论。修订后用 7 档：</p>
''' + tier_table + '''

<h3>4. 评估的两个核心维度</h3>
<ul>
<li><strong>Journey 维度</strong>：Quality（主题准确性）/ Type Diversity（journey 类型多样性）/ Coverage Diversity（品类覆盖）/ Relevance（用户意图吻合度）</li>
<li><strong>Product 维度</strong>：Quality（产品-query 匹配）/ Seller Authority（按场景细分：垂直专业 / 探索百货 / 手作长尾）/ Diversity（seller/brand 多样性）/ Relevance（价格/偏好执行）</li>
</ul>

<h3>5. Per-user 分析的 7 章结构</h3>
<ul>
<li>0. 数据校验（与 paired_data 预计算值比对）</li>
<li>1. 用户历史对比</li>
<li>2. Journey 语义匹配（HIGH/MEDIUM/NO MATCH 矩阵）</li>
<li>3. 相似 Journey 深入对比</li>
<li>4. 差异归因（L1-L4）</li>
<li>5. 关键发现（5.1 P1 优势 / P2 差距 + 5.2 P2 优势 / P1 差距，对称呈现）</li>
<li>6. 改进建议</li>
<li>7. Seller Authority 根因分析</li>
</ul>
'''

    # Overview KPIs
    overview_html = f'''
<h2>核心 KPI</h2>
<div class="kpi-grid">
  <div class="kpi info"><div class="label">Deep 用户数</div><div class="value">{total_users}</div><div class="sub">两端均有 journey</div></div>
  <div class="kpi info"><div class="label">P1 Journey 总数</div><div class="value">{total_p1}</div><div class="sub">人均 {total_p1/max(total_users,1):.1f}</div></div>
  <div class="kpi info"><div class="label">P2 Journey 总数</div><div class="value">{total_p2}</div><div class="sub">人均 {total_p2/max(total_users,1):.1f}</div></div>
  <div class="kpi info"><div class="label">P2 浏览事件总数</div><div class="value">{total_events}</div><div class="sub">recent_events 累计</div></div>
</div>

<h2>Seller Tier 分布对比</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch p1"></div>P1</div>
  <div class="legend-item"><div class="legend-swatch p2"></div>P2</div>
</div>
<div class="tier-grid">{tier_bars}</div>
{tier_table}

<h2>用户卡片（点击查看详情）</h2>
{cards_html}
'''

    # Final HTML
    CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; margin: 0; background: #f6f8fa; color: #1f2328; line-height: 1.55; }
header { background: linear-gradient(135deg, #0969da 0%, #6f42c1 100%); color: #fff; padding: 32px 40px; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
header h1 { margin: 0 0 6px; font-size: 24px; }
header .meta { font-size: 13px; opacity: .9; }
nav { background: #fff; border-bottom: 1px solid #d1d9e0; padding: 0 40px; display: flex; gap: 8px; flex-wrap: wrap; position: sticky; top: 92px; z-index: 99; box-shadow: 0 2px 4px rgba(0,0,0,.04); }
nav button { background: none; border: none; padding: 14px 18px; cursor: pointer; font-size: 14px; font-weight: 600; color: #57606a; border-bottom: 3px solid transparent; transition: all .15s; }
nav button:hover { color: #0969da; }
nav button.active { color: #0969da; border-bottom-color: #0969da; }
nav button.verdict-btn { color: #cf222e; font-weight: 700; }
nav button.verdict-btn.active { color: #cf222e; border-bottom-color: #cf222e; }
main { padding: 28px 40px; max-width: 1400px; margin: 0 auto; }
.tab { display: none; }
.tab.active { display: block; animation: fadein .25s; }
@keyframes fadein { from {opacity:0; transform:translateY(4px);} to {opacity:1; transform:none;} }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 20px 0 28px; }
.kpi { background: #fff; border: 1px solid #d1d9e0; border-radius: 10px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
.kpi .label { font-size: 12px; color: #57606a; text-transform: uppercase; letter-spacing: .5px; }
.kpi .value { font-size: 28px; font-weight: 700; margin: 6px 0; color: #1f2328; }
.kpi .sub { font-size: 12px; color: #656d76; }
.kpi.p0 { border-left: 4px solid #cf222e; }
.kpi.warn { border-left: 4px solid #d4a72c; }
.kpi.good { border-left: 4px solid #1a7f37; }
.kpi.info { border-left: 4px solid #0969da; }
.user-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin: 20px 0; }
.user-card { background: #fff; border: 1px solid #d1d9e0; border-radius: 8px; padding: 14px; cursor: pointer; transition: all .15s; }
.user-card:hover { border-color: #0969da; box-shadow: 0 4px 12px rgba(9,105,218,.15); transform: translateY(-2px); }
.user-card .sid { font-family: ui-monospace, monospace; font-size: 13px; color: #0969da; font-weight: 600; }
.user-card .stats { display: flex; gap: 12px; margin-top: 8px; font-size: 12px; }
.user-card .stat-pill { background: #ddf4ff; color: #0969da; padding: 3px 10px; border-radius: 12px; font-weight: 600; }
.user-card .delta { padding: 3px 10px; border-radius: 12px; font-weight: 600; font-size: 12px; }
.delta.p2-more { background: #dafbe1; color: #1a7f37; }
.delta.p1-more { background: #fff8c5; color: #9a6700; }
.delta.neutral { background: #eaeef2; color: #57606a; }
.user-card .pref { font-size: 11px; color: #656d76; margin-top: 8px; }
.user-link { display: inline-block; margin-top: 8px; color: #0969da; font-size: 12px; font-weight: 600; text-decoration: none; }
.user-link:hover { text-decoration: underline; }
h1 { font-size: 26px; }
h2 { font-size: 20px; border-bottom: 1px solid #d1d9e0; padding-bottom: 6px; margin-top: 28px; }
h3 { font-size: 17px; margin-top: 22px; }
h4 { font-size: 15px; margin-top: 16px; color: #57606a; }
.table-wrap { overflow-x: auto; margin: 12px 0; }
table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,.04); font-size: 13.5px; }
th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid #eaeef2; vertical-align: top; }
th { background: #f6f8fa; font-weight: 600; color: #1f2328; font-size: 12.5px; text-transform: uppercase; letter-spacing: .3px; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f6f8fa; }
blockquote { border-left: 4px solid #0969da; background: #ddf4ff; padding: 10px 16px; margin: 12px 0; border-radius: 0 6px 6px 0; color: #1f2328; }
blockquote p { margin: 6px 0; }
code { background: rgba(175,184,193,.2); padding: 2px 6px; border-radius: 4px; font-size: 12.5px; font-family: ui-monospace, monospace; }
pre.code { background: #1f2328; color: #f0f6fc; padding: 14px 16px; border-radius: 8px; overflow-x: auto; font-size: 12.5px; line-height: 1.5; }
pre.code code { background: none; padding: 0; color: inherit; }
hr { border: none; border-top: 1px solid #d1d9e0; margin: 24px 0; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; letter-spacing: .3px; }
.badge.p0 { background: #cf222e; color: #fff; }
.badge.p1 { background: #d4a72c; color: #1f2328; }
.badge.p2 { background: #57606a; color: #fff; }
.ok { color: #1a7f37; font-weight: 700; }
.bad { color: #cf222e; font-weight: 700; }
.bar-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 13px; }
.bar-row .label { width: 160px; font-weight: 600; }
.bar-row .bar-bg { flex: 1; background: #eaeef2; border-radius: 4px; height: 18px; position: relative; overflow: hidden; }
.bar-row .bar { background: linear-gradient(90deg, #0969da, #6f42c1); height: 100%; border-radius: 4px; transition: width .4s; }
.bar-row .bar.p1 { background: linear-gradient(90deg, #1a7f37, #1a7f37); }
.bar-row .bar.p2 { background: linear-gradient(90deg, #cf222e, #d4a72c); }
.bar-row .pct { width: 70px; text-align: right; font-weight: 600; }
.tier-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }
.tier-block { background: #fff; padding: 16px; border-radius: 8px; border: 1px solid #d1d9e0; }
.tier-block h4 { margin: 0 0 10px; }
.legend { display: flex; gap: 20px; font-size: 12px; margin: 8px 0 16px; color: #57606a; }
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-swatch { width: 14px; height: 14px; border-radius: 3px; }
.legend-swatch.p1 { background: linear-gradient(90deg, #1a7f37, #1a7f37); }
.legend-swatch.p2 { background: linear-gradient(90deg, #cf222e, #d4a72c); }
.user-detail { display: none; padding-top: 12px; }
.user-detail.active { display: block; }
.user-tabs { display: flex; gap: 4px; flex-wrap: wrap; margin: 16px 0 8px; padding-bottom: 12px; border-bottom: 1px solid #d1d9e0; }
.user-tabs button { background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 6px; padding: 6px 12px; font-family: ui-monospace, monospace; font-size: 12px; font-weight: 600; color: #57606a; cursor: pointer; }
.user-tabs button:hover { background: #ddf4ff; color: #0969da; }
.user-tabs button.active { background: #0969da; color: #fff; border-color: #0969da; }
td.p1-win { background: #dafbe1; color: #1a7f37; }
td.p2-win { background: #ddf4ff; color: #0969da; }
td.neutral-win { background: #fff8c5; color: #9a6700; }
footer { padding: 20px 40px; color: #656d76; font-size: 12px; text-align: center; }
"""

    JS = """
function showTab(name, btn) {
  document.querySelectorAll('main > .tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelectorAll('nav > button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function showUser(sid, btn) {
  document.querySelectorAll('.user-detail').forEach(t => t.classList.remove('active'));
  document.getElementById('user-' + sid).classList.add('active');
  document.querySelectorAll('.user-tabs button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('user-' + sid).scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function jumpToUser(sid) {
  showTab('users', document.getElementById('nav-users'));
  setTimeout(() => {
    const btns = document.querySelectorAll('.user-tabs button');
    for (const b of btns) if (b.textContent === sid) { b.click(); break; }
  }, 80);
}
"""

    run_name = os.path.basename(run_dir)
    html_out = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Shopping Pipeline 对比报告</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Shopping Pipeline 对比分析报告</h1>
  <div class="meta">{run_name} · {total_users} Deep 用户</div>
</header>
<nav>
  <button class="active" onclick="showTab('overview', this)">概览</button>
  <button class="verdict-btn" onclick="showTab('verdict', this)">⭐ 最终结论</button>
  <button onclick="showTab('gap', this)">差距分析</button>
  <button onclick="showTab('summary', this)">综合总结</button>
  <button id="nav-users" onclick="showTab('users', this)">逐用户深入分析</button>
  <button onclick="showTab('seller', this)">Seller / 价格分析</button>
  <button onclick="showTab('method', this)">📋 评估方法</button>
</nav>
<main>
<section id="tab-overview" class="tab active">{overview_html}</section>
<section id="tab-verdict" class="tab">{verdict_html}</section>
<section id="tab-gap" class="tab">{gap_html}</section>
<section id="tab-summary" class="tab">{summary_html}</section>
<section id="tab-users" class="tab">
  <h2>逐用户深入分析</h2>
  <p>选择一个用户 ID 查看完整 7 章分析</p>
  {user_tabs_html}
  {user_details_html}
</section>
<section id="tab-seller" class="tab">
  <h2>Seller Tier 分布</h2>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch p1"></div>P1</div>
    <div class="legend-item"><div class="legend-swatch p2"></div>P2</div>
  </div>
  <div class="tier-grid">{tier_bars}</div>
  {tier_table}
  <h2>Top Sellers</h2>
  {top_sellers_html}
</section>
<section id="tab-method" class="tab">{method_html}</section>
</main>
<footer>shopping-journey-pipeline-compare · {total_users} Deep users</footer>
<script>{JS}</script>
<script>
document.addEventListener('DOMContentLoaded', () => {{
  const first = document.querySelector('.user-tabs button');
  if (first) first.click();
}});
</script>
</body>
</html>
'''

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"Wrote {out_html}")
    print(f"Size: {os.path.getsize(out_html)/1024:.1f} KB")


if __name__ == "__main__":
    main()
