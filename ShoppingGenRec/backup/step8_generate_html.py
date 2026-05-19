#!/usr/bin/env python3
"""Rule-based L3 reranker on top of SLM-ranker output (full JSONL).

Reads the flat per-journey product list (sorted by SLM `Rank`) produced by
`step1_run_slm_ranker_v3.py` and reorders within each journey to improve
diversity in the visible Top-K, applying the following rules:

  (1) same OriginalQuery: at most ``--max_per_query_topk`` products in the
      first ``--top_k`` positions (default: 6 in 12).
  (2) same Seller       : at most ``--max_per_seller_total`` products kept
      in the "primary" list (default: 4). Surplus same-Seller items are
      *demoted* to the tail (kept in JSONL, but pushed past Top-K).
  (3) same Seller       : at most ``--max_consecutive_seller`` adjacent
      positions (default: 2). e.g. Rank=1,2,3 share Seller-A and Rank=4,5
      share Seller-B  ->  reorder to 1,2,4,3,5.
  (4) same Brand        : symmetric to (2)+(3) — same-Brand blocks look
      almost as repetitive as same-Seller blocks.
  (5) excluded-Seller safety filter: drops sellers in
      {shein, temu, ebay, aliexpress, wish, dhgate} as a defensive pass
      (the SLM should already exclude these per its prompt).

Constraints are applied with greedy "look-ahead within remaining pool":
at each output slot, scan the remaining (rank-sorted) candidates and pick
the *first* one that satisfies all live constraints. If none fits, the
constraints are relaxed in order: consecutive-Seller -> consecutive-Brand
-> per-query cap -> take next. This guarantees we never silently drop
items, only reorder them.

Outputs:
  results/<prefix>.jsonl  – same shape as input, products reordered.
                           Each product carries its original `Rank` and a
                           new `DisplayPosition`. Stats block gets a
                           `_L3Rerank` sub-dict with diagnostic counts.
  results/<prefix>.html   – per-user collapsible AND per-journey
                           collapsible view. Product card layout puts
                           Seller below Title (Brand below price).

Usage:
    python step2_run_ruleBase_L3.py
    python step2_run_ruleBase_L3.py --input results/foo_full.jsonl
    python step2_run_ruleBase_L3.py --top_k 12 --max_per_seller_total 4
"""

import argparse
import html as html_module
import json
import os
import sys
from collections import Counter

# ── Defaults ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(
    _HERE, "results",
    "exp_offline_ranker_expandMore0504Index_v3_4bModel_SLM_rankerV3_full.jsonl",
)
DEFAULT_OUTPUT_PREFIX = (
    "exp_offline_ranker_expandMore0504Index_v3_4bModel_SLM_rankerV3_L3"
)

EXCLUDED_SELLERS_LOWER = {
    "shein", "temu", "ebay", "aliexpress", "wish", "dhgate",
}
PLACEHOLDER_IMG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1'/>"
)


# ═══════════════════════════════════════════════════════════════════════════
#  Reranker
# ═══════════════════════════════════════════════════════════════════════════

def _rank_key(p):
    r = p.get("Rank")
    return r if isinstance(r, int) else 10**9


def _max_run(seq):
    """Length of the longest consecutive run of identical (truthy) values."""
    best = cur = 0
    last = None
    for v in seq:
        if v and v == last:
            cur += 1
        else:
            cur = 1 if v else 0
        last = v
        if cur > best:
            best = cur
    return best


def filter_excluded_sellers(products):
    kept, dropped = [], []
    for p in products:
        s = (p.get("Seller", "") or "").strip().lower()
        # Match if any excluded token appears as a whole word/substring of
        # the seller name (e.g., "Shein US" -> shein).
        if any(tok in s for tok in EXCLUDED_SELLERS_LOWER):
            dropped.append(p)
        else:
            kept.append(p)
    return kept, dropped


def rerank_journey(
    products,
    top_k=12,
    max_per_query_topk=6,
    max_per_seller_total=4,
    max_consecutive_seller=2,
    max_per_brand_total=4,
    max_consecutive_brand=2,
):
    """Rerank a single journey's product list.

    Returns: (final_list, stats_dict)
      final_list – reordered list (no items removed unless excluded-seller).
                   Each item gets a new key ``DisplayPosition`` (1-based).
      stats_dict – diagnostic counts for the journey-level summary.
    """
    if not products:
        return [], {
            "input_n": 0, "kept_n": 0, "excluded_seller_dropped": 0,
            "demoted_n": 0,
            "consec_seller_before": 0, "consec_seller_after": 0,
            "consec_brand_before": 0, "consec_brand_after": 0,
            "topk_unique_sellers_before": 0, "topk_unique_sellers_after": 0,
        }

    # 0) Sort by SLM Rank (stable for ties).
    products = sorted(products, key=_rank_key)
    n_in = len(products)

    # 0a) Excluded-seller safety filter.
    products, excluded = filter_excluded_sellers(products)

    # Pre-stats on the original (rank-sorted) Top-K.
    orig_top = products[:top_k]
    orig_consec_seller = _max_run([p.get("Seller", "") for p in orig_top])
    orig_consec_brand = _max_run([p.get("Brand", "") for p in orig_top])
    orig_topk_unique_sellers = len({
        p.get("Seller", "") for p in orig_top if p.get("Seller", "")
    })

    # 1) Soft cap per-Seller / per-Brand totals: items beyond the cap are
    #    "demoted" to the tail. The cap is RELAXED if the candidate pool is
    #    not large enough to fill the visible Top-K window — we promote the
    #    best-Rank demoted items back so the primary pool always has at
    #    least min(top_k, n_after_excluded) items.
    seller_cnt = Counter()
    brand_cnt = Counter()
    primary, demoted = [], []
    for p in products:
        s = p.get("Seller", "")
        b = p.get("Brand", "")
        over_s = bool(s) and seller_cnt[s] >= max_per_seller_total
        over_b = bool(b) and brand_cnt[b] >= max_per_brand_total
        if over_s or over_b:
            demoted.append(p)
        else:
            primary.append(p)
            if s:
                seller_cnt[s] += 1
            if b:
                brand_cnt[b] += 1

    # 1b) Candidate-aware relaxation. If we don't have enough candidates to
    #     fill Top-K under the totals caps, promote the best-Rank demoted
    #     items back so Top-K stays full. (`demoted` is already in Rank
    #     order because we iterated `products` in Rank order above.)
    target_primary = min(top_k, len(products))
    if len(primary) < target_primary and demoted:
        n_promote = target_primary - len(primary)
        promoted = demoted[:n_promote]
        demoted = demoted[n_promote:]
        primary.extend(promoted)
        primary.sort(key=_rank_key)  # restore Rank order before greedy

    # 2) Greedy reorder of `primary` to enforce per-query cap (in Top-K only)
    #    and consecutive-Seller / consecutive-Brand caps. We never drop
    #    items here — at worst we relax constraints in order.
    output = []
    pool = list(primary)  # already rank-sorted
    query_cnt_topk = Counter()
    last_seller = None
    last_brand = None
    consec_s = 0
    consec_b = 0

    def pick(pool, predicate):
        for i, p in enumerate(pool):
            if predicate(p):
                return i
        return -1

    while pool:
        in_topk = len(output) < top_k

        def fully_ok(p):
            s = p.get("Seller", "")
            b = p.get("Brand", "")
            q = p.get("OriginalQuery", "")
            if in_topk and q and query_cnt_topk[q] >= max_per_query_topk:
                return False
            if s and s == last_seller and consec_s >= max_consecutive_seller:
                return False
            if b and b == last_brand and consec_b >= max_consecutive_brand:
                return False
            return True

        idx = pick(pool, fully_ok)

        if idx == -1:
            # Relax 1: ignore consecutive-Brand cap
            def no_consec_brand(p):
                s = p.get("Seller", "")
                q = p.get("OriginalQuery", "")
                if in_topk and q and query_cnt_topk[q] >= max_per_query_topk:
                    return False
                if s and s == last_seller and consec_s >= max_consecutive_seller:
                    return False
                return True
            idx = pick(pool, no_consec_brand)

        if idx == -1:
            # Relax 2: drop the per-query cap (insufficient query diversity)
            #          but keep consecutive-Seller — visual quality matters
            #          most for adjacent items.
            def no_query(p):
                s = p.get("Seller", "")
                if s and s == last_seller and consec_s >= max_consecutive_seller:
                    return False
                return True
            idx = pick(pool, no_query)

        if idx == -1:
            # Relax 3: drop consecutive-Seller too (pool is dominated by a
            #          single Seller). At this point all rules are off.
            idx = 0

        p = pool.pop(idx)
        output.append(p)
        s = p.get("Seller", "")
        b = p.get("Brand", "")
        q = p.get("OriginalQuery", "")
        consec_s = consec_s + 1 if s and s == last_seller else (1 if s else 0)
        consec_b = consec_b + 1 if b and b == last_brand else (1 if b else 0)
        last_seller = s if s else None
        last_brand = b if b else None
        if in_topk and q:
            query_cnt_topk[q] += 1

    # 3) Append demoted items (still useful, just past Top-K).
    final = output + demoted

    # 4) Annotate display positions. Preserve the original SLM rank under
    #    `OriginalSLMRank`, then renumber `Rank` to match the L3 order so
    #    downstream consumers reading `Rank` see the reranked positions.
    for i, p in enumerate(final, start=1):
        if "OriginalSLMRank" not in p:
            p["OriginalSLMRank"] = p.get("Rank")
        p["Rank"] = i
        p["DisplayPosition"] = i

    # Post-stats on the new Top-K.
    new_top = final[:top_k]
    new_consec_seller = _max_run([p.get("Seller", "") for p in new_top])
    new_consec_brand = _max_run([p.get("Brand", "") for p in new_top])
    new_topk_unique_sellers = len({
        p.get("Seller", "") for p in new_top if p.get("Seller", "")
    })

    stats = {
        "input_n": n_in,
        "kept_n": len(final),
        "excluded_seller_dropped": len(excluded),
        "demoted_n": len(demoted),
        "consec_seller_before": orig_consec_seller,
        "consec_seller_after": new_consec_seller,
        "consec_brand_before": orig_consec_brand,
        "consec_brand_after": new_consec_brand,
        "topk_unique_sellers_before": orig_topk_unique_sellers,
        "topk_unique_sellers_after": new_topk_unique_sellers,
    }
    return final, stats


# ═══════════════════════════════════════════════════════════════════════════
#  HTML rendering
# ═══════════════════════════════════════════════════════════════════════════

def esc(text):
    return html_module.escape(str(text))


def render_product_card(p, position):
    """Card layout: Title -> Seller (prominent) -> Price -> Brand (small).

    Original SLM `Rank` is shown as a secondary badge next to the
    display-position rank.
    """
    pid = p.get("global_offer_id", "")
    title = p.get("Title", "—") or "—"
    seller = p.get("Seller", "")
    brand = p.get("Brand", "")
    price = p.get("OriginalPrice", "")
    image_url = p.get("ImageUrl", "") or PLACEHOLDER_IMG
    offer_url = p.get("OfferUrl", "")
    orig_query = p.get("OriginalQuery", "")
    orig_rank = p.get("OriginalSLMRank", p.get("Rank"))
    ann_score = p.get("ANNScore", None)
    gender = (p.get("Gender", "") or "").strip()
    age_group = (p.get("AgeGroup", "") or "").strip()

    link_o = (
        f'<a href="{esc(offer_url)}" target="_blank" rel="noopener">'
        if offer_url else ""
    )
    link_c = "</a>" if offer_url else ""

    orig_badge = (
        f'<span class="orig-rank" title="Original SLM rank">orig #{orig_rank}</span>'
        if isinstance(orig_rank, int) else ""
    )

    chips = []
    if gender:
        chips.append(f'<span class="attr-chip attr-gender">{esc(gender)}</span>')
    if age_group:
        chips.append(f'<span class="attr-chip attr-age">{esc(age_group)}</span>')
    attrs_html = (
        f'<div class="product-attrs">{"".join(chips)}</div>' if chips else ""
    )

    return f"""<div class="product-card">
  <div class="rank-row">
    <span class="product-rank">#{position}</span>
    {orig_badge}
  </div>
  {link_o}<div class="product-img-wrap">
    <img src="{esc(image_url)}" alt="{esc(title)}" loading="lazy"
         onerror="this.src='{PLACEHOLDER_IMG}'">
  </div>{link_c}
  <div class="product-info">
    <div class="product-title" title="{esc(title)}">{esc(title[:80])}</div>
    {f'<div class="product-seller">{esc(seller)}</div>' if seller else ''}
    {f'<div class="product-price">{esc(price)}</div>' if price else ''}
    {attrs_html}
    {f'<div class="product-brand">{esc(brand)}</div>' if brand else ''}
    {f'<div class="product-query">Q: {esc(orig_query[:50])}</div>' if orig_query else ''}
    {f'<div class="product-ann-score">ANN: {ann_score:.4f}</div>' if isinstance(ann_score, (int, float)) and ann_score else ''}
  </div>
</div>"""


def render_journey_block(j, ji, top_k):
    title = j.get("title", "Untitled") or "Untitled"
    jtype = j.get("journeyType", "explicit") or "explicit"
    desc = j.get("description", "") or ""
    reason = j.get("reason", "") or ""
    cs = j.get("conversationStarter", "") or ""
    products = j.get("products", []) or []
    stats = j.get("stats", {}) or {}
    rr = stats.get("_L3Rerank", {}) or {}
    type_cls = "tag-explicit" if jtype == "explicit" else "tag-related"

    # Recommended products (Top-K of the reordered list)
    visible = products[:top_k]
    cards = "".join(
        render_product_card(p, i + 1) for i, p in enumerate(visible)
    )
    prods_html = (
        f'<div class="prods-section">'
        f'<div class="prods-label">Recommended Products '
        f'(showing {len(visible)} of {len(products)})</div>'
        f'<div class="prods-grid">{cards}</div></div>'
        if visible else
        '<div class="muted">No products</div>'
    )

    # Tail (the rest)
    tail = products[top_k:]
    tail_html = ""
    if tail:
        tail_cards = "".join(
            render_product_card(p, i + 1 + top_k) for i, p in enumerate(tail)
        )
        tail_html = (
            f'<details class="tail-section">'
            f'<summary class="tail-label">More Products ({len(tail)})</summary>'
            f'<div class="prods-grid">{tail_cards}</div></details>'
        )

    cs_html = (
        f'<div class="cs-section"><div class="csl">Conversation Starter</div>'
        f'<div class="cst">&ldquo;{esc(cs)}&rdquo;</div></div>'
        if cs else ""
    )

    # Journey-level Queries chip list, reconstructed from products' OriginalQuery
    # (preserve first-seen order). Lives under Description, like run_demo_e2e.
    seen_q = set()
    unique_queries = []
    for p in products:
        q = p.get("OriginalQuery", "") or ""
        if q and q not in seen_q:
            seen_q.add(q)
            unique_queries.append(q)
    q_html = ""
    if unique_queries:
        chips = "".join(
            f'<span class="query-chip">{esc(q)}</span>' for q in unique_queries
        )
        q_html = (
            f'<div class="queries-section"><div class="ql">Queries</div>'
            f'<div class="qc">{chips}</div></div>'
        )

    # SLM ranker stats (totalCandidates / kept / filtered)
    rsumm_html = ""
    if stats:
        total = stats.get("totalCandidates", 0)
        kept = stats.get("selectedCount", 0)
        filt = stats.get("filteredCount", 0)
        rsumm_html = (
            f'<div class="rsumm">'
            f'<span class="rs">SLM Total: {total}</span>'
            f'<span class="rs rsk">Kept: {kept}</span>'
            f'<span class="rs rsf">Filtered: {filt}</span>'
            f'</div>'
        )

    # L3 rerank diagnostics
    l3_html = ""
    if rr:
        l3_html = (
            f'<div class="l3summ" title="Rule-based L3 reranker diagnostics">'
            f'<span class="l3">L3 in: {rr.get("input_n", 0)}</span>'
            f'<span class="l3 l3d">demoted: {rr.get("demoted_n", 0)}</span>'
            f'<span class="l3 l3x">excluded: {rr.get("excluded_seller_dropped", 0)}</span>'
            f'<span class="l3">consec-Seller: '
            f'{rr.get("consec_seller_before", 0)}→{rr.get("consec_seller_after", 0)}</span>'
            f'<span class="l3">consec-Brand: '
            f'{rr.get("consec_brand_before", 0)}→{rr.get("consec_brand_after", 0)}</span>'
            f'<span class="l3">unique-Sellers in Top-{top_k}: '
            f'{rr.get("topk_unique_sellers_before", 0)}→{rr.get("topk_unique_sellers_after", 0)}</span>'
            f'</div>'
        )

    # Wrap each journey in a collapsible <details>. Default collapsed
    # so the user doesn't get a wall of products on first load.
    summary_meta = (
        f'<span class="ji">#{ji}</span>'
        f'<span class="jt {type_cls}">{esc(jtype)}</span>'
        f'<span class="jtitle">{esc(title)}</span>'
        f'<span class="jct">{len(products)} products</span>'
    )

    body = (
        (f'<div class="jd">{esc(desc)}</div>' if desc else '')
        + q_html
        + cs_html
        + (f'<div class="jr"><span class="jrl">Why am I seeing this</span> '
           f'{esc(reason)}</div>' if reason else '')
        + rsumm_html + l3_html + prods_html + tail_html
    )

    return (
        f'<details class="journey-card"><summary class="jh">{summary_meta}</summary>'
        f'<div class="jbody">{body}</div></details>'
    )


def render_profile_html(profile_obj):
    if not profile_obj:
        return '<span class="muted">No profile</span>'
    data = profile_obj.get("userShoppingProfile", profile_obj)
    if not isinstance(data, dict):
        return (
            f'<pre class="praw">'
            f'{esc(json.dumps(profile_obj, indent=2)[:500])}</pre>'
        )
    parts = []
    for k, v in data.items():
        if isinstance(v, list):
            disp = ", ".join(str(x) for x in v)
        else:
            disp = str(v).strip() or "—"
        parts.append(
            f'<div class="prow"><span class="pk">{esc(k)}</span>'
            f'<span class="pv">{esc(disp)}</span></div>'
        )
    return "\n".join(parts) if parts else '<span class="muted">Empty</span>'


def render_user_block(u, ui, top_k):
    uid = u.get("stableid", "") or ""
    journeys = u.get("journeys", []) or []
    profile = u.get("userShoppingProfile", {})
    events_text = u.get("recentShoppingEvents", "") or ""
    events = [e.strip() for e in events_text.split("\n") if e.strip()]

    nj = len(journeys)
    j_cards = "".join(
        render_journey_block(j, i + 1, top_k) for i, j in enumerate(journeys)
    ) or '<div class="muted">No journeys</div>'

    ev_rows = "".join(f'<div class="ev">{esc(e)}</div>' for e in events) \
        or '<div class="muted">No events</div>'
    prof_html = render_profile_html(profile)

    return f"""<div class="ub">
  <details>
    <summary class="us">
      <span class="ui">User {ui}</span>
      <span class="uid">{esc(uid[:16])}…</span>
      <span class="ub2">{len(events)} events</span>
      <span class="ub2 ubj">{nj} journeys</span>
    </summary>
    <div class="uc">
      <details class="sd"><summary class="st"><span>📋 Events</span><span class="sc">{len(events)}</span></summary><div class="ec">{ev_rows}</div></details>
      <details class="sd"><summary class="st"><span>👤 Shopping Profile</span></summary><div class="pc">{prof_html}</div></details>
      <div class="sj"><div class="sts"><span>🛒 Shopping Journeys</span><span class="sc">{nj}</span></div><div class="jg">{j_cards}</div></div>
    </div>
  </details>
</div>"""


HTML_CSS = """
:root{--bg:#f5f7fa;--cb:#fff;--bd:#e2e8f0;--tx:#1a202c;--t2:#4a5568;--mt:#a0aec0;
--ac:#4f46e5;--al:#eef2ff;--ex:#059669;--exb:#ecfdf5;--rel:#7c3aed;--rlb:#f5f3ff;
--qb:#f0f9ff;--qbd:#bae6fd;--sb:#fffbeb;--r:12px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--tx);line-height:1.6;padding:2rem;max-width:1400px;margin:0 auto}
h1{font-size:1.8rem;font-weight:700;margin-bottom:.5rem;
background:linear-gradient(135deg,var(--ac),#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:var(--t2);margin-bottom:2rem;font-size:.95rem}
.ctrl{display:flex;gap:.5rem;margin-bottom:1.5rem;flex-wrap:wrap}
.ctrl button{padding:.4rem 1rem;border-radius:8px;border:1px solid var(--bd);background:var(--cb);cursor:pointer;font-size:.85rem;transition:all .15s}
.ctrl button:hover{background:var(--al);border-color:var(--ac)}
.ub{background:var(--cb);border-radius:var(--r);border:1px solid var(--bd);margin-bottom:1rem;box-shadow:0 1px 3px rgba(0,0,0,.04);transition:box-shadow .2s}
.ub:hover{box-shadow:0 4px 12px rgba(0,0,0,.08)}
.us{padding:1rem 1.25rem;cursor:pointer;display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;list-style:none}
.us::-webkit-details-marker{display:none}
.us::before{content:'▶';font-size:.7rem;color:var(--mt);transition:transform .2s}
details[open]>.us::before{transform:rotate(90deg)}
.ui{font-weight:700;font-size:1rem;color:var(--ac);min-width:65px}
.uid{font-family:'SF Mono',Monaco,monospace;font-size:.8rem;color:var(--mt);background:var(--bg);padding:.15rem .5rem;border-radius:6px}
.ub2{font-size:.78rem;padding:.2rem .6rem;border-radius:20px;background:var(--bg);color:var(--t2);border:1px solid var(--bd)}
.ubj{background:var(--al);color:var(--ac);border-color:#c7d2fe}
.uc{padding:0 1.25rem 1.25rem}
.sd{margin-bottom:1rem}
.st{font-weight:600;font-size:.95rem;padding:.6rem 0;cursor:pointer;display:flex;align-items:center;gap:.5rem;border-bottom:1px solid var(--bd);margin-bottom:.5rem;list-style:none}
.st::-webkit-details-marker{display:none}
.sts{font-weight:600;font-size:.95rem;padding:.6rem 0;display:flex;align-items:center;gap:.5rem;border-bottom:1px solid var(--bd);margin-bottom:.75rem}
.sc{font-size:.75rem;background:var(--bg);color:var(--t2);padding:.1rem .5rem;border-radius:10px;margin-left:auto}
.ec{max-height:400px;overflow-y:auto;padding:.5rem;background:var(--bg);border-radius:8px}
.ev{font-size:.82rem;padding:.35rem .6rem;border-radius:4px;font-family:'SF Mono',Monaco,monospace;color:var(--t2);border-bottom:1px solid #f0f0f0}
.ev:nth-child(odd){background:rgba(255,255,255,.6)}
.pc{padding:.5rem}
.prow{display:flex;gap:.75rem;padding:.3rem 0;border-bottom:1px dashed #f0f0f0;font-size:.88rem}
.pk{font-weight:600;color:var(--ac);min-width:220px;flex-shrink:0;font-size:.82rem}
.pv{color:var(--t2)}
.praw{font-size:.8rem;background:var(--bg);padding:.75rem;border-radius:8px;overflow-x:auto}
.jg{display:flex;flex-direction:column;gap:.6rem}
/* Journey cards: collapsible <details> */
.journey-card{border:1px solid var(--bd);border-radius:10px;background:var(--cb);transition:box-shadow .15s;overflow:hidden}
.journey-card:hover{box-shadow:0 4px 12px rgba(0,0,0,.06)}
.jh{padding:.75rem 1rem;cursor:pointer;display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;list-style:none;background:#fafbff}
.jh::-webkit-details-marker{display:none}
.jh::before{content:'▶';font-size:.65rem;color:var(--mt);transition:transform .2s}
.journey-card[open]>.jh::before{transform:rotate(90deg)}
.journey-card[open]>.jh{border-bottom:1px solid var(--bd)}
.jbody{padding:1rem}
.ji{font-weight:700;font-size:.8rem;color:var(--mt);width:24px}
.jt{font-size:.7rem;font-weight:600;padding:.15rem .5rem;border-radius:6px;text-transform:uppercase;letter-spacing:.5px}
.tag-explicit{background:var(--exb);color:var(--ex)}
.tag-related{background:var(--rlb);color:var(--rel)}
.jtitle{font-weight:600;font-size:1rem;flex:1}
.jct{font-size:.72rem;color:var(--mt);background:var(--bg);padding:.15rem .55rem;border-radius:10px;border:1px solid var(--bd)}
.jd{font-size:.88rem;color:var(--t2);margin-bottom:.5rem;line-height:1.5}
.queries-section{margin-bottom:.5rem}
.ql{font-size:.75rem;font-weight:600;color:#0369a1;margin-bottom:.3rem}
.qc{display:flex;flex-wrap:wrap;gap:.4rem}
.query-chip{font-size:.78rem;padding:.2rem .6rem;border-radius:6px;background:var(--qb);border:1px solid var(--qbd);color:#0369a1}
.cs-section{background:var(--sb);border-radius:8px;padding:.6rem .8rem;margin-bottom:.5rem}
.csl{font-size:.75rem;font-weight:600;color:#92400e;margin-bottom:.3rem}
.cst{font-size:.85rem;color:#78350f;font-style:italic;line-height:1.5}
.jr{font-size:.88rem;color:var(--t2);line-height:1.5;padding:.5rem .75rem;margin-top:.4rem;background:#f8fafc;border-radius:8px;border-left:3px solid var(--ac)}
.jrl{font-size:.75rem;font-weight:600;color:var(--ac);margin-right:.4rem;text-transform:uppercase;letter-spacing:.3px}
.rsumm,.l3summ{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;margin:.4rem 0;padding:.4rem .6rem;background:#f8fafc;border-radius:8px}
.rs,.l3{font-size:.78rem;padding:.15rem .5rem;border-radius:6px;background:var(--bg)}
.rsk{background:#ecfdf5;color:#059669}
.rsf{background:#fef2f2;color:#dc2626}
.l3summ{background:#fff7ed}
.l3{background:#fff;border:1px solid #fed7aa;color:#9a3412}
.l3d{background:#fff7ed;color:#92400e}
.l3x{background:#fef2f2;color:#dc2626;border-color:#fecaca}
.prods-section{margin-top:.75rem}
.prods-label{font-size:.8rem;font-weight:600;color:#059669;margin-bottom:.5rem}
.prods-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:.75rem}
.product-card{border:1px solid var(--bd);border-radius:10px;overflow:hidden;background:var(--cb);transition:transform .15s,box-shadow .15s;position:relative}
.product-card:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(0,0,0,.1)}
.rank-row{position:absolute;top:6px;left:6px;display:flex;gap:.3rem;z-index:1}
.product-rank{background:var(--ac);color:#fff;font-size:.7rem;font-weight:700;padding:.1rem .4rem;border-radius:6px}
.orig-rank{background:rgba(255,255,255,.92);color:var(--t2);font-size:.62rem;font-weight:600;padding:.1rem .35rem;border-radius:6px;border:1px solid var(--bd)}
.product-img-wrap{width:100%;height:170px;overflow:hidden;background:#f8f8f8;display:flex;align-items:center;justify-content:center}
.product-img-wrap img{max-width:100%;max-height:170px;object-fit:contain}
.product-info{padding:.6rem}
.product-title{font-size:.82rem;font-weight:600;line-height:1.3;margin-bottom:.3rem;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.product-seller{font-size:.78rem;color:var(--ac);font-weight:600;margin-bottom:.25rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.product-price{font-size:.9rem;font-weight:700;color:#dc2626;margin-bottom:.2rem}
.product-brand{font-size:.7rem;color:var(--mt);margin-bottom:.15rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.product-attrs{display:flex;flex-wrap:wrap;gap:.25rem;margin:.15rem 0 .25rem}
.attr-chip{font-size:.65rem;font-weight:600;padding:.08rem .4rem;border-radius:10px;text-transform:capitalize;letter-spacing:.2px;line-height:1.3}
.attr-gender{background:#fce7f3;color:#9d174d;border:1px solid #fbcfe8}
.attr-age{background:#dcfce7;color:#166534;border:1px solid #bbf7d0}
.product-query{font-size:.68rem;color:var(--mt);margin-top:.15rem;font-style:italic;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.product-ann-score{font-size:.68rem;color:#2563eb;margin-top:.15rem;font-weight:600}
.tail-section{margin-top:.6rem;border:1px dashed var(--bd);border-radius:8px;padding:.5rem .6rem;background:#fafbff}
.tail-label{font-size:.78rem;font-weight:600;color:var(--t2);cursor:pointer;list-style:none;padding:.15rem 0}
.tail-label::-webkit-details-marker{display:none}
.tail-section[open]>.tail-label{margin-bottom:.5rem;color:var(--ac)}
.muted{color:var(--mt);font-style:italic;font-size:.9rem}
"""


def render_html(users, output_path, top_k):
    page_title = os.path.splitext(os.path.basename(output_path))[0]
    n = len(users)
    blocks = "\n".join(
        render_user_block(u, i + 1, top_k) for i, u in enumerate(users)
    )
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{page_title} ({n} users)</title>
<style>{HTML_CSS}</style></head>
<body>
<h1>🛍️ {page_title}</h1>
<p class="sub">{n} users · Rule-based L3 reranker on SLM ranker output · Top-{top_k} diversity-shaped</p>
<div class="ctrl">
  <button onclick="document.querySelectorAll('details').forEach(d=>d.open=true)">Expand All</button>
  <button onclick="document.querySelectorAll('details').forEach(d=>d.open=false)">Collapse All</button>
  <button onclick="document.querySelectorAll('.journey-card').forEach(d=>d.open=true)">Expand Journeys</button>
  <button onclick="document.querySelectorAll('.journey-card').forEach(d=>d.open=false)">Collapse Journeys</button>
</div>
{blocks}
</body></html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)
    sz = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved HTML: {output_path} ({sz:.2f} MB)")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Rule-based L3 reranker")
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help="Input full JSONL from step1 SLM ranker")
    ap.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX,
                    help="Output prefix (relative to results/)")
    ap.add_argument("--results_dir", default=os.path.join(_HERE, "results"))
    ap.add_argument("--top_k", type=int, default=12,
                    help="Visible Top-K window (constraints apply here)")
    ap.add_argument("--max_per_query_topk", type=int, default=6)
    ap.add_argument("--max_per_seller_total", type=int, default=4)
    ap.add_argument("--max_consecutive_seller", type=int, default=2)
    ap.add_argument("--max_per_brand_total", type=int, default=4)
    ap.add_argument("--max_consecutive_brand", type=int, default=2)
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 70)
    print("  Rule-based L3 Reranker")
    print("=" * 70)
    print(f"  Input         : {args.input}")
    print(f"  Top-K window  : {args.top_k}")
    print(f"  per-query  cap: {args.max_per_query_topk} (in Top-{args.top_k})")
    print(f"  per-seller cap: {args.max_per_seller_total} total / "
          f"{args.max_consecutive_seller} consecutive")
    print(f"  per-brand  cap: {args.max_per_brand_total} total / "
          f"{args.max_consecutive_brand} consecutive")
    print()

    # Load all users
    users = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            users.append(json.loads(line))
    print(f"Loaded {len(users)} users")

    # Rerank every journey
    n_journeys = 0
    sum_consec_s_before = sum_consec_s_after = 0
    sum_consec_b_before = sum_consec_b_after = 0
    sum_uniq_s_before = sum_uniq_s_after = 0
    sum_excluded = sum_demoted = 0

    for u in users:
        for j in u.get("journeys", []):
            products = j.get("products", []) or []
            new_list, rr_stats = rerank_journey(
                products,
                top_k=args.top_k,
                max_per_query_topk=args.max_per_query_topk,
                max_per_seller_total=args.max_per_seller_total,
                max_consecutive_seller=args.max_consecutive_seller,
                max_per_brand_total=args.max_per_brand_total,
                max_consecutive_brand=args.max_consecutive_brand,
            )
            j["products"] = new_list
            stats = j.setdefault("stats", {})
            if not isinstance(stats, dict):
                stats = {}
                j["stats"] = stats
            stats["_L3Rerank"] = rr_stats

            n_journeys += 1
            sum_consec_s_before += rr_stats["consec_seller_before"]
            sum_consec_s_after += rr_stats["consec_seller_after"]
            sum_consec_b_before += rr_stats["consec_brand_before"]
            sum_consec_b_after += rr_stats["consec_brand_after"]
            sum_uniq_s_before += rr_stats["topk_unique_sellers_before"]
            sum_uniq_s_after += rr_stats["topk_unique_sellers_after"]
            sum_excluded += rr_stats["excluded_seller_dropped"]
            sum_demoted += rr_stats["demoted_n"]

    if n_journeys:
        print(f"\nReranked {n_journeys} journeys")
        print(f"  Avg max-consecutive Seller in Top-{args.top_k}: "
              f"{sum_consec_s_before/n_journeys:.2f} → "
              f"{sum_consec_s_after/n_journeys:.2f}")
        print(f"  Avg max-consecutive Brand  in Top-{args.top_k}: "
              f"{sum_consec_b_before/n_journeys:.2f} → "
              f"{sum_consec_b_after/n_journeys:.2f}")
        print(f"  Avg unique Sellers         in Top-{args.top_k}: "
              f"{sum_uniq_s_before/n_journeys:.2f} → "
              f"{sum_uniq_s_after/n_journeys:.2f}")
        print(f"  Excluded-seller items dropped: {sum_excluded}")
        print(f"  Items demoted to tail (over caps): {sum_demoted}")

    # Write outputs
    out_jsonl = os.path.join(args.results_dir, f"{args.output_prefix}.jsonl")
    out_html = os.path.join(args.results_dir, f"{args.output_prefix}.html")

    # JSONL: keep only the visible Top-K per journey (trimmed copy so the
    # full list is still available for HTML rendering below).
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for u in users:
            u_out = dict(u)
            u_out["journeys"] = []
            for j in u.get("journeys", []):
                j_out = dict(j)
                j_out["products"] = (j.get("products") or [])[:args.top_k]
                u_out["journeys"].append(j_out)
            f.write(json.dumps(u_out, ensure_ascii=False) + "\n")
    sz = os.path.getsize(out_jsonl) / (1024 * 1024)
    print(f"\n  Saved JSONL (Top-{args.top_k} per journey): "
          f"{out_jsonl} ({sz:.2f} MB)")

    render_html(users, out_html, args.top_k)

    print("\n" + "=" * 70)
    print("  L3 Rerank complete!")
    print(f"  JSONL: {out_jsonl}")
    print(f"  HTML : {out_html}")
    print("=" * 70)


if __name__ == "__main__":
    main()
