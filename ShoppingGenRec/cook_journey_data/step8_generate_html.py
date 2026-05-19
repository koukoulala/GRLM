#!/usr/bin/env python3
"""Rule-based L3 reranker + HTML visualization.

Supports two input formats:
  1. Legacy JSONL (from original pipeline)
  2. Step6 Ranked TSV (from our cook_journey_data pipeline)

When reading Step6 TSV, the script:
  - Parses RankedJourneys JSON column
  - Maps PascalCase fields to the internal camelCase format
  - Loads ImageUrl / OfferUrl from item.json (--item_json)
  - Falls back to reading the original index TSV (--original_index_file)
    when item.json doesn't have URL fields

Outputs:
  results/<prefix>.jsonl  – reranked, same shape as original JSONL
  results/<prefix>.html   – per-user/per-journey collapsible view

Usage:
    # From step6 TSV:
    python step8_generate_html.py --input results/XXX_Ranked.tsv --item_json .../item.json

    # From original JSONL:
    python step8_generate_html.py --input results/foo_full.jsonl

    # With original index TSV fallback for URLs:
    python step8_generate_html.py --input results/XXX_Ranked.tsv \\
        --item_json .../item.json --original_index_file .../ProductBestOffer.tsv
"""

import argparse
import csv
import html as html_module
import json
import os
import sys
import time
from collections import Counter
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

# ── Defaults ────────────────────────────────────────────────────────────────


EXCLUDED_SELLERS_LOWER = {
    "shein", "temu", "ebay", "aliexpress", "wish", "dhgate",
}
PLACEHOLDER_IMG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1'/>"
)


# ═══════════════════════════════════════════════════════════════════════════
#  URL Lookup: item.json + fallback to original index TSV
# ═══════════════════════════════════════════════════════════════════════════

def load_urls_from_item_json(item_json_path):
    """Load offer_url and image_url from item.json.

    Returns dict: {gid: {"offer_url": ..., "image_url": ...}} for items
    that have at least one URL field.
    """
    if not item_json_path or not os.path.isfile(item_json_path):
        return {}
    print(f"  Loading URLs from item.json: {item_json_path}")
    t0 = time.time()
    try:
        import orjson
        with open(item_json_path, "rb") as f:
            items = orjson.loads(f.read())
    except ImportError:
        with open(item_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    url_map = {}
    for gid, item in items.items():
        offer_url = item.get("offer_url", "")
        image_url = item.get("image_url", "")
        if offer_url or image_url:
            url_map[str(gid)] = {"offer_url": offer_url, "image_url": image_url}
    print(f"  Loaded {len(url_map):,} items with URLs "
          f"(of {len(items):,} total) in {time.time()-t0:.1f}s")
    del items
    return url_map


def load_urls_from_index_tsv(index_tsv_path, needed_gids=None):
    """Load OfferURL and ImageUrl from the original index TSV.

    Only reads the 3 columns: GlobalOfferId, OfferURL, ImageUrl.
    If needed_gids is provided, only loads those GIDs (memory-efficient).

    Returns dict: {gid: {"offer_url": ..., "image_url": ...}}
    """
    if not index_tsv_path or not os.path.isfile(index_tsv_path):
        return {}
    print(f"  Loading URLs from index TSV: {index_tsv_path}")
    t0 = time.time()
    url_map = {}
    file_size = os.path.getsize(index_tsv_path)

    with open(index_tsv_path, "r", encoding="utf-8", buffering=64*1024*1024) as f:
        header_line = f.readline().rstrip("\r\n")
        cols = header_line.split("\t")
        # Find column indices
        try:
            gid_idx = cols.index("GlobalOfferId")
        except ValueError:
            gid_idx = 2  # fallback: column 3
        # OfferURL (note: TSV uses "OfferURL" with capital URL)
        offer_idx = None
        for name in ("OfferURL", "OfferUrl"):
            if name in cols:
                offer_idx = cols.index(name)
                break
        image_idx = None
        for name in ("ImageUrl", "ImageURL"):
            if name in cols:
                image_idx = cols.index(name)
                break

        if offer_idx is None and image_idx is None:
            print(f"  WARNING: Neither OfferURL nor ImageUrl found in header")
            return {}

        max_idx = max(x for x in (gid_idx, offer_idx, image_idx) if x is not None)

        pbar = tqdm(total=file_size, unit="B", unit_scale=True,
                    desc="    Reading index TSV", mininterval=5)
        pbar.update(len(header_line.encode("utf-8")) + 1)

        for line in f:
            pbar.update(len(line.encode("utf-8")))
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) <= max_idx:
                continue
            gid = fields[gid_idx].strip()
            if not gid:
                continue
            if needed_gids is not None and gid not in needed_gids:
                continue
            entry = {}
            if offer_idx is not None:
                entry["offer_url"] = fields[offer_idx].strip()
            if image_idx is not None:
                entry["image_url"] = fields[image_idx].strip()
            if entry.get("offer_url") or entry.get("image_url"):
                url_map[gid] = entry
        pbar.close()

    print(f"  Loaded {len(url_map):,} items with URLs in {time.time()-t0:.1f}s")
    return url_map


# ═══════════════════════════════════════════════════════════════════════════
#  Step6 TSV → Internal Format Adapter
# ═══════════════════════════════════════════════════════════════════════════

def _fix_backslash_quotes(text):
    """Fix backslash-escaped quotes from CSV roundtrips."""
    if not text or '\\"' not in text:
        return text
    for _ in range(3):
        if '\\"' not in text:
            break
        placeholder = "\x00BSQFIX\x00"
        text = text.replace('\\\\"', placeholder)
        text = text.replace('\\"', '"')
        text = text.replace(placeholder, '\\"')
    return text


def detect_input_format(filepath):
    """Detect whether input is JSONL or step6 TSV.

    Returns 'jsonl' or 'tsv'.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".jsonl":
        return "jsonl"
    if ext == ".tsv":
        return "tsv"
    # Peek at first line
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if first_line.startswith("{"):
        return "jsonl"
    if "RankedJourneys" in first_line or "UserId" in first_line:
        return "tsv"
    return "jsonl"


def load_jsonl_input(filepath):
    """Load original JSONL format."""
    users = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            users.append(json.loads(line))
    return users


def load_step6_tsv(filepath, url_map=None):
    """Load step6 merged TSV and convert to internal format.

    Step6 format (TSV columns):
        UserId, ReadableUserEvents, ShoppingProfile, JourneyWithProducts, RankedJourneys

    RankedJourneys column contains JSON:
        {"ContinuedJourneys": [{
            "JourneyType": "...", "Title": "...",
            "Description": "...", "ConversationStarter": "...",
            "WhyAmISeeingThis": "...",
            "Products": [{"Rank":1, "OfferId":"...", "Title":"...",
                          "Seller":"...", "Price":"...", "Brand":"...",
                          "Category":"...", "OriginalQuery":"..."}]
        }]}

    Converts to step8 internal format (matching original JSONL):
        {"stableid": "...", "userShoppingProfile": {...},
         "recentShoppingEvents": "...",
         "journeys": [{"journeyType": "...", "title": "...",
                       "description": "...", "conversationStarter": "...",
                       "reason": "...",
                       "products": [{"global_offer_id": "...", "Title": "...",
                                     "Seller": "...", "OriginalPrice": "...",
                                     "Brand": "...", "ImageUrl": "...",
                                     "OfferUrl": "...", "OriginalQuery": "...",
                                     "Rank": 1}]}]}

    Args:
        filepath: Path to step6 merged TSV.
        url_map: Optional dict {gid: {"offer_url": ..., "image_url": ...}}
                 for enriching products with URLs.

    Returns:
        List of user dicts in internal format.
    """
    if url_map is None:
        url_map = {}

    users = []
    n_users = 0
    n_journeys = 0
    n_products = 0
    n_urls_found = 0
    n_parse_errors = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc="Loading step6 TSV", mininterval=2):
            uid = row.get("UserId", "")
            n_users += 1

            # Parse ShoppingProfile
            profile_raw = _fix_backslash_quotes(
                row.get("ShoppingProfile", ""))
            try:
                profile = json.loads(profile_raw) if profile_raw else {}
            except (json.JSONDecodeError, TypeError):
                profile = {}

            # Parse events: #N# separated → newline separated
            events_raw = row.get("ReadableUserEvents", "")
            events = events_raw.replace("#N#", "\n")

            # Parse RankedJourneys
            rj_raw = _fix_backslash_quotes(
                row.get("RankedJourneys", ""))
            try:
                rj_obj = json.loads(rj_raw) if rj_raw else {}
            except (json.JSONDecodeError, TypeError):
                n_parse_errors += 1
                rj_obj = {}

            continued = rj_obj.get("ContinuedJourneys", [])
            journeys = []
            for j in continued:
                n_journeys += 1
                products = []
                for p in j.get("Products", []):
                    oid = str(p.get("OfferId", ""))
                    n_products += 1

                    # URL lookup
                    urls = url_map.get(oid, {})
                    offer_url = urls.get("offer_url", "")
                    image_url = urls.get("image_url", "")
                    if offer_url or image_url:
                        n_urls_found += 1

                    products.append({
                        "global_offer_id": oid,
                        "Title": p.get("Title", ""),
                        "Seller": p.get("Seller", ""),
                        "OriginalPrice": p.get("Price", ""),
                        "Brand": p.get("Brand", ""),
                        "Gender": p.get("Gender", ""),
                        "AgeGroup": p.get("AgeGroup", ""),
                        "OriginalQuery": p.get("OriginalQuery", ""),
                        "Rank": p.get("Rank"),
                        "ImageUrl": image_url,
                        "OfferUrl": offer_url,
                    })

                journeys.append({
                    "journeyType": j.get("JourneyType", "explicit"),
                    "title": j.get("Title", ""),
                    "description": j.get("Description", ""),
                    "conversationStarter": j.get("ConversationStarter", ""),
                    "reason": j.get("WhyAmISeeingThis", ""),
                    "products": products,
                })

            users.append({
                "stableid": uid,
                "userShoppingProfile": profile,
                "recentShoppingEvents": events,
                "journeys": journeys,
            })

    print(f"  Loaded {n_users:,} users, {n_journeys:,} journeys, "
          f"{n_products:,} products")
    if url_map:
        pct = n_urls_found / n_products * 100 if n_products else 0
        print(f"  URL coverage: {n_urls_found:,}/{n_products:,} "
              f"({pct:.1f}%) products have URLs")
    if n_parse_errors:
        print(f"  WARNING: {n_parse_errors:,} users had unparseable "
              f"RankedJourneys")
    return users


def collect_all_offer_ids(filepath):
    """Scan step6 TSV and collect all unique OfferIds for targeted URL loading."""
    gids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rj_raw = _fix_backslash_quotes(
                row.get("RankedJourneys", ""))
            try:
                rj_obj = json.loads(rj_raw) if rj_raw else {}
            except (json.JSONDecodeError, TypeError):
                continue
            for j in rj_obj.get("ContinuedJourneys", []):
                for p in j.get("Products", []):
                    oid = str(p.get("OfferId", ""))
                    if oid:
                        gids.add(oid)
    return gids


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
    ap = argparse.ArgumentParser(description="Rule-based L3 reranker + HTML visualization")
    ap.add_argument("--input", 
                    default="",
                    help="Input file: step6 Ranked TSV or legacy JSONL (auto-detected)")
    ap.add_argument("--output_prefix", default=None,
                    help="Output prefix (default: derived from input filename)")
    ap.add_argument("--results_dir", 
                    default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/vip_case_study/",
                    help="Directory to save outputs (HTML files)")
    ap.add_argument("--item_json", 
                    default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/item.json",
                    help="Path to item.json with offer_url/image_url fields")
    ap.add_argument("--original_index_file", 
                    default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/ProductGroup/20260515_ProductBestOffer_Sampled.tsv",
                    help="Fallback: original index TSV with OfferURL/ImageUrl columns. ")
    ap.add_argument("--top_k", type=int, default=12,
                    help="Visible Top-K window (constraints apply here)")
    ap.add_argument("--max_per_query_topk", type=int, default=6)
    ap.add_argument("--max_per_seller_total", type=int, default=4)
    ap.add_argument("--max_consecutive_seller", type=int, default=2)
    ap.add_argument("--max_per_brand_total", type=int, default=4)
    ap.add_argument("--max_consecutive_brand", type=int, default=2)
    ap.add_argument("--skip_rerank", action="store_true", default=False,
                    help="Skip L3 reranking; only convert + generate HTML")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.results_dir, exist_ok=True)

    # Default output prefix from input filename
    if args.output_prefix is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output_prefix = f"{base}_L3"

    fmt = detect_input_format(args.input)
    print("=" * 70)
    print("  Step8: L3 Reranker + HTML Visualization")
    print("=" * 70)
    print(f"  Input         : {args.input}")
    print(f"  Format        : {fmt}")
    print(f"  Top-K window  : {args.top_k}")
    print(f"  Skip rerank   : {args.skip_rerank}")
    print()

    # Load users
    if fmt == "tsv":
        # --- Step6 TSV path: load URLs, then convert ---
        print("  Loading URL data for product enrichment...")
        url_map = {}

        # 1) Try item.json first
        if args.item_json and os.path.isfile(args.item_json):
            url_map = load_urls_from_item_json(args.item_json)

        # 2) Check coverage: if item.json has few URLs, try index TSV fallback
        #    Collect all needed GIDs first
        needed_gids = collect_all_offer_ids(args.input)
        print(f"  Unique OfferIds in input: {len(needed_gids):,}")

        missing_gids = needed_gids - set(url_map.keys())
        if missing_gids:
            pct_covered = (len(needed_gids) - len(missing_gids)) / len(needed_gids) * 100 if needed_gids else 100
            print(f"  item.json coverage: {pct_covered:.1f}% "
                  f"({len(missing_gids):,} missing)")

            # Try fallback to original index TSV
            fallback = args.original_index_file
            if fallback and os.path.isfile(fallback):
                fallback_urls = load_urls_from_index_tsv(
                    fallback, needed_gids=missing_gids)
                url_map.update(fallback_urls)
                new_missing = needed_gids - set(url_map.keys())
                print(f"  After fallback: {len(new_missing):,} GIDs still missing URLs")

        print()
        users = load_step6_tsv(args.input, url_map=url_map)
        del url_map  # free memory
    else:
        users = load_jsonl_input(args.input)

    print(f"\nLoaded {len(users)} users")

    # Rerank every journey
    if not args.skip_rerank:
        print(f"\n  per-query  cap: {args.max_per_query_topk} (in Top-{args.top_k})")
        print(f"  per-seller cap: {args.max_per_seller_total} total / "
              f"{args.max_consecutive_seller} consecutive")
        print(f"  per-brand  cap: {args.max_per_brand_total} total / "
              f"{args.max_consecutive_brand} consecutive")
        print()

    # Rerank every journey
    n_journeys = 0
    sum_consec_s_before = sum_consec_s_after = 0
    sum_consec_b_before = sum_consec_b_after = 0
    sum_uniq_s_before = sum_uniq_s_after = 0
    sum_excluded = sum_demoted = 0

    if not args.skip_rerank:
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
    else:
        for u in users:
            for j in u.get("journeys", []):
                n_journeys += 1

    if n_journeys and not args.skip_rerank:
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
    print("  Step8 complete!")
    print(f"  Input format  : {fmt}")
    print(f"  JSONL: {out_jsonl}")
    print(f"  HTML : {out_html}")
    print(f"  Users: {len(users)}, Journeys: {n_journeys}")
    print("=" * 70)


if __name__ == "__main__":
    main()
