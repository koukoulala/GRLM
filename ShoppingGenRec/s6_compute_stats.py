#!/usr/bin/env python3
"""
Shopping Journey Evaluation Statistics Analyzer.

Parses TSV evaluation data files and produces comprehensive numerical
statistics across journey diversity, quality, relevance, product diversity,
and product quality dimensions — aligned with UHRS evaluation guidelines.

Usage:
    python compute_stats.py data/file1.tsv
    python compute_stats.py data/file1.tsv data/file2.tsv data/file3.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_json_loads(text: str):
    """Parse a JSON string, returning None on failure."""
    if not text or text.strip() == "":
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_price(price_str: str) -> float | None:
    """Strip '$' and parse price to float."""
    if not price_str:
        return None
    try:
        return float(str(price_str).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def desc_stats(values: list[float | int]) -> dict:
    """Compute descriptive statistics for a numeric list."""
    if not values:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "median": np.nan, "count": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "median": float(np.nanmedian(arr)),
        "count": len(arr),
    }


def score_distribution(values: list[int], possible_scores=(0, 1, 2)) -> dict:
    """Count and percentage for each possible score value."""
    c = Counter(values)
    total = len(values) if values else 1
    return {s: {"count": c.get(s, 0), "pct": 100.0 * c.get(s, 0) / total} for s in possible_scores}


def fmt_stats(stats: dict, indent: int = 4) -> str:
    """Format descriptive stats dict into a readable line."""
    prefix = " " * indent
    return (
        f"{prefix}Count: {stats['count']}  |  "
        f"Mean: {stats['mean']:.4f}  |  Std: {stats['std']:.4f}  |  "
        f"Min: {stats['min']:.1f}  |  Median: {stats['median']:.1f}  |  Max: {stats['max']:.1f}"
    )


def fmt_dist(dist: dict, indent: int = 4) -> str:
    """Format a score distribution dict."""
    prefix = " " * indent
    parts = [f"Score {s}: {d['count']:>6d} ({d['pct']:5.1f}%)" for s, d in sorted(dist.items())]
    return prefix + "  |  ".join(parts)


def fmt_counter(counter: Counter, indent: int = 4, top_n: int | None = None) -> str:
    """Format a Counter as aligned rows."""
    prefix = " " * indent
    total = sum(counter.values()) or 1
    items = counter.most_common(top_n)
    if not items:
        return prefix + "(no data)"
    max_label_len = max(len(str(k)) for k, _ in items)
    lines = []
    for label, count in items:
        pct = 100.0 * count / total
        lines.append(f"{prefix}{str(label):<{max_label_len}}  {count:>6d}  ({pct:5.1f}%)")
    return "\n".join(lines)


def histogram_buckets(values: list[int], max_buckets: int = 10) -> Counter:
    """Create histogram-style buckets for integer values."""
    if not values:
        return Counter()
    lo, hi = min(values), max(values)
    if hi - lo < max_buckets:
        # Each value is its own bucket
        return Counter(values)
    # Create ranges
    edges = np.linspace(lo, hi, max_buckets + 1)
    buckets = Counter()
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v <= edges[i + 1] or (i == len(edges) - 2 and v == edges[i + 1]):
                label = f"{int(edges[i])}-{int(edges[i+1])}"
                buckets[label] += 1
                break
    return buckets


def safe_corr(a, b) -> float | None:
    """Compute Pearson correlation, returning None if insufficient data or constant."""
    if len(a) < 3:
        return None
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    if np.std(a_arr) == 0 or np.std(b_arr) == 0:
        return None
    return float(np.corrcoef(a_arr, b_arr)[0, 1])


def parse_rule_ids(explanation: str) -> list[str]:
    """Extract semicolon-separated rule IDs from an explanation string.

    Handles formats like:
        "RULE_STEP1_DIRECT_CATEGORY; RULE_STEP1_BRAND"
        "RULE_STEP2_INTEREST_EXTENSION"
        "RULE_STEP1_DIRECT_PRODUCT"
    """
    if not explanation:
        return []
    # Split on semicolons, strip whitespace, keep non-empty tokens
    tokens = [t.strip() for t in explanation.split(";") if t.strip()]
    # Filter to rule-like tokens (start with RULE_ or contain STEP)
    rules = [t for t in tokens if t.startswith("RULE_") or "STEP" in t.upper()]
    # If no rule-like tokens found, return the raw tokens (some data may
    # not follow the RULE_ prefix convention)
    return rules if rules else tokens


def classify_rule_step(rule_id: str) -> str | None:
    """Classify a rule ID as Step1 or Step2 based on naming convention."""
    upper = rule_id.upper()
    if "STEP1" in upper:
        return "Step1"
    if "STEP2" in upper:
        return "Step2"
    return None


# ---------------------------------------------------------------------------
# Data Parsing
# ---------------------------------------------------------------------------

def parse_file(filepath: str) -> tuple[list[dict], int, int]:
    """
    Parse a TSV evaluation file.

    Returns:
        (parsed_rows, total_rows, skipped_rows)
        Each parsed_row is a dict with structured fields.
    """
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB per field

    rows = []
    total = 0
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            total += 1
            try:
                row = _parse_row(raw)
                if row is not None:
                    rows.append(row)
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    return rows, total, skipped


def _parse_row(raw: dict) -> dict | None:
    """Parse a single TSV row into structured data. Returns None on critical parse failure."""
    user_id = raw.get("UserId", "")
    if not user_id:
        return None

    # --- ShoppingJourneys (required) ---
    journeys = safe_json_loads(raw.get("ShoppingJourneys", ""))
    if not isinstance(journeys, list) or len(journeys) == 0:
        return None

    # Parse products within journeys
    for j in journeys:
        for p in j.get("Products", []):
            p["_parsed_price"] = parse_price(p.get("Price"))

    # --- Evaluation columns (all optional — gracefully degrade) ---
    journey_diversity = safe_json_loads(raw.get("journey_diversity", ""))
    journey_quality = safe_json_loads(raw.get("journey_quality", ""))
    journey_relevance = safe_json_loads(raw.get("journey_relevance", ""))
    product_diversity = safe_json_loads(raw.get("product_diversity", ""))
    product_quality_raw = safe_json_loads(raw.get("product_quality", ""))

    # Normalize product_quality: single object -> list
    if isinstance(product_quality_raw, dict):
        product_quality = [product_quality_raw]
    elif isinstance(product_quality_raw, list):
        product_quality = product_quality_raw
    else:
        product_quality = None

    # Normalize journey_quality: ensure list
    if isinstance(journey_quality, dict):
        journey_quality = [journey_quality]

    # Normalize journey_relevance: ensure list
    if isinstance(journey_relevance, dict):
        journey_relevance = [journey_relevance]

    # Normalize product_diversity: ensure list
    if isinstance(product_diversity, dict):
        product_diversity = [product_diversity]

    return {
        "user_id": user_id,
        "journeys": journeys,
        "journey_diversity": journey_diversity,
        "journey_quality": journey_quality,
        "journey_relevance": journey_relevance,
        "product_diversity": product_diversity,
        "product_quality": product_quality,
    }


# ---------------------------------------------------------------------------
# Statistics Computation
# ---------------------------------------------------------------------------

def compute_stats(rows: list[dict]) -> dict:
    """Compute all statistics for a list of parsed rows."""
    stats = {}

    # ===== A. Volume Statistics =====
    num_users = len(rows)
    user_journey_counts = []
    user_product_counts = []
    journey_product_counts = []  # products per journey
    total_journeys = 0
    total_products = 0

    for r in rows:
        jc = len(r["journeys"])
        user_journey_counts.append(jc)
        total_journeys += jc
        pc = 0
        for j in r["journeys"]:
            n_prod = len(j.get("Products", []))
            journey_product_counts.append(n_prod)
            pc += n_prod
        user_product_counts.append(pc)
        total_products += pc

    stats["volume"] = {
        "num_users": num_users,
        "total_journeys": total_journeys,
        "total_products": total_products,
        "journeys_per_user": desc_stats(user_journey_counts),
        "products_per_user": desc_stats(user_product_counts),
        "products_per_journey": desc_stats(journey_product_counts),
        "journey_count_dist": histogram_buckets(user_journey_counts),
        "product_per_journey_dist": histogram_buckets(journey_product_counts),
    }

    # ===== B. Journey Diversity Metrics =====
    jd_scores = []
    jd_num_groups = []
    jd_group_sizes = []
    jd_diversity_ratios = []  # computed num_groups / total_journeys per user

    for r in rows:
        jd = r["journey_diversity"]
        if not isinstance(jd, dict):
            continue
        score = jd.get("diversityScore")
        if score is not None:
            jd_scores.append(int(score))
        groups = jd.get("journeyGroups", [])
        jd_num_groups.append(len(groups))
        for g in groups:
            if isinstance(g, list):
                jd_group_sizes.append(len(g))
        # Compute diversity ratio = num_groups / total_journeys for this user
        n_journeys_user = len(r["journeys"])
        if n_journeys_user > 0 and len(groups) > 0:
            jd_diversity_ratios.append(len(groups) / n_journeys_user)

    stats["journey_diversity"] = {
        "diversity_score": desc_stats(jd_scores),
        "diversity_score_dist": score_distribution(jd_scores),
        "groups_per_user": desc_stats(jd_num_groups),
        "group_size": desc_stats(jd_group_sizes),
        "diversity_ratio": desc_stats(jd_diversity_ratios),
    }

    # ===== C. Journey Quality Metrics =====
    jq_types = Counter()
    jq_journey_value = []
    jq_content_compliance = []
    jq_tone = []
    jq_self_coherence = []
    jq_composite = []
    jq_appropriateness_composite = []  # avg of journeyValue + contentCompliance
    jq_title_quality_composite = []     # avg of tone + selfCoherence

    # Per journey-type breakdowns
    jq_value_by_type = defaultdict(list)   # journeyType -> list of journeyValue scores
    jq_all_by_type = defaultdict(list)     # journeyType -> list of {jv, cc, tn, sc} dicts

    for r in rows:
        jq_list = r["journey_quality"]
        if not isinstance(jq_list, list):
            continue
        for jq in jq_list:
            if not isinstance(jq, dict):
                continue
            jtype = jq.get("journeyType", "unknown")
            jq_types[jtype] += 1

            approp = jq.get("journeyAppropriateness", {})
            title_q = jq.get("journeyTitleQuality", {})

            jv = approp.get("journeyValue")
            cc = approp.get("contentCompliance")
            tn = title_q.get("tone")
            sc = title_q.get("selfCoherence")

            if jv is not None:
                jq_journey_value.append(int(jv))
                jq_value_by_type[jtype].append(int(jv))
            if cc is not None:
                jq_content_compliance.append(int(cc))
            if tn is not None:
                jq_tone.append(int(tn))
            if sc is not None:
                jq_self_coherence.append(int(sc))

            # Journey Appropriateness composite (journeyValue + contentCompliance)
            approp_sub = [x for x in [jv, cc] if x is not None]
            if approp_sub:
                jq_appropriateness_composite.append(sum(int(x) for x in approp_sub) / len(approp_sub))

            # Journey Title Quality composite (tone + selfCoherence)
            title_sub = [x for x in [tn, sc] if x is not None]
            if title_sub:
                jq_title_quality_composite.append(sum(int(x) for x in title_sub) / len(title_sub))

            # Overall composite: average of available sub-scores
            sub = [x for x in [jv, cc, tn, sc] if x is not None]
            if sub:
                jq_composite.append(sum(int(x) for x in sub) / len(sub))

            # Store for per-type breakdowns
            jq_all_by_type[jtype].append({
                "jv": int(jv) if jv is not None else None,
                "cc": int(cc) if cc is not None else None,
                "tn": int(tn) if tn is not None else None,
                "sc": int(sc) if sc is not None else None,
            })

    stats["journey_quality"] = {
        "journey_type_dist": jq_types,
        "journey_value": desc_stats(jq_journey_value),
        "journey_value_dist": score_distribution(jq_journey_value),
        "content_compliance": desc_stats(jq_content_compliance),
        "content_compliance_dist": score_distribution(jq_content_compliance),
        "tone": desc_stats(jq_tone),
        "tone_dist": score_distribution(jq_tone),
        "self_coherence": desc_stats(jq_self_coherence),
        "self_coherence_dist": score_distribution(jq_self_coherence),
        "appropriateness_composite": desc_stats(jq_appropriateness_composite),
        "title_quality_composite": desc_stats(jq_title_quality_composite),
        "composite": desc_stats(jq_composite),
        "journey_value_by_type": {
            jtype: desc_stats(vals) for jtype, vals in jq_value_by_type.items()
        },
    }

    # ===== D. Journey Relevance Metrics =====
    jr_scores = []
    jr_explanations = Counter()
    jr_rule_ids = Counter()           # individual rule ID counts
    jr_step_classification = Counter()  # Step1 vs Step2 counts
    jr_scores_by_step = defaultdict(list)  # Step1/Step2 -> list of scores
    jr_journeys_with_step = Counter()  # per-journey Step1 vs Step2

    for r in rows:
        jr_list = r["journey_relevance"]
        if not isinstance(jr_list, list):
            continue
        for jr in jr_list:
            if not isinstance(jr, dict):
                continue
            score = jr.get("shoppingRelevanceScore")
            if score is not None:
                jr_scores.append(int(score))
            expl = jr.get("explanation", "")
            if expl:
                jr_explanations[expl] += 1

            # Parse rule IDs from explanation
            rules = parse_rule_ids(expl)
            journey_steps = set()
            for rule_id in rules:
                jr_rule_ids[rule_id] += 1
                step = classify_rule_step(rule_id)
                if step:
                    jr_step_classification[step] += 1
                    journey_steps.add(step)

            # Classify this journey as Step1 or Step2
            # If any Step1 rule is present, it's a Step1 journey (direct interaction)
            # Otherwise if Step2 rules are present, it's Step2 (extension-based)
            if "Step1" in journey_steps:
                jr_journeys_with_step["Step1"] += 1
                if score is not None:
                    jr_scores_by_step["Step1"].append(int(score))
            elif "Step2" in journey_steps:
                jr_journeys_with_step["Step2"] += 1
                if score is not None:
                    jr_scores_by_step["Step2"].append(int(score))
            elif rules:
                # Rules present but couldn't classify as Step1/Step2
                jr_journeys_with_step["Unclassified"] += 1
                if score is not None:
                    jr_scores_by_step["Unclassified"].append(int(score))

    total_classified_journeys = sum(jr_journeys_with_step.values())

    stats["journey_relevance"] = {
        "relevance_score": desc_stats(jr_scores),
        "relevance_score_dist": score_distribution(jr_scores),
        "explanation_dist": jr_explanations,
        "rule_id_dist": jr_rule_ids,
        "step_classification": jr_step_classification,
        "journey_step_dist": jr_journeys_with_step,
        "total_classified_journeys": total_classified_journeys,
        "step1_pct": (100.0 * jr_journeys_with_step.get("Step1", 0) / total_classified_journeys)
            if total_classified_journeys > 0 else 0.0,
        "step2_pct": (100.0 * jr_journeys_with_step.get("Step2", 0) / total_classified_journeys)
            if total_classified_journeys > 0 else 0.0,
        "scores_by_step": {
            step: desc_stats(vals) for step, vals in jr_scores_by_step.items()
        },
        "score_dist_by_step": {
            step: score_distribution(vals) for step, vals in jr_scores_by_step.items()
        },
    }

    # ===== E. Product Diversity Metrics =====
    pd_scores = []
    pd_num_groups = []
    pd_group_sizes = []
    pd_computed_ratios = []      # actual num_groups / num_products per journey
    pd_product_counts = []       # number of products per journey (for correlation)

    for r in rows:
        pd_list = r["product_diversity"]
        if not isinstance(pd_list, list):
            continue
        for pd_idx, pd_item in enumerate(pd_list):
            if not isinstance(pd_item, dict):
                continue
            score = pd_item.get("diversityScore")
            if score is not None:
                pd_scores.append(int(score))
            groups = pd_item.get("productGroups", [])
            n_groups = len(groups)
            pd_num_groups.append(n_groups)
            for g in groups:
                if isinstance(g, list):
                    pd_group_sizes.append(len(g))

            # Compute actual diversity ratio from group structure
            # Total products = sum of all group sizes
            n_products_in_groups = sum(
                len(g) for g in groups if isinstance(g, list)
            )
            if n_products_in_groups > 0:
                ratio = n_groups / n_products_in_groups
                pd_computed_ratios.append(ratio)
                pd_product_counts.append(n_products_in_groups)
                # Also pair with score for correlation
            elif n_groups > 0:
                # Groups exist but no products countable from group structure
                # Fall back: try matching journey product count
                if pd_idx < len(r["journeys"]):
                    n_prod = len(r["journeys"][pd_idx].get("Products", []))
                    if n_prod > 0:
                        pd_computed_ratios.append(n_groups / n_prod)
                        pd_product_counts.append(n_prod)

    # Correlation: number of products vs diversity score
    pd_score_product_pairs = list(zip(pd_product_counts[:len(pd_scores)], pd_scores[:len(pd_product_counts)]))
    if len(pd_score_product_pairs) >= 3:
        a_vals, b_vals = zip(*pd_score_product_pairs)
        pd_product_count_vs_score_corr = safe_corr(a_vals, b_vals)
    else:
        pd_product_count_vs_score_corr = None

    stats["product_diversity"] = {
        "diversity_score": desc_stats(pd_scores),
        "diversity_score_dist": score_distribution(pd_scores),
        "groups_per_journey": desc_stats(pd_num_groups),
        "group_size": desc_stats(pd_group_sizes),
        "computed_diversity_ratio": desc_stats(pd_computed_ratios),
        "product_count_vs_score_corr": pd_product_count_vs_score_corr,
    }

    # ===== F. Product Quality Metrics =====
    pq_intent = []
    pq_attribute = []
    pq_gender = []
    pq_compliance = []
    pq_seller = []
    pq_composite = []
    pq_relevance_composite = []  # avg of intent + attribute + gender only
    pq_has_reason = 0
    pq_total = 0
    pq_reason_categories = Counter()

    # Per-seller-authority breakdown of product quality
    pq_by_seller_tier = defaultdict(list)  # seller_score -> list of product composites

    for r in rows:
        pq_list = r["product_quality"]
        if not isinstance(pq_list, list):
            continue
        for pq_journey in pq_list:
            if not isinstance(pq_journey, dict):
                continue
            products = pq_journey.get("productQuality", [])
            if not isinstance(products, list):
                continue
            for prod in products:
                if not isinstance(prod, dict):
                    continue
                pq_total += 1
                rel = prod.get("productToJourneyRelevance", {})
                ia = rel.get("intentAlignment")
                aa = rel.get("attributeAlignment")
                ga = rel.get("genderAlignment")
                pc_val = prod.get("productCompliance")
                sa = prod.get("sellerAuthority")
                reason = prod.get("qualityReason", "")

                if ia is not None:
                    pq_intent.append(int(ia))
                if aa is not None:
                    pq_attribute.append(int(aa))
                if ga is not None:
                    pq_gender.append(int(ga))
                if pc_val is not None:
                    pq_compliance.append(int(pc_val))
                if sa is not None:
                    pq_seller.append(int(sa))
                if reason and str(reason).strip():
                    pq_has_reason += 1
                    # Categorize reason: extract first meaningful phrase
                    reason_text = str(reason).strip()
                    # Use first ~60 chars or up to first period/semicolon as category
                    for sep in [";", ".", ","]:
                        if sep in reason_text:
                            reason_text = reason_text[:reason_text.index(sep)]
                            break
                    reason_text = reason_text[:80].strip()
                    if reason_text:
                        pq_reason_categories[reason_text] += 1

                # Product-to-Journey Relevance composite (intent + attribute + gender)
                rel_sub = [x for x in [ia, aa, ga] if x is not None]
                if rel_sub:
                    rel_comp = sum(int(s) for s in rel_sub) / len(rel_sub)
                    pq_relevance_composite.append(rel_comp)

                # Overall composite: average of all 5 sub-metrics
                sub = [x for x in [ia, aa, ga, pc_val, sa] if x is not None]
                if sub:
                    comp = sum(int(s) for s in sub) / len(sub)
                    pq_composite.append(comp)
                    # Store by seller authority tier
                    if sa is not None:
                        pq_by_seller_tier[int(sa)].append(comp)

    stats["product_quality"] = {
        "intent_alignment": desc_stats(pq_intent),
        "intent_alignment_dist": score_distribution(pq_intent),
        "attribute_alignment": desc_stats(pq_attribute),
        "attribute_alignment_dist": score_distribution(pq_attribute),
        "gender_alignment": desc_stats(pq_gender),
        "gender_alignment_dist": score_distribution(pq_gender),
        "product_compliance": desc_stats(pq_compliance),
        "product_compliance_dist": score_distribution(pq_compliance),
        "seller_authority": desc_stats(pq_seller),
        "seller_authority_dist": score_distribution(pq_seller),
        "relevance_composite": desc_stats(pq_relevance_composite),
        "composite": desc_stats(pq_composite),
        "total_products_evaluated": pq_total,
        "pct_with_reason": 100.0 * pq_has_reason / pq_total if pq_total > 0 else 0.0,
        "reason_categories": pq_reason_categories,
        "quality_by_seller_tier": {
            tier: desc_stats(vals) for tier, vals in sorted(pq_by_seller_tier.items())
        },
    }

    # ===== G. Journey-Type Breakdown Analysis =====
    # For explicit vs exploratory journeys, show how downstream metrics differ.
    # We need to align journey_quality journeyType with journey_relevance,
    # product_diversity, and product_quality by journey index within each user.
    jtype_relevance = defaultdict(list)
    jtype_prod_quality = defaultdict(list)
    jtype_prod_diversity = defaultdict(list)

    for r in rows:
        jq_list = r["journey_quality"]
        jr_list = r["journey_relevance"]
        pd_list = r["product_diversity"]
        pq_list = r["product_quality"]

        if not isinstance(jq_list, list):
            continue

        for idx, jq in enumerate(jq_list):
            if not isinstance(jq, dict):
                continue
            jtype = jq.get("journeyType", "unknown")

            # Journey relevance score for this journey index
            if isinstance(jr_list, list) and idx < len(jr_list):
                jr_item = jr_list[idx]
                if isinstance(jr_item, dict):
                    jr_s = jr_item.get("shoppingRelevanceScore")
                    if jr_s is not None:
                        jtype_relevance[jtype].append(int(jr_s))

            # Product diversity score for this journey index
            if isinstance(pd_list, list) and idx < len(pd_list):
                pd_item = pd_list[idx]
                if isinstance(pd_item, dict):
                    pd_s = pd_item.get("diversityScore")
                    if pd_s is not None:
                        jtype_prod_diversity[jtype].append(int(pd_s))

            # Product quality composite for this journey index
            if isinstance(pq_list, list) and idx < len(pq_list):
                pq_item = pq_list[idx]
                if isinstance(pq_item, dict):
                    products = pq_item.get("productQuality", [])
                    if isinstance(products, list):
                        journey_composites = []
                        for prod in products:
                            if not isinstance(prod, dict):
                                continue
                            rel = prod.get("productToJourneyRelevance", {})
                            sub = [
                                rel.get("intentAlignment"),
                                rel.get("attributeAlignment"),
                                rel.get("genderAlignment"),
                                prod.get("productCompliance"),
                                prod.get("sellerAuthority"),
                            ]
                            sub = [int(s) for s in sub if s is not None]
                            if sub:
                                journey_composites.append(np.mean(sub))
                        if journey_composites:
                            jtype_prod_quality[jtype].append(float(np.mean(journey_composites)))

    stats["journey_type_breakdown"] = {
        "relevance_by_type": {
            jtype: desc_stats(vals) for jtype, vals in jtype_relevance.items()
        },
        "product_quality_by_type": {
            jtype: desc_stats(vals) for jtype, vals in jtype_prod_quality.items()
        },
        "product_diversity_by_type": {
            jtype: desc_stats(vals) for jtype, vals in jtype_prod_diversity.items()
        },
    }

    # ===== H. Cross-metric Correlations =====
    corr = {}

    # Journey diversity vs product diversity (per-user, avg product diversity)
    pairs_jd_pd = []
    for r in rows:
        jd = r["journey_diversity"]
        pd_list = r["product_diversity"]
        if isinstance(jd, dict) and isinstance(pd_list, list):
            jd_s = jd.get("diversityScore")
            pd_scores_user = [
                pd_item.get("diversityScore")
                for pd_item in pd_list
                if isinstance(pd_item, dict) and pd_item.get("diversityScore") is not None
            ]
            if jd_s is not None and pd_scores_user:
                pairs_jd_pd.append((int(jd_s), np.mean(pd_scores_user)))

    if len(pairs_jd_pd) >= 3:
        a, b = zip(*pairs_jd_pd)
        corr["journey_div_vs_product_div"] = safe_corr(a, b)
    else:
        corr["journey_div_vs_product_div"] = None

    # Journey relevance vs product quality (per-journey)
    pairs_jr_pq = []
    for r in rows:
        jr_list = r["journey_relevance"]
        pq_list = r["product_quality"]
        if not isinstance(jr_list, list) or not isinstance(pq_list, list):
            continue
        # Match by index (both are per-journey arrays)
        for idx in range(min(len(jr_list), len(pq_list))):
            jr_item = jr_list[idx] if isinstance(jr_list[idx], dict) else {}
            pq_item = pq_list[idx] if isinstance(pq_list[idx], dict) else {}
            jr_score = jr_item.get("shoppingRelevanceScore")
            products = pq_item.get("productQuality", [])
            if not isinstance(products, list) or jr_score is None:
                continue
            # Average product composite for this journey
            composites = []
            for prod in products:
                if not isinstance(prod, dict):
                    continue
                rel = prod.get("productToJourneyRelevance", {})
                sub = [
                    rel.get("intentAlignment"),
                    rel.get("attributeAlignment"),
                    rel.get("genderAlignment"),
                    prod.get("productCompliance"),
                    prod.get("sellerAuthority"),
                ]
                sub = [int(s) for s in sub if s is not None]
                if sub:
                    composites.append(np.mean(sub))
            if composites:
                pairs_jr_pq.append((int(jr_score), np.mean(composites)))

    if len(pairs_jr_pq) >= 3:
        a, b = zip(*pairs_jr_pq)
        corr["journey_rel_vs_product_qual"] = safe_corr(a, b)
    else:
        corr["journey_rel_vs_product_qual"] = None

    # Journey value vs product diversity (per-journey)
    pairs_jv_pd = []
    for r in rows:
        jq_list = r["journey_quality"]
        pd_list = r["product_diversity"]
        if not isinstance(jq_list, list) or not isinstance(pd_list, list):
            continue
        for idx in range(min(len(jq_list), len(pd_list))):
            jq_item = jq_list[idx] if isinstance(jq_list[idx], dict) else {}
            pd_item = pd_list[idx] if isinstance(pd_list[idx], dict) else {}
            approp = jq_item.get("journeyAppropriateness", {})
            jv = approp.get("journeyValue")
            pd_s = pd_item.get("diversityScore")
            if jv is not None and pd_s is not None:
                pairs_jv_pd.append((int(jv), int(pd_s)))

    if len(pairs_jv_pd) >= 3:
        a, b = zip(*pairs_jv_pd)
        corr["journey_value_vs_product_div"] = safe_corr(a, b)
    else:
        corr["journey_value_vs_product_div"] = None

    # Content compliance vs product compliance (per-journey, avg product compliance)
    pairs_cc_pc = []
    for r in rows:
        jq_list = r["journey_quality"]
        pq_list = r["product_quality"]
        if not isinstance(jq_list, list) or not isinstance(pq_list, list):
            continue
        for idx in range(min(len(jq_list), len(pq_list))):
            jq_item = jq_list[idx] if isinstance(jq_list[idx], dict) else {}
            pq_item = pq_list[idx] if isinstance(pq_list[idx], dict) else {}
            approp = jq_item.get("journeyAppropriateness", {})
            cc = approp.get("contentCompliance")
            products = pq_item.get("productQuality", [])
            if not isinstance(products, list) or cc is None:
                continue
            pc_vals = [
                int(prod.get("productCompliance"))
                for prod in products
                if isinstance(prod, dict) and prod.get("productCompliance") is not None
            ]
            if pc_vals:
                pairs_cc_pc.append((int(cc), np.mean(pc_vals)))

    if len(pairs_cc_pc) >= 3:
        a, b = zip(*pairs_cc_pc)
        corr["content_compliance_vs_product_compliance"] = safe_corr(a, b)
    else:
        corr["content_compliance_vs_product_compliance"] = None

    # Number of signals vs journey count (per-user)
    corr["num_signals_vs_journey_count"] = None  # Requires UserSignals parsing

    stats["correlations"] = corr

    return stats


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------

SECTION_SEP = "=" * 80
SUBSEC_SEP = "-" * 60


def print_report(filepath: str, stats: dict, total: int, skipped: int) -> None:
    """Print a formatted statistics report for one file."""
    name = Path(filepath).stem
    print()
    print(SECTION_SEP)
    print(f"  STATISTICS REPORT: {name}")
    print(f"  File: {filepath}")
    print(SECTION_SEP)

    # --- Parse Summary ---
    print(f"\n  Rows total: {total}  |  Parsed: {total - skipped}  |  Skipped: {skipped}")

    # --- A. Volume ---
    v = stats["volume"]
    print(f"\n{SUBSEC_SEP}")
    print("  A. VOLUME STATISTICS")
    print(SUBSEC_SEP)
    print(f"    Users (valid rows):    {v['num_users']}")
    print(f"    Total journeys:        {v['total_journeys']}")
    print(f"    Total products:        {v['total_products']}")
    print(f"\n    Journeys per user:")
    print(fmt_stats(v["journeys_per_user"], 6))
    print(f"\n    Products per user:")
    print(fmt_stats(v["products_per_user"], 6))
    print(f"\n    Products per journey:")
    print(fmt_stats(v["products_per_journey"], 6))
    print(f"\n    Journey count distribution (per user):")
    print(fmt_counter(v["journey_count_dist"], 6))
    print(f"\n    Product count distribution (per journey):")
    print(fmt_counter(v["product_per_journey_dist"], 6))

    # --- B. Journey Diversity ---
    jd = stats["journey_diversity"]
    print(f"\n{SUBSEC_SEP}")
    print("  B. JOURNEY DIVERSITY METRICS")
    print(f"     Evaluates whether the set of shopping journeys collectively provides")
    print(f"     meaningful diversity across user intent and decision space.")
    print(f"     Journeys sharing same Recipient + Primary Category + Brand are grouped.")
    print(f"     diversity_ratio = num_journeyGroups / total_journeys")
    print(f"     Score: 2 (ratio=1.0, all distinct) | 1 (0.6<=ratio<1.0) | 0 (ratio<0.6)")
    print(SUBSEC_SEP)
    print(f"    Diversity Score:")
    print(fmt_stats(jd["diversity_score"], 6))
    print(f"    Score distribution:")
    print(fmt_dist(jd["diversity_score_dist"], 6))
    print(f"\n    Computed Diversity Ratio (num_groups / total_journeys per user):")
    print(fmt_stats(jd["diversity_ratio"], 6))
    print(f"\n    Journey groups per user:")
    print(fmt_stats(jd["groups_per_user"], 6))
    print(f"\n    Journey group size:")
    print(fmt_stats(jd["group_size"], 6))

    # --- C. Journey Quality ---
    jq = stats["journey_quality"]
    print(f"\n{SUBSEC_SEP}")
    print("  C. JOURNEY QUALITY METRICS")
    print(f"     Two sub-dimensions: Journey Appropriateness (category suitability +")
    print(f"     content compliance) and Journey Title Quality (tone + self-coherence).")
    print(f"     journeyType: 'explicit' (user directly interacted) vs 'exploratory' (inferred)")
    print(SUBSEC_SEP)
    print(f"    Journey Type distribution:")
    print(fmt_counter(jq["journey_type_dist"], 6))

    # Sub-dimension 1: Journey Appropriateness
    print(f"\n    --- Journey Appropriateness (category suitability for curation) ---")
    print(f"        journeyValue: Whether category benefits from curation")
    print(f"          2=Benefits from curation | 1=Limited curation value | 0=Unsuitable")
    print(f"\n    Journey Value:")
    print(fmt_stats(jq["journey_value"], 6))
    print(f"    Distribution:")
    print(fmt_dist(jq["journey_value_dist"], 6))

    # Journey Value by type breakdown
    jv_by_type = jq.get("journey_value_by_type", {})
    if jv_by_type:
        print(f"\n    Journey Value by Journey Type:")
        for jtype, type_stats in sorted(jv_by_type.items()):
            print(f"      {jtype}:")
            print(fmt_stats(type_stats, 8))

    print(f"\n        contentCompliance: Safety + online shopping suitability")
    print(f"          2=Completely safe & suitable | 1=Borderline | 0=Prohibited/unsuitable")
    print(f"\n    Content Compliance:")
    print(fmt_stats(jq["content_compliance"], 6))
    print(f"    Distribution:")
    print(fmt_dist(jq["content_compliance_dist"], 6))

    print(f"\n    >> Journey Appropriateness Composite (avg of journeyValue + contentCompliance):")
    print(fmt_stats(jq["appropriateness_composite"], 6))

    # Sub-dimension 2: Journey Title Quality
    print(f"\n    --- Journey Title Quality (language quality of journey title) ---")
    print(f"        tone: Natural, grammatically correct, idiomatic language")
    print(f"          2=Excellent human-like | 1=Acceptable, minor issues | 0=Poor/robotic")
    print(f"\n    Tone:")
    print(fmt_stats(jq["tone"], 6))
    print(f"    Distribution:")
    print(fmt_dist(jq["tone_dist"], 6))

    print(f"\n        selfCoherence: Title internally consistent without contradictions")
    print(f"          2=Logically consistent | 1=Minor inconsistencies | 0=Contradictory")
    print(f"\n    Self-Coherence:")
    print(fmt_stats(jq["self_coherence"], 6))
    print(f"    Distribution:")
    print(fmt_dist(jq["self_coherence_dist"], 6))

    print(f"\n    >> Journey Title Quality Composite (avg of tone + selfCoherence):")
    print(fmt_stats(jq["title_quality_composite"], 6))

    # Overall composite
    print(f"\n    >> Overall Journey Quality Composite (avg of all 4 sub-metrics):")
    print(fmt_stats(jq["composite"], 6))

    # --- D. Journey Relevance ---
    jr = stats["journey_relevance"]
    print(f"\n{SUBSEC_SEP}")
    print("  D. JOURNEY RELEVANCE METRICS")
    print(f"     Whether the journey matches user's observed shopping behavior.")
    print(f"     Two-step classification:")
    print(f"       Step1: Direct interaction with journey product/category/brand")
    print(f"       Step2: No direct interaction; evaluate extension logic")
    print(f"     Score: 2=Clearly shopping | 1=Partially/ambiguous | 0=Not shopping")
    print(SUBSEC_SEP)
    print(f"    Shopping Relevance Score:")
    print(fmt_stats(jr["relevance_score"], 6))
    print(f"    Score distribution:")
    print(fmt_dist(jr["relevance_score_dist"], 6))

    # Step1 vs Step2 breakdown
    print(f"\n    --- Step Classification (from rule IDs in explanation) ---")
    step1_pct = jr["step1_pct"]
    step2_pct = jr["step2_pct"]
    total_classified = jr["total_classified_journeys"]
    print(f"    Journeys with rule classification: {total_classified}")
    print(f"    Step1 (direct interaction):   {jr['journey_step_dist'].get('Step1', 0):>6d} ({step1_pct:5.1f}%)")
    print(f"    Step2 (extension-based):      {jr['journey_step_dist'].get('Step2', 0):>6d} ({step2_pct:5.1f}%)")
    unclassified = jr["journey_step_dist"].get("Unclassified", 0)
    if unclassified > 0:
        uncl_pct = 100.0 * unclassified / total_classified if total_classified > 0 else 0.0
        print(f"    Unclassified:                 {unclassified:>6d} ({uncl_pct:5.1f}%)")

    # Score distribution by step
    scores_by_step = jr.get("scores_by_step", {})
    score_dist_by_step = jr.get("score_dist_by_step", {})
    for step_label in ["Step1", "Step2", "Unclassified"]:
        if step_label in scores_by_step:
            print(f"\n    Relevance scores for {step_label} journeys:")
            print(fmt_stats(scores_by_step[step_label], 6))
            if step_label in score_dist_by_step:
                print(f"    Distribution:")
                print(fmt_dist(score_dist_by_step[step_label], 6))

    # Rule ID distribution
    rule_dist = jr.get("rule_id_dist", Counter())
    if rule_dist:
        print(f"\n    Rule ID distribution (top 20):")
        print(fmt_counter(rule_dist, 6, top_n=20))

    print(f"\n    Explanation type distribution (top 20):")
    print(fmt_counter(jr["explanation_dist"], 6, top_n=20))

    # --- E. Product Diversity ---
    pd_s = stats["product_diversity"]
    print(f"\n{SUBSEC_SEP}")
    print("  E. PRODUCT DIVERSITY METRICS")
    print(f"     Diversity of products within each journey, after collapsing near-duplicates.")
    print(f"     Group criteria: Same brand + product type + key variant attributes.")
    print(f"     diversity_ratio = num_productGroups / num_products")
    print(f"     Score: 2 (ratio=1.0) | 1 (0.6<=ratio<1.0) | 0 (ratio<0.6)")
    print(SUBSEC_SEP)
    print(f"    Diversity Score:")
    print(fmt_stats(pd_s["diversity_score"], 6))
    print(f"    Score distribution:")
    print(fmt_dist(pd_s["diversity_score_dist"], 6))
    print(f"\n    Computed Diversity Ratio (num_groups / num_products per journey):")
    print(fmt_stats(pd_s["computed_diversity_ratio"], 6))
    pd_corr_val = pd_s.get("product_count_vs_score_corr")
    print(f"\n    Correlation: product count vs diversity score: ", end="")
    if pd_corr_val is not None:
        print(f"r = {pd_corr_val:+.4f}")
    else:
        print("N/A (insufficient data)")
    print(f"\n    Product groups per journey:")
    print(fmt_stats(pd_s["groups_per_journey"], 6))
    print(f"\n    Product group size:")
    print(fmt_stats(pd_s["group_size"], 6))

    # --- F. Product Quality ---
    pq = stats["product_quality"]
    print(f"\n{SUBSEC_SEP}")
    print("  F. PRODUCT QUALITY METRICS")
    print(f"     Three core dimensions: Product-to-Journey Relevance (intent, attribute,")
    print(f"     gender alignment), Product Compliance (safety/legality), Seller Authority.")
    print(SUBSEC_SEP)
    print(f"    Total products evaluated: {pq['total_products_evaluated']}")
    print(f"    Products with qualityReason: {pq['pct_with_reason']:.1f}%")

    # Sub-dimension 1: Product-to-Journey Relevance
    print(f"\n    --- Product-to-Journey Relevance ---")

    print(f"\n        intentAlignment: Whether product directly supports journey's stated intent")
    print(f"          2=Clearly fulfills core intent | 1=Related but secondary | 0=Cannot resolve intent")
    print(f"\n    Intent Alignment:")
    print(fmt_stats(pq["intent_alignment"], 6))
    print(f"    Distribution:")
    print(fmt_dist(pq["intent_alignment_dist"], 6))

    print(f"\n        attributeAlignment: Whether product attributes match journey constraints")
    print(f"          2=Fully matches all constraints | 1=Category fits but attributes missing | 0=Contradicts core")
    print(f"\n    Attribute Alignment:")
    print(fmt_stats(pq["attribute_alignment"], 6))
    print(f"    Distribution:")
    print(fmt_dist(pq["attribute_alignment_dist"], 6))

    print(f"\n        genderAlignment: Whether product matches journey/user gender preference")
    print(f"          2=Clearly matches | 1=Reasonably compatible (unisex) | 0=Explicitly conflicts")
    print(f"\n    Gender Alignment:")
    print(fmt_stats(pq["gender_alignment"], 6))
    print(f"    Distribution:")
    print(fmt_dist(pq["gender_alignment_dist"], 6))

    print(f"\n    >> Product-to-Journey Relevance Composite (avg of intent + attribute + gender):")
    print(fmt_stats(pq["relevance_composite"], 6))

    # Sub-dimension 2: Product Compliance
    print(f"\n    --- Product Compliance ---")
    print(f"        NOTE: Effectively binary per UHRS guidelines (2=compliant, 0=fails).")
    print(f"        Score 1 may appear in data but is not a standard guideline value.")
    print(f"          2=Fully compliant and safe | 0=Fails compliance")
    print(f"\n    Product Compliance:")
    print(fmt_stats(pq["product_compliance"], 6))
    print(f"    Distribution:")
    print(fmt_dist(pq["product_compliance_dist"], 6))
    # Highlight binary nature
    compliance_dist = pq["product_compliance_dist"]
    score_1_count = compliance_dist.get(1, {}).get("count", 0)
    if score_1_count > 0:
        print(f"    ** Note: {score_1_count} products have compliance score=1 (not standard per guidelines) **")

    # Sub-dimension 3: Seller Authority
    print(f"\n    --- Seller Authority ---")
    print(f"        sellerAuthority: Trustworthiness/appropriateness of seller")
    print(f"          2=Reputable retailer/brand store | 1=Legitimate but not top-tier | 0=Unknown/suspicious")
    print(f"\n    Seller Authority:")
    print(fmt_stats(pq["seller_authority"], 6))
    print(f"    Distribution:")
    print(fmt_dist(pq["seller_authority_dist"], 6))

    # Overall composite
    print(f"\n    >> Overall Product Quality Composite (avg of all 5 sub-metrics):")
    print(fmt_stats(pq["composite"], 6))

    # Per-seller-authority quality breakdown
    quality_by_seller = pq.get("quality_by_seller_tier", {})
    if quality_by_seller:
        print(f"\n    --- Product Quality by Seller Authority Tier ---")
        for tier in sorted(quality_by_seller.keys()):
            tier_stats = quality_by_seller[tier]
            tier_label = {0: "Unknown/Suspicious", 1: "Legitimate (not top-tier)", 2: "Reputable/Brand Store"}.get(tier, f"Tier {tier}")
            print(f"    Seller Authority={tier} ({tier_label}):")
            print(fmt_stats(tier_stats, 6))

    # Quality reason categories
    reason_cats = pq.get("reason_categories", Counter())
    if reason_cats:
        print(f"\n    --- Top Quality Reason Categories (top 20) ---")
        print(fmt_counter(reason_cats, 6, top_n=20))

    # --- G. Journey-Type Breakdown ---
    jtb = stats.get("journey_type_breakdown", {})
    print(f"\n{SUBSEC_SEP}")
    print("  G. JOURNEY-TYPE BREAKDOWN ANALYSIS")
    print(f"     How metrics differ between 'explicit' (user directly interacted with")
    print(f"     products in category) and 'exploratory' (inferred from broader patterns) journeys.")
    print(SUBSEC_SEP)

    rel_by_type = jtb.get("relevance_by_type", {})
    pq_by_type = jtb.get("product_quality_by_type", {})
    pd_by_type = jtb.get("product_diversity_by_type", {})

    all_types = sorted(set(list(rel_by_type.keys()) + list(pq_by_type.keys()) + list(pd_by_type.keys())))

    if all_types:
        for jtype in all_types:
            print(f"\n    Journey Type: {jtype}")
            if jtype in rel_by_type:
                print(f"      Avg Relevance Score:")
                print(fmt_stats(rel_by_type[jtype], 8))
            if jtype in pd_by_type:
                print(f"      Avg Product Diversity Score:")
                print(fmt_stats(pd_by_type[jtype], 8))
            if jtype in pq_by_type:
                print(f"      Avg Product Quality Composite:")
                print(fmt_stats(pq_by_type[jtype], 8))
    else:
        print(f"    (no journey type data available)")

    # --- H. Correlations ---
    corr = stats["correlations"]
    print(f"\n{SUBSEC_SEP}")
    print("  H. CROSS-METRIC CORRELATIONS")
    print(SUBSEC_SEP)
    for key, label in [
        ("journey_div_vs_product_div", "Journey Diversity vs Avg Product Diversity (per-user)"),
        ("journey_rel_vs_product_qual", "Journey Relevance vs Avg Product Quality (per-journey)"),
        ("journey_value_vs_product_div", "Journey Value vs Product Diversity (per-journey)"),
        ("content_compliance_vs_product_compliance", "Content Compliance vs Avg Product Compliance (per-journey)"),
        ("num_signals_vs_journey_count", "Num Signals vs Journey Count (per-user)"),
    ]:
        val = corr.get(key)
        if val is not None:
            print(f"    {label}: r = {val:+.4f}")
        else:
            print(f"    {label}: N/A (insufficient data)")

    print()


def print_comparison(all_results: list[tuple[str, dict, dict]]) -> None:
    """Print a side-by-side comparison table for multiple files."""
    print()
    print(SECTION_SEP)
    print("  COMPARISON SUMMARY")
    print(SECTION_SEP)

    names = [Path(fp).stem for fp, _, _ in all_results]
    col_width = max(20, max(len(n) for n in names) + 2)

    def header_line():
        h = f"{'Metric':<55}"
        for n in names:
            h += f"{n:>{col_width}}"
        return h

    def data_line(label: str, values: list[str]):
        line = f"{label:<55}"
        for v in values:
            line += f"{v:>{col_width}}"
        return line

    print(f"\n{header_line()}")
    print("-" * (55 + col_width * len(names)))

    # Helper to safely extract journey type percentages
    def _jtype_pct(s, jtype):
        dist = s["journey_quality"]["journey_type_dist"]
        total = sum(dist.values()) or 1
        return f"{100.0 * dist.get(jtype, 0) / total:.1f}%"

    # Define comparison metrics (label, accessor returning formatted string)
    metrics = [
        # Volume
        ("Users", lambda s: f"{s['volume']['num_users']}"),
        ("Total Journeys", lambda s: f"{s['volume']['total_journeys']}"),
        ("Total Products", lambda s: f"{s['volume']['total_products']}"),
        ("Journeys/User (mean)", lambda s: f"{s['volume']['journeys_per_user']['mean']:.2f}"),
        ("Products/Journey (mean)", lambda s: f"{s['volume']['products_per_journey']['mean']:.2f}"),
        # Journey Diversity
        ("Journey Diversity Score (mean)", lambda s: f"{s['journey_diversity']['diversity_score']['mean']:.4f}"),
        ("Journey Diversity Score=2 (%)", lambda s: f"{s['journey_diversity']['diversity_score_dist'][2]['pct']:.1f}%"),
        ("Journey Diversity Ratio (mean)", lambda s: f"{s['journey_diversity']['diversity_ratio']['mean']:.4f}"),
        # Journey Quality
        ("Journey Value (mean)", lambda s: f"{s['journey_quality']['journey_value']['mean']:.4f}"),
        ("Content Compliance (mean)", lambda s: f"{s['journey_quality']['content_compliance']['mean']:.4f}"),
        ("Appropriateness Composite (mean)", lambda s: f"{s['journey_quality']['appropriateness_composite']['mean']:.4f}"),
        ("Tone (mean)", lambda s: f"{s['journey_quality']['tone']['mean']:.4f}"),
        ("Self-Coherence (mean)", lambda s: f"{s['journey_quality']['self_coherence']['mean']:.4f}"),
        ("Title Quality Composite (mean)", lambda s: f"{s['journey_quality']['title_quality_composite']['mean']:.4f}"),
        ("Overall Journey Quality Composite", lambda s: f"{s['journey_quality']['composite']['mean']:.4f}"),
        ("Explicit Journeys (%)", lambda s: _jtype_pct(s, "explicit")),
        ("Exploratory Journeys (%)", lambda s: _jtype_pct(s, "exploratory")),
        # Journey Relevance
        ("Shopping Relevance (mean)", lambda s: f"{s['journey_relevance']['relevance_score']['mean']:.4f}"),
        ("Shopping Relevance Score=2 (%)", lambda s: f"{s['journey_relevance']['relevance_score_dist'][2]['pct']:.1f}%"),
        ("Step1 (direct interaction) (%)", lambda s: f"{s['journey_relevance']['step1_pct']:.1f}%"),
        ("Step2 (extension-based) (%)", lambda s: f"{s['journey_relevance']['step2_pct']:.1f}%"),
        # Product Diversity
        ("Product Diversity Score (mean)", lambda s: f"{s['product_diversity']['diversity_score']['mean']:.4f}"),
        ("Product Diversity Score=2 (%)", lambda s: f"{s['product_diversity']['diversity_score_dist'][2]['pct']:.1f}%"),
        ("Computed Diversity Ratio (mean)", lambda s: f"{s['product_diversity']['computed_diversity_ratio']['mean']:.4f}"),
        # Product Quality
        ("Intent Alignment (mean)", lambda s: f"{s['product_quality']['intent_alignment']['mean']:.4f}"),
        ("Attribute Alignment (mean)", lambda s: f"{s['product_quality']['attribute_alignment']['mean']:.4f}"),
        ("Gender Alignment (mean)", lambda s: f"{s['product_quality']['gender_alignment']['mean']:.4f}"),
        ("P2J Relevance Composite (mean)", lambda s: f"{s['product_quality']['relevance_composite']['mean']:.4f}"),
        ("Product Compliance (mean)", lambda s: f"{s['product_quality']['product_compliance']['mean']:.4f}"),
        ("Seller Authority (mean)", lambda s: f"{s['product_quality']['seller_authority']['mean']:.4f}"),
        ("Overall Product Quality Composite", lambda s: f"{s['product_quality']['composite']['mean']:.4f}"),
        ("Products w/ Reason (%)", lambda s: f"{s['product_quality']['pct_with_reason']:.1f}%"),
        # Journey-Type Breakdown (explicit vs exploratory)
        ("Relevance (explicit journeys)", lambda s: f"{s['journey_type_breakdown']['relevance_by_type'].get('explicit', {}).get('mean', np.nan):.4f}"),
        ("Relevance (exploratory journeys)", lambda s: f"{s['journey_type_breakdown']['relevance_by_type'].get('exploratory', {}).get('mean', np.nan):.4f}"),
        ("Prod Quality (explicit journeys)", lambda s: f"{s['journey_type_breakdown']['product_quality_by_type'].get('explicit', {}).get('mean', np.nan):.4f}"),
        ("Prod Quality (exploratory journeys)", lambda s: f"{s['journey_type_breakdown']['product_quality_by_type'].get('exploratory', {}).get('mean', np.nan):.4f}"),
        ("Prod Diversity (explicit journeys)", lambda s: f"{s['journey_type_breakdown']['product_diversity_by_type'].get('explicit', {}).get('mean', np.nan):.4f}"),
        ("Prod Diversity (exploratory journeys)", lambda s: f"{s['journey_type_breakdown']['product_diversity_by_type'].get('exploratory', {}).get('mean', np.nan):.4f}"),
        # Correlations
        (
            "Corr: JourneyDiv~ProductDiv",
            lambda s: f"{s['correlations']['journey_div_vs_product_div']:+.4f}"
            if s["correlations"]["journey_div_vs_product_div"] is not None
            else "N/A",
        ),
        (
            "Corr: JourneyRel~ProductQual",
            lambda s: f"{s['correlations']['journey_rel_vs_product_qual']:+.4f}"
            if s["correlations"]["journey_rel_vs_product_qual"] is not None
            else "N/A",
        ),
        (
            "Corr: JourneyValue~ProductDiv",
            lambda s: f"{s['correlations']['journey_value_vs_product_div']:+.4f}"
            if s["correlations"]["journey_value_vs_product_div"] is not None
            else "N/A",
        ),
        (
            "Corr: ContentCompl~ProductCompl",
            lambda s: f"{s['correlations']['content_compliance_vs_product_compliance']:+.4f}"
            if s["correlations"]["content_compliance_vs_product_compliance"] is not None
            else "N/A",
        ),
    ]

    for label, accessor in metrics:
        values = []
        for _, stats, _ in all_results:
            try:
                values.append(accessor(stats))
            except (KeyError, TypeError, ZeroDivisionError):
                values.append("N/A")
        print(data_line(label, values))

    print()


# ---------------------------------------------------------------------------
# Concise Summary Report
# ---------------------------------------------------------------------------

def compute_concise_metrics(rows: list[dict], total: int, skipped: int) -> dict:
    """
    Compute the concise summary metrics from raw rows.

    Follows the calculation logic in Step3_CalculateMetrics_select_metrics.py:
    - Good Rate: % of items where ALL sub-scores >= 1 (for composites) or score >= 1 (for single)
    - Strict Good Rate: % of items where ALL sub-scores == 2 (for composites) or score == 2 (for single)
    """
    # --- Coverage metrics ---
    num_sampled = total
    num_succeed = total - skipped
    num_with_journey = len(rows)  # rows with at least 1 journey (parse_file skips empty)
    coverage = 100.0 * num_with_journey / num_succeed if num_succeed > 0 else 0.0
    total_journeys = sum(len(r["journeys"]) for r in rows)
    density = total_journeys / num_with_journey if num_with_journey > 0 else 0.0

    # --- Journey to User Relevance (per journey, single score) ---
    jr_total, jr_good, jr_strict = 0, 0, 0
    for r in rows:
        jr_list = r["journey_relevance"]
        if not isinstance(jr_list, list):
            continue
        for jr in jr_list:
            if not isinstance(jr, dict):
                continue
            s = jr.get("shoppingRelevanceScore")
            if s is None:
                continue
            s = int(s)
            jr_total += 1
            if s >= 1:
                jr_good += 1
            if s >= 2:
                jr_strict += 1

    # --- Journey Appropriateness (per journey, composite: journeyValue AND contentCompliance) ---
    ja_total, ja_good, ja_strict = 0, 0, 0
    for r in rows:
        jq_list = r["journey_quality"]
        if not isinstance(jq_list, list):
            continue
        for jq in jq_list:
            if not isinstance(jq, dict):
                continue
            approp = jq.get("journeyAppropriateness", {})
            jv = approp.get("journeyValue")
            cc = approp.get("contentCompliance")
            if jv is None or cc is None:
                continue
            jv, cc = int(jv), int(cc)
            ja_total += 1
            if jv >= 1 and cc >= 1:
                ja_good += 1
            if jv == 2 and cc == 2:
                ja_strict += 1

    # --- Journey Title Quality (per journey, composite: tone AND selfCoherence) ---
    jtq_total, jtq_good, jtq_strict = 0, 0, 0
    for r in rows:
        jq_list = r["journey_quality"]
        if not isinstance(jq_list, list):
            continue
        for jq in jq_list:
            if not isinstance(jq, dict):
                continue
            tq = jq.get("journeyTitleQuality", {})
            tn = tq.get("tone")
            sc = tq.get("selfCoherence")
            if tn is None or sc is None:
                continue
            tn, sc = int(tn), int(sc)
            jtq_total += 1
            if tn >= 1 and sc >= 1:
                jtq_good += 1
            if tn == 2 and sc == 2:
                jtq_strict += 1

    # --- Journey Diversity (per user, single score) ---
    jd_total, jd_good, jd_strict = 0, 0, 0
    for r in rows:
        jd = r["journey_diversity"]
        if not isinstance(jd, dict):
            continue
        s = jd.get("diversityScore")
        if s is None:
            continue
        s = int(s)
        jd_total += 1
        if s >= 1:
            jd_good += 1
        if s >= 2:
            jd_strict += 1

    # --- Product to Journey Relevance (per product, composite: ia AND aa AND ga) ---
    p2j_total, p2j_good, p2j_strict = 0, 0, 0
    # --- Product Compliance (per product, single score) ---
    pc_total, pc_good, pc_strict = 0, 0, 0
    # --- Seller Authority (per product, single score) ---
    sa_total, sa_good, sa_strict = 0, 0, 0

    for r in rows:
        pq_list = r["product_quality"]
        if not isinstance(pq_list, list):
            continue
        for pq_journey in pq_list:
            if not isinstance(pq_journey, dict):
                continue
            products = pq_journey.get("productQuality", [])
            if not isinstance(products, list):
                continue
            for prod in products:
                if not isinstance(prod, dict):
                    continue
                rel = prod.get("productToJourneyRelevance", {})
                ia = rel.get("intentAlignment")
                aa = rel.get("attributeAlignment")
                ga = rel.get("genderAlignment")
                pc = prod.get("productCompliance")
                sa = prod.get("sellerAuthority")

                # P2J Relevance
                if ia is not None and aa is not None and ga is not None:
                    ia, aa, ga = int(ia), int(aa), int(ga)
                    p2j_total += 1
                    if ia >= 1 and aa >= 1 and ga >= 1:
                        p2j_good += 1
                    if ia == 2 and aa == 2 and ga == 2:
                        p2j_strict += 1

                # Product Compliance
                if pc is not None:
                    pc = int(pc)
                    pc_total += 1
                    if pc >= 1:
                        pc_good += 1
                    if pc >= 2:
                        pc_strict += 1

                # Seller Authority
                if sa is not None:
                    sa = int(sa)
                    sa_total += 1
                    if sa >= 1:
                        sa_good += 1
                    if sa >= 2:
                        sa_strict += 1

    # --- Product Diversity (per journey, single score) ---
    pd_total, pd_good, pd_strict = 0, 0, 0
    for r in rows:
        pd_list = r["product_diversity"]
        if not isinstance(pd_list, list):
            continue
        for pd_item in pd_list:
            if not isinstance(pd_item, dict):
                continue
            s = pd_item.get("diversityScore")
            if s is None:
                continue
            s = int(s)
            pd_total += 1
            if s >= 1:
                pd_good += 1
            if s >= 2:
                pd_strict += 1

    def _pct(num, den):
        return 100.0 * num / den if den > 0 else 0.0

    return {
        "coverage": {
            "num_sampled": num_sampled,
            "num_succeed": num_succeed,
            "num_with_journey": num_with_journey,
            "coverage_pct": coverage,
            "density": density,
        },
        "metrics": {
            "journey_relevance":      (_pct(jr_good, jr_total),  _pct(jr_strict, jr_total)),
            "journey_appropriateness": (_pct(ja_good, ja_total),  _pct(ja_strict, ja_total)),
            "journey_title_quality":   (_pct(jtq_good, jtq_total), _pct(jtq_strict, jtq_total)),
            "journey_diversity":       (_pct(jd_good, jd_total),  _pct(jd_strict, jd_total)),
            "p2j_relevance":           (_pct(p2j_good, p2j_total), _pct(p2j_strict, p2j_total)),
            "product_compliance":      (_pct(pc_good, pc_total),  _pct(pc_strict, pc_total)),
            "seller_authority":        (_pct(sa_good, sa_total),  _pct(sa_strict, sa_total)),
            "product_diversity":       (_pct(pd_good, pd_total),  _pct(pd_strict, pd_total)),
        },
    }


def print_concise_report(filepath: str, concise: dict) -> None:
    """Print the concise summary report for one file."""
    name = Path(filepath).stem
    cov = concise["coverage"]
    met = concise["metrics"]

    print()
    print(SECTION_SEP)
    print(f"  CONCISE SUMMARY: {name}")
    print(SECTION_SEP)

    # --- Table 1: Coverage ---
    print()
    print(f"  {'No.':<6}{'Data':<55}{'Value':>12}")
    print(f"  {'-'*6}{'-'*55}{'-'*12}")
    coverage_rows = [
        ("1", "# Random sampled online users", f"{cov['num_sampled']}"),
        ("2", "# Succeed running users", f"{cov['num_succeed']}"),
        ("3", "# of users with at least 1 journey", f"{cov['num_with_journey']}"),
        ("4", "Coverage (% of users with at least 1 journey)", f"{cov['coverage_pct']:.2f}"),
        ("5", "Density (avg. # of journeys per covered user)", f"{cov['density']:.2f}"),
    ]
    for no, data, val in coverage_rows:
        print(f"  {no:<6}{data:<55}{val:>12}")

    # --- Table 2: Metrics ---
    print()
    print(f"  {'No.':<6}{'Metrics':<40}{'Good Rate':>12}{'Strict Good Rate':>18}")
    print(f"  {'-'*6}{'-'*40}{'-'*12}{'-'*18}")
    metrics_rows = [
        ("1", "Journey to User Relevance", met["journey_relevance"]),
        ("2", "Journey Appropriateness", met["journey_appropriateness"]),
        ("3", "Journey Title Quality", met["journey_title_quality"]),
        ("4", "Journey Diversity", met["journey_diversity"]),
        ("5", "Product to Journey Relevance", met["p2j_relevance"]),
        ("6", "Product Compliance", met["product_compliance"]),
        ("7", "Seller Authority", met["seller_authority"]),
        ("8", "Product Diversity", met["product_diversity"]),
    ]
    for no, label, (good, strict) in metrics_rows:
        print(f"  {no:<6}{label:<40}{good:>11.2f}%{strict:>17.2f}%")

    print()


def print_concise_comparison(all_results: list[tuple[str, dict, dict, dict]]) -> None:
    """Print a side-by-side concise comparison for multiple files."""
    print()
    print(SECTION_SEP)
    print("  CONCISE COMPARISON")
    print(SECTION_SEP)

    names = [Path(fp).stem for fp, _, _, _ in all_results]
    concise_list = [c for _, _, _, c in all_results]
    name_width = max(20, max(len(n) for n in names) + 2)

    def _header():
        h = f"  {'Data':<50}"
        for n in names:
            h += f"{n:>{name_width}}"
        return h

    def _sep():
        return f"  {'-'*50}" + f"{'-'*name_width}" * len(names)

    # --- Coverage comparison ---
    print()
    print(_header())
    print(_sep())

    cov_fields = [
        ("# Random sampled online users", lambda c: f"{c['coverage']['num_sampled']}"),
        ("# Succeed running users", lambda c: f"{c['coverage']['num_succeed']}"),
        ("# of users with at least 1 journey", lambda c: f"{c['coverage']['num_with_journey']}"),
        ("Coverage (%)", lambda c: f"{c['coverage']['coverage_pct']:.2f}"),
        ("Density (avg. # journeys per covered)", lambda c: f"{c['coverage']['density']:.2f}"),
    ]
    for label, accessor in cov_fields:
        line = f"  {label:<50}"
        for concise in concise_list:
            line += f"{accessor(concise):>{name_width}}"
        print(line)

    # --- Metrics: Good Rate ---
    metric_keys = [
        ("Journey to User Relevance", "journey_relevance"),
        ("Journey Appropriateness", "journey_appropriateness"),
        ("Journey Title Quality", "journey_title_quality"),
        ("Journey Diversity", "journey_diversity"),
        ("Product to Journey Relevance", "p2j_relevance"),
        ("Product Compliance", "product_compliance"),
        ("Seller Authority", "seller_authority"),
        ("Product Diversity", "product_diversity"),
    ]

    print()
    header = f"  {'Metrics (Good Rate: score >= 1)':<50}"
    for n in names:
        header += f"{n:>{name_width}}"
    print(header)
    print(_sep())
    for label, key in metric_keys:
        line = f"  {label:<50}"
        for concise in concise_list:
            good, _ = concise["metrics"][key]
            line += f"{f'{good:.2f}%':>{name_width}}"
        print(line)

    # --- Metrics: Strict Good Rate ---
    print()
    header = f"  {'Metrics (Strict Good Rate: score == 2)':<50}"
    for n in names:
        header += f"{n:>{name_width}}"
    print(header)
    print(_sep())
    for label, key in metric_keys:
        line = f"  {label:<50}"
        for concise in concise_list:
            _, strict = concise["metrics"][key]
            line += f"{f'{strict:.2f}%':>{name_width}}"
        print(line)

    print()


# ---------------------------------------------------------------------------
# Correlation with UserSignals (deferred enrichment)
# ---------------------------------------------------------------------------

def enrich_signal_correlation(filepath: str, stats: dict) -> None:
    """
    Parse UserSignals to compute num_signals vs journey_count correlation.
    Done as a second pass to keep the main parse clean.
    """
    csv.field_size_limit(10 * 1024 * 1024)

    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            signals = safe_json_loads(raw.get("UserSignals", ""))
            journeys = safe_json_loads(raw.get("ShoppingJourneys", ""))
            if isinstance(signals, list) and isinstance(journeys, list):
                pairs.append((len(signals), len(journeys)))

    if len(pairs) >= 3:
        a, b = zip(*pairs)
        stats["correlations"]["num_signals_vs_journey_count"] = safe_corr(a, b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics for shopping journey evaluation TSV files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more TSV file paths to analyze.",
    )
    args = parser.parse_args()

    all_results = []  # (filepath, stats_dict, meta_dict, concise_dict)

    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"ERROR: File not found: {filepath}", file=sys.stderr)
            continue

        print(f"\nParsing {filepath} ...", file=sys.stderr)
        rows, total, skipped = parse_file(filepath)
        print(f"  -> {total} total rows, {len(rows)} parsed, {skipped} skipped", file=sys.stderr)

        if not rows:
            print(f"  -> No valid data rows. Skipping.", file=sys.stderr)
            continue

        stats = compute_stats(rows)

        # Enrich with signal correlation (second pass)
        enrich_signal_correlation(filepath, stats)

        # Compute concise metrics from raw rows
        concise = compute_concise_metrics(rows, total, skipped)

        meta = {"total": total, "skipped": skipped}
        all_results.append((filepath, stats, meta, concise))

        # Print concise summary first
        print_concise_report(filepath, concise)
        # Then print detailed report
        print_report(filepath, stats, total, skipped)

    # Comparison tables when multiple files are provided
    if len(all_results) > 1:
        print_concise_comparison(all_results)
        # Adapt for print_comparison which expects 3-tuples
        comparison_results = [(fp, st, mt) for fp, st, mt, _ in all_results]
        print_comparison(comparison_results)


if __name__ == "__main__":
    main()
