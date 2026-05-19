"""
step7_stats.py
==============

Statistics tool for pipeline outputs (step3, step5, step6).

Auto-detects which step produced the input TSV based on column names,
then prints comprehensive statistics.

Usage:
    python step7_stats.py --input_file /path/to/output.tsv
    python step7_stats.py --input_file /path/to/output.tsv --max_rows 10000
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)


# ============================================================================ #
# Backslash-quote cleanup (same as step5/step6)                                #
# ============================================================================ #
def _fix_backslash_json(text):
    if not text or not text.strip():
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    _BS, _Q = chr(92), chr(34)
    _PH = "\x00_ESC_Q_\x00"
    cur = text
    for _ in range(3):
        if _BS + _Q not in cur:
            break
        cur = cur.replace(_BS + _BS + _Q, _PH)
        cur = cur.replace(_BS + _Q, _Q)
        cur = cur.replace(_PH, _BS + _Q)
        try:
            return json.loads(cur)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


# ============================================================================ #
# Distribution helper                                                          #
# ============================================================================ #
def _dist(values, label, unit=""):
    """Print distribution stats for a list of numeric values."""
    if not values:
        print(f"  {label}: (no data)")
        return
    s = sorted(values)
    n = len(s)
    total = sum(s)
    avg = total / n
    p25 = s[int(n * 0.25)]
    p50 = s[int(n * 0.50)]
    p75 = s[int(n * 0.75)]
    p90 = s[min(int(n * 0.90), n - 1)]
    p99 = s[min(int(n * 0.99), n - 1)]
    u = f" {unit}" if unit else ""
    print(f"  {label}:")
    print(f"    N={n:,}  Total={total:,}{u}  "
          f"Min={s[0]}{u}  P25={p25}{u}  P50={p50}{u}  "
          f"P75={p75}{u}  P90={p90}{u}  P99={p99}{u}  Max={s[-1]}{u}  "
          f"Mean={avg:.1f}{u}")


def _bucket_dist(values, label, buckets=None):
    """Print bucketed distribution."""
    if not values:
        return
    if buckets is None:
        buckets = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 10),
                   (11, 15), (16, 20), (21, 30), (31, 50), (51, None)]
    counts = Counter()
    for v in values:
        for lo, hi in buckets:
            if hi is None:
                if v >= lo:
                    counts[f"{lo}+"] = counts.get(f"{lo}+", 0) + 1
                    break
            elif lo <= v <= hi:
                lbl = f"{lo}" if lo == hi else f"{lo}-{hi}"
                counts[lbl] = counts.get(lbl, 0) + 1
                break
    print(f"  {label} distribution:")
    for lo, hi in buckets:
        if hi is None:
            lbl = f"{lo}+"
        elif lo == hi:
            lbl = f"{lo}"
        else:
            lbl = f"{lo}-{hi}"
        cnt = counts.get(lbl, 0)
        pct = cnt / len(values) * 100 if values else 0
        bar = "█" * int(pct / 2)
        print(f"    {lbl:>6s}: {cnt:>8,} ({pct:5.1f}%) {bar}")


# ============================================================================ #
# Extract journeys/queries/products from a JSON column                         #
# ============================================================================ #
def _parse_journeys_column(text):
    """Parse a ContinuedJourneys JSON column. Returns list of journey dicts."""
    if not text or not text.strip():
        return None
    obj = _fix_backslash_json(text)
    if obj and "ContinuedJourneys" in obj:
        return obj["ContinuedJourneys"]
    return None


def _count_queries_and_products(journeys):
    """From a list of journey dicts, count total queries and products."""
    n_queries = 0
    n_products = 0
    products_per_journey = []
    queries_per_journey = []
    for j in journeys:
        if not isinstance(j, dict):
            continue
        # Products directly on journey (step6 ranked format)
        if "Products" in j and isinstance(j["Products"], list):
            products_per_journey.append(len(j["Products"]))
            n_products += len(j["Products"])
        # Queries with Products (step5 JWP format)
        queries = j.get("Queries", [])
        if isinstance(queries, list):
            queries_per_journey.append(len(queries))
            n_queries += len(queries)
            for q in queries:
                if isinstance(q, dict) and "Products" in q:
                    prods = q.get("Products", [])
                    if isinstance(prods, list):
                        n_products += len(prods)
                        products_per_journey.append(len(prods))
    return n_queries, n_products, queries_per_journey, products_per_journey


# ============================================================================ #
# Detect step and analyze                                                      #
# ============================================================================ #
def detect_step(fieldnames):
    """Detect which pipeline step produced the TSV."""
    fset = set(fieldnames)
    if "RankedJourneys" in fset:
        return "step6"
    if "JourneyWithProducts" in fset:
        return "step5"
    if "OUTPUT" in fset or "ShoppingJourneys" in fset:
        return "step3"
    return "unknown"


def run_stats(input_file, max_rows=0):
    print("=" * 70)
    print(f"  Pipeline Statistics: {os.path.basename(input_file)}")
    print("=" * 70)

    file_size = os.path.getsize(input_file)
    print(f"  File: {input_file}")
    print(f"  Size: {file_size / (1024**3):.2f} GB ({file_size / (1024**2):.1f} MB)")
    if max_rows > 0:
        print(f"  Max rows: {max_rows:,}")
    print()

    # Read data
    rows = []
    with open(input_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)
            if max_rows > 0 and len(rows) >= max_rows:
                break

    step = detect_step(fieldnames)
    print(f"  Columns: {fieldnames}")
    print(f"  Detected step: {step}")
    print(f"  Total rows: {len(rows):,}")
    print()

    # ================================================================
    # Basic field coverage
    # ================================================================
    print("-" * 70)
    print("Field Coverage")
    print("-" * 70)
    for col in fieldnames:
        non_empty = sum(1 for r in rows if r.get(col, "").strip())
        pct = non_empty / len(rows) * 100 if rows else 0
        print(f"  {col:<30s}  {non_empty:>10,} / {len(rows):,}  ({pct:.1f}%)")
    print()

    # ================================================================
    # Events statistics
    # ================================================================
    events_col = "ReadableUserEvents"
    if events_col in fieldnames:
        print("-" * 70)
        print("User Events")
        print("-" * 70)
        event_counts = []
        for r in rows:
            ev = r.get(events_col, "")
            # Count events: split by #N# or newline
            if ev:
                lines = [l for l in ev.replace("#N#", "\n").split("\n")
                         if l.strip()]
                event_counts.append(len(lines))
            else:
                event_counts.append(0)
        _dist(event_counts, "Events per user")
        print()

    # ================================================================
    # Profile statistics
    # ================================================================
    profile_col = "ShoppingProfile"
    if profile_col in fieldnames:
        print("-" * 70)
        print("Shopping Profile")
        print("-" * 70)
        has_profile = sum(1 for r in rows if r.get(profile_col, "").strip())
        print(f"  Users with profile: {has_profile:,} / {len(rows):,} "
              f"({has_profile / len(rows) * 100:.1f}%)")
        print()

    # ================================================================
    # Journey column analysis (step3: OUTPUT, step5: ShoppingJourneys)
    # ================================================================
    journey_col = None
    if step == "step3":
        journey_col = "OUTPUT" if "OUTPUT" in fieldnames else "ShoppingJourneys"
    elif step in ("step5", "step6"):
        journey_col = "ShoppingJourneys" if "ShoppingJourneys" in fieldnames else None

    if journey_col and journey_col in fieldnames:
        print("-" * 70)
        print(f"Journey Column: {journey_col}")
        print("-" * 70)
        j_counts = []  # journeys per user
        q_counts = []  # queries per user
        p_counts = []  # products per user (if in queries)
        j_per_journey_queries = []
        empty_journey_users = 0
        invalid_json_users = 0
        for r in rows:
            raw = r.get(journey_col, "").strip()
            if not raw:
                empty_journey_users += 1
                j_counts.append(0)
                continue
            journeys = _parse_journeys_column(raw)
            if journeys is None:
                invalid_json_users += 1
                j_counts.append(0)
                continue
            j_counts.append(len(journeys))
            if not journeys:
                empty_journey_users += 1
            nq, np_, qpj, ppj = _count_queries_and_products(journeys)
            q_counts.append(nq)
            p_counts.append(np_)
            j_per_journey_queries.extend(qpj)

        print(f"  Users with empty/no journey: {empty_journey_users:,}")
        print(f"  Users with invalid JSON: {invalid_json_users:,}")
        _dist(j_counts, "Journeys per user")
        _bucket_dist(j_counts, "Journeys per user",
                      [(0, 0), (1, 3), (4, 5), (6, 10), (11, 15), (16, 20), (21, None)])
        if q_counts:
            _dist(q_counts, "Queries per user (total)")
            _dist(j_per_journey_queries, "Queries per journey")
        if any(p > 0 for p in p_counts):
            _dist(p_counts, "Products per user (total)")
        print()

    # ================================================================
    # JourneyWithProducts (step5, step6)
    # ================================================================
    jwp_col = "JourneyWithProducts"
    if jwp_col in fieldnames:
        print("-" * 70)
        print(f"JourneyWithProducts (before ranking)")
        print("-" * 70)
        jwp_j_counts = []
        jwp_q_counts = []
        jwp_p_counts = []
        jwp_prods_per_query = []
        jwp_empty = 0
        for r in rows:
            raw = r.get(jwp_col, "").strip()
            if not raw:
                jwp_empty += 1
                jwp_j_counts.append(0)
                continue
            journeys = _parse_journeys_column(raw)
            if journeys is None:
                jwp_j_counts.append(0)
                continue
            jwp_j_counts.append(len(journeys))
            nq, np_, qpj, ppq = _count_queries_and_products(journeys)
            jwp_q_counts.append(nq)
            jwp_p_counts.append(np_)
            jwp_prods_per_query.extend(ppq)

        print(f"  Users with empty JWP: {jwp_empty:,}")
        _dist(jwp_j_counts, "Journeys per user")
        _dist(jwp_q_counts, "Queries per user")
        _dist(jwp_p_counts, "Products per user")
        if jwp_prods_per_query:
            _dist(jwp_prods_per_query, "Products per query")
        print()

    # ================================================================
    # RankedJourneys (step6 only)
    # ================================================================
    ranked_col = "RankedJourneys"
    if ranked_col in fieldnames:
        print("-" * 70)
        print(f"RankedJourneys (after ranking)")
        print("-" * 70)
        rj_j_counts = []
        rj_p_counts = []
        rj_prods_per_journey = []
        rj_empty = 0
        rj_total_candidates = []
        rj_selected = []
        rj_filtered = []
        for r in rows:
            raw = r.get(ranked_col, "").strip()
            if not raw:
                rj_empty += 1
                rj_j_counts.append(0)
                continue
            journeys = _parse_journeys_column(raw)
            if journeys is None:
                rj_j_counts.append(0)
                continue
            rj_j_counts.append(len(journeys))
            user_prods = 0
            for j in journeys:
                if not isinstance(j, dict):
                    continue
                prods = j.get("Products", [])
                n_p = len(prods) if isinstance(prods, list) else 0
                rj_prods_per_journey.append(n_p)
                user_prods += n_p
                # RankingSummary
                summary = j.get("RankingSummary", {})
                if isinstance(summary, dict):
                    tc = summary.get("totalCandidates")
                    sc = summary.get("selectedCount")
                    fc = summary.get("filteredCount")
                    if tc is not None:
                        rj_total_candidates.append(int(tc))
                    if sc is not None:
                        rj_selected.append(int(sc))
                    if fc is not None:
                        rj_filtered.append(int(fc))
            rj_p_counts.append(user_prods)

        print(f"  Users with empty RankedJourneys: {rj_empty:,}")
        _dist(rj_j_counts, "Journeys per user (after ranking)")
        _dist(rj_p_counts, "Products per user (after ranking)")
        _dist(rj_prods_per_journey, "Products per journey (after ranking)")
        _bucket_dist(rj_prods_per_journey, "Products per journey",
                      [(0, 0), (1, 5), (6, 10), (11, 15), (16, 20),
                       (21, 25), (26, 30), (31, None)])

        # Ranking effectiveness
        if rj_total_candidates and rj_selected:
            print()
            print("  --- Ranking Effectiveness (from RankingSummary) ---")
            _dist(rj_total_candidates, "Candidates per journey (before)")
            _dist(rj_selected, "Selected per journey (after)")
            _dist(rj_filtered, "Filtered per journey")
            total_before = sum(rj_total_candidates)
            total_after = sum(rj_selected)
            if total_before > 0:
                retention = total_after / total_before * 100
                print(f"    Overall retention: {total_after:,}/{total_before:,} "
                      f"({retention:.1f}%)")

        # Before vs after comparison (if both JWP and Ranked exist)
        if jwp_col in fieldnames and jwp_j_counts and rj_j_counts:
            print()
            print("  --- Before vs After Ranking ---")
            jwp_total_j = sum(jwp_j_counts)
            rj_total_j = sum(rj_j_counts)
            jwp_total_p = sum(jwp_p_counts) if jwp_p_counts else 0
            rj_total_p = sum(rj_p_counts)
            print(f"    Journeys: {jwp_total_j:,} -> {rj_total_j:,} "
                  f"({rj_total_j / max(jwp_total_j, 1) * 100:.1f}%)")
            if jwp_total_p > 0:
                print(f"    Products: {jwp_total_p:,} -> {rj_total_p:,} "
                      f"({rj_total_p / max(jwp_total_p, 1) * 100:.1f}%)")
            # Users who lost all journeys
            lost_all = sum(1 for jc, rc in zip(jwp_j_counts, rj_j_counts)
                           if jc > 0 and rc == 0)
            print(f"    Users who lost all journeys: {lost_all:,}")

        print()

    # ================================================================
    # Summary
    # ================================================================
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Step:       {step}")
    print(f"  Users:      {len(rows):,}")
    if journey_col and journey_col in fieldnames:
        valid_j = sum(1 for c in j_counts if c > 0)
        print(f"  With journey: {valid_j:,} ({valid_j / len(rows) * 100:.1f}%)")
    if jwp_col in fieldnames:
        valid_jwp = sum(1 for c in jwp_j_counts if c > 0)
        print(f"  With JWP:   {valid_jwp:,} ({valid_jwp / len(rows) * 100:.1f}%)")
    if ranked_col in fieldnames:
        valid_rj = sum(1 for c in rj_j_counts if c > 0)
        print(f"  With ranked: {valid_rj:,} ({valid_rj / len(rows) * 100:.1f}%)")
    print("=" * 70)


# ============================================================================ #
# CLI                                                                          #
# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(
        description="Step 7: Statistics for pipeline outputs (step3/5/6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_file", type=str, required=True,
                   help="Path to TSV file (output of step3, step5, or step6).")
    p.add_argument("--max_rows", type=int, default=0,
                   help="If >0, only analyze first N rows (for quick preview).")
    return p.parse_args()


def main():
    args = parse_args()
    run_stats(args.input_file, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
