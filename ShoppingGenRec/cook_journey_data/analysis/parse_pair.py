"""
Parse two pipeline output files (JSONL from step8 or raw Schema B/C),
pair users by StableId, compute stats, output paired_data.json.

Usage:
  python3 parse_pair.py --run-name <name> \
    --p1-file <path> --p2-file <path> \
    [--p1-name <label>] [--p2-name <label>] \
    [--map-file <path>] [--output-dir <path>]

Output:
  analysis/<run_name>/paired_data.json  (local)
  <output_dir>/<run_name>/              (remote copy, if --output-dir given)
"""
import json, os, sys, argparse, io, csv

csv.field_size_limit(2**30)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Phase 1: parse & pair two pipeline outputs")
    ap.add_argument("--run-name", required=True, help="Run directory name under analysis/")
    ap.add_argument("--p1-file", required=True, help="Path to P1 output (JSONL or TSV)")
    ap.add_argument("--p2-file", required=True, help="Path to P2 output (JSONL or TSV)")
    ap.add_argument("--p1-name", default="P1", help="Label for P1 pipeline")
    ap.add_argument("--p2-name", default="P2", help="Label for P2 pipeline")
    ap.add_argument("--map-file", default=None, help="PicassoId↔StableId mapping TSV")
    ap.add_argument("--output-dir", default=None,
                    help="Additional output directory (e.g. /cosmos/...); results are copied there too")
    return ap.parse_args()


# ── Format / Schema detection ────────────────────────────────────────────────

def detect_format(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    if first.startswith("{"):
        return "jsonl"
    if first.startswith("["):
        return "json_array"
    if "\t" in first:
        return "tsv"
    return "unknown"


def detect_schema(record):
    """Return one of: A, B, B_ranked, C, unknown"""
    # Schema A: TSV with response.journeyProfiles
    if "response" in record and "journeyProfiles" in str(record.get("response", "")):
        return "A"
    # Schema B/C/B_ranked
    journeys = record.get("journeys", [])
    if journeys:
        j0 = journeys[0]
        products = j0.get("products", [])
        if products:
            p0 = products[0]
            if "tid" in p0:
                return "C"
            if "query" in p0 and "matched_products" in p0:
                return "B"
            if "OriginalQuery" in p0 or "Title" in p0:
                return "B_ranked"  # step8 post-reranked flat format
    if "stableid" in record:
        return "B"
    return "unknown"


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_tsv(path):
    """Load TSV with possible large fields (step3/5/6 outputs)."""
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Try to parse JSON columns
            for col in ("RankedJourneys", "JourneyWithProducts", "OUTPUT", "response"):
                if col in row and row[col]:
                    try:
                        row[col] = json.loads(row[col])
                    except (json.JSONDecodeError, TypeError):
                        pass
            records.append(row)
    return records


def load_file(path):
    fmt = detect_format(path)
    if fmt == "jsonl":
        return load_jsonl(path), fmt
    if fmt == "json_array":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), fmt
    if fmt == "tsv":
        return load_tsv(path), fmt
    raise ValueError(f"Cannot detect format for {path}")


# ── TSV → unified record ────────────────────────────────────────────────────

def tsv_row_to_record(row):
    """Convert a step6-merged TSV row to a JSONL-like record."""
    uid = row.get("UserId", "")
    # Parse RankedJourneys or JourneyWithProducts
    ranked = row.get("RankedJourneys")
    jwp = row.get("JourneyWithProducts")
    profile_str = row.get("ShoppingProfile", "{}")
    events = row.get("ReadableUserEvents", "")

    journeys_raw = []
    if isinstance(ranked, dict):
        journeys_raw = ranked.get("ContinuedJourneys", [])
    elif isinstance(jwp, dict):
        journeys_raw = jwp.get("ContinuedJourneys", [])

    # Parse profile
    profile = {}
    if isinstance(profile_str, str):
        try:
            profile = json.loads(profile_str)
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(profile_str, dict):
        profile = profile_str

    return {
        "stableid": uid,
        "userShoppingProfile": profile,
        "recentShoppingEvents": events,
        "journeys": journeys_raw,
    }


# ── Extract standardized user data ──────────────────────────────────────────

def extract_user_data(record, schema):
    # ── Profile ──
    profile_raw = record.get("userShoppingProfile", {})
    if isinstance(profile_raw, dict) and "userShoppingProfile" in profile_raw:
        profile = profile_raw["userShoppingProfile"]
    else:
        profile = profile_raw if isinstance(profile_raw, dict) else {}

    # ── Recent events ──
    recent_events = record.get("recentShoppingEvents", "") or ""
    recent_events_count = len([l for l in recent_events.split("\n") if l.strip()]) if recent_events else 0

    # ── Retailer preferences ──
    retailer_prefs = profile.get("retailerPreferences", []) if isinstance(profile, dict) else []

    # ── Journeys ──
    raw_journeys = record.get("journeys", [])
    journeys = []
    for j in raw_journeys:
        journey = {
            "title": j.get("title", j.get("Title", "")),
            "journeyType": j.get("journeyType", j.get("JourneyType", "")),
            "description": j.get("description", j.get("Description", "")),
            "reason": j.get("reason", j.get("WhyAmISeeingThis", j.get("Reason", ""))),
            "conversationStarter": j.get("conversationStarter", j.get("ConversationStarter", "")),
            "stats": j.get("stats", j.get("RankingSummary", {})),
        }

        products = j.get("products", j.get("Products", j.get("Queries", [])))
        if not isinstance(products, list):
            products = []

        if schema == "B_ranked":
            # Step8 output: flat products with OriginalQuery — regroup by query
            query_groups = {}
            for p in products:
                q = p.get("OriginalQuery", "unknown")
                if q not in query_groups:
                    query_groups[q] = []
                query_groups[q].append({
                    "Title": p.get("Title", ""),
                    "Seller": p.get("Seller", ""),
                    "Brand": p.get("Brand", ""),
                    "OriginalPrice": p.get("OriginalPrice", ""),
                    "global_offer_id": p.get("global_offer_id", ""),
                    "Gender": p.get("Gender", ""),
                    "AgeGroup": p.get("AgeGroup", ""),
                    "ImageUrl": p.get("ImageUrl", ""),
                    "OfferUrl": p.get("OfferUrl", ""),
                    "Rank": p.get("Rank", ""),
                    "OriginalSLMRank": p.get("OriginalSLMRank", ""),
                    "DisplayPosition": p.get("DisplayPosition", ""),
                })
            journey["products"] = [
                {"query": q, "matched_products": prods}
                for q, prods in query_groups.items()
            ]
        else:
            journey["products"] = products

        journeys.append(journey)

    return {
        "journey_count": len(journeys),
        "recent_events_count": recent_events_count,
        "recent_events": recent_events,
        "profile": profile,
        "profile_retailer_preferences": retailer_prefs,
        "journeys": journeys,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(script_dir, args.run_name)

    if os.path.exists(run_dir):
        print(f"WARNING: Run directory already exists: {run_dir}")
        print("         Overwriting...")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 70)
    print(f"  Phase 1: Parse & Pair")
    print("=" * 70)
    print(f"  Run name : {args.run_name}")
    print(f"  Output   : {run_dir}")
    print(f"  P1 ({args.p1_name}): {args.p1_file}")
    print(f"  P2 ({args.p2_name}): {args.p2_file}")
    print()

    # ── Load ──
    p1_records, fmt1 = load_file(args.p1_file)
    p2_records, fmt2 = load_file(args.p2_file)
    print(f"  P1 format: {fmt1}  ({len(p1_records)} records)")
    print(f"  P2 format: {fmt2}  ({len(p2_records)} records)")

    # ── Convert TSV rows if needed ──
    if fmt1 == "tsv":
        p1_records = [tsv_row_to_record(r) for r in p1_records]
    if fmt2 == "tsv":
        p2_records = [tsv_row_to_record(r) for r in p2_records]

    # ── Detect schemas ──
    schema1 = detect_schema(p1_records[0]) if p1_records else "unknown"
    schema2 = detect_schema(p2_records[0]) if p2_records else "unknown"
    print(f"  P1 schema: {schema1}")
    print(f"  P2 schema: {schema2}")

    # ── ID extraction ──
    def get_uid(r):
        return r.get("stableid", r.get("user_id", r.get("UserId", "")))

    p1_by_id = {get_uid(r): r for r in p1_records}
    p2_by_id = {get_uid(r): r for r in p2_records}

    # ── Pairing (with prefix matching for truncated IDs) ──
    common_ids = set(p1_by_id.keys()) & set(p2_by_id.keys())

    # If no exact match, try prefix matching (HTML-extracted IDs may be truncated)
    if not common_ids and p1_by_id and p2_by_id:
        p1_ids = list(p1_by_id.keys())
        p2_ids = list(p2_by_id.keys())
        min_len = min(min(len(x) for x in p1_ids), min(len(x) for x in p2_ids))
        if min_len >= 8:
            print(f"\n  No exact ID match. Trying prefix matching (first {min_len} chars)...")
            p1_prefix = {uid[:min_len]: uid for uid in p1_ids}
            p2_prefix = {uid[:min_len]: uid for uid in p2_ids}
            prefix_common = set(p1_prefix.keys()) & set(p2_prefix.keys())
            if prefix_common:
                # Rebuild dicts with matched full IDs
                # Use P1's full ID as canonical
                new_p1 = {}
                new_p2 = {}
                for pfx in prefix_common:
                    canonical = p1_prefix[pfx]
                    new_p1[canonical] = p1_by_id[p1_prefix[pfx]]
                    new_p2[canonical] = p2_by_id[p2_prefix[pfx]]
                p1_by_id = new_p1
                p2_by_id = new_p2
                common_ids = set(p1_by_id.keys()) & set(p2_by_id.keys())
                print(f"  Prefix-matched: {len(common_ids)} users")

    p1_only = set(p1_by_id.keys()) - common_ids
    p2_only = set(p2_by_id.keys()) - common_ids

    print(f"\n  User pairing:")
    print(f"    Common (inner join): {len(common_ids)}")
    print(f"    P1 only:            {len(p1_only)}")
    print(f"    P2 only:            {len(p2_only)}")

    # ── Build paired data ──
    paired = []
    triage = {"deep": 0, "skip_both_empty": 0, "skip_p1_only": 0, "skip_p2_only": 0}

    for uid in sorted(common_ids):
        d1 = extract_user_data(p1_by_id[uid], schema1)
        d2 = extract_user_data(p2_by_id[uid], schema2)

        if d1["journey_count"] > 0 and d2["journey_count"] > 0:
            t = "deep"
        elif d1["journey_count"] == 0 and d2["journey_count"] == 0:
            t = "skip_both_empty"
        elif d1["journey_count"] > 0:
            t = "skip_p1_only"
        else:
            t = "skip_p2_only"
        triage[t] += 1

        paired.append({
            "stableid": uid,
            "triage": t,
            "p1_schema": schema1.replace("_ranked", ""),
            "p2_schema": schema2.replace("_ranked", ""),
            "p1_name": args.p1_name,
            "p2_name": args.p2_name,
            "p1": d1,
            "p2": d2,
        })

    for uid in sorted(p1_only):
        d1 = extract_user_data(p1_by_id[uid], schema1)
        triage["skip_p1_only"] += 1
        paired.append({
            "stableid": uid, "triage": "skip_p1_only",
            "p1_schema": schema1.replace("_ranked", ""),
            "p2_schema": schema2.replace("_ranked", ""),
            "p1_name": args.p1_name, "p2_name": args.p2_name,
            "p1": d1, "p2": None,
        })

    for uid in sorted(p2_only):
        d2 = extract_user_data(p2_by_id[uid], schema2)
        triage["skip_p2_only"] += 1
        paired.append({
            "stableid": uid, "triage": "skip_p2_only",
            "p1_schema": schema1.replace("_ranked", ""),
            "p2_schema": schema2.replace("_ranked", ""),
            "p1_name": args.p1_name, "p2_name": args.p2_name,
            "p1": None, "p2": d2,
        })

    # ── Write ──
    out_path = os.path.join(run_dir, "paired_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(paired, f, ensure_ascii=False, indent=2)

    print(f"\n  Triage summary:")
    for k, v in triage.items():
        print(f"    {k:20s}: {v}")
    print(f"\n  Wrote: {out_path}")
    print(f"  Total entries: {len(paired)}")

    # ── Mirror to --output-dir ──
    if args.output_dir:
        mirror_dir = os.path.join(args.output_dir, args.run_name)
        os.makedirs(mirror_dir, exist_ok=True)
        mirror_path = os.path.join(mirror_dir, "paired_data.json")
        with open(mirror_path, "w", encoding="utf-8") as f:
            json.dump(paired, f, ensure_ascii=False, indent=2)
        print(f"  Mirrored: {mirror_path}")
    print()


if __name__ == "__main__":
    main()
