"""
step6_call_LLM_ranker.py
========================

Pipeline step 6 — call LLM to rank/filter products per journey.

Reads step5 output TSV (per-user, with JourneyWithProducts containing
multiple journeys), splits each journey into an independent LLM call
using JourneyRankerPromptV4, and reassembles ranked results per user.

Three modes
-----------
1. **split**  (``--split_n N``): Read step5 output → flatten to per-journey
   rows → write chunk TSVs of N journeys each. No LLM inference.
2. **inference** (default): Read input TSV (either step5 output or a split
   chunk) → build per-journey prompts → call LLM → write ``*_Results.tsv``.
3. **merge** (``--merge_dir /path/``): Collect all ``*_Results.tsv`` files
   → validate JSON → reassemble per-user → write final output TSV.

Split chunk format (per-journey)::

    JourneyKey \\t UserId \\t JourneyIdx \\t ShoppingProfile \\t JourneyWithProducts

Inference result format::

    JourneyKey \\t UserId \\t JourneyIdx \\t OUTPUT

Final merge output format::

    UserId \\t ReadableUserEvents \\t ShoppingProfile \\t JourneyWithProducts \\t RankedJourneys

Usage examples::

    # Split step5 output into chunks of 50K journeys
    python step6_call_LLM_ranker.py --split_n 50000 \\
        --input_file /path/to/step5_output.tsv \\
        --output_dir /path/to/output/

    # Run inference on a split chunk (or full step5 output)
    python step6_call_LLM_ranker.py \\
        --input_file /path/to/chunk_001.tsv \\
        --output_dir /path/to/output/

    # Debug: only 50 journeys
    python step6_call_LLM_ranker.py --debug --input_file /path/to/input.tsv

    # Merge all chunk results
    python step6_call_LLM_ranker.py --merge_dir /path/to/output/ \\
        --input_file /path/to/step5_output.tsv \\
        --output_dir /path/to/output/
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "resources"))
from llm_utils import (run_llm_parallel_with_checkpoint,
                        load_checkpoint, cleanup_checkpoint)


# ============================================================================ #
# Constants                                                                    #
# ============================================================================ #
PROMPT_FILE = os.path.join(PROJECT_DIR, "prompts", "JourneyRankerPromptV5.md")
TOKEN_FILE = os.path.join(PROJECT_DIR, "resources", "tokens_full.txt")

# Column names for split chunk TSV (per-journey)
SPLIT_COLUMNS = ["JourneyKey", "UserId", "JourneyIdx",
                  "ShoppingProfile", "JourneyWithProducts"]

# Column names for inference result TSV
RESULT_COLUMNS = ["JourneyKey", "UserId", "JourneyIdx", "OUTPUT"]

# Column names for final merged output
MERGE_OUTPUT_COLUMNS = ["UserId", "ReadableUserEvents", "ShoppingProfile",
                        "JourneyWithProducts", "RankedJourneys"]


# ============================================================================ #
# Backslash-quote cleanup (same as step5)                                      #
# ============================================================================ #
def _fix_backslash_json(text):
    """Iteratively fix backslash-escaped quotes and parse as JSON.

    Returns parsed dict/list on success, None on failure.
    """
    if not text or not text.strip():
        return None
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Iterative backslash fix
    _BS = chr(92)
    _Q = chr(34)
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


def _clean_json_field(text):
    """Parse JSON with backslash fix, then re-serialize cleanly."""
    obj = _fix_backslash_json(text)
    if obj is not None:
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    return text  # fallback: return as-is


# ============================================================================ #
# Data loading: step5 output → per-journey rows                                #
# ============================================================================ #
def load_step5_and_flatten(input_file, max_journeys=0):
    """Read step5 output TSV and flatten to per-journey rows.

    Optimized for large files:
      - Uses 8 MB binary read buffer for NFS performance
      - Skips unused columns (ShoppingJourneys) to save I/O + memory
      - Does NOT store user_rows during inference/split (only user_rows
        is populated during merge, which calls run_merge separately)

    Args:
        input_file: Path to step5 output TSV (or a split chunk TSV).
        max_journeys: If >0, stop after this many journeys.

    Returns:
        Tuple of (journey_rows, user_rows, is_split_format).
        journey_rows: List of dicts with keys matching SPLIT_COLUMNS.
        user_rows: Dict of UserId -> original row dict.
                   Empty for inference/split (not needed).
        is_split_format: True if input is already per-journey format.
    """
    journey_rows = []
    # user_rows is NOT populated here — it's only needed by merge mode,
    # which reads the file separately in run_merge().
    user_rows = {}

    try:
        file_size = os.path.getsize(input_file)
    except OSError:
        file_size = 0

    # Use binary mode with large buffer for NFS performance, then decode
    # line by line. This avoids the Python default 8KB buffer which causes
    # thousands of tiny NFS reads on /cosmos.
    with open(input_file, "rb", buffering=8 << 20) as fb:
        header_line = fb.readline()
        if not header_line:
            return [], {}, False
        header_text = header_line.decode("utf-8", errors="replace")
        header = next(csv.reader([header_text], delimiter="\t"))

        # Detect format: split chunk has JourneyKey column
        is_split = "JourneyKey" in header

        # Pre-resolve column indices for fast access
        col_idx = {name: i for i, name in enumerate(header)}
        idx_uid = col_idx.get("UserId", -1)
        idx_jwp = col_idx.get("JourneyWithProducts", -1)
        idx_profile = col_idx.get("ShoppingProfile", -1)
        idx_jkey = col_idx.get("JourneyKey", -1)
        idx_jidx = col_idx.get("JourneyIdx", -1)
        # NOTE: ShoppingJourneys is intentionally NOT read — it's unused
        # by step6 and constitutes ~3-5% of file size.

        pbar = tqdm(total=file_size or None, unit="B", unit_scale=True,
                    desc="[load] reading", mininterval=60, smoothing=0.1)
        pbar.update(len(header_line))

        # Track profiles by uid to avoid re-parsing JSON for every row
        uid_profile_cache = {}

        for line_bytes in fb:
            pbar.update(len(line_bytes))
            line = line_bytes.decode("utf-8", errors="replace")
            fields = next(csv.reader([line], delimiter="\t"))
            nf = len(fields)

            if is_split:
                jr = {
                    "JourneyKey": fields[idx_jkey] if idx_jkey < nf else "",
                    "UserId": fields[idx_uid] if idx_uid < nf else "",
                    "JourneyIdx": int(fields[idx_jidx])
                        if idx_jidx < nf and fields[idx_jidx].strip() else 0,
                    "ShoppingProfile": _clean_json_field(
                        fields[idx_profile] if idx_profile < nf else ""),
                    "JourneyWithProducts": _clean_json_field(
                        fields[idx_jwp] if idx_jwp < nf else ""),
                }
                journey_rows.append(jr)
                if max_journeys > 0 and len(journey_rows) >= max_journeys:
                    break
                continue

            # Step5 per-user format — flatten to per-journey rows
            uid = (fields[idx_uid] if idx_uid < nf else "").strip()
            if not uid:
                continue

            jwp_raw = (fields[idx_jwp] if idx_jwp < nf else "").strip()
            if not jwp_raw:
                continue

            jwp = _fix_backslash_json(jwp_raw)
            if not jwp or "ContinuedJourneys" not in jwp:
                continue

            # Cache cleaned profile per user (avoid re-parsing same JSON)
            if uid not in uid_profile_cache:
                raw_prof = fields[idx_profile] if idx_profile < nf else ""
                uid_profile_cache[uid] = _clean_json_field(raw_prof)
            profile = uid_profile_cache[uid]

            for idx, journey in enumerate(jwp["ContinuedJourneys"]):
                if not isinstance(journey, dict):
                    continue
                key = f"{uid}_{idx}"
                journey_rows.append({
                    "JourneyKey": key,
                    "UserId": uid,
                    "JourneyIdx": idx,
                    "ShoppingProfile": profile,
                    "JourneyWithProducts": json.dumps(
                        journey, ensure_ascii=False, separators=(',', ':')),
                })
                if max_journeys > 0 and len(journey_rows) >= max_journeys:
                    break
            if max_journeys > 0 and len(journey_rows) >= max_journeys:
                break

        pbar.close()

    return journey_rows, user_rows, is_split

    return journey_rows, user_rows, False


# ============================================================================ #
# Split mode                                                                   #
# ============================================================================ #
def run_split(journey_rows, output_dir, split_n, input_basename):
    """Write journey rows into chunk TSVs of split_n journeys each."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(journey_rows)
    n_chunks = (n + split_n - 1) // split_n
    print(f"\n[split] {n:,} journeys -> {n_chunks} chunks of {split_n:,}")

    for ci in range(n_chunks):
        start = ci * split_n
        end = min(start + split_n, n)
        chunk = journey_rows[start:end]
        chunk_file = os.path.join(
            output_dir, f"{input_basename}_split_{ci + 1:03d}.tsv")
        with open(chunk_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SPLIT_COLUMNS,
                                    delimiter="\t",
                                    extrasaction="ignore")
            writer.writeheader()
            for jr in chunk:
                writer.writerow(jr)
        sz = os.path.getsize(chunk_file) / (1024 * 1024)
        print(f"  Chunk {ci + 1}/{n_chunks}: {len(chunk):,} journeys "
              f"({sz:.1f} MB) -> {chunk_file}")

    print(f"\n[split] Done. Run each chunk with:")
    print(f"  python step6_call_LLM_ranker.py "
          f"--input_file <chunk_file> --output_dir {output_dir}")


# ============================================================================ #
# Prompt construction                                                          #
# ============================================================================ #
def load_prompt_template(prompt_file):
    """Load prompt template from file."""
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def build_ranker_prompt(profile_str, journey_str, prompt_template):
    """Build LLM prompt for one journey.

    Replaces:
      ##Profile##              <- ShoppingProfile JSON
      ##JourneyWithProducts##  <- single journey JSON object
    """
    result = prompt_template.replace("##Profile##", profile_str)
    result = result.replace("##JourneyWithProducts##", journey_str)
    return result


# ============================================================================ #
# Result extraction / validation                                               #
# ============================================================================ #
def extract_ranked_journey(raw_text):
    """Extract and validate ranked journey JSON from LLM output.

    Expects the simplified V5 format: a JSON object with "Products" key
    where each product has at least "Rank" and "OfferId".
    Returns compact JSON string if valid, empty string otherwise.
    """
    if not raw_text or not raw_text.strip():
        return ""
    text = raw_text.strip()

    # Strip <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Extract from <OUTPUT>...</OUTPUT>
    tag_match = re.search(r'<OUTPUT>\s*(.*?)\s*</OUTPUT>', text, re.DOTALL)
    if tag_match:
        text = tag_match.group(1).strip()

    # Strip markdown code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to parse as JSON
    def _try_validate(candidate):
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict) or "Products" not in parsed:
                return None
            products = parsed["Products"]
            if not isinstance(products, list):
                return None
            # Validate each product has at least OfferId
            valid_products = []
            for p in products:
                if isinstance(p, dict) and p.get("OfferId"):
                    valid_products.append(p)
            if not valid_products:
                return None
            parsed["Products"] = valid_products
            return json.dumps(parsed, ensure_ascii=False,
                              separators=(',', ':'))
        except (json.JSONDecodeError, TypeError):
            return None

    result = _try_validate(text)
    if result:
        return result

    # Fallback: find outermost { ... }
    bs = text.find('{')
    be = text.rfind('}')
    if bs != -1 and be > bs:
        result = _try_validate(text[bs:be + 1])
        if result:
            return result

    return ""


# ============================================================================ #
# Inference mode                                                               #
# ============================================================================ #
def run_inference(journey_rows, prompt_template, args):
    """Run LLM inference for each journey and save results TSV."""
    # Build (key, prompt) inputs
    print(f"\n[infer] Building prompts for {len(journey_rows):,} journeys ...")
    inputs = []
    for jr in journey_rows:
        prompt = build_ranker_prompt(
            jr["ShoppingProfile"],
            jr["JourneyWithProducts"],
            prompt_template,
        )
        inputs.append((jr["JourneyKey"], prompt))

    prompt_lens = [len(p) for _, p in inputs]
    print(f"  Prompt lengths: min={min(prompt_lens):,}  "
          f"max={max(prompt_lens):,}  "
          f"avg={sum(prompt_lens) / len(prompt_lens):,.0f}")

    # Checkpoint dir
    input_base = os.path.splitext(os.path.basename(args.input_file))[0]
    checkpoint_dir = os.path.join(args.output_dir,
                                  f"_ranker_ckpt_{input_base}")

    # Run LLM
    print(f"\n[infer] Calling LLM ({args.copilot_model}, "
          f"{args.num_workers} workers) ...")
    t0 = time.time()
    results = run_llm_parallel_with_checkpoint(
        inputs=inputs,
        token_file=args.token_file,
        checkpoint_dir=checkpoint_dir,
        num_workers=args.num_workers,
        model=args.copilot_model,
        temperature=0,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
    )
    elapsed = time.time() - t0
    print(f"[infer] Done in {elapsed:.1f}s "
          f"({len(inputs) / max(elapsed, 1):.1f} journeys/s)")

    # Build key -> journey_row lookup
    key_to_jr = {jr["JourneyKey"]: jr for jr in journey_rows}

    # Write results TSV
    output_file = os.path.join(
        args.output_dir,
        f"{input_base}_Results.tsv")
    os.makedirs(args.output_dir, exist_ok=True)

    success = 0
    valid_json = 0
    empty_products = 0
    fail_keys = []

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS,
                                delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for key, raw_result in results:
            jr = key_to_jr[key]
            if not raw_result:
                fail_keys.append(key)
                writer.writerow({
                    "JourneyKey": key,
                    "UserId": jr["UserId"],
                    "JourneyIdx": jr["JourneyIdx"],
                    "OUTPUT": "",
                })
                continue
            success += 1
            clean = extract_ranked_journey(raw_result)
            if clean:
                # Check if Products is empty
                try:
                    obj = json.loads(clean)
                    if not obj.get("Products"):
                        empty_products += 1
                except (json.JSONDecodeError, TypeError):
                    pass
                valid_json += 1
                output_val = clean
            else:
                output_val = raw_result.replace("\n", " ").replace("\t", " ")
            writer.writerow({
                "JourneyKey": key,
                "UserId": jr["UserId"],
                "JourneyIdx": jr["JourneyIdx"],
                "OUTPUT": output_val,
            })

    sz = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n[infer] Results: {output_file} ({sz:.1f} MB)")
    print(f"  Total journeys:    {len(results):,}")
    print(f"  API success:       {success:,}/{len(results):,}")
    print(f"  Valid JSON:        {valid_json:,}/{success:,}")
    print(f"  Empty Products:    {empty_products:,}")
    print(f"  Inference failed:  {len(fail_keys):,}")
    if fail_keys[:10]:
        print(f"  Failed keys (first 10): {fail_keys[:10]}")

    # Cleanup checkpoint
    if args.cleanup_checkpoint:
        cleanup_checkpoint(checkpoint_dir)
    else:
        print(f"  Checkpoint preserved: {checkpoint_dir}")

    return output_file


# ============================================================================ #
# Merge mode                                                                   #
# ============================================================================ #
def run_merge(args):
    """Merge all *_Results.tsv into a per-user output aligned with step5."""
    merge_dir = args.merge_dir
    input_file = args.input_file  # original step5 output (for user rows)
    output_dir = args.output_dir or merge_dir

    # 1. Find all result files
    pattern = os.path.join(merge_dir, "*_Results.tsv")
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        print(f"[merge] No *_Results.tsv found in {merge_dir}")
        return
    print(f"[merge] Found {len(result_files)} result files in {merge_dir}")

    # 2. Load all results: key -> OUTPUT
    all_results = {}  # JourneyKey -> OUTPUT string
    for fpath in result_files:
        count = 0
        with open(fpath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = row.get("JourneyKey", "").strip()
                output = row.get("OUTPUT", "").strip()
                if key and output:
                    all_results[key] = output
                    count += 1
        print(f"  {os.path.basename(fpath)}: {count:,} valid results")
    print(f"  Total unique results: {len(all_results):,}")

    # 3. Load original step5 output to get per-user data
    #    Uses binary buffered read for NFS performance.
    #    Skips ShoppingJourneys column (unused, ~3-5% of file).
    print(f"\n[merge] Reading original step5 output: {input_file}")
    user_order = []  # preserve user order
    user_data = {}   # uid -> row dict
    user_journey_counts = {}  # uid -> number of journeys

    try:
        file_size = os.path.getsize(input_file)
    except OSError:
        file_size = 0

    with open(input_file, "rb", buffering=8 << 20) as fb:
        header_line = fb.readline()
        header = next(csv.reader(
            [header_line.decode("utf-8", errors="replace")],
            delimiter="\t"))
        col_idx = {name: i for i, name in enumerate(header)}
        idx_uid = col_idx.get("UserId", -1)
        idx_events = col_idx.get("ReadableUserEvents", -1)
        idx_profile = col_idx.get("ShoppingProfile", -1)
        idx_jwp = col_idx.get("JourneyWithProducts", -1)

        pbar = tqdm(total=file_size or None, unit="B", unit_scale=True,
                    desc="[merge] read users", mininterval=60)
        pbar.update(len(header_line))

        for line_bytes in fb:
            pbar.update(len(line_bytes))
            line = line_bytes.decode("utf-8", errors="replace")
            fields = next(csv.reader([line], delimiter="\t"))
            nf = len(fields)

            uid = (fields[idx_uid] if idx_uid < nf else "").strip()
            if not uid:
                continue
            if uid not in user_data:
                user_order.append(uid)
                user_data[uid] = {
                    "UserId": uid,
                    "ReadableUserEvents":
                        fields[idx_events] if idx_events < nf else "",
                    "ShoppingProfile": _clean_json_field(
                        fields[idx_profile] if idx_profile < nf else ""),
                    "JourneyWithProducts":
                        fields[idx_jwp] if idx_jwp < nf else "",
                }
            # Count journeys
            jwp_raw = (fields[idx_jwp] if idx_jwp < nf else "").strip()
            jwp = _fix_backslash_json(jwp_raw)
            if jwp and "ContinuedJourneys" in jwp:
                user_journey_counts[uid] = len(
                    jwp.get("ContinuedJourneys", []))
            else:
                user_journey_counts.setdefault(uid, 0)

        pbar.close()

    print(f"  Users: {len(user_order):,}, "
          f"Total journeys: {sum(user_journey_counts.values()):,}")

    # 4. Reassemble per-user RankedJourneys (with full metadata backfill)
    stats = {
        "users_total": len(user_order),
        "users_with_ranked": 0,
        "users_partial_fail": 0,
        "journeys_total": sum(user_journey_counts.values()),
        "journeys_ranked": 0,
        "journeys_infer_fail": 0,
        "journeys_invalid_json": 0,
        "journeys_empty_products": 0,
    }

    def _build_journey_lookup(jwp_raw):
        """Parse JWP and build per-journey OfferId->product dict + journey meta."""
        jwp = _fix_backslash_json(jwp_raw)
        if not jwp or "ContinuedJourneys" not in jwp:
            return []
        result = []  # list of (journey_meta_dict, offerid_to_product, offerid_to_query)
        for j in jwp["ContinuedJourneys"]:
            if not isinstance(j, dict):
                result.append(({}, {}, {}))
                continue
            meta = {
                "JourneyType": j.get("JourneyType", ""),
                "Title": j.get("Title", ""),
                "Description": j.get("Description", ""),
                "ConversationStarter": j.get("ConversationStarter", ""),
                "WhyAmISeeingThis": j.get("WhyAmISeeingThis", ""),
            }
            oid_to_prod = {}
            oid_to_query = {}
            for q in j.get("Queries") or []:
                query_text = q.get("Query", "") if isinstance(q, dict) else ""
                for p in (q.get("Products") or []) if isinstance(q, dict) else []:
                    if isinstance(p, dict) and p.get("OfferId"):
                        oid = str(p["OfferId"])
                        if oid not in oid_to_prod:
                            oid_to_prod[oid] = p
                            oid_to_query[oid] = query_text
            result.append((meta, oid_to_prod, oid_to_query))
        return result

    input_base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{input_base}_Ranked.tsv")

    with open(output_file, "w", encoding="utf-8",
              buffering=8 << 20, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MERGE_OUTPUT_COLUMNS,
                                delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()

        for uid in tqdm(user_order, desc="[merge] assemble",
                        mininterval=60):
            ud = user_data[uid]
            n_journeys = user_journey_counts.get(uid, 0)
            ranked_journeys = []
            has_fail = False

            # Build per-journey lookup from original JWP
            jwp_lookup = _build_journey_lookup(
                ud.get("JourneyWithProducts", ""))

            for idx in range(n_journeys):
                key = f"{uid}_{idx}"
                raw_output = all_results.get(key, "")

                if not raw_output:
                    stats["journeys_infer_fail"] += 1
                    has_fail = True
                    continue

                clean = extract_ranked_journey(raw_output)
                if not clean:
                    stats["journeys_invalid_json"] += 1
                    has_fail = True
                    continue

                try:
                    obj = json.loads(clean)
                except (json.JSONDecodeError, TypeError):
                    stats["journeys_invalid_json"] += 1
                    has_fail = True
                    continue

                products = obj.get("Products", [])
                if not products:
                    stats["journeys_empty_products"] += 1
                    continue  # skip empty-product journeys

                # Backfill: enrich ranked output with full metadata from JWP
                if idx < len(jwp_lookup):
                    j_meta, oid_to_prod, oid_to_query = jwp_lookup[idx]
                    # Fill journey-level fields from original
                    for field in ("Description", "ConversationStarter",
                                  "WhyAmISeeingThis"):
                        if field not in obj or not obj[field]:
                            obj[field] = j_meta.get(field, "")
                    # Fill product-level fields from original
                    enriched = []
                    for p in products:
                        oid = str(p.get("OfferId", ""))
                        orig = oid_to_prod.get(oid, {})
                        enriched.append({
                            "Rank": p.get("Rank"),
                            "OfferId": oid,
                            "Title": orig.get("Title", ""),
                            "Seller": orig.get("Seller", ""),
                            "Price": orig.get("Price", ""),
                            "Brand": orig.get("Brand", ""),
                            "Category": orig.get("Category", ""),
                            "OriginalQuery": oid_to_query.get(oid, ""),
                        })
                    obj["Products"] = enriched

                ranked_journeys.append(obj)
                stats["journeys_ranked"] += 1

            if ranked_journeys:
                stats["users_with_ranked"] += 1
            if has_fail:
                stats["users_partial_fail"] += 1

            ranked_json = json.dumps(
                {"ContinuedJourneys": ranked_journeys},
                ensure_ascii=False, separators=(',', ':'))

            writer.writerow({
                "UserId": uid,
                "ReadableUserEvents": ud["ReadableUserEvents"],
                "ShoppingProfile": ud["ShoppingProfile"],
                "JourneyWithProducts": ud["JourneyWithProducts"],
                "RankedJourneys": ranked_json,
            })

    sz = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n[merge] Output: {output_file} ({sz:.1f} MB)")
    print(f"\n{'=' * 60}")
    print(f"  Merge Statistics")
    print(f"{'=' * 60}")
    print(f"  Users total:              {stats['users_total']:>10,}")
    print(f"  Users with ranked:        {stats['users_with_ranked']:>10,}")
    print(f"  Users with partial fail:  {stats['users_partial_fail']:>10,}")
    print(f"  Journeys total:           {stats['journeys_total']:>10,}")
    print(f"  Journeys ranked:          {stats['journeys_ranked']:>10,}")
    print(f"  Journeys infer failed:    {stats['journeys_infer_fail']:>10,}")
    print(f"  Journeys invalid JSON:    {stats['journeys_invalid_json']:>10,}")
    print(f"  Journeys empty products:  {stats['journeys_empty_products']:>10,}")
    print(f"{'=' * 60}")


# ============================================================================ #
# CLI                                                                          #
# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(
        description="Step 6: Call LLM ranker per journey",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_io = p.add_argument_group("I/O")
    g_io.add_argument(
        "--input_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
                "/Data/LLMTrainingData/20260516/raw_data"
                "/UserEvents_clean_combined_full_journey_with_products.tsv",
        help="Step5 output TSV or a split chunk TSV.",
    )
    g_io.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec"
                "/Data/LLMTrainingData/20260516/raw_data/ranker_output_full",
        help="Directory for output files (results, splits, merged).",
    )
    g_io.add_argument(
        "--prompt_file", type=str, default=PROMPT_FILE,
        help="Path to JourneyRankerPromptV5.md prompt template.",
    )
    g_io.add_argument(
        "--token_file", type=str, default=TOKEN_FILE,
        help="Path to GitHub tokens file.",
    )

    g_llm = p.add_argument_group("LLM")
    g_llm.add_argument("--copilot_model", type=str, default="gpt-5.2",
                        help="Copilot model name.")
    g_llm.add_argument("--num_workers", type=int, default=60,
                        help="Number of parallel workers.")
    g_llm.add_argument("--max_tokens", type=int, default=10000,
                        help="Max output tokens per API call.")
    g_llm.add_argument("--chunk_size", type=int, default=10000,
                        help="Journeys per checkpoint chunk.")

    g_mode = p.add_argument_group("Mode selection")
    g_mode.add_argument(
        "--split_n", type=int, default=0,
        help="Split mode: write chunks of N journeys each (no inference). "
             "Set 0 to disable (run inference instead).",
    )
    g_mode.add_argument(
        "--merge_dir", type=str, default=None,
        help="Merge mode: path to directory containing *_Results.tsv files. "
             "Requires --input_file pointing to the original step5 output.",
    )
    g_mode.add_argument(
        "--cleanup_checkpoint", action="store_true", default=False,
        help="Delete checkpoint dir after successful inference.",
    )

    g_dbg = p.add_argument_group("Debug")
    g_dbg.add_argument("--debug", action="store_true",
                        help="Debug mode: process only --debug_rows journeys.")
    g_dbg.add_argument("--debug_rows", type=int, default=50,
                        help="Max journeys in debug mode.")

    return p.parse_args()


# ============================================================================ #
# Main                                                                         #
# ============================================================================ #
def main():
    args = parse_args()

    print("=" * 70)
    print("Step 6: Call LLM Ranker (per-journey)")
    print("=" * 70)
    print(f"  Input:      {args.input_file}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Prompt:     {args.prompt_file}")
    print(f"  Model:      {args.copilot_model}")
    if args.split_n > 0:
        print(f"  Mode:       SPLIT (n={args.split_n:,})")
    elif args.merge_dir:
        print(f"  Mode:       MERGE (dir={args.merge_dir})")
    else:
        print(f"  Mode:       INFERENCE")
        print(f"  Workers:    {args.num_workers}")
        print(f"  Max tokens: {args.max_tokens}")
        print(f"  Chunk size: {args.chunk_size:,}")
    if args.debug:
        print(f"  *** DEBUG: limit to {args.debug_rows} journeys ***")
    print()

    # ---- Merge mode ----
    if args.merge_dir:
        run_merge(args)
        return

    # ---- Load data ----
    max_j = args.debug_rows if args.debug else 0
    print("[load] Loading and flattening input ...")
    journey_rows, user_rows, is_split = load_step5_and_flatten(
        args.input_file, max_journeys=max_j)

    n_users = len(set(jr["UserId"] for jr in journey_rows))
    print(f"  {len(journey_rows):,} journeys from {n_users:,} users"
          f" (split_format={is_split})")

    if not journey_rows:
        print("No journeys found!")
        return

    # ---- Split mode ----
    if args.split_n > 0:
        input_base = os.path.splitext(
            os.path.basename(args.input_file))[0]
        run_split(journey_rows, args.output_dir, args.split_n, input_base)
        return

    # ---- Inference mode ----
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[infer] Loading prompt: {args.prompt_file}")
    prompt_template = load_prompt_template(args.prompt_file)
    print(f"  Template: {len(prompt_template):,} chars")

    # Show sample
    sample = journey_rows[0]
    print(f"\n  Sample journey: {sample['JourneyKey']}")
    try:
        j_obj = json.loads(sample["JourneyWithProducts"])
        n_q = len(j_obj.get("Queries", []))
        n_p = sum(len(q.get("Products", []))
                  for q in j_obj.get("Queries", []))
        print(f"    Title: {j_obj.get('Title', '')[:80]}")
        print(f"    Queries: {n_q}, Products: {n_p}")
    except (json.JSONDecodeError, TypeError):
        print(f"    JWP: (parse error)")

    run_inference(journey_rows, prompt_template, args)

    print("\nStep 6 Done!")


if __name__ == "__main__":
    main()
