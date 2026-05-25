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

# Use orjson for 5-10x faster JSON parsing/serialization when available
try:
    import orjson
    def _json_loads(s):
        if isinstance(s, str):
            return orjson.loads(s.encode("utf-8") if isinstance(s, str) else s)
        return orjson.loads(s)
    def _json_dumps_compact(obj):
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")
    _JSON_ENGINE = "orjson"
except ImportError:
    _json_loads = json.loads
    def _json_dumps_compact(obj):
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    _JSON_ENGINE = "stdlib"

csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "resources"))
from llm_utils import (run_llm_parallel_with_checkpoint,
                        run_llm_parallel, load_tokens, validate_tokens,
                        load_checkpoint, save_checkpoint, cleanup_checkpoint)


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
        return _json_loads(text)
    except (json.JSONDecodeError, TypeError, orjson.JSONDecodeError
            if _JSON_ENGINE == "orjson" else json.JSONDecodeError):
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
            return _json_loads(cur)
        except (json.JSONDecodeError, TypeError, orjson.JSONDecodeError
                if _JSON_ENGINE == "orjson" else json.JSONDecodeError):
            pass
    return None


def _clean_json_field(text):
    """Parse JSON with backslash fix, then re-serialize cleanly."""
    obj = _fix_backslash_json(text)
    if obj is not None:
        return _json_dumps_compact(obj)
    return text  # fallback: return as-is


# ============================================================================ #
# Data loading: step5 output → per-journey rows                                #
# ============================================================================ #
def load_step5_and_flatten(input_file, max_journeys=0):
    """Read step5 output TSV and flatten to per-journey rows.

    Optimized for large files:
      - Uses 64 MB binary read buffer for NFS performance
      - str.split('\\t') instead of csv.reader (faster)
      - orjson for JSON parse/serialize when available
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
    user_rows = {}

    try:
        file_size = os.path.getsize(input_file)
    except OSError:
        file_size = 0

    # 64 MB buffer for better NFS throughput
    with open(input_file, "rb", buffering=64 << 20) as fb:
        header_line = fb.readline()
        if not header_line:
            return [], {}, False
        header_text = header_line.decode("utf-8", errors="replace").rstrip("\r\n")
        header = header_text.split("\t")

        # Detect format: split chunk has JourneyKey column
        is_split = "JourneyKey" in header

        # Pre-resolve column indices for fast access
        col_idx = {name: i for i, name in enumerate(header)}
        idx_uid = col_idx.get("UserId", -1)
        idx_jwp = col_idx.get("JourneyWithProducts", -1)
        idx_profile = col_idx.get("ShoppingProfile", -1)
        idx_jkey = col_idx.get("JourneyKey", -1)
        idx_jidx = col_idx.get("JourneyIdx", -1)

        pbar = tqdm(total=file_size or None, unit="B", unit_scale=True,
                    desc="[load] reading", mininterval=60, smoothing=0.1)
        pbar.update(len(header_line))

        # Track profiles by uid to avoid re-parsing JSON for every row
        uid_profile_cache = {}

        for line_bytes in fb:
            pbar.update(len(line_bytes))
            line = line_bytes.decode("utf-8", errors="replace")
            if is_split:
                # Split format is clean TSV, safe to use str.split
                fields = line.rstrip("\r\n").split("\t")
            else:
                # Step5 per-user format: JWP JSON may contain quoted
                # fields with tabs/newlines, must use csv.reader.
                fields = next(csv.reader([line], delimiter="\t"))
            nf = len(fields)

            if is_split:
                jr = {
                    "JourneyKey": fields[idx_jkey] if idx_jkey < nf else "",
                    "UserId": fields[idx_uid] if idx_uid < nf else "",
                    "JourneyIdx": int(fields[idx_jidx])
                        if idx_jidx < nf and fields[idx_jidx].strip() else 0,
                    "ShoppingProfile": fields[idx_profile]
                        if idx_profile < nf else "",
                    "JourneyWithProducts": fields[idx_jwp]
                        if idx_jwp < nf else "",
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
                    "JourneyWithProducts": _json_dumps_compact(journey),
                })
                if max_journeys > 0 and len(journey_rows) >= max_journeys:
                    break
            if max_journeys > 0 and len(journey_rows) >= max_journeys:
                break

        pbar.close()

    return journey_rows, user_rows, is_split


# ============================================================================ #
# Chunked journey reader (for streaming inference)                             #
# ============================================================================ #
def _iter_journey_chunks(input_file, chunk_size, max_journeys=0):
    """Generator: stream-read input file and yield chunks of journey_rows.

    Each chunk is a list of dicts with keys matching SPLIT_COLUMNS.
    Memory usage is O(chunk_size) instead of O(total_journeys).

    Args:
        input_file: Path to step5 output TSV or split chunk TSV.
        chunk_size: Max journeys per yielded chunk.
        max_journeys: If >0, stop after this many total journeys.

    Yields:
        list[dict]: Chunk of journey rows.
    """
    try:
        file_size = os.path.getsize(input_file)
    except OSError:
        file_size = 0

    chunk = []
    total = 0

    with open(input_file, "rb", buffering=64 << 20) as fb:
        header_line = fb.readline()
        if not header_line:
            return
        header_text = header_line.decode("utf-8", errors="replace").rstrip("\r\n")
        header = header_text.split("\t")

        is_split = "JourneyKey" in header
        col_idx = {name: i for i, name in enumerate(header)}
        idx_uid = col_idx.get("UserId", -1)
        idx_jwp = col_idx.get("JourneyWithProducts", -1)
        idx_profile = col_idx.get("ShoppingProfile", -1)
        idx_jkey = col_idx.get("JourneyKey", -1)
        idx_jidx = col_idx.get("JourneyIdx", -1)

        pbar = tqdm(total=file_size or None, unit="B", unit_scale=True,
                    desc="[load] streaming", mininterval=60, smoothing=0.1)
        pbar.update(len(header_line))

        uid_profile_cache = {}

        for line_bytes in fb:
            pbar.update(len(line_bytes))
            line = line_bytes.decode("utf-8", errors="replace")

            if is_split:
                fields = line.rstrip("\r\n").split("\t")
            else:
                fields = next(csv.reader([line], delimiter="\t"))
            nf = len(fields)

            if is_split:
                jr = {
                    "JourneyKey": fields[idx_jkey] if idx_jkey < nf else "",
                    "UserId": fields[idx_uid] if idx_uid < nf else "",
                    "JourneyIdx": int(fields[idx_jidx])
                        if idx_jidx < nf and fields[idx_jidx].strip() else 0,
                    "ShoppingProfile": fields[idx_profile]
                        if idx_profile < nf else "",
                    "JourneyWithProducts": fields[idx_jwp]
                        if idx_jwp < nf else "",
                }
                chunk.append(jr)
                total += 1
                if max_journeys > 0 and total >= max_journeys:
                    yield chunk
                    pbar.close()
                    return
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
                continue

            # Step5 per-user format
            uid = (fields[idx_uid] if idx_uid < nf else "").strip()
            if not uid:
                continue
            jwp_raw = (fields[idx_jwp] if idx_jwp < nf else "").strip()
            if not jwp_raw:
                continue
            jwp = _fix_backslash_json(jwp_raw)
            if not jwp or "ContinuedJourneys" not in jwp:
                continue

            if uid not in uid_profile_cache:
                raw_prof = fields[idx_profile] if idx_profile < nf else ""
                uid_profile_cache[uid] = _clean_json_field(raw_prof)
            profile = uid_profile_cache[uid]

            for idx, journey in enumerate(jwp["ContinuedJourneys"]):
                if not isinstance(journey, dict):
                    continue
                key = f"{uid}_{idx}"
                chunk.append({
                    "JourneyKey": key,
                    "UserId": uid,
                    "JourneyIdx": idx,
                    "ShoppingProfile": profile,
                    "JourneyWithProducts": _json_dumps_compact(journey),
                })
                total += 1
                if max_journeys > 0 and total >= max_journeys:
                    yield chunk
                    pbar.close()
                    return
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []

        pbar.close()

    if chunk:
        yield chunk


# ============================================================================ #
# Split mode (in-memory, for small files)                                      #
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
# Streaming split mode (constant memory, for large files)                      #
# ============================================================================ #
def _count_existing_splits(output_dir, input_basename, split_n):
    """Scan output_dir for existing split files and count completed journeys.

    Returns (skip_journeys, resume_chunk_idx):
      - skip_journeys: number of journeys in fully completed chunks to skip
      - resume_chunk_idx: chunk index to resume writing from (0-based)

    The last split file is deleted if it has fewer than split_n rows,
    since it may be a partial write from a crashed run.
    """
    pattern = os.path.join(
        output_dir, f"{input_basename}_split_*.tsv")
    existing = sorted(glob.glob(pattern))
    if not existing:
        return 0, 0

    skip = 0
    last_chunk_idx = 0
    for fpath in existing:
        # Extract chunk index from filename: ..._split_003.tsv -> 3
        base = os.path.splitext(os.path.basename(fpath))[0]
        try:
            ci = int(base.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            continue
        # Count data rows (exclude header)
        n_rows = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for i, _ in enumerate(f):
                pass
            n_rows = i  # i is 0-indexed, last line index = total-1 = data rows
        last_chunk_idx = max(last_chunk_idx, ci)

        if n_rows < split_n:
            # Incomplete chunk — delete and re-do from this point
            print(f"  [resume] Incomplete chunk ({n_rows:,}/{split_n:,} rows),"
                  f" will re-write: {os.path.basename(fpath)}")
            os.remove(fpath)
        else:
            skip += n_rows
            print(f"  [resume] Complete chunk ({n_rows:,} rows): "
                  f"{os.path.basename(fpath)}")

    # resume_chunk_idx = number of complete chunks
    complete_chunks = skip // split_n
    return skip, complete_chunks


def run_split_streaming(input_file, output_dir, split_n):
    """Stream-read step5 TSV and write split chunk files on the fly.

    Supports resume: on restart, scans output_dir for existing complete
    split files, skips the corresponding journeys in the input, and
    continues writing from the next chunk.

    Optimizations vs naive approach:
      - 64 MB binary read buffer (better NFS throughput)
      - str.split('\\t') instead of csv.reader (no per-line object creation)
      - No _clean_json_field during split (raw passthrough for profile)
      - orjson for JSON parse/serialize when available
      - Lightweight skip: during resume, skipped rows only count journeys
        without building dicts

    Memory usage is O(split_n) instead of O(total_journeys).
    """
    os.makedirs(output_dir, exist_ok=True)
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    print(f"  JSON engine: {_JSON_ENGINE}")

    # ---- Resume: detect existing complete chunks ----
    skip_journeys, resume_chunk_idx = _count_existing_splits(
        output_dir, input_basename, split_n)
    if skip_journeys > 0:
        print(f"\n[split-stream] Resuming: skipping first {skip_journeys:,} "
              f"journeys ({resume_chunk_idx} complete chunks found)")

    try:
        file_size = os.path.getsize(input_file)
    except OSError:
        file_size = 0

    chunk_buf = []       # current chunk buffer
    chunk_idx = resume_chunk_idx  # continue numbering from last complete chunk
    total_journeys = 0   # journeys processed (including skipped)
    written_journeys = 0 # journeys actually written in this run
    n_users = 0          # approximate user count (avoid set overhead)
    uid_seen = set()

    def _flush_chunk():
        """Write current buffer to a chunk TSV file."""
        nonlocal chunk_buf, chunk_idx, written_journeys
        if not chunk_buf:
            return
        chunk_idx += 1
        chunk_file = os.path.join(
            output_dir, f"{input_basename}_split_{chunk_idx:03d}.tsv")
        # Write using raw tab-join for speed (avoid csv.DictWriter overhead)
        with open(chunk_file, "w", encoding="utf-8",
                  buffering=8 << 20) as f:
            f.write("\t".join(SPLIT_COLUMNS) + "\n")
            for jr in chunk_buf:
                f.write(f"{jr[0]}\t{jr[1]}\t{jr[2]}\t{jr[3]}\t{jr[4]}\n")
        written_journeys += len(chunk_buf)
        sz = os.path.getsize(chunk_file) / (1024 * 1024)
        print(f"  Chunk {chunk_idx}: {len(chunk_buf):,} journeys "
              f"({sz:.1f} MB) -> {chunk_file}")
        chunk_buf = []

    # Use 64 MB buffer for better NFS throughput
    BUF_SIZE = 64 << 20

    with open(input_file, "rb", buffering=BUF_SIZE) as fb:
        header_line = fb.readline()
        if not header_line:
            print("[split-stream] Empty input file!")
            return
        header_text = header_line.decode("utf-8", errors="replace").rstrip("\r\n")
        header = header_text.split("\t")

        is_split = "JourneyKey" in header
        col_idx = {name: i for i, name in enumerate(header)}
        idx_uid = col_idx.get("UserId", -1)
        idx_jwp = col_idx.get("JourneyWithProducts", -1)
        idx_profile = col_idx.get("ShoppingProfile", -1)
        idx_jkey = col_idx.get("JourneyKey", -1)
        idx_jidx = col_idx.get("JourneyIdx", -1)
        n_cols = len(header)

        desc = "[split-stream] reading"
        if skip_journeys > 0:
            desc = "[split-stream] scanning+writing"
        pbar = tqdm(total=file_size or None, unit="B", unit_scale=True,
                    desc=desc, mininterval=60, smoothing=0.1)
        pbar.update(len(header_line))

        for line_bytes in fb:
            pbar.update(len(line_bytes))

            # --- Fast path: skip phase (resume) ---
            # During skip, do minimal work: just count journeys, don't
            # build dicts or parse profile JSON.
            if total_journeys + 50 <= skip_journeys and not is_split:
                line = line_bytes.decode("utf-8", errors="replace")
                # Use csv.reader for step5 format — JWP JSON may contain
                # tabs/quotes that str.split('\t') can't handle correctly.
                fields = next(csv.reader([line], delimiter="\t"))
                nf = len(fields)
                uid = (fields[idx_uid] if idx_uid < nf else "").strip()
                if not uid:
                    continue
                jwp_raw = (fields[idx_jwp] if idx_jwp < nf else "").strip()
                if not jwp_raw:
                    continue
                jwp = _fix_backslash_json(jwp_raw)
                if not jwp or "ContinuedJourneys" not in jwp:
                    continue
                n_j = sum(1 for j in jwp["ContinuedJourneys"]
                          if isinstance(j, dict))
                total_journeys += n_j
                continue

            line = line_bytes.decode("utf-8", errors="replace")

            if is_split:
                # Split format is clean TSV (no embedded tabs), safe to
                # use str.split.
                fields = line.rstrip("\r\n").split("\t")
            else:
                # Step5 per-user format: JWP JSON may contain quoted
                # fields with tabs/newlines, must use csv.reader.
                fields = next(csv.reader([line], delimiter="\t"))
            nf = len(fields)

            if is_split:
                total_journeys += 1
                if total_journeys <= skip_journeys:
                    continue
                # For already-split format: pass through as-is (no JSON re-parse)
                jkey = fields[idx_jkey] if idx_jkey < nf else ""
                uid = fields[idx_uid] if idx_uid < nf else ""
                jidx = fields[idx_jidx] if idx_jidx < nf else "0"
                profile = fields[idx_profile] if idx_profile < nf else ""
                jwp = fields[idx_jwp] if idx_jwp < nf else ""
                # Tuple instead of dict for lower memory + faster write
                chunk_buf.append((jkey, uid, jidx, profile, jwp))
                if len(chunk_buf) >= split_n:
                    _flush_chunk()
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

            # Raw profile passthrough — no _clean_json_field (saves parse+dump)
            if uid not in uid_seen:
                uid_seen.add(uid)
                n_users += 1
            profile_raw = fields[idx_profile] if idx_profile < nf else ""

            for idx, journey in enumerate(jwp["ContinuedJourneys"]):
                if not isinstance(journey, dict):
                    continue
                total_journeys += 1
                if total_journeys <= skip_journeys:
                    continue
                key = f"{uid}_{idx}"
                # Tuple: (JourneyKey, UserId, JourneyIdx, Profile, JWP)
                chunk_buf.append((
                    key, uid, str(idx), profile_raw,
                    _json_dumps_compact(journey),
                ))
                if len(chunk_buf) >= split_n:
                    _flush_chunk()

        pbar.close()

    # Flush remaining
    _flush_chunk()

    print(f"\n[split-stream] Done: {total_journeys:,} total journeys from "
          f"{n_users:,} users -> {chunk_idx} chunks total")
    if skip_journeys > 0:
        print(f"  Resumed from chunk {resume_chunk_idx + 1}, "
              f"wrote {written_journeys:,} new journeys")
    print(f"  Run each chunk with:")
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
# Pre-inference dedup: remove duplicate/variant products per journey           #
# ============================================================================ #
_DEDUP_SIZE_RE = re.compile(
    r',?\s*\b(?:XXS|XS|XXL|XL|2XL|3XL|4XL|5XL|2XLarge|3XLarge|XLarge'
    r'|Small|Medium|Large|Womens|Women\'s|Men\'s|Mens)\b'
    r'|,\s*\b[SML]\b'
    r'|\bEU\s*\d{2,3}\b'                   # EU 32, EU 38, EU 42
    r'|\bUS\s*\d{1,2}(?:\.\d)?\b'          # US 6, US 10.5
    r'|\b\d{1,2}(?:\.\d)?\s*(?:Petite|Tall|Regular|Short)\b',  # 8 Petite
    re.IGNORECASE)
_DEDUP_NUMSIZE_RE = re.compile(
    r',?\s*Size:?\s*\d{1,2}(?:\.\d)?(?:\s*[A-Z]{1,3})?\b'  # Size 6, Size: 9.5M
    r'|\bSize\s+\d{1,2}(?:\.\d)?(?:\s*[A-Z]{1,3})?\b'
    r'|\b\d{1,2}(?:\.\d)?\s*(?:[MW]{1,2})\b'               # 9.5M, 8.5WW
    r'|\b\d{1,2}(?:\.\d)?\s*(?:Regular|Wide|Narrow|D - Medium|4E|EE)\b'
    r'|:\s*\d{1,2}(?:\.\d)?\s*$',                           # trailing ": 6"
    re.IGNORECASE)
_DEDUP_COLOR_RE = re.compile(
    r'\b(?:Black|White|Blue|Red|Grey|Gray|Navy|Green|Brown|Orange|Yellow|'
    r'Purple|Pink|Silver|Gold|Beige|Ivory|Charcoal|Titanium|Platinum|'
    r'Indigo|Slate|Taupe|Coral|Cream|Nude|Peach|Teal|Burgundy|Maroon|'
    r'Olive|Sage|Rust|Lilac|Lavender|Mauve|Mint|Khaki|Tan|Camel|Wine|'
    r'Light Beige|Light Blue|Light Brown|Light Grey|Off White|'
    r'Slate Blue|Dark Blue|Dark Brown|Dark Grey|Rose Gold)\b'
    r'|(?<=\s)(?:Petal|Blush|Driftwood|Moonstone|Alpine|Chai|'
    r'Midnight|Storm|Frost|Sand|Oat|Espresso|Cognac|Mocha|'
    r'Truffle|Biscuit|Mushroom|Pewter)(?:\s|$)',
    re.IGNORECASE)


def _dedup_normalize_title(title):
    """Normalize product title for same-seller variant dedup."""
    t = _DEDUP_SIZE_RE.sub('', title)
    t = _DEDUP_NUMSIZE_RE.sub('', t)
    t = _DEDUP_COLOR_RE.sub('', t)
    t = re.sub(r'\s+(?:upper|suede|leather|fabric)\b.*$', '', t,
               flags=re.IGNORECASE)
    # Strip trailing material/color descriptors after comma
    t = re.sub(r',\s*(?:Leather|Suede|Canvas|Mesh|Knit|Satin|Silk)\s+\w+.*$',
               '', t, flags=re.IGNORECASE)
    t = re.sub(r'[,\-\s:]+$', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def dedup_journey_products(jwp_json_str):
    """Remove duplicate products from a journey's JourneyWithProducts JSON.

    Two layers:
      1. Exact OfferId dedup (cross-query): same product returned by
         multiple queries → keep first occurrence.
      2. Same-seller size/color variant dedup: same seller + normalized
         title → keep first occurrence (highest ANN score).

    Modifies the journey JSON in-place and returns the updated string.
    Also returns (n_oid_dup, n_variant_dup) counts.
    """
    try:
        jwp = _json_loads(jwp_json_str)
    except Exception:
        return jwp_json_str, 0, 0

    if not isinstance(jwp, dict) or "Queries" not in jwp:
        return jwp_json_str, 0, 0

    seen_oids = set()
    seen_norm = set()
    n_oid_dup = 0
    n_variant_dup = 0

    for q in jwp.get("Queries", []):
        if not isinstance(q, dict):
            continue
        products = q.get("Products", [])
        if not isinstance(products, list):
            continue

        deduped = []
        for p in products:
            if not isinstance(p, dict):
                continue
            oid = str(p.get("OfferId", ""))

            # Layer 1: exact OfferId dedup
            if oid and oid in seen_oids:
                n_oid_dup += 1
                continue
            if oid:
                seen_oids.add(oid)

            # Layer 2: same-seller variant dedup
            title = p.get("Title", "")
            seller = p.get("Seller", "")
            norm_key = (_dedup_normalize_title(title), seller)
            if norm_key[0] and norm_key in seen_norm:
                n_variant_dup += 1
                continue
            if norm_key[0]:
                seen_norm.add(norm_key)

            deduped.append(p)

        q["Products"] = deduped

    return _json_dumps_compact(jwp), n_oid_dup, n_variant_dup


# ============================================================================ #
# Inference mode (streaming, memory-efficient)                                 #
# ============================================================================ #
def run_inference_streaming(prompt_template, args):
    """Memory-efficient streaming inference.

    Reads input file in chunks of args.chunk_size, deduplicates products
    (sequential, no multiprocessing fork), builds prompts, calls LLM,
    and writes results — all per chunk. Then frees the chunk before
    reading the next one.

    Peak memory: O(chunk_size) instead of O(total_journeys).
    For chunk_size=10,000 with avg 50KB prompts: ~1.5 GB per chunk
    vs ~120 GB for 900K journeys loaded all at once.

    Supports checkpoint/resume: on restart, previously completed LLM
    calls are loaded from checkpoint files and skipped.
    """
    input_base = os.path.splitext(os.path.basename(args.input_file))[0]
    checkpoint_dir = os.path.join(args.output_dir,
                                  f"_ranker_ckpt_{input_base}")
    output_file = os.path.join(args.output_dir, f"{input_base}_Results.tsv")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint for resume
    completed = load_checkpoint(checkpoint_dir)

    # Load and validate tokens once
    tokens = load_tokens(args.token_file)
    tokens = validate_tokens(tokens, model=args.copilot_model)

    # Determine checkpoint chunk offset (continue numbering)
    existing_ckpt = []
    if os.path.exists(checkpoint_dir):
        existing_ckpt = [f for f in os.listdir(checkpoint_dir)
                         if f.endswith('.jsonl')]
    chunk_offset = len(existing_ckpt)

    max_j = args.debug_rows if args.debug else 0

    # Stats
    total_journeys = 0
    total_success = 0
    total_valid_json = 0
    total_empty_products = 0
    total_failed = 0
    total_oid_dup = 0
    total_variant_dup = 0
    fail_keys = []

    t0_all = time.time()
    chunk_num = 0
    sample_shown = False

    print(f"\n[infer] Streaming inference (chunk_size={args.chunk_size:,}) ...")

    # Always write output from scratch (checkpoint handles LLM resume)
    with open(output_file, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=RESULT_COLUMNS,
                                delimiter="\t", extrasaction="ignore")
        writer.writeheader()

        for chunk_rows in _iter_journey_chunks(args.input_file,
                                               args.chunk_size,
                                               max_journeys=max_j):
            chunk_num += 1
            n = len(chunk_rows)
            total_journeys += n

            # Show sample from first chunk
            if not sample_shown:
                sample = chunk_rows[0]
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
                sample_shown = True

            # --- Sequential dedup + build prompts (skip checkpoint items) ---
            inputs = []
            for jr in chunk_rows:
                key = jr["JourneyKey"]
                if key in completed:
                    continue  # skip dedup + prompt for already-done items
                cleaned, n_oid, n_var = dedup_journey_products(
                    jr["JourneyWithProducts"])
                jr["JourneyWithProducts"] = cleaned
                total_oid_dup += n_oid
                total_variant_dup += n_var
                prompt = build_ranker_prompt(
                    jr["ShoppingProfile"],
                    jr["JourneyWithProducts"],
                    prompt_template,
                )
                inputs.append((key, prompt))

            # --- Call LLM for new items ---
            skipped = n - len(inputs)
            if inputs:
                print(f"\n  Chunk {chunk_num} ({n:,} journeys): "
                      f"{len(inputs):,} to infer"
                      + (f", {skipped:,} from checkpoint" if skipped else ""))
                t0 = time.time()
                results = run_llm_parallel(
                    inputs=inputs,
                    num_workers=args.num_workers,
                    model=args.copilot_model,
                    temperature=0,
                    max_tokens=args.max_tokens,
                    _tokens=tokens,
                )
                elapsed = time.time() - t0

                # Save checkpoint
                chunk_results = []
                for key, response in results:
                    completed[key] = response
                    chunk_results.append((key, response))
                save_checkpoint(chunk_results, checkpoint_dir,
                                chunk_offset + chunk_num - 1)

                success_n = sum(1 for _, r in chunk_results if r)
                print(f"  Chunk done in {elapsed:.1f}s "
                      f"({len(inputs) / max(elapsed, 1):.1f} items/s), "
                      f"{success_n}/{len(inputs)} succeeded")
                del results, chunk_results
            else:
                print(f"\n  Chunk {chunk_num} ({n:,} journeys): "
                      f"all {skipped:,} from checkpoint")

            # --- Write results for this chunk ---
            for jr in chunk_rows:
                key = jr["JourneyKey"]
                raw_result = completed.get(key, "")

                if not raw_result:
                    total_failed += 1
                    fail_keys.append(key)
                    writer.writerow({
                        "JourneyKey": key,
                        "UserId": jr["UserId"],
                        "JourneyIdx": jr["JourneyIdx"],
                        "OUTPUT": "",
                    })
                    continue

                total_success += 1
                clean = extract_ranked_journey(raw_result)
                if clean:
                    try:
                        obj = json.loads(clean)
                        if not obj.get("Products"):
                            total_empty_products += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
                    total_valid_json += 1
                    output_val = clean
                else:
                    output_val = raw_result.replace("\n", " ").replace(
                        "\t", " ")

                writer.writerow({
                    "JourneyKey": key,
                    "UserId": jr["UserId"],
                    "JourneyIdx": jr["JourneyIdx"],
                    "OUTPUT": output_val,
                })

            # Free chunk memory
            del chunk_rows, inputs

    elapsed_all = time.time() - t0_all
    sz = os.path.getsize(output_file) / (1024 * 1024)

    if total_oid_dup or total_variant_dup:
        print(f"\n[infer] Dedup totals: removed {total_oid_dup:,} OfferId "
              f"dups + {total_variant_dup:,} size/color variants "
              f"(total {total_oid_dup + total_variant_dup:,})")

    print(f"\n[infer] Results: {output_file} ({sz:.1f} MB)")
    print(f"  Total journeys:    {total_journeys:,}")
    print(f"  API success:       {total_success:,}/{total_journeys:,}")
    print(f"  Valid JSON:        {total_valid_json:,}/{total_success:,}")
    print(f"  Empty Products:    {total_empty_products:,}")
    print(f"  Inference failed:  {total_failed:,}")
    if fail_keys[:10]:
        print(f"  Failed keys (first 10): {fail_keys[:10]}")
    print(f"  Total time: {elapsed_all:.1f}s "
          f"({total_journeys / max(elapsed_all, 1):.1f} journeys/s)")

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
    """Merge all *_Results.tsv into a per-user output aligned with step5.

    Optimized for large datasets:
      - 64 MB read buffer for NFS throughput
      - str.split('\\t') instead of csv.reader
      - Single JWP parse per user (count + raw stored together)
      - Progressive cleanup: removes used keys from all_results
      - Uses _json_dumps_compact (orjson) for output serialization
    """
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
    #    Results TSV is written by csv.DictWriter which quotes fields
    #    containing quotes/tabs, so we must use csv.reader to unquote.
    all_results = {}  # JourneyKey -> OUTPUT string
    for fpath in result_files:
        count = 0
        with open(fpath, "r", encoding="utf-8", buffering=8 << 20) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = (row.get("JourneyKey") or "").strip()
                output = (row.get("OUTPUT") or "").strip()
                if key and output:
                    all_results[key] = output
                    count += 1
        print(f"  {os.path.basename(fpath)}: {count:,} valid results")
    print(f"  Total unique results: {len(all_results):,}")

    # 3. Stream-read original step5 output, build per-user data, and write
    #    merged output in a single pass. This avoids storing all user data
    #    in memory — we process each user as we encounter them.
    print(f"\n[merge] Reading step5 output + writing merged: {input_file}")

    try:
        file_size = os.path.getsize(input_file)
    except OSError:
        file_size = 0

    input_base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{input_base}_Ranked.tsv")
    os.makedirs(output_dir, exist_ok=True)

    stats = {
        "users_total": 0,
        "users_with_ranked": 0,
        "users_partial_fail": 0,
        "journeys_total": 0,
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
        result = []
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

    seen_uids = set()

    with open(input_file, "rb", buffering=64 << 20) as fb, \
         open(output_file, "w", encoding="utf-8",
              buffering=8 << 20, newline="") as fout:

        header_line = fb.readline()
        header_text = header_line.decode("utf-8", errors="replace").rstrip("\r\n")
        header = header_text.split("\t")
        col_idx = {name: i for i, name in enumerate(header)}
        idx_uid = col_idx.get("UserId", -1)
        idx_events = col_idx.get("ReadableUserEvents", -1)
        idx_profile = col_idx.get("ShoppingProfile", -1)
        idx_jwp = col_idx.get("JourneyWithProducts", -1)

        # Write output header
        fout.write("\t".join(MERGE_OUTPUT_COLUMNS) + "\n")

        pbar = tqdm(total=file_size or None, unit="B", unit_scale=True,
                    desc="[merge] read+write", mininterval=60)
        pbar.update(len(header_line))

        for line_bytes in fb:
            pbar.update(len(line_bytes))
            line = line_bytes.decode("utf-8", errors="replace")
            # Step5 per-user format: JWP JSON may contain quoted fields
            # with tabs/newlines, must use csv.reader for correct parsing.
            fields = next(csv.reader([line], delimiter="\t"))
            nf = len(fields)

            uid = (fields[idx_uid] if idx_uid < nf else "").strip()
            if not uid or uid in seen_uids:
                continue
            seen_uids.add(uid)
            stats["users_total"] += 1

            # Extract fields
            events = fields[idx_events] if idx_events < nf else ""
            profile = fields[idx_profile] if idx_profile < nf else ""
            jwp_raw = (fields[idx_jwp] if idx_jwp < nf else "").strip()

            # Parse JWP once: get journey count + build lookup
            jwp_lookup = _build_journey_lookup(jwp_raw) if jwp_raw else []
            n_journeys = len(jwp_lookup)
            stats["journeys_total"] += n_journeys

            # Assemble ranked journeys for this user
            ranked_journeys = []
            has_fail = False

            for idx in range(n_journeys):
                key = f"{uid}_{idx}"
                raw_output = all_results.pop(key, "")  # pop to free memory

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
                    obj = _json_loads(clean)
                except Exception:
                    stats["journeys_invalid_json"] += 1
                    has_fail = True
                    continue

                products = obj.get("Products", [])
                if not products:
                    stats["journeys_empty_products"] += 1
                    continue

                # Backfill metadata from JWP
                if idx < len(jwp_lookup):
                    j_meta, oid_to_prod, oid_to_query = jwp_lookup[idx]
                    for field in ("Description", "ConversationStarter",
                                  "WhyAmISeeingThis"):
                        if field not in obj or not obj[field]:
                            obj[field] = j_meta.get(field, "")
                    total_candidates = len(oid_to_prod)
                    selected_count = len(products)
                    if "RankingSummary" not in obj:
                        obj["RankingSummary"] = {
                            "totalCandidates": total_candidates,
                            "selectedCount": selected_count,
                            "filteredCount": total_candidates - selected_count,
                        }
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

            ranked_json = _json_dumps_compact(
                {"ContinuedJourneys": ranked_journeys})

            # Write directly with tab-join (avoid csv.DictWriter overhead)
            # Columns: UserId, ReadableUserEvents, ShoppingProfile,
            #          JourneyWithProducts, RankedJourneys
            out_fields = [uid, events, profile, jwp_raw, ranked_json]
            fout.write("\t".join(out_fields) + "\n")

        pbar.close()

    sz = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n[merge] Output: {output_file} ({sz:.1f} MB)")
    if all_results:
        print(f"  WARNING: {len(all_results):,} result keys had no matching "
              f"user in step5 input (orphaned)")
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

    # ---- Streaming split mode (constant memory) ----
    if args.split_n > 0:
        run_split_streaming(args.input_file, args.output_dir, args.split_n)
        return

    # ---- Inference mode (streaming, memory-efficient) ----
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[infer] Loading prompt: {args.prompt_file}")
    prompt_template = load_prompt_template(args.prompt_file)
    print(f"  Template: {len(prompt_template):,} chars")

    run_inference_streaming(prompt_template, args)

    print("\nStep 6 Done!")


if __name__ == "__main__":
    main()
