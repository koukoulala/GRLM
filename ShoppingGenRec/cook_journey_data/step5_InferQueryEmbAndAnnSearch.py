"""
step5_InferQueryEmbAndAnnSearch.py
==================================

Pipeline step 5 — for the journey output TSV from step3:

  1. Parse each row's ``OUTPUT`` column (ContinuedJourneys JSON).
  2. Collect every ``Query`` string across all journeys, dedup.
  3. Embed all queries via the MatadorEmb ONNX model (in-process, GPU).
  4. ANN-search the FAISS index built by step4 to get the top-K
     ``GlobalOfferId`` for every query.
  5. Load product metadata from ``item.json`` (step0 output).
  6. Attach a ``Products`` list (with metadata) to each ``Query``.
  7. Write a TSV with columns:
     ``UserId, ReadableUserEvents, ShoppingProfile, ShoppingJourneys,
     JourneyWithProducts``

Examples
--------
    python step5_InferQueryEmbAndAnnSearch.py

    python step5_InferQueryEmbAndAnnSearch.py --num_users 100
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sqlite3
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback so the script still runs without tqdm
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else iter(())

csv.field_size_limit(sys.maxsize)


# ============================================================================ #
# Constants — paths relative to this script's directory                        #
# ============================================================================ #
SCRIPT_DIR = "/vc_data/users/wangying/OneRec/ShoppingJourney/Pipeline/run_matador_emb"

_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INFERENCE_DIR = os.path.join(_PROJECT_DIR, "run_matador_emb")
TOKENIZER_PATH = os.path.join(
    INFERENCE_DIR,
    "simiaozuo_dense_retrieval_url_data_20250415_checkpoints_model_1_checkpoint-keyword",
)
ONNX_MODEL_PATH = os.path.join(TOKENIZER_PATH, "model_dynamic.onnx")


# ============================================================================ #
# Stage 1 — parse the SLM journey TSV                                          #
# ============================================================================ #
def parse_journey_json(raw: str):
    """Best-effort parse of the ``ParsedJourneys`` column."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    tag_match = re.search(r"<OUTPUT>\s*(.*?)\s*</OUTPUT>", text, re.DOTALL)
    if tag_match:
        text = tag_match.group(1).strip()
    bs = text.find("{")
    if bs == -1:
        return None
    depth, be = 0, -1
    for i in range(bs, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                be = i
                break
    cand = text[bs : be + 1] if be != -1 else text[bs:] + "}"
    for t in (cand, text):
        try:
            data = json.loads(t)
            if "ContinuedJourneys" in data:
                return data
        except Exception:
            pass
    # Fallback: iteratively unescape backslash-quoted JSON that appears when
    # step3 output goes through csv.writer/reader roundtrips.  Some rows have
    # multiple levels of escaping (\" → \\\" → …), so we loop up to 3 rounds.
    _BS = chr(92)   # backslash
    _Q = chr(34)    # double-quote
    _PH = "\x00_ESC_Q_\x00"
    cur = text
    for _round in range(3):
        if _BS + _Q not in cur:
            break
        cur = cur.replace(_BS + _BS + _Q, _PH)   # \\" -> placeholder
        cur = cur.replace(_BS + _Q, _Q)           # \"  -> "
        cur = cur.replace(_PH, _BS + _Q)          # placeholder -> \"
        bs2 = cur.find("{")
        if bs2 == -1:
            break
        depth2, be2 = 0, -1
        for i in range(bs2, len(cur)):
            if cur[i] == "{":
                depth2 += 1
            elif cur[i] == "}":
                depth2 -= 1
                if depth2 == 0:
                    be2 = i
                    break
        cand2 = cur[bs2 : be2 + 1] if be2 != -1 else cur[bs2:]
        try:
            data = json.loads(cand2)
            if "ContinuedJourneys" in data:
                return data
        except Exception:
            pass
    return None


def load_input_rows(input_tsv: str, num_users: int = 0,
                    n_workers: int = 4):
    """Read the input TSV and parse the ``ShoppingJourneys`` JSON per row.

    Implementation is a bounded producer-consumer pipeline so the read,
    JSON parse, and progress reporting truly overlap (using
    ``ThreadPoolExecutor.map`` would submit every task before yielding any
    result, defeating both streaming and the progress bar).

    The file is opened in binary mode with a large buffer to make NFS
    readahead happy, and progress is reported in real bytes consumed.
    """
    rows = []
    skipped = 0

    try:
        total_bytes = os.path.getsize(input_tsv)
    except OSError:
        total_bytes = 0

    pbar = tqdm(total=total_bytes or None,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc="[stage1] read+parse",
                mininterval=60, smoothing=0.1)

    # 8 MB binary read buffer -- big win on /cosmos NFS where the kernel
    # default of 8 KB causes a tiny-RTT-bound read pattern.
    with open(input_tsv, "rb", buffering=8 << 20) as fb:
        header_line = fb.readline()
        if not header_line:
            pbar.close()
            return [], [], 0
        pbar.update(len(header_line))
        header_text = header_line.decode("utf-8", errors="replace")
        header = next(csv.reader([header_text], delimiter="\t"))
        fieldnames = list(header)

        # Pre-resolve column indices once -> fast per-row access in workers.
        # Support both step3 output ("OUTPUT") and legacy column names.
        try:
            idx_journey = header.index("OUTPUT")
        except ValueError:
            try:
                idx_journey = header.index("ShoppingJourneys")
            except ValueError:
                idx_journey = -1
        try:
            idx_raw = header.index("RawShoppingJourneys")
        except ValueError:
            idx_raw = -1

        def parse_one(line_bytes):
            # Returns (row_dict, parsed_or_None, byte_len) so the writer side
            # can advance the byte-based progress bar accurately.
            n = len(line_bytes)
            line = line_bytes.decode("utf-8", errors="replace")
            row_list = next(csv.reader([line], delimiter="\t"))
            row = dict(zip(header, row_list))
            if 0 <= idx_journey < len(row_list):
                raw = row_list[idx_journey]
            else:
                raw = ""
            if not raw and 0 <= idx_raw < len(row_list):
                raw = row_list[idx_raw]
            raw = raw.strip().strip('"') if raw else ""
            parsed = parse_journey_json(raw) if raw else None
            # Rename OUTPUT -> ShoppingJourneys for output column consistency.
            # Store clean JSON (re-serialized from parsed data) instead of the
            # raw backslash-escaped TSV encoding.
            if "OUTPUT" in row:
                row.pop("OUTPUT")
            if parsed:
                row["ShoppingJourneys"] = json.dumps(
                    parsed, ensure_ascii=False, separators=(',', ':'))
            else:
                row["ShoppingJourneys"] = raw
            # Clean backslash-escaped JSON in ShoppingProfile column too.
            _BS_Q = chr(92) + chr(34)
            for col in ("ShoppingProfile",):
                val = row.get(col, "")
                if val and _BS_Q in val:
                    try:
                        _PH = "\x00_PH_\x00"
                        cleaned = val
                        for _ in range(3):
                            if _BS_Q not in cleaned:
                                break
                            cleaned = cleaned.replace(
                                chr(92) + chr(92) + chr(34), _PH)
                            cleaned = cleaned.replace(_BS_Q, chr(34))
                            cleaned = cleaned.replace(_PH, chr(92) + chr(34))
                        obj = json.loads(cleaned)
                        row[col] = json.dumps(
                            obj, ensure_ascii=False, separators=(',', ':'))
                    except Exception:
                        row[col] = val  # keep original if can't parse
            return row, parsed, n

        # Bounded pipeline: keep at most `max_pending` futures in flight so
        # the reader naturally throttles to the workers' pace and tqdm gets
        # a steady stream of completions.
        max_pending = max(2 * n_workers, 16)
        pending = deque()

        def _drain_one():
            row, parsed, nb = pending.popleft().result()
            return row, parsed, nb

        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
            n_submitted = 0
            for line_bytes in fb:
                if not line_bytes:
                    break
                if num_users and n_submitted >= num_users:
                    break
                pending.append(ex.submit(parse_one, line_bytes))
                n_submitted += 1
                while len(pending) >= max_pending:
                    row, parsed, nb = _drain_one()
                    if parsed is None:
                        skipped += 1
                    rows.append((row, parsed))
                    pbar.update(nb)
            while pending:
                row, parsed, nb = _drain_one()
                if parsed is None:
                    skipped += 1
                rows.append((row, parsed))
                pbar.update(nb)

        pbar.set_postfix(rows=len(rows), skipped=skipped)
        pbar.close()

    # Rename OUTPUT -> ShoppingJourneys in fieldnames to match row dicts
    fieldnames = ["ShoppingJourneys" if f == "OUTPUT" else f for f in fieldnames]

    return fieldnames, rows, skipped


def collect_queries(rows) -> Set[str]:
    """Pull every Query string out of every journey of every row."""
    queries: Set[str] = set()
    for _, parsed in rows:
        if not parsed:
            continue
        for j in parsed.get("ContinuedJourneys") or []:
            if not isinstance(j, dict):
                continue
            for q in j.get("Queries") or []:
                if isinstance(q, dict):
                    qt = q.get("Query")
                    if not isinstance(qt, str):
                        # A handful of upstream LLM rows emit `Query` as a
                        # nested dict / list / null instead of a string.
                        # Skip those rather than crash.
                        continue
                    qt = qt.strip()
                    if qt:
                        queries.add(qt)
    return queries


# ============================================================================ #
# Stage 2 — query embedding inference (in-process ONNX)                        #
# ============================================================================ #
def load_onnx_session(gpu_ids: str):
    """Load tokenizer + ONNX session once; reused across all chunks.

    ``gpu_ids`` is a comma-separated string (e.g. ``"0,1,2,3"``). All listed
    GPUs are made visible via ``CUDA_VISIBLE_DEVICES`` so the same env can be
    reused by FAISS for multi-GPU ANN search; the ONNX session itself only
    binds to the first visible device.
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

    print(f"[infer] tokenizer: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    providers = ["CPUExecutionProvider"]
    try:
        if ort.get_device() == "GPU":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    print(f"[infer] ONNX providers (preferred order): {providers}")
    t0 = time.time()
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    print(f"[infer] session loaded in {time.time() - t0:.1f}s; "
          f"actual providers: {session.get_providers()}")
    return tokenizer, session


def encode_queries_to_matrix(queries: List[str],
                             tokenizer,
                             session,
                             batch_size: int,
                             max_length: int) -> np.ndarray:
    """Encode a list of queries into an (N, D) L2-normalized float32 matrix."""
    out = None
    write_pos = 0
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        inputs_onnx = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        outputs = session.run(None, inputs_onnx)
        # Model output: [dummy_score (scalar), vec (B, D)]
        vecs = outputs[1].astype(np.float32, copy=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        if out is None:
            out = np.empty((len(queries), vecs.shape[1]), dtype=np.float32)
        out[write_pos : write_pos + len(batch)] = vecs
        write_pos += len(batch)
    return out


# ============================================================================ #
# Stage 3 — FAISS ANN search (chunked, with disk shards)                       #
# ============================================================================ #
def load_faiss_index(index_path: str,
                     id_map_path: str,
                     ef_search: int,
                     gpu_ids: str = ""):
    """Load FAISS index + id map once; reused across all chunks.

    If ``gpu_ids`` is non-empty (e.g. ``"0"`` or ``"0,1,2,3"``), the index is
    moved to GPU(s):
      * 1 GPU  -> ``index_cpu_to_gpu(StandardGpuResources(), 0, ...)``
      * >1 GPU -> ``index_cpu_to_all_gpus(...)`` (sharded across all visible)
    HNSW indexes are not supported on GPU in FAISS and stay on CPU.

    Note: ``CUDA_VISIBLE_DEVICES`` is expected to already be set to ``gpu_ids``
    (this is done by ``load_onnx_session``), so ``index_cpu_to_all_gpus`` will
    automatically pick up exactly the requested devices.
    """
    import faiss

    print(f"[ann] loading FAISS index: {index_path}")
    t0 = time.time()
    index = faiss.read_index(index_path)
    print(f"[ann] index loaded in {time.time() - t0:.1f}s "
          f"(ntotal={index.ntotal:,})")

    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = [line.rstrip("\n") for line in f if line.strip()]
    print(f"[ann] id map loaded: {len(id_map):,} ids")

    is_hnsw = hasattr(index, "hnsw")
    if is_hnsw:
        index.hnsw.efSearch = ef_search
        print(f"[ann] hnsw.efSearch = {ef_search}")
    if hasattr(index, "nprobe"):
        index.nprobe = max(getattr(index, "nprobe", 1), ef_search)
        print(f"[ann] index.nprobe   = {index.nprobe}")

    gpu_id_list = [g.strip() for g in str(gpu_ids).split(",") if g.strip()]
    if gpu_id_list:
        if is_hnsw:
            print("[ann] HNSW index is CPU-only in FAISS; staying on CPU")
        else:
            n_gpus = len(gpu_id_list)
            try:
                t0 = time.time()
                if n_gpus > 1:
                    print(f"[ann] moving index to {n_gpus} GPUs "
                          f"(CUDA_VISIBLE_DEVICES={gpu_ids}) ...")
                    index = faiss.index_cpu_to_all_gpus(index)
                else:
                    print(f"[ann] moving index to 1 GPU "
                          f"(CUDA_VISIBLE_DEVICES={gpu_ids}) ...")
                    index = faiss.index_cpu_to_gpu(
                        faiss.StandardGpuResources(), 0, index)
                print(f"[ann] index on GPU in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"[ann] WARNING: failed to move index to GPU ({e}); "
                      f"falling back to CPU")
    else:
        print("[ann] no gpu_ids given for ANN; using CPU")
    return index, id_map


def _index_search_batched(index, embs: np.ndarray, top_k: int,
                          search_batch: int) -> Tuple[np.ndarray, np.ndarray]:
    """Call ``index.search`` in sub-batches of ``search_batch`` rows.

    A single GPU IVF kernel launch over a very large ``nq`` (especially with
    a high ``nprobe``) can exceed CUDA's per-launch grid limits and crash
    with ``cudaErrorInvalidConfiguration`` (CUDA error 9) inside
    ``ivfInterleavedScanImpl``. Splitting the search into smaller batches
    keeps each launch well within limits and recombines the results.

    ``search_batch <= 0`` falls back to a single call (legacy behavior).
    """
    n = embs.shape[0]
    if search_batch <= 0 or n <= search_batch:
        return index.search(embs, top_k)
    dists = np.empty((n, top_k), dtype=np.float32)
    idxs = np.empty((n, top_k), dtype=np.int64)
    for s in range(0, n, search_batch):
        e = min(s + search_batch, n)
        d_b, i_b = index.search(embs[s:e], top_k)
        dists[s:e] = d_b
        idxs[s:e] = i_b
    return dists, idxs


def chunked_embed_and_search(queries: List[str],
                             tokenizer,
                             session,
                             index,
                             id_map: List[str],
                             top_k: int,
                             batch_size: int,
                             max_length: int,
                             chunk_size: int,
                             shard_dir: str,
                             search_batch: int = 16384) -> Set[str]:
    """Embed + ANN search ``queries`` in chunks of ``chunk_size``.

    For each chunk, writes a shard ``chunk_NNNNN.npz`` into ``shard_dir``
    containing the chunk's queries, top-k product ids, and scores.
    Returns the union of all matched product ids (so Stage 4 can scan the
    product TSV exactly once).

    Memory peak per chunk is O(chunk_size * (D + top_k)) — not O(N).
    """
    os.makedirs(shard_dir, exist_ok=True)

    needed_pids: Set[str] = set()
    n = len(queries)
    n_chunks = (n + chunk_size - 1) // chunk_size
    print(f"[chunk] processing {n:,} queries in {n_chunks} chunks "
          f"of {chunk_size:,}; shard_dir={shard_dir}")

    t_total = time.time()
    for ci, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        chunk_q = queries[start:end]

        # ---- embed (mini-batches handled inside) ----
        t0 = time.time()
        chunk_embs = encode_queries_to_matrix(
            chunk_q, tokenizer, session, batch_size, max_length)
        t_emb = time.time() - t0

        # ---- ANN search (sub-batched to keep GPU kernel launches small) ----
        t0 = time.time()
        dists, idxs = _index_search_batched(
            index, chunk_embs, top_k, search_batch)
        t_ann = time.time() - t0

        # ---- resolve to product ids; accumulate needed_pids ----
        pids_2d = np.empty(idxs.shape, dtype=object)
        for i in range(idxs.shape[0]):
            for r in range(idxs.shape[1]):
                ix = idxs[i, r]
                if ix >= 0:
                    pid = id_map[ix]
                    pids_2d[i, r] = pid
                    needed_pids.add(pid)
                else:
                    pids_2d[i, r] = ""

        shard_path = os.path.join(shard_dir, f"chunk_{ci:05d}.npz")
        np.savez(shard_path,
                 queries=np.array(chunk_q, dtype=object),
                 pids=pids_2d,
                 scores=dists.astype(np.float32))
        # Log every 10 chunks or the last one to reduce log noise
        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            elapsed = time.time() - t_total
            print(f"[chunk] {ci + 1}/{n_chunks}  "
                  f"matched_pids={len(needed_pids):,}  "
                  f"elapsed={elapsed:.1f}s")

        # Free per-chunk buffers immediately.
        del chunk_embs, dists, idxs, pids_2d

    print(f"[chunk] all chunks done in {time.time() - t_total:.1f}s, "
          f"unique matched pids: {len(needed_pids):,}")
    return needed_pids


def load_topk_from_shards(shard_dir: str,
                          top_k: int) -> Dict[str, List[Tuple[str, float]]]:
    """Read every ``chunk_*.npz`` in ``shard_dir`` and merge into one dict.

    WARNING: For large runs (>10M queries), this can use >100 GB of RAM
    due to Python object overhead.  Prefer ``load_topk_to_sqlite`` for
    production runs.
    """
    files = sorted(f for f in os.listdir(shard_dir)
                   if f.startswith("chunk_") and f.endswith(".npz"))
    print(f"[chunk] loading {len(files)} shards from {shard_dir}")
    out: Dict[str, List[Tuple[str, float]]] = {}
    t0 = time.time()
    for fn in files:
        path = os.path.join(shard_dir, fn)
        with np.load(path, allow_pickle=True) as d:
            qs = d["queries"]
            pids = d["pids"]
            scores = d["scores"]
            for i in range(len(qs)):
                row: List[Tuple[str, float]] = []
                for r in range(top_k):
                    pid = pids[i, r]
                    if pid:
                        row.append((str(pid), float(scores[i, r])))
                out[str(qs[i])] = row
    print(f"[chunk] merged {len(out):,} query->topK entries "
          f"from {len(files)} shards in {time.time() - t0:.1f}s")
    return out


# ============================================================================ #
# SQLite-backed ANN results (memory-efficient alternative)                      #
# ============================================================================ #
def load_topk_to_sqlite(shard_dir: str, top_k: int,
                        db_path: str) -> sqlite3.Connection:
    """Load ANN results from chunk shards into a SQLite database.

    Instead of holding 27M+ entries × 20 results in Python dicts (~100 GB),
    this stores them in a SQLite database on local disk (~8-12 GB).
    Lookup speed is ~50K queries/sec which is sufficient for streaming
    stage 5.

    Returns an open sqlite3.Connection for lookups.
    """
    files = sorted(f for f in os.listdir(shard_dir)
                   if f.startswith("chunk_") and f.endswith(".npz"))
    n_files = len(files)
    if not n_files:
        print(f"[sqlite] WARNING: no chunk files found in {shard_dir}")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("CREATE TABLE IF NOT EXISTS ann "
                     "(query TEXT PRIMARY KEY, results TEXT)")
        return conn

    # Check if DB already exists and is complete
    if os.path.isfile(db_path):
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            n_rows = conn.execute("SELECT COUNT(*) FROM ann").fetchone()[0]
            if n_rows > 0:
                # Enable mmap for fast random lookups
                db_sz = os.path.getsize(db_path)
                conn.execute(f"PRAGMA mmap_size={db_sz + 1024*1024}")
                print(f"[sqlite] reusing existing DB: {db_path} "
                      f"({n_rows:,} entries, mmap={db_sz/(1024**3):.1f}GB)")
                return conn
            conn.close()
        except Exception:
            pass
        os.remove(db_path)

    print(f"[sqlite] building ANN DB from {n_files} shards -> {db_path}")
    t0 = time.time()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-65536")  # 64 MB cache
    conn.execute("CREATE TABLE IF NOT EXISTS ann "
                 "(query TEXT PRIMARY KEY, results TEXT)")

    n_total = 0
    for fi, fn in enumerate(files):
        path = os.path.join(shard_dir, fn)
        batch = []
        with np.load(path, allow_pickle=True) as d:
            qs = d["queries"]
            pids = d["pids"]
            scores = d["scores"]
            for i in range(len(qs)):
                row = []
                for r in range(top_k):
                    pid = pids[i, r]
                    if pid:
                        row.append([str(pid), float(scores[i, r])])
                batch.append((str(qs[i]),
                              json.dumps(row, separators=(',', ':'))))
        conn.executemany("INSERT OR REPLACE INTO ann VALUES (?, ?)", batch)
        conn.commit()
        n_total += len(batch)
        if (fi + 1) % 20 == 0 or fi + 1 == n_files:
            elapsed = time.time() - t0
            db_mb = os.path.getsize(db_path) / (1024 * 1024)
            print(f"[sqlite] {fi+1}/{n_files}  entries={n_total:,}  "
                  f"db={db_mb:.0f}MB  elapsed={elapsed:.1f}s")

    # Enable memory-mapped I/O: maps the entire DB into virtual memory.
    # On a 661GB machine this makes random lookups ~10-100x faster than
    # regular file I/O (direct memory reads vs syscalls + page cache).
    db_size_bytes = os.path.getsize(db_path)
    conn.execute(f"PRAGMA mmap_size={db_size_bytes + 1024*1024}")
    print(f"[sqlite] done: {n_total:,} entries in {time.time()-t0:.1f}s  "
          f"db_size={db_size_bytes/(1024**3):.2f}GB  mmap=enabled")
    return conn


def lookup_ann_sqlite(conn: sqlite3.Connection,
                      query: str) -> List[Tuple[str, float]]:
    """Look up ANN results for a single query from SQLite."""
    row = conn.execute(
        "SELECT results FROM ann WHERE query=?", (query,)).fetchone()
    if row:
        return [(p, s) for p, s in json.loads(row[0])]
    return []


def batch_lookup_ann_sqlite(conn: sqlite3.Connection,
                            queries: List[str]
                            ) -> Dict[str, List[Tuple[str, float]]]:
    """Batch lookup: fetch ANN results for multiple queries in one SQL call.

    Uses ``WHERE query IN (?, ?, ...)`` to reduce SQLite roundtrips from
    N individual queries to 1 batch query. Typically 10-30x faster than
    individual lookups when N > 5.
    """
    if not queries:
        return {}
    out = {q: [] for q in queries}
    # SQLite has a limit of 999 bind params; batch in groups
    BATCH = 900
    for i in range(0, len(queries), BATCH):
        batch = queries[i:i + BATCH]
        placeholders = ','.join('?' * len(batch))
        cursor = conn.execute(
            f"SELECT query, results FROM ann WHERE query IN ({placeholders})",
            batch)
        for q, r in cursor:
            out[q] = [(p, s) for p, s in json.loads(r)]
    return out


# ============================================================================ #
# Stage 4 — load product metadata from item.json                               #
# ============================================================================ #
def load_product_meta_from_json(
        item_json_file: str,
        needed_pids) -> Dict[str, Dict[str, str]]:
    """Load product metadata from item.json (step0 output).

    If ``needed_pids`` is a set, only entries in that set are kept.
    If ``needed_pids`` is None, ALL items are loaded (used in resume mode).

    Returns:
        Dict mapping GlobalOfferId -> {"OfferId", "Title", "Seller",
        "Price", "Brand", "Category", "Gender"}.
    """
    import json as _json

    if not os.path.isfile(item_json_file):
        print(f"[meta] WARNING: item.json not found: {item_json_file}")
        return {}
    # needed_pids=set() means nothing needed; needed_pids=None means load all
    if needed_pids is not None and len(needed_pids) == 0:
        return {}

    load_all = needed_pids is None
    print(f"[meta] loading item.json: {item_json_file}"
          f"{'  (all items — resume mode)' if load_all else ''}")
    t0 = time.time()
    try:
        import orjson
        with open(item_json_file, "rb") as f:
            all_items = orjson.loads(f.read())
    except ImportError:
        with open(item_json_file, "r", encoding="utf-8") as f:
            all_items = _json.load(f)
    print(f"[meta] loaded {len(all_items):,} items in {time.time() - t0:.1f}s")

    out: Dict[str, Dict[str, str]] = {}
    iter_pids = all_items.keys() if load_all else needed_pids
    for pid in iter_pids:
        item = all_items.get(pid)
        if item is None:
            continue
        attrs = item.get("attributes", {})
        out[pid] = {
            "OfferId":  pid,
            "Title":    item.get("title", ""),
            "Seller":   str(attrs.get("Seller", "")),
            "Price":    str(attrs.get("Price", "")),
            "Brand":    str(attrs.get("Brand", "")),
            "Category": item.get("categories", ""),
            "Gender":   str(attrs.get("Gender", "")),
        }

    n_target = len(all_items) if load_all else len(needed_pids)
    del all_items
    print(f"[meta] built {len(out):,}/{n_target:,} product entries "
          f"({time.time() - t0:.1f}s)")
    return out


# ============================================================================ #
# Stage 5 — build JourneyWithProducts JSON per row                             #
# ============================================================================ #
def build_journey_with_products(parsed: Dict,
                                ann_results: Dict[str, List[Tuple[str, float]]],
                                product_meta: Dict[str, Dict[str, str]] = None
                                ) -> str:
    """Attach Products list to each Query in each journey of one user.

    Each Product is a dict with metadata (OfferId, Title, Seller, Price,
    Brand, Category, Gender) loaded from item.json.  If ``product_meta``
    is ``None`` or a product is not found, falls back to the bare
    ``GlobalOfferId`` string.
    """
    journeys_out = []
    for j in parsed.get("ContinuedJourneys") or []:
        if not isinstance(j, dict):
            continue
        queries_out = []
        for q in j.get("Queries") or []:
            if not isinstance(q, dict):
                continue
            qt = q.get("Query")
            if not isinstance(qt, str):
                continue
            qt = qt.strip()
            products = []
            for pid, _score in ann_results.get(qt, []):
                if product_meta and pid in product_meta:
                    products.append(product_meta[pid])
                else:
                    products.append(pid)
            queries_out.append({"Query": qt, "Products": products})
        journeys_out.append({
            "JourneyType":         j.get("JourneyType", ""),
            "Title":               j.get("Title", ""),
            "Queries":             queries_out,
            "Description":         j.get("Description", ""),
            "ConversationStarter": j.get("ConversationStarter", ""),
            "WhyAmISeeingThis":    j.get("WhyAmISeeingThis", ""),
        })
    return json.dumps({"ContinuedJourneys": journeys_out}, ensure_ascii=False)


# Name of the new column written to the output TSV.
OUT_COLUMN = "JourneyWithProducts"


def _build_jwp_with_lookup(parsed: Dict,
                            ann_lookup,
                            product_meta: Dict[str, Dict[str, str]] = None,
                            ann_batch_lookup=None,
                            ) -> str:
    """Like build_journey_with_products but uses a callable for ANN lookup.

    If ``ann_batch_lookup`` is provided, all queries are collected first
    and looked up in a single batch call (much faster for SQLite).
    Otherwise falls back to per-query ``ann_lookup(query_str)``.
    """
    # Collect all unique queries across all journeys for batch lookup
    all_queries = []
    for j in parsed.get("ContinuedJourneys") or []:
        if not isinstance(j, dict):
            continue
        for q in j.get("Queries") or []:
            if isinstance(q, dict):
                qt = q.get("Query")
                if isinstance(qt, str) and qt.strip():
                    all_queries.append(qt.strip())

    # Batch lookup if available (one SQL query instead of N)
    if ann_batch_lookup and all_queries:
        unique_qs = list(set(all_queries))
        ann_cache = ann_batch_lookup(unique_qs)
    else:
        ann_cache = None

    def _get_results(query):
        if ann_cache is not None:
            return ann_cache.get(query, [])
        return ann_lookup(query)

    journeys_out = []
    for j in parsed.get("ContinuedJourneys") or []:
        if not isinstance(j, dict):
            continue
        queries_out = []
        for q in j.get("Queries") or []:
            if not isinstance(q, dict):
                continue
            qt = q.get("Query")
            if not isinstance(qt, str):
                continue
            qt = qt.strip()
            products = []
            for pid, _score in _get_results(qt):
                if product_meta and pid in product_meta:
                    products.append(product_meta[pid])
                else:
                    products.append(pid)
            queries_out.append({"Query": qt, "Products": products})
        journeys_out.append({
            "JourneyType":         j.get("JourneyType", ""),
            "Title":               j.get("Title", ""),
            "Queries":             queries_out,
            "Description":         j.get("Description", ""),
            "ConversationStarter": j.get("ConversationStarter", ""),
            "WhyAmISeeingThis":    j.get("WhyAmISeeingThis", ""),
        })
    return json.dumps({"ContinuedJourneys": journeys_out}, ensure_ascii=False)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 5: query embedding + ANN search + JourneyWithProducts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O ---------------------------------------------------------------------
    g_io = p.add_argument_group("I/O")
    g_io.add_argument("--input_tsv", 
                      default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/UserEvents_clean_profiles_results_Journey_Results_combined.tsv",
                      help="Journey output TSV from step3 (UserId, "
                           "ReadableUserEvents, ShoppingProfile, RequestTime, "
                           "HisCount, OUTPUT).")
    g_io.add_argument("--item_json", 
                      default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/item.json",
                      help="Path to item.json (output of step0). Used to "
                           "attach product metadata to search results.")
    g_io.add_argument("--work_dir", 
                      default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data",
                      help="Output directory.")
    g_io.add_argument("--output_prefix", default="UserEvents_clean_combined_full",
                      help="Filename prefix for the output TSV.")

    # Step4 index (for ANN search) -------------------------------------------
    g_idx = p.add_argument_group("Step4 index")
    g_idx.add_argument("--index_dir",
                      default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/MatadorEmb_Index",
                      help="Directory containing the FAISS index and id-map "
                           "files built by step4.")
    g_idx.add_argument("--index_prefix", default="Items_full",
                      help="Filename prefix used in step4 (--output_prefix).")
    g_idx.add_argument("--index_type", default="hnsw",
                      choices=["flat", "ivf", "hnsw"],
                      help="Index type used in step4 (determines filename).")
    g_idx.add_argument("--ann_index", default=None,
                      help="Override: explicit path to the FAISS index file. "
                           "If not set, auto-resolved from "
                           "index_dir/index_prefix_ann_<type>.index.")
    g_idx.add_argument("--ann_id_map", default=None,
                      help="Override: explicit path to the id-map txt file. "
                           "If not set, auto-resolved from "
                           "index_dir/index_prefix_ann_ids.txt.")

    # Embedding inference -----------------------------------------------------
    g_inf = p.add_argument_group("Query embedding inference")
    g_inf.add_argument("--gpu_ids", default="0,1,2,3",
                       help="Comma-separated GPU ids, e.g. '0' or '0,1,2,3'. "
                            "All listed GPUs are made visible via "
                            "CUDA_VISIBLE_DEVICES; the ONNX query encoder uses "
                            "the first one, FAISS ANN search uses all of them "
                            "(unless the index is HNSW, which is CPU-only).")
    g_inf.add_argument("--max_length", type=int, default=64,
                       help="Tokenizer max_length for query encoding.")
    g_inf.add_argument("--batch_size", type=int, default=512)

    # ANN search --------------------------------------------------------------
    g_ann = p.add_argument_group("ANN search")
    g_ann.add_argument("--top_k", type=int, default=20)
    g_ann.add_argument("--ef_search", type=int, default=128,
                       help="HNSW efSearch / IVF nprobe.")
    g_ann.add_argument("--chunk_size", type=int, default=200_000,
                       help="Process queries in chunks of this size; each "
                            "chunk's (queries, top_k pids, scores) is written "
                            "to a shard file under "
                            "<work_dir>/_step5_chunks_<output_prefix>/.")
    g_ann.add_argument("--ann_search_batch", type=int, default=16384,
                       help="Sub-batch size for the FAISS index.search() call. "
                            "Large nq * nprobe combos can exceed the CUDA "
                            "kernel launch limits on GPU IVF indexes (CUDA "
                            "error 9 'invalid configuration argument'); "
                            "sub-batching avoids that. Set 0 to disable.")
    g_ann.add_argument("--keep_chunks", action="store_true",
                       help="Keep the per-chunk shard files after the output "
                            "TSV is written (default: delete them).")
    g_ann.add_argument("--ann_db", default="",
                       help="Path for the SQLite ANN results database. "
                            "Defaults to /tmp/step5_ann_<output_prefix>.db. "
                            "Using SQLite saves ~100GB RAM vs in-memory dict.")
    g_ann.add_argument("--ann_in_memory", action="store_true",
                       default=True,
                       help="Load ANN results into memory dict instead of "
                            "SQLite. Much faster (~100x). Needs ~100GB RAM. "
                            "Use --no_ann_in_memory to force SQLite on "
                            "low-memory machines.")
    g_ann.add_argument("--no_ann_in_memory", action="store_false",
                       dest="ann_in_memory",
                       help="Force SQLite-backed ANN lookups (slow but "
                            "uses ~0 RAM). Only for machines with <200GB.")

    # Resume ------------------------------------------------------------------
    g_res = p.add_argument_group("Resume")
    g_res.add_argument("--resume", action="store_true",
                       help="Resume from existing chunk shards: skip stages "
                            "1-3 (TSV parsing, embedding, ANN search). "
                            "Requires chunk shard dir to exist. "
                            "Stage 5 re-reads the input TSV in streaming "
                            "mode (no rows kept in memory).")
    g_res.add_argument("--resume_offset", type=int, default=0,
                       help="Skip the first N data rows in stage 5 output "
                            "(useful to resume a partially-written file). "
                            "The header row is always written.")

    # Debug -------------------------------------------------------------------
    g_dbg = p.add_argument_group("Debug")
    g_dbg.add_argument("--num_users", type=int, default=0,
                       help="Limit to first N rows (0 = all).")
    g_dbg.add_argument("--io_workers", type=int, default=16,
                       help="Thread-pool size used to parallelize JSON "
                            "parsing (Stage 1) and JourneyWithProducts "
                            "building (Stage 5). Set to 1 to disable.")

    return p.parse_args()


def default_output_path(work_dir: str, output_prefix: str) -> str:
    return os.path.join(work_dir, f"{output_prefix}_journey_with_products.tsv")


def main() -> None:
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    args.output_tsv = default_output_path(args.work_dir, args.output_prefix)
    os.makedirs(os.path.dirname(args.output_tsv) or ".", exist_ok=True)

    # Resolve ANN index paths from index_dir/index_prefix if not explicitly set
    if args.ann_index is None:
        args.ann_index = os.path.join(
            args.index_dir,
            f"{args.index_prefix}_ann_{args.index_type}.index")
    if args.ann_id_map is None:
        args.ann_id_map = os.path.join(
            args.index_dir, f"{args.index_prefix}_ann_ids.txt")

    index_path = args.ann_index
    id_map_path = args.ann_id_map
    shard_dir = os.path.join(args.work_dir,
                             f"_step5_chunks_{args.output_prefix}")

    # Default SQLite DB path
    if not args.ann_db:
        args.ann_db = f"/tmp/step5_ann_{args.output_prefix}.db"

    print("=" * 80)
    print("  step5: query embedding + ANN search -> JourneyWithProducts")
    print("=" * 80)
    print(f"  input_tsv:    {args.input_tsv}")
    print(f"  item_json:    {args.item_json}")
    print(f"  work_dir:     {args.work_dir}")
    print(f"  output_tsv:   {args.output_tsv}")
    print(f"  num_users:    {args.num_users} (0=all)")
    print(f"  resume:       {args.resume}")
    print(f"  resume_offset:{args.resume_offset}")
    print(f"  ann_db:       {args.ann_db}")
    print(f"  ann_in_memory:{args.ann_in_memory}")
    if not args.resume:
        print(f"  ann_index:    {index_path}")
        print(f"  ann_id_map:   {id_map_path}")
        print(f"  GPUs:         {args.gpu_ids}  "
              f"(batch={args.batch_size}, max_len={args.max_length})")
    print(f"  ANN:          top_k={args.top_k}, ef_search={args.ef_search}, "
          f"chunk_size={args.chunk_size:,}, keep_chunks={args.keep_chunks}")
    print(f"  out column:   {OUT_COLUMN}")
    print()

    # ================================================================
    # Stages 1-3: parse + embed + ANN search (skip if --resume)
    # ================================================================
    needed_pids = set()

    if args.resume:
        print("-" * 80)
        print("RESUME MODE: skipping stages 1-3, using existing chunk shards")
        print("-" * 80)
        if not os.path.isdir(shard_dir):
            print(f"[resume] ERROR: shard dir not found: {shard_dir}")
            sys.exit(1)
        n_shards = len([f for f in os.listdir(shard_dir)
                        if f.startswith("chunk_") and f.endswith(".npz")])
        if n_shards == 0:
            print(f"[resume] ERROR: no chunk files in {shard_dir}")
            sys.exit(1)
        print(f"[resume] found {n_shards} chunk shards in {shard_dir}")
        # In resume mode, skip the slow per-shard PID collection.
        # Instead, load ALL items from item.json — the extra ~5M items
        # beyond the 12.3M actually needed cost only ~2.5GB extra memory,
        # but saves ~10 min of NFS-bound shard scanning.
        needed_pids = None  # signal to load everything
    else:
        # ---- Stage 1: parse input TSV ----
        print("-" * 80)
        print("Stage 1/5: parse input TSV")
        print("-" * 80)
        fieldnames, rows, n_skipped = load_input_rows(
            args.input_tsv, args.num_users, n_workers=args.io_workers)
        print(f"[parse] loaded {len(rows):,} rows  "
              f"(unparseable journeys: {n_skipped:,})")
        queries = collect_queries(rows)
        queries = sorted(queries)
        print(f"[parse] unique queries to encode: {len(queries):,}")
        if not queries:
            print("[parse] no queries found, nothing to do.")
            sys.exit(0)

        # FREE rows immediately — they consumed ~25GB.
        # Stage 5 will re-read the input TSV in streaming mode.
        del rows
        gc.collect()
        print(f"[parse] freed rows from memory (gc.collect done)")

        # ---- Stage 2 + 3: chunked embed + ANN search ----
        print("-" * 80)
        print("Stage 2+3/5: chunked query embedding + FAISS ANN search")
        print("-" * 80)
        if not os.path.isfile(index_path):
            print(f"[ann] ERROR: index file not found: {index_path}")
            sys.exit(1)
        if not os.path.isfile(id_map_path):
            print(f"[ann] ERROR: id map file not found: {id_map_path}")
            sys.exit(1)

        tokenizer, session = load_onnx_session(args.gpu_ids)
        index, id_map = load_faiss_index(index_path, id_map_path,
                                         args.ef_search, args.gpu_ids)
        needed_pids = chunked_embed_and_search(
            queries=queries,
            tokenizer=tokenizer,
            session=session,
            index=index,
            id_map=id_map,
            top_k=args.top_k,
            batch_size=args.batch_size,
            max_length=args.max_length,
            chunk_size=args.chunk_size,
            shard_dir=shard_dir,
            search_batch=args.ann_search_batch,
        )
        del index, id_map, session, tokenizer, queries
        gc.collect()

    # ================================================================
    # Stage 4: product metadata
    # ================================================================
    print("-" * 80)
    print("Stage 4/5: load product metadata from item.json")
    print("-" * 80)
    product_meta = load_product_meta_from_json(args.item_json, needed_pids)
    del needed_pids
    gc.collect()

    # ================================================================
    # Stage 5: build JourneyWithProducts — STREAMING from disk
    # ================================================================
    print("-" * 80)
    print("Stage 5/5: build JourneyWithProducts and write output TSV")
    print("-" * 80)

    # Load ANN results: SQLite (default, ~0 RAM) or in-memory dict (~100GB)
    ann_conn = None   # SQLite connection (if using SQLite)
    ann_dict = None   # In-memory dict (if using --ann_in_memory)

    if args.ann_in_memory:
        print("[stage5] Loading ANN results into memory (--ann_in_memory)...")
        ann_dict = load_topk_from_shards(shard_dir, args.top_k)
    else:
        print("[stage5] Loading ANN results into SQLite (memory-efficient)...")
        ann_conn = load_topk_to_sqlite(shard_dir, args.top_k, args.ann_db)

    def ann_lookup(query: str) -> List[Tuple[str, float]]:
        """Unified ANN lookup: uses dict or SQLite transparently."""
        if ann_dict is not None:
            return ann_dict.get(query, [])
        return lookup_ann_sqlite(ann_conn, query)

    def ann_batch(queries: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Batch ANN lookup: dict or SQLite batch."""
        if ann_dict is not None:
            return {q: ann_dict.get(q, []) for q in queries}
        return batch_lookup_ann_sqlite(ann_conn, queries)

    # --- Streaming stage 5: re-read input TSV row by row ---
    # This avoids keeping all 901K rows in memory (~25GB saved).
    KEEP_COLUMNS = ["UserId", "ReadableUserEvents", "ShoppingProfile",
                    "ShoppingJourneys"]
    out_fieldnames = KEEP_COLUMNS + [OUT_COLUMN]

    # Count total rows for progress bar
    try:
        total_bytes = os.path.getsize(args.input_tsv)
    except OSError:
        total_bytes = 0

    n_written = 0
    n_no_journey = 0
    n_skipped_resume = 0
    t_start = time.time()

    def _parse_and_build_one(line_bytes, header):
        """Parse one input TSV line and build JourneyWithProducts."""
        line = line_bytes.decode("utf-8", errors="replace")
        row_list = next(csv.reader([line], delimiter="\t"))
        row = dict(zip(header, row_list))

        # Parse OUTPUT / ShoppingJourneys column (same logic as stage 1)
        _BS_Q = chr(92) + chr(34)
        idx_journey = header.index("OUTPUT") if "OUTPUT" in header else (
            header.index("ShoppingJourneys")
            if "ShoppingJourneys" in header else -1)
        raw = row_list[idx_journey].strip().strip('"') if (
            0 <= idx_journey < len(row_list)) else ""
        parsed = parse_journey_json(raw) if raw else None

        # Rename OUTPUT -> ShoppingJourneys
        if "OUTPUT" in row:
            row.pop("OUTPUT")
        if parsed:
            row["ShoppingJourneys"] = json.dumps(
                parsed, ensure_ascii=False, separators=(',', ':'))
        else:
            row["ShoppingJourneys"] = raw

        # Clean backslash-escaped ShoppingProfile
        for col in ("ShoppingProfile",):
            val = row.get(col, "")
            if val and _BS_Q in val:
                try:
                    _PH = "\x00_PH_\x00"
                    cleaned = val
                    for _ in range(3):
                        if _BS_Q not in cleaned:
                            break
                        cleaned = cleaned.replace(
                            chr(92) + chr(92) + chr(34), _PH)
                        cleaned = cleaned.replace(_BS_Q, chr(34))
                        cleaned = cleaned.replace(_PH, chr(92) + chr(34))
                    obj = json.loads(cleaned)
                    row[col] = json.dumps(
                        obj, ensure_ascii=False, separators=(',', ':'))
                except Exception:
                    pass

        # Build JourneyWithProducts
        if parsed:
            row[OUT_COLUMN] = _build_jwp_with_lookup(
                parsed, ann_lookup, product_meta,
                ann_batch_lookup=ann_batch)
            had_journey = True
        else:
            row[OUT_COLUMN] = ""
            had_journey = False

        return row, had_journey

    # 8 MB write buffer for NFS
    write_mode = "a" if args.resume_offset > 0 else "w"
    with open(args.output_tsv, write_mode, encoding="utf-8",
              buffering=8 << 20, newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames,
                                delimiter="\t",
                                quoting=csv.QUOTE_MINIMAL,
                                extrasaction="ignore")
        if write_mode == "w":
            writer.writeheader()

        with open(args.input_tsv, "rb", buffering=8 << 20) as fin:
            header_line = fin.readline()
            header = next(csv.reader(
                [header_line.decode("utf-8", errors="replace")],
                delimiter="\t"))

            pbar = tqdm(total=total_bytes or None,
                        unit="B", unit_scale=True, unit_divisor=1024,
                        desc="[stage5] stream+write",
                        mininterval=60, smoothing=0.1)
            pbar.update(len(header_line))

            # Bounded thread pool for parallel JSON building
            max_pending = max(2 * args.io_workers, 16)
            pending = deque()
            row_idx = 0

            def _drain_one():
                nonlocal n_written, n_no_journey
                row, had_journey = pending.popleft().result()
                writer.writerow({k: row.get(k, "") for k in out_fieldnames})
                if not had_journey:
                    n_no_journey += 1
                n_written += 1

            with ThreadPoolExecutor(
                    max_workers=max(1, args.io_workers)) as ex:
                for line_bytes in fin:
                    pbar.update(len(line_bytes))
                    if not line_bytes.strip():
                        continue
                    if args.num_users and row_idx >= args.num_users:
                        break
                    row_idx += 1

                    # Skip already-written rows when resuming
                    if row_idx <= args.resume_offset:
                        n_skipped_resume += 1
                        continue

                    pending.append(
                        ex.submit(_parse_and_build_one, line_bytes, header))
                    while len(pending) >= max_pending:
                        _drain_one()

                # Drain remaining
                while pending:
                    _drain_one()

            pbar.close()

    # Close SQLite if used
    if ann_conn is not None:
        ann_conn.close()

    elapsed = time.time() - t_start
    sz_mb = os.path.getsize(args.output_tsv) / (1024 * 1024)
    print(f"\n[write] wrote {n_written:,} rows  "
          f"({n_no_journey:,} without journeys) "
          f"{sz_mb:,.1f} MB -> {args.output_tsv}")
    if n_skipped_resume:
        print(f"[write] skipped {n_skipped_resume:,} rows (resume_offset)")
    print(f"[write] columns: {out_fieldnames}")
    print(f"[write] stage 5 elapsed: {elapsed:.1f}s "
          f"({n_written/max(elapsed,1):.1f} rows/s)")

    # ---- Cleanup ----
    if args.keep_chunks:
        print(f"[cleanup] keeping shard dir (--keep_chunks): {shard_dir}")
    else:
        import shutil
        shutil.rmtree(shard_dir, ignore_errors=True)
        print(f"[cleanup] removed shard dir: {shard_dir}")
    # Remove SQLite DB (it's in /tmp, no need to keep)
    if not args.ann_in_memory and os.path.isfile(args.ann_db):
        os.remove(args.ann_db)
        print(f"[cleanup] removed ANN DB: {args.ann_db}")

    print("=" * 80)
    print("  Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
