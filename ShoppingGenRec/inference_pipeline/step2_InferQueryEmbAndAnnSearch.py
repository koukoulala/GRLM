"""
step2_InferQueryEmbAndannSearch.py
==================================

Pipeline step 2 — for an SLM-journey output TSV:

  1. Parse each row's ``ShoppingJourneys`` (ContinuedJourneys list).
  2. Collect every ``Query`` string across all journeys, dedup.
  3. Embed all queries via the MatadorEmb ONNX model (in-process, GPU).
  4. ANN-search the FAISS index built by step 1 to get the top-K
     ``GlobalOfferId`` for every query.
  5. Attach a ``Products`` list of ``GlobalOfferId`` strings to each ``Query``
      in every journey, producing a JSON in the JourneyRanker training format.
  6. Write a TSV with columns
     ``UserId, ReadableUserSignals, UserProfile, ShoppingJourneys,
     JourneyWithProducts``  (the last one is the new JSON column).

Reference for the JourneyRanker training format:
  ``/scratch/workspaceblobstore/users/wangying/OneRec/Journey/Demo/step1_run_slm_ranker_v3.py``

All paths default to relative locations alongside this script so the
pipeline is self-contained:

  ./run_matador_emb/                                            (tokenizer + model)
  --ann_index   <path>.index                                    (read: FAISS index, explicit input file)
  --ann_id_map  <path>.txt                                      (read: id map for the index)
  --work_dir    ./data/Index_debug/                             (output dir + shard scratch)
  --work_dir/<output_prefix>_journey_with_products.tsv          (write: default output)

Examples
--------
    # Default paths (uses ./data/Index_debug/EnUs_Product_ann_ivf.index)
    python step2_InferQueryEmbAndAnnSearch.py

    # Explicit ANN index file (the recommended way)
    python step2_InferQueryEmbAndAnnSearch.py \\
        --input_tsv     /cosmos/.../only_journey_output_*.tsv \\
        --ann_index     /path/to/EnUs_Product_ann_ivf.index \\
        --ann_id_map    /path/to/EnUs_Product_ann_ids.txt \\
        --work_dir      ./data/Index_2026_05_04 \\
        --output_prefix EnUs_Product \\
        --gpu_ids 0,1,2,3  --top_k 20

    # Quick debug: only 5 users
    python step2_InferQueryEmbAndAnnSearch.py --num_users 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INFERENCE_DIR = os.path.join(SCRIPT_DIR, "run_matador_emb")
TOKENIZER_PATH = os.path.join(
    INFERENCE_DIR,
    "simiaozuo_dense_retrieval_url_data_20250415_checkpoints_model_1_checkpoint-keyword",
)
ONNX_MODEL_PATH = os.path.join(TOKENIZER_PATH, "model_dynamic.onnx")

# Keep only these original input columns in memory and in the final output.
# This is intentionally small: carrying the full input row dict for large TSVs
# can dominate RAM, while Stage 5 only needs these fields plus OUT_COLUMN.
KEEP_COLUMNS = ["UserId", "ReadableUserSignals", "UserProfile", "ShoppingJourneys"]


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
    return None


def collect_queries_from_parsed(parsed) -> Set[str]:
    """Pull Query strings out of one parsed journey object."""
    out: Set[str] = set()
    if not parsed:
        return out
    for journey in parsed.get("ContinuedJourneys") or []:
        if not isinstance(journey, dict):
            continue
        for q in journey.get("Queries") or []:
            if not isinstance(q, dict):
                continue
            qt = q.get("Query")
            if isinstance(qt, str):
                qt = qt.strip()
                if qt:
                    out.add(qt)
    return out


def load_input_rows(input_tsv: str,
                    num_users: int = 0,
                    reader_threads: int = 4):
    """Read input TSV with multiple file descriptors and collect queries.

    For full runs, the file is split into byte ranges and each reader thread
    opens its own fd, seeks to its range, aligns to a line boundary, reads,
    and parses rows independently. Row order is intentionally not preserved.

    To reduce memory pressure, each retained row only contains KEEP_COLUMNS.
    The parsed journey JSON is used immediately to collect the query set and
    then discarded; Stage 5 reparses ``ShoppingJourneys`` while writing.
    ``num_users`` keeps the old debug meaning: first N rows only, single fd.
    """
    try:
        total_bytes = os.path.getsize(input_tsv)
    except OSError:
        total_bytes = 0

    with open(input_tsv, "rb", buffering=8 << 20) as fb:
        header_line = fb.readline()
        if not header_line:
            return [], [], set(), 0
        data_start = fb.tell()
        header_text = header_line.decode("utf-8", errors="replace")
        header = next(csv.reader([header_text], delimiter="\t"))
        fieldnames = list(header)

    idx = {name: i for i, name in enumerate(fieldnames)}
    idx_journey = idx.get("ShoppingJourneys", -1)
    idx_raw = idx.get("RawShoppingJourneys", -1)
    keep_idx = [(name, idx[name]) for name in KEEP_COLUMNS if name in idx]

    rows = []
    rows_lock = Lock()

    if num_users and num_users > 0:
        n_readers = 1
        ranges = [(data_start, total_bytes, num_users)]
        pbar = tqdm(total=num_users, unit="row",
                    desc="[stage1] read+parse",
                    mininterval=0.5, smoothing=0.1)
    else:
        n_readers = max(1, min(int(reader_threads or 1), 64))
        span = max(0, total_bytes - data_start)
        ranges = [
            (data_start + span * i // n_readers,
             data_start + span * (i + 1) // n_readers,
             0)
            for i in range(n_readers)
        ]
        pbar = tqdm(total=total_bytes or None,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f"[stage1] read+parse ({n_readers} fd)",
                    mininterval=0.5, smoothing=0.1)
        pbar.update(data_start)

    def parse_line(line_bytes):
        line = line_bytes.decode("utf-8", errors="replace")
        row_list = next(csv.reader([line], delimiter="\t"))
        row = {
            name: (row_list[col_i] if col_i < len(row_list) else "")
            for name, col_i in keep_idx
        }
        raw = row_list[idx_journey] if 0 <= idx_journey < len(row_list) else ""
        if not raw and 0 <= idx_raw < len(row_list):
            raw = row_list[idx_raw]
        raw = raw.strip().strip('"') if raw else ""
        parsed = parse_journey_json(raw) if raw else None
        return row, parsed

    def flush_rows(batch):
        if not batch:
            return
        with rows_lock:
            rows.extend(batch)

    def read_range(start: int, end: int, limit_rows: int = 0):
        local_rows = []
        local_queries: Set[str] = set()
        local_skipped = 0
        progress_bytes = 0
        progress_rows = 0
        n_rows = 0

        with open(input_tsv, "rb", buffering=8 << 20) as fb:
            fb.seek(start)
            if start != data_start:
                # If the split point lands in the middle of a line, discard
                # that partial line. If it is already on a line boundary,
                # keep the next line.
                fb.seek(start - 1)
                prev = fb.read(1)
                fb.seek(start)
                if prev != b"\n":
                    fb.readline()

            while True:
                line_start = fb.tell()
                if line_start >= end:
                    break
                line_bytes = fb.readline()
                if not line_bytes:
                    break
                row, parsed = parse_line(line_bytes)
                local_rows.append(row)
                n_rows += 1
                if parsed is None:
                    local_skipped += 1
                else:
                    local_queries.update(collect_queries_from_parsed(parsed))

                if len(local_rows) >= 2048:
                    flush_rows(local_rows)
                    local_rows = []

                if limit_rows:
                    progress_rows += 1
                    if progress_rows >= 256:
                        pbar.update(progress_rows)
                        progress_rows = 0
                    if n_rows >= limit_rows:
                        break
                else:
                    progress_bytes += len(line_bytes)
                    if progress_bytes >= (4 << 20):
                        pbar.update(progress_bytes)
                        progress_bytes = 0

        flush_rows(local_rows)
        if limit_rows and progress_rows:
            pbar.update(progress_rows)
        elif not limit_rows and progress_bytes:
            pbar.update(progress_bytes)
        return local_queries, local_skipped

    queries: Set[str] = set()
    skipped = 0
    with ThreadPoolExecutor(max_workers=n_readers) as ex:
        futures = [
            ex.submit(read_range, start, end, limit_rows)
            for start, end, limit_rows in ranges
            if start < end
        ]
        for fut in as_completed(futures):
            part_queries, part_skipped = fut.result()
            queries.update(part_queries)
            skipped += part_skipped

    pbar.set_postfix(rows=len(rows), skipped=skipped, queries=len(queries))
    pbar.close()

    return fieldnames, rows, queries, skipped
def collect_queries(rows) -> Set[str]:
    """Pull every Query string out of every journey of every row."""
    queries: Set[str] = set()
    for item in rows:
        if isinstance(item, tuple) and len(item) == 2:
            _row, parsed = item
        elif isinstance(item, dict):
            raw = (item.get("ShoppingJourneys") or "").strip().strip('"')
            parsed = parse_journey_json(raw) if raw else None
        else:
            parsed = None
        if not parsed:
            continue
        queries.update(collect_queries_from_parsed(parsed))
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
                             search_batch: int = 16384) -> None:
    """Embed + ANN search ``queries`` in chunks of ``chunk_size``.

    For each chunk, writes a shard ``chunk_NNNNN.npz`` into ``shard_dir``
    containing the chunk's queries, top-k product ids, and scores.

    Memory peak per chunk is O(chunk_size * (D + top_k)) — not O(N).
    """
    os.makedirs(shard_dir, exist_ok=True)

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

        # ---- resolve to product ids ----
        pids_2d = np.empty(idxs.shape, dtype=object)
        for i in range(idxs.shape[0]):
            for r in range(idxs.shape[1]):
                ix = idxs[i, r]
                if ix >= 0:
                    pid = id_map[ix]
                    pids_2d[i, r] = pid
                else:
                    pids_2d[i, r] = ""

        shard_path = os.path.join(shard_dir, f"chunk_{ci:05d}.npz")
        np.savez(shard_path,
                 queries=np.array(chunk_q, dtype=object),
                 pids=pids_2d,
                 scores=dists.astype(np.float32))
        print(f"[chunk] {ci + 1}/{n_chunks}  rows={len(chunk_q):,}  "
              f"emb={t_emb:.1f}s  ann={t_ann:.1f}s  "
              f"{os.path.basename(shard_path)}")

        # Free per-chunk buffers immediately.
        del chunk_embs, dists, idxs, pids_2d

    print(f"[chunk] all chunks done in {time.time() - t_total:.1f}s")


def load_topk_from_shards(shard_dir: str,
                          top_k: int) -> Dict[str, List[Tuple[str, float]]]:
    """Read every ``chunk_*.npz`` in ``shard_dir`` and merge into one dict."""
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
# Stage 4 — build JourneyWithProducts JSON per row                             #
# ============================================================================ #
def build_journey_with_products(parsed: Dict,
                                ann_results: Dict[str, List[Tuple[str, float]]]
                                ) -> str:
    """Attach Products list to each Query in each journey of one user.

    Each Product is emitted as the bare ``GlobalOfferId`` string (not a
    metadata dict), matching the downstream ranker's expected schema.
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
                # Match the defensive skip in collect_queries(): some upstream
                # rows emit `Query` as a nested dict / list / null.
                continue
            qt = qt.strip()
            products = [pid for pid, _score in ann_results.get(qt, [])]
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


# ============================================================================ #
# CLI                                                                          #
# ============================================================================ #
# Name of the new column written to the output TSV.
OUT_COLUMN = "JourneyWithProducts"

DEFAULT_INPUT_TSV = (
    "./data/only_journey_output_only_journey_datav3_ckpt960_cut16384.tsv"
)
DEFAULT_WORK_DIR = os.path.join(SCRIPT_DIR, "data", "Index_debug")
DEFAULT_OUTPUT_PREFIX = "EnUs_Product"
DEFAULT_ANN_INDEX = os.path.join(
    DEFAULT_WORK_DIR, f"{DEFAULT_OUTPUT_PREFIX}_ann_ivf.index")
DEFAULT_ANN_ID_MAP = os.path.join(
    DEFAULT_WORK_DIR, f"{DEFAULT_OUTPUT_PREFIX}_ann_ids.txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 2: query embedding + ANN search + JourneyWithProducts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O ---------------------------------------------------------------------
    g_io = p.add_argument_group("I/O")
    g_io.add_argument("--input_tsv", default=DEFAULT_INPUT_TSV,
                      help="SLM journey output TSV (UserId, ReadableUserSignals, "
                           "UserProfile, ShoppingJourneys, ...).")
    g_io.add_argument("--work_dir", default=DEFAULT_WORK_DIR,
                      help="Output directory; step2's output TSV is written "
                           "here as <work_dir>/<output_prefix>_journey_with_products.tsv "
                           "and per-chunk shards under "
                           "<work_dir>/_step2_chunks_<output_prefix>/.")
    g_io.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX,
                      help="Filename prefix for the output TSV / shard dir.")

    # Embedding inference -----------------------------------------------------
    g_inf = p.add_argument_group("Query embedding inference")
    g_inf.add_argument("--gpu_ids", default="0",
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
    g_ann.add_argument("--ann_index", default=DEFAULT_ANN_INDEX,
                       help="Path to the FAISS ANN index file (built by step1).")
    g_ann.add_argument("--ann_id_map", default=DEFAULT_ANN_ID_MAP,
                       help="Path to the id-map txt file paired with --ann_index "
                            "(line N -> GlobalOfferId for vector N).")
    g_ann.add_argument("--top_k", type=int, default=20)
    g_ann.add_argument("--ef_search", type=int, default=128,
                       help="HNSW efSearch / IVF nprobe.")
    g_ann.add_argument("--chunk_size", type=int, default=200_000,
                       help="Process queries in chunks of this size; each "
                            "chunk's (queries, top_k pids, scores) is written "
                            "to a shard file under "
                            "<work_dir>/_step2_chunks_<output_prefix>/.")
    g_ann.add_argument("--ann_search_batch", type=int, default=16384,
                       help="Sub-batch size for the FAISS index.search() call. "
                            "Large nq * nprobe combos can exceed the CUDA "
                            "kernel launch limits on GPU IVF indexes (CUDA "
                            "error 9 'invalid configuration argument'); "
                            "sub-batching avoids that. Set 0 to disable.")
    g_ann.add_argument("--keep_chunks", action="store_true",
                       help="Keep the per-chunk shard files after the output "
                            "TSV is written (default: delete them).")

    # Debug -------------------------------------------------------------------
    g_dbg = p.add_argument_group("Debug")
    g_dbg.add_argument("--num_users", type=int, default=0,
                       help="Limit to first N rows (0 = all).")
    g_dbg.add_argument("--reader_threads", type=int, default=4,
                       help="Number of independent file readers for Stage 1. "
                           "Full runs split the input TSV into byte ranges "
                           "and read them through this many file descriptors. "
                           "Ignored when --num_users is set.")
    g_dbg.add_argument("--io_workers", type=int, default=16,
                       help="Thread-pool size used to parallelize "
                           "JourneyWithProducts building (Stage 5). Set to "
                           "1 to disable Stage 5 parallelism.")

    return p.parse_args()


def default_output_path(work_dir: str, output_prefix: str) -> str:
    return os.path.join(work_dir, f"{output_prefix}_journey_with_products.tsv")


def main() -> None:
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    args.output_tsv = default_output_path(args.work_dir, args.output_prefix)
    os.makedirs(os.path.dirname(args.output_tsv) or ".", exist_ok=True)

    index_path = args.ann_index
    id_map_path = args.ann_id_map

    print("=" * 80)
    print("  step2: query embedding + ANN search -> JourneyWithProducts")
    print("=" * 80)
    print(f"  input_tsv:    {args.input_tsv}")
    print(f"  work_dir:     {args.work_dir}")
    print(f"  output_tsv:   {args.output_tsv}")
    print(f"  num_users:    {args.num_users} (0=all)")
    print(f"  ann_index:    {index_path}")
    print(f"  ann_id_map:   {id_map_path}")
    print(f"  GPUs:         {args.gpu_ids}  (batch={args.batch_size}, max_len={args.max_length})")
    print(f"  ANN:          top_k={args.top_k}, ef_search={args.ef_search}, "
          f"chunk_size={args.chunk_size:,}, keep_chunks={args.keep_chunks}")
    print(f"  I/O:          reader_threads={args.reader_threads}, "
          f"io_workers={args.io_workers}")
    print(f"  out column:   {OUT_COLUMN}")
    print()

    # ---- Stage 1: parse input TSV ----
    print("-" * 80)
    print("Stage 1/4: parse input TSV")
    print("-" * 80)
    fieldnames, rows, queries, n_skipped = load_input_rows(
        args.input_tsv, args.num_users, reader_threads=args.reader_threads)
    print(f"[parse] loaded {len(rows):,} rows  "
          f"(unparseable journeys: {n_skipped:,})")
    queries = sorted(queries)
    print(f"[parse] unique queries to encode: {len(queries):,}")
    if not queries:
        print("[parse] no queries found, nothing to do.")
        sys.exit(0)

    # ---- Stage 2 + 3: chunked embed + ANN search (writes shards) ----
    print("-" * 80)
    print("Stage 2+3/4: chunked query embedding + FAISS ANN search")
    print("-" * 80)
    if not os.path.isfile(index_path):
        print(f"[ann] ERROR: index file not found: {index_path}")
        sys.exit(1)
    if not os.path.isfile(id_map_path):
        print(f"[ann] ERROR: id map file not found: {id_map_path}")
        sys.exit(1)

    shard_dir = os.path.join(args.work_dir,
                             f"_step2_chunks_{args.output_prefix}")
    tokenizer, session = load_onnx_session(args.gpu_ids)
    index, id_map = load_faiss_index(index_path, id_map_path,
                                     args.ef_search, args.gpu_ids)
    chunked_embed_and_search(
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
    del queries
    # Free the heavy resources before the final write stage.
    del index, id_map, session, tokenizer

    # ---- Stage 4: build JourneyWithProducts and write output TSV ----
    print("-" * 80)
    print("Stage 4/4: build JourneyWithProducts and write output TSV")
    print("-" * 80)
    # Reload the chunked top-K results into a dict for the join.
    ann_results = load_topk_from_shards(shard_dir, args.top_k)
    # Keep only the columns the downstream ranker needs, plus the new column.
    out_fieldnames = [c for c in KEEP_COLUMNS if c in fieldnames]
    missing_keep = [c for c in KEEP_COLUMNS if c not in fieldnames]
    if missing_keep:
        print(f"[write] WARNING: input TSV missing expected columns: {missing_keep}")
    if OUT_COLUMN not in out_fieldnames:
        out_fieldnames.append(OUT_COLUMN)

    n_written = 0
    n_no_journey = 0
    # 8 MB write buffer -- big win on /cosmos NFS for the same reason as the
    # input read.
    with open(args.output_tsv, "w", encoding="utf-8",
             buffering=8 << 20, newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames,
                                delimiter="\t",
                                quoting=csv.QUOTE_MINIMAL,
                                extrasaction="ignore")
        writer.writeheader()

        # Build the JourneyWithProducts JSON in a thread pool: that JSON
        # serialization is the dominant CPU cost per row, and ann_results is
        # read-only here so it's safe to share.
        # Input row order may already be shuffled by Stage 1's multi-fd reader;
        # output columns and per-row content remain the same.
        def build_one(row):
            raw = (row.get("ShoppingJourneys") or "").strip().strip('"')
            parsed = parse_journey_json(raw) if raw else None
            out_row = {k: row.get(k, "") for k in out_fieldnames}
            if parsed:
                out_row[OUT_COLUMN] = build_journey_with_products(
                    parsed, ann_results)
                had_journey = True
            else:
                out_row[OUT_COLUMN] = ""
                had_journey = False
            return out_row, had_journey

        # Bounded pipeline: keep at most `max_pending` futures in flight so
        # the writer (NFS-bound) paces the workers and tqdm advances steadily.
        max_pending = max(2 * args.io_workers, 16)
        pending = deque()
        pbar = tqdm(total=len(rows), unit="row",
                    desc="[stage5] build+write",
                    mininterval=0.5, smoothing=0.1)

        def _drain_one():
            row, had_journey = pending.popleft().result()
            writer.writerow(row)
            return had_journey

        with ThreadPoolExecutor(max_workers=max(1, args.io_workers)) as ex:
            for item in rows:
                pending.append(ex.submit(build_one, item))
                while len(pending) >= max_pending:
                    had_journey = _drain_one()
                    if not had_journey:
                        n_no_journey += 1
                    n_written += 1
                    pbar.update(1)
            while pending:
                had_journey = _drain_one()
                if not had_journey:
                    n_no_journey += 1
                n_written += 1
                pbar.update(1)

        pbar.set_postfix(no_journey=n_no_journey)
        pbar.close()
    sz_mb = os.path.getsize(args.output_tsv) / (1024 * 1024)
    print(f"[write] wrote {n_written:,} rows  ({n_no_journey:,} without journeys) "
          f"{sz_mb:,.1f} MB -> {args.output_tsv}")
    print(f"[write] columns: {out_fieldnames}")

    # ---- Cleanup: delete intermediate chunk shards ----
    if args.keep_chunks:
        print(f"[cleanup] keeping shard dir (--keep_chunks): {shard_dir}")
    else:
        import shutil
        shutil.rmtree(shard_dir, ignore_errors=True)
        print(f"[cleanup] removed shard dir: {shard_dir}")

    print("=" * 80)
    print("  Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
