"""step4_InferIndexEmbAndAnnBuild.py
=================================

Pipeline step 4 — read item data, infer item embeddings via MatadorEmb ONNX,
and build a FAISS ANN index for downstream query search.

Supports **two input formats** (auto-detected by file extension):

* **JSON** (``item.json`` from step0): dict keyed by GlobalOfferId.
* **TSV** (``IndexData_*.tsv`` product catalogue): header-based TSV, uses
  ``LLMCatMapping.tsv`` for category-id → name mapping.

Stages
------
1. extract  : read input file (JSON *or* TSV), build a text per product
              (Title, Brand, Seller, Category, Gender, Price), write
              ``<work_dir>/<prefix>_text.tsv`` (id\\ttext) and
              ``<work_dir>/<prefix>_ids.tsv``.
2. inference: run MatadorEmb ONNX inference, write
              ``<work_dir>/<prefix>_text_embeddings.tsv``.
3. merge    : join id+text+embedding, write
              ``<work_dir>/<prefix>_final_embeddings.tsv``.
4. index    : build a FAISS index, write
              ``<work_dir>/<prefix>_ann_<type>.index`` and
              ``<work_dir>/<prefix>_ann_ids.txt``.

Examples
--------
    # JSON input (item.json from step0)
    python step4_InferIndexEmbAndAnnBuild.py \\
        --input_file ./raw_data/item.json \\
        --work_dir   ./data/Index_2026_05_04 \\
        --output_prefix EnUs_Product \\
        --gpu_ids 0,1,2,3 \\
        --index_type hnsw

    # TSV input (product catalogue, same as step1)
    python step4_InferIndexEmbAndAnnBuild.py \\
        --input_file /cosmos/.../IndexData_en_us.tsv \\
        --input_format tsv \\
        --cat_mapping ./res/LLMCatMapping.tsv \\
        --work_dir   ./data/Index_2026_05_04 \\
        --output_prefix EnUs_Product \\
        --gpu_ids 0,1,2,3 \\
        --index_type hnsw

    # only rebuild ANN index from existing final embeddings
    python step4_InferIndexEmbAndAnnBuild.py \\
        --work_dir     ./data/Index_2026_05_04 \\
        --output_prefix EnUs_Product \\
        --only_index --index_type hnsw

    # resume from a previous superset run
    python step4_InferIndexEmbAndAnnBuild.py \\
        --input_file ./raw_data/item.json \\
        --resume_emb_dir .../raw_data/MatadorEmb_Index \\
        --work_dir ./data/Index_new \\
        --output_prefix Items_full

    # debug
    python step4_InferIndexEmbAndAnnBuild.py ... --debug --debug_rows 100 --gpu_ids 0
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import faiss
import numpy as np
from tqdm import tqdm


# ============================================================================ #
# Constants — all paths relative to this script's directory                    #
# ============================================================================ #
SCRIPT_DIR = "/vc_data/users/wangying/OneRec/ShoppingJourney/Pipeline/run_matador_emb"

# Inference assets (under ../run_matador_emb relative to this script's parent)
_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INFERENCE_DIR = os.path.join(_PROJECT_DIR, "run_matador_emb")
INFERENCE_SCRIPT = os.path.join(INFERENCE_DIR, "inference_onnx_distributed_entry.py")
TOKENIZER_PATH = os.path.join(
    INFERENCE_DIR,
    "simiaozuo_dense_retrieval_url_data_20250415_checkpoints_model_1_checkpoint-keyword",
)
MODEL_PATH = os.path.join(TOKENIZER_PATH, "model_dynamic.onnx")

# Category id -> name mapping (used only for TSV input)
# LLMCatMapping.tsv lives under Pipeline/res/, not Pipeline/run_matador_emb/res/
LLM_CAT_MAPPING_PATH = os.path.join(_PROJECT_DIR, "res", "LLMCatMapping.tsv")


# ============================================================================ #
# Helpers                                                                      #
# ============================================================================ #
def load_llm_cat_mapping(path: str) -> Dict[str, str]:
    """Load id -> CategoryName from a 2-col TSV (`<name>\\t<id>`)."""
    mapping: Dict[str, str] = {}
    if not os.path.isfile(path):
        print(f"[cat] WARNING: mapping file not found: {path}")
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            name, cat_id = parts[0].strip(), parts[1].strip()
            if cat_id and name:
                mapping[cat_id] = name
    print(f"[cat] loaded {len(mapping):,} category id->name entries")
    return mapping


def _clean(text: str) -> str:
    """Strip + collapse whitespace + remove disruptive characters."""
    if not text:
        return ""
    text = text.strip().replace(",", " ").replace("|", " - ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def build_item_text(item: Dict) -> str:
    """Build embedding text from an item.json entry.

    Concatenates Title, Brand, Seller, Category, Gender, Price with ", "
    (same format as the original TSV-based build_product_text).
    Only non-empty fields are included.
    """
    title = _clean(item.get("title", ""))
    categories = _clean(item.get("categories", ""))
    attrs = item.get("attributes", {})

    brand = _clean(str(attrs.get("Brand", "")))
    if brand.lower() == "other":
        brand = ""
    seller = _clean(str(attrs.get("Seller", "")))
    gender = _clean(str(attrs.get("Gender", "")))
    price = _clean(str(attrs.get("Price", "")))

    segments = [s for s in (title, brand, seller, categories, gender, price) if s]
    return ", ".join(segments)


def build_product_text(parts: List[str],
                      cat_mapping: Dict[str, str],
                      col_title: int,
                      col_seller: int,
                      col_gender: int,
                      col_price: int,
                      col_brand: int,
                      col_llm_cat_id: int) -> str:
    """Concatenate Title, Brand, Seller, Category, Gender, Price (TSV input)."""
    def get(idx: int) -> str:
        if 0 <= idx < len(parts):
            return _clean(parts[idx])
        return ""

    title = get(col_title)
    seller = get(col_seller)
    gender = get(col_gender)
    price = get(col_price)
    brand = get(col_brand)
    if brand.lower() == "other":
        brand = ""
    cat_id = parts[col_llm_cat_id].strip() if 0 <= col_llm_cat_id < len(parts) else ""
    category = _clean(cat_mapping.get(cat_id, ""))

    segments = [s for s in (title, brand, seller, category, gender, price) if s]
    return ", ".join(segments)


# ============================================================================ #
# Stage 1 — extract product text (JSON or TSV)                                 #
# ============================================================================ #
def extract_product_text_from_json(item_json_file: str,
                                   text_file: str,
                                   id_file: str,
                                   max_rows: int = 0) -> int:
    """Read item.json (from step0); write `id\\ttext` and `id`.

    item.json is keyed by GlobalOfferId, each value has:
      {"title": str, "description": str, "categories": str,
       "attributes": {"Brand": ..., "Seller": ..., "Gender": ...,
                      "Price": ..., ...}}

    Returns the number of valid items written.
    """
    import json as _json

    file_size = os.path.getsize(item_json_file)
    print(f"[extract] reading {item_json_file} ({file_size / (1024**3):.2f} GB)"
          + (f"  [DEBUG: limit={max_rows:,}]" if max_rows > 0 else ""))

    # Load JSON
    t0 = time.time()
    try:
        import orjson
        with open(item_json_file, "rb") as f:
            items = orjson.loads(f.read())
    except ImportError:
        with open(item_json_file, "r", encoding="utf-8") as f:
            items = _json.load(f)
    print(f"[extract] loaded {len(items):,} items in {time.time() - t0:.1f}s")

    valid = 0
    with open(text_file, "w", encoding="utf-8") as f_text, \
         open(id_file, "w", encoding="utf-8") as f_id:
        for goid, item in tqdm(items.items(), desc="extract", mininterval=10):
            if 0 < max_rows <= valid:
                break
            text = build_item_text(item)
            if not text:
                continue
            f_text.write(f"{goid}\t{text}\n")
            f_id.write(goid + "\n")
            valid += 1

    print(f"[extract] {len(items):,} items -> {valid:,} valid products")
    return valid


def extract_product_text_from_tsv(product_file: str,
                                  text_file: str,
                                  id_file: str,
                                  cat_mapping: Dict[str, str],
                                  col_id: str,
                                  col_title: str,
                                  col_seller: str,
                                  col_gender: str,
                                  col_price: str,
                                  col_brand: str,
                                  col_llm_cat_id: str,
                                  max_rows: int = 0) -> int:
    """Read product TSV (with header); write `id\\ttext` and `id`.

    Column arguments are header names resolved to 0-based indices.
    Returns the number of valid items written.
    """
    print(f"[extract] reading TSV: {product_file}")
    if max_rows > 0:
        print(f"[extract] DEBUG: limiting to first {max_rows:,} rows")

    total = 0
    valid = 0
    with open(product_file, "r", encoding="utf-8") as fin, \
         open(text_file, "w", encoding="utf-8") as f_text, \
         open(id_file, "w", encoding="utf-8") as f_id:

        header_line = fin.readline()
        if not header_line:
            print("[extract] empty file")
            return 0
        header_cols = header_line.rstrip("\n").split("\t")
        name_to_idx = {name: i for i, name in enumerate(header_cols)}
        wanted = {
            "id":        col_id,
            "title":     col_title,
            "seller":    col_seller,
            "gender":    col_gender,
            "price":     col_price,
            "brand":     col_brand,
            "llm_cat":   col_llm_cat_id,
        }
        missing = [f"--col_{k} ({v!r})" for k, v in wanted.items()
                   if v not in name_to_idx]
        if missing:
            raise ValueError(
                f"Header is missing required columns: {missing}\n"
                f"Available header: {header_cols}"
            )
        idx = {k: name_to_idx[v] for k, v in wanted.items()}
        print(f"[extract] resolved column indices: {idx}")

        for line in tqdm(fin, desc="extract", mininterval=10):
            total += 1
            if 0 < max_rows < total:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= idx["id"]:
                continue
            goid = parts[idx["id"]].strip()
            if not goid:
                continue
            text = build_product_text(parts, cat_mapping,
                                      col_title=idx["title"],
                                      col_seller=idx["seller"],
                                      col_gender=idx["gender"],
                                      col_price=idx["price"],
                                      col_brand=idx["brand"],
                                      col_llm_cat_id=idx["llm_cat"])
            if not text:
                continue
            f_text.write(f"{goid}\t{text}\n")
            f_id.write(goid + "\n")
            valid += 1

    print(f"[extract] {total:,} rows scanned -> {valid:,} valid products")
    return valid


# ============================================================================ #
# Stage 2 — run MatadorEmb ONNX inference                                      #
# ============================================================================ #
def run_inference(text_file: str,
                  emb_output: str,
                  temp_folder: str,
                  gpu_ids: str,
                  num_sessions_per_gpu: int,
                  max_length: int,
                  batch_size: int) -> int:
    """Spawn the distributed ONNX inference entrypoint as a subprocess."""
    gpu_id_list = [g.strip() for g in gpu_ids.split(",") if g.strip()]
    num_gpus = len(gpu_id_list)
    gpu_ids_str = ",".join(gpu_id_list)

    # Subprocess uses cwd=INFERENCE_DIR, so all data paths must be absolute.
    text_file_abs = os.path.abspath(text_file)
    emb_output_abs = os.path.abspath(emb_output)
    temp_folder_abs = os.path.abspath(temp_folder)
    os.makedirs(temp_folder_abs, exist_ok=True)
    os.makedirs(os.path.dirname(emb_output_abs) or ".", exist_ok=True)

    print(f"[infer] script:        {INFERENCE_SCRIPT}")
    print(f"[infer] tokenizer:     {TOKENIZER_PATH}")
    print(f"[infer] input  text:   {text_file_abs}")
    print(f"[infer] output emb:    {emb_output_abs}")
    print(f"[infer] temp folder:   {temp_folder_abs}")
    print(f"[infer] GPUs: {gpu_ids_str} ({num_gpus} GPUs, "
          f"{num_sessions_per_gpu} sess/GPU)")
    print(f"[infer] max_length={max_length} batch_size={batch_size}")

    cmd = [
        sys.executable,
        INFERENCE_SCRIPT,
        "--tokenizer_path", TOKENIZER_PATH,
        "--model_path", MODEL_PATH,
        "--data_file", text_file_abs,
        "--output_file", emb_output_abs,
        "--temp_folder", temp_folder_abs,
        "--inference_type", "keyword",
        "--compute_file", "v0",
        "--normalize_and_quantize", "0",
        "--num_gpus", str(num_gpus),
        "--num_sessions_per_gpu", str(num_sessions_per_gpu),
        "--max_length", str(max_length),
        "--max_length_entity", "-1",
        "--batch_size", str(batch_size),
        "--include_id_num", "1",
    ]
    print(f"[infer] command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    rc = subprocess.run(cmd, cwd=INFERENCE_DIR, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"Embedding inference failed with return code {rc}")

    line_count = sum(1 for _ in open(emb_output_abs, "r", encoding="utf-8"))
    print(f"[infer] embedding output: {line_count:,} lines")

    # Clean up the per-shard scratch directory; the merged emb_output is what
    # downstream stages need.
    if os.path.isdir(temp_folder_abs):
        import shutil
        try:
            shutil.rmtree(temp_folder_abs)
            print(f"[infer] cleaned temp folder: {temp_folder_abs}")
        except OSError as e:
            print(f"[infer] WARNING: failed to remove {temp_folder_abs}: {e}")

    return line_count


# ============================================================================ #
# Stage 3 — merge id + text + embedding                                        #
# ============================================================================ #
def merge_results(text_file: str, emb_output: str, final_file: str) -> int:
    """Combine `id\\ttext` and `id\\tembedding` into `id\\ttext\\tembedding`."""
    print(f"[merge] text file: {text_file}")
    print(f"[merge] emb  file: {emb_output}")
    print(f"[merge] final out: {final_file}")

    id_to_text = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                id_to_text[parts[0]] = parts[1]
    print(f"[merge] loaded {len(id_to_text):,} text entries")

    emb_count = sum(1 for _ in open(emb_output, "r", encoding="utf-8"))
    written = 0
    with open(emb_output, "r", encoding="utf-8") as f_emb, \
         open(final_file, "w", encoding="utf-8") as f_out:
        f_out.write("GlobalOfferId\tText\tEmbedding\n")
        for line in tqdm(f_emb, desc="merge", total=emb_count, mininterval=10):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            goid, embedding = parts[0], parts[1]
            f_out.write(f"{goid}\t{id_to_text.get(goid, '')}\t{embedding}\n")
            written += 1
    sz_mb = os.path.getsize(final_file) / (1024 * 1024)
    print(f"[merge] wrote {written:,} lines ({sz_mb:,.1f} MB) -> {final_file}")
    return written


# ============================================================================ #
# Resume — filter existing embeddings to a subset                               #
# ============================================================================ #
def filter_final_embeddings(src_emb_file: str,
                            dst_emb_file: str,
                            id_file: str) -> Tuple[int, set]:
    """Filter an existing final_embeddings.tsv to only IDs in ``id_file``.

    Used by ``--resume_emb_dir``: reuse embeddings from a previous (superset)
    run instead of re-running the expensive ONNX inference.

    Returns (n_kept, missing_ids) where ``missing_ids`` is the set of IDs
    that were requested but not found in the source embeddings.
    """
    print(f"[resume] loading target IDs from {id_file}")
    with open(id_file, "r", encoding="utf-8") as f:
        target_ids = set(line.strip() for line in f if line.strip())
    print(f"[resume] target IDs: {len(target_ids):,}")

    print(f"[resume] filtering {src_emb_file} -> {dst_emb_file}")
    t0 = time.time()
    found_ids = set()
    kept, total = 0, 0
    with open(src_emb_file, "r", encoding="utf-8") as fin, \
         open(dst_emb_file, "w", encoding="utf-8") as fout:
        header = fin.readline()
        fout.write(header)
        for line in tqdm(fin, desc="resume-filter", mininterval=10):
            total += 1
            goid = line.split("\t", 1)[0]
            if goid in target_ids:
                fout.write(line)
                found_ids.add(goid)
                kept += 1
    elapsed = time.time() - t0
    missing_ids = target_ids - found_ids
    print(f"[resume] kept {kept:,}/{total:,} embeddings "
          f"(target={len(target_ids):,}) in {elapsed:.1f}s")
    if missing_ids:
        print(f"[resume] {len(missing_ids):,} target IDs "
              f"not found in source embeddings — will infer these")
    else:
        print(f"[resume] all target IDs found, no extra inference needed")
    return kept, missing_ids


def write_missing_text_file(src_text_file: str,
                            dst_text_file: str,
                            missing_ids: set) -> int:
    """Write a text.tsv containing only the missing IDs for inference."""
    written = 0
    with open(src_text_file, "r", encoding="utf-8") as fin, \
         open(dst_text_file, "w", encoding="utf-8") as fout:
        for line in fin:
            goid = line.split("\t", 1)[0]
            if goid in missing_ids:
                fout.write(line)
                written += 1
    print(f"[resume] wrote {written:,} missing items to {dst_text_file}")
    return written


def append_to_final_embeddings(main_emb_file: str,
                               extra_emb_file: str) -> int:
    """Append lines from extra_emb_file to main_emb_file (skip header)."""
    appended = 0
    with open(extra_emb_file, "r", encoding="utf-8") as fin, \
         open(main_emb_file, "a", encoding="utf-8") as fout:
        fin.readline()  # skip header
        for line in fin:
            fout.write(line)
            appended += 1
    print(f"[resume] appended {appended:,} extra embeddings to {main_emb_file}")
    return appended


# ============================================================================ #
# Stage 4 — build FAISS ANN index                                              #
# ============================================================================ #
def load_final_embeddings(final_file: str,
                          emb_dim: int) -> Tuple[List[str], np.ndarray]:
    """Read the 3-column merged TSV (with header) and return (ids, matrix)."""
    print(f"[index] loading embeddings from {final_file}")
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    skipped = 0
    t0 = time.time()
    with open(final_file, "r", encoding="utf-8") as f:
        header = f.readline()
        print(f"[index] header: {header.strip()}")
        for line in tqdm(f, desc="index-load", mininterval=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                skipped += 1
                continue
            try:
                emb = np.fromstring(parts[2], dtype=np.float32, sep=" ")
            except ValueError:
                skipped += 1
                continue
            if emb.shape[0] != emb_dim:
                skipped += 1
                continue
            ids.append(parts[0])
            vecs.append(emb)
    matrix = np.vstack(vecs).astype(np.float32)
    print(f"[index] loaded {len(ids):,} vectors ({matrix.shape}) "
          f"skipped={skipped:,} elapsed={time.time()-t0:.1f}s")
    return ids, matrix


def build_ann_index(embeddings: np.ndarray,
                    index_type: str,
                    nlist: int,
                    hnsw_m: int,
                    hnsw_ef_construction: int,
                    use_all_gpus: bool) -> faiss.Index:
    """Build FAISS ANN index. Returns a CPU index ready to save."""
    n, d = embeddings.shape
    faiss.normalize_L2(embeddings)  # cosine == inner product after norm
    t0 = time.time()

    if index_type == "flat":
        cpu_index = faiss.IndexFlatIP(d)
        gpu_index = (faiss.index_cpu_to_all_gpus(cpu_index) if use_all_gpus
                     else faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index))
        gpu_index.add(embeddings)
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)

    elif index_type == "ivf":
        nlist = max(1, min(nlist, n // 40))
        quantizer = faiss.IndexFlatIP(d)
        cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        gpu_index = (faiss.index_cpu_to_all_gpus(cpu_index) if use_all_gpus
                     else faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index))
        print(f"[index] training IVF (nlist={nlist}) ...")
        gpu_index.train(embeddings)
        print(f"[index] adding vectors ...")
        gpu_index.add(embeddings)
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)

    elif index_type == "hnsw":
        # HNSW is CPU-only in FAISS
        print(f"[index] building HNSW (M={hnsw_m}, "
              f"efConstruction={hnsw_ef_construction}) on CPU ...")
        cpu_index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        cpu_index.hnsw.efConstruction = hnsw_ef_construction
        cpu_index.add(embeddings)

    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    print(f"[index] built in {time.time()-t0:.1f}s -> ntotal={cpu_index.ntotal:,}")
    return cpu_index


def save_id_mapping(ids: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for gid in ids:
            f.write(gid + "\n")
    print(f"[index] id map saved -> {path} ({len(ids):,} ids)")


# ============================================================================ #
# Stage 5 — compute top-k similarities                                         #
# ============================================================================ #
def compute_similarities(index: faiss.Index,
                        ids: List[str],
                        embeddings: np.ndarray,
                        top_k: int,
                        output_path: str) -> None:
    """Search the FAISS index for top-k neighbors per item and write
    similarities.json in the same format as step1_init_item_emb.py:

        {"item_id_1": [{"item_id": "...", "similarity": 0.95}, ...], ...}

    Self-matches are excluded.
    """
    n = len(ids)
    search_k = min(top_k + 1, n)  # +1 to exclude self
    print(f"[sim] searching top-{search_k} neighbors for {n:,} items ...")
    t0 = time.time()

    # Batch search for memory efficiency
    batch_size = 100000
    all_D, all_I = [], []
    for start in tqdm(range(0, n, batch_size), desc="sim-search", mininterval=5):
        end = min(start + batch_size, n)
        D_batch, I_batch = index.search(embeddings[start:end], search_k)
        all_D.append(D_batch)
        all_I.append(I_batch)
    D = np.vstack(all_D)
    I = np.vstack(all_I)

    # Build results dict, excluding self-matches
    results = {}
    for qi in range(n):
        similar_items = []
        for j in range(search_k):
            idx = int(I[qi, j])
            if idx != qi and idx >= 0:
                similar_items.append({
                    "item_id": ids[idx],
                    "similarity": round(float(D[qi, j]), 6),
                })
            if len(similar_items) >= top_k:
                break
        results[ids[qi]] = similar_items

    # Write JSON
    print(f"[sim] writing {len(results):,} entries to {output_path} ...")
    try:
        import orjson
        raw = orjson.dumps(results)
    except ImportError:
        import json as _json
        raw = _json.dumps(results, ensure_ascii=False).encode("utf-8")
    with open(output_path, "wb") as f:
        f.write(raw)
    sz_mb = len(raw) / (1024 * 1024)
    print(f"[sim] done in {time.time()-t0:.1f}s ({sz_mb:,.1f} MB)")


# ============================================================================ #
# CLI                                                                          #
# ============================================================================ #
DEFAULT_WORK_DIR = os.path.join(SCRIPT_DIR, "data", "Index_debug")


def _detect_input_format(path: str) -> str:
    """Auto-detect input format from file extension."""
    lower = path.lower()
    if lower.endswith(".json"):
        return "json"
    elif lower.endswith(".tsv") or lower.endswith(".csv") or lower.endswith(".txt"):
        return "tsv"
    return "json"  # fallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 4: infer item embeddings (JSON or TSV) + build FAISS ANN index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O ---------------------------------------------------------------------
    g_io = p.add_argument_group("I/O")
    g_io.add_argument("--input_file",
                      #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/item.json",
                      default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_PG/item.json",
                      #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/item.json",
                      #default="/cosmos/projects/Recommendations/Pipelines/QualityIndexAllMarketUnify/dev_Updated/2026/05/22/IndexData_en_us_PerSellerCapped.tsv",
                      #default="/cosmos/projects/Recommendations/Pipelines/QualityIndexAllMarketUnify/dev_Updated/2026/05/26/IndexData_en_us_all.tsv",
                      help="Input file: item.json (from step0) OR product TSV "
                           "(from IndexData pipeline). Format auto-detected "
                           "from extension, or set --input_format explicitly.")
    g_io.add_argument("--input_format", choices=["json", "tsv", "auto"],
                      default="auto",
                      help="Input format. 'auto' detects from file extension "
                           "(.json -> json, .tsv/.csv/.txt -> tsv).")
    g_io.add_argument("--work_dir", 
                      #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/MatadorEmb_Index",
                      #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/MatadorEmb_Index",
                      #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260522/raw_data/MatadorEmb_Index",
                      #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/MatadorEmb_Index",
                      default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_PG/MatadorEmb_Index",
                      help="Directory for all intermediate + final artefacts.")
    g_io.add_argument("--output_prefix", default="Items_full",
                      help="Filename prefix used for all output files.")

    # TSV-specific columns (only used when input_format=tsv) ------------------
    g_col = p.add_argument_group("TSV columns (only used when --input_format=tsv)")
    g_col.add_argument("--cat_mapping", default=LLM_CAT_MAPPING_PATH,
                       help="LLMCatMapping.tsv: '<CategoryName>\\t<LLMCatId>'.")
    g_col.add_argument("--col_id", default="GlobalOfferId")
    g_col.add_argument("--col_title", default="Title")
    g_col.add_argument("--col_seller", default="Seller")
    g_col.add_argument("--col_gender", default="Gender")
    g_col.add_argument("--col_price", default="OriginalPrice")
    g_col.add_argument("--col_brand", default="Brand")
    g_col.add_argument("--col_llm_cat_id", default="LLMCatId",
                       help="Column whose value is mapped via cat_mapping.")

    # Embedding inference -----------------------------------------------------
    g_inf = p.add_argument_group("Embedding inference")
    g_inf.add_argument("--gpu_ids", default="0,1,2,3",
                       help="Comma-separated GPU ids, e.g. '0,1,2,3'.")
    g_inf.add_argument("--num_sessions_per_gpu", type=int, default=1)
    g_inf.add_argument("--max_length", type=int, default=512)
    g_inf.add_argument("--batch_size", type=int, default=256)

    # ANN index ---------------------------------------------------------------
    g_ann = p.add_argument_group("ANN index")
    g_ann.add_argument("--index_type", choices=["flat", "ivf", "hnsw"],
                       default="hnsw")
    g_ann.add_argument("--emb_dim", type=int, default=64,
                       help="Expected embedding dimensionality.")
    g_ann.add_argument("--nlist", type=int, default=4096,
                       help="IVF nlist (only used for --index_type=ivf).")
    g_ann.add_argument("--hnsw_m", type=int, default=32,
                       help="HNSW M (only used for --index_type=hnsw).")
    g_ann.add_argument("--hnsw_ef_construction", type=int, default=200,
                       help="HNSW efConstruction.")
    g_ann.add_argument("--use_all_gpus", action="store_true", default=True,
                       help="Spread Flat / IVF training across all visible GPUs.")
    g_ann.add_argument("--single_gpu", dest="use_all_gpus", action="store_false",
                       help="Use only the first visible GPU for index building.")

    # Stage selection ---------------------------------------------------------
    g_st = p.add_argument_group("Stage selection")
    g_st.add_argument("--skip_extract", action="store_true",
                      help="Skip stage 1 (text extraction); reuse existing files.")
    g_st.add_argument("--skip_inference", action="store_true",
                      help="Skip stage 2 (embedding inference).")
    g_st.add_argument("--skip_merge", action="store_true",
                      help="Skip stage 3 (id+text+emb merge).")
    g_st.add_argument("--skip_index", action="store_true",
                      help="Skip stage 4 (ANN index build).")
    g_st.add_argument("--only_index", action="store_true",
                      help="Shortcut for --skip_extract --skip_inference --skip_merge.")
    g_st.add_argument("--resume_emb_dir", nargs="*", 
                      default=["/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/MatadorEmb_Index/",
                               "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_PG/MatadorEmb_Index/"],
                      help="One or more paths to existing MatadorEmb_Index dirs "
                           "(from previous runs). If set, Stage 1 (extract) "
                           "runs to get the new IDs, then existing "
                           "final_embeddings.tsv files are scanned to reuse "
                           "embeddings, skipping the expensive Stage 2 "
                           "(inference) and Stage 3 (merge). Multiple dirs "
                           "are scanned in order; later dirs fill in IDs "
                           "not found in earlier dirs. "
                           "Example: --resume_emb_dir dir1 dir2 dir3")

    # Debug -------------------------------------------------------------------
    g_dbg = p.add_argument_group("Debug")
    g_dbg.add_argument("--debug", action="store_true",
                       help="Limit text extraction to --debug_rows lines.")
    g_dbg.add_argument("--debug_rows", type=int, default=100)

    # Similarity computation --------------------------------------------------
    g_sim = p.add_argument_group("Similarity computation (Stage 5)")
    g_sim.add_argument("--skip_similarity", action="store_true",
                       help="Skip stage 5 (similarity computation).")
    g_sim.add_argument("--sim_top_k", type=int, default=8,
                       help="Number of top similar items per item.")
    g_sim.add_argument("--sim_output", default="",
                       help="Output path for similarities.json. "
                            "Default: <work_dir>/similarities.json.")

    return p.parse_args()


def resolve_paths(work_dir: str, prefix: str, index_type: str) -> Dict[str, str]:
    os.makedirs(work_dir, exist_ok=True)
    return {
        "text_file":     os.path.join(work_dir, f"{prefix}_text.tsv"),
        "id_file":       os.path.join(work_dir, f"{prefix}_ids.tsv"),
        "emb_inference": os.path.join(work_dir, f"{prefix}_text_embeddings.tsv"),
        "final_emb":     os.path.join(work_dir, f"{prefix}_final_embeddings.tsv"),
        "temp_folder":   os.path.join(work_dir, "temp_product_emb"),
        "index_file":    os.path.join(work_dir, f"{prefix}_ann_{index_type}.index"),
        "id_map_file":   os.path.join(work_dir, f"{prefix}_ann_ids.txt"),
        "sim_file":      os.path.join(work_dir, "similarities.json"),
    }


def main() -> None:
    args = parse_args()

    # Resolve input format
    if args.input_format == "auto":
        args.input_format = _detect_input_format(args.input_file)
    print(f"[init] input format: {args.input_format}")

    if args.only_index:
        args.skip_extract = True
        args.skip_inference = True
        args.skip_merge = True

    # --resume_emb_dir implies: run extract (Stage 1) but skip inference
    # and merge (Stage 2+3); we'll filter existing embeddings instead.
    resume_emb_dirs = [d for d in (args.resume_emb_dir or []) if d]
    if resume_emb_dirs:
        # Validate that at least one source embedding file exists
        valid_resume_dirs = []
        for d in resume_emb_dirs:
            src = os.path.join(d, f"{args.output_prefix}_final_embeddings.tsv")
            if os.path.isfile(src):
                valid_resume_dirs.append(d)
            else:
                print(f"WARNING: resume dir missing embeddings, skipping: {src}")
        if not valid_resume_dirs:
            print(f"ERROR: --resume_emb_dir given but no valid source "
                  f"embeddings found in any dir")
            sys.exit(1)
        resume_emb_dirs = valid_resume_dirs
        args.skip_inference = True
        args.skip_merge = True

    # Pin the visible GPUs for THIS process too (the inference subprocess sets
    # its own CUDA_VISIBLE_DEVICES later, but FAISS GPU calls during stage 4
    # otherwise see every device on the host -> easily OOMs on busy GPUs).
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    paths = resolve_paths(args.work_dir, args.output_prefix, args.index_type)

    print("=" * 80)
    print("  step4: Infer item embeddings + build ANN index")
    print("=" * 80)
    print(f"  input_file:     {args.input_file}")
    print(f"  input_format:   {args.input_format}")
    print(f"  work_dir:       {args.work_dir}")
    print(f"  output_prefix:  {args.output_prefix}")
    print(f"  inference_dir:  {INFERENCE_DIR}")
    print(f"  index_type:     {args.index_type}  (emb_dim={args.emb_dim})")
    print(f"  GPUs:           {args.gpu_ids}")
    print(f"  debug:          {args.debug} (rows={args.debug_rows})")
    if args.input_format == "tsv":
        print(f"  cat_mapping:    {args.cat_mapping}")
    print()

    # ---- Stage 1: extract product text ----
    if not args.skip_extract:
        print("-" * 80)
        print(f"Stage 1/4: extract product text from {args.input_format.upper()} input")
        print("-" * 80)
        max_rows = args.debug_rows if args.debug else 0

        if args.input_format == "json":
            n = extract_product_text_from_json(
                item_json_file=args.input_file,
                text_file=paths["text_file"],
                id_file=paths["id_file"],
                max_rows=max_rows,
            )
        else:  # tsv
            cat_mapping = load_llm_cat_mapping(args.cat_mapping)
            n = extract_product_text_from_tsv(
                product_file=args.input_file,
                text_file=paths["text_file"],
                id_file=paths["id_file"],
                cat_mapping=cat_mapping,
                col_id=args.col_id,
                col_title=args.col_title,
                col_seller=args.col_seller,
                col_gender=args.col_gender,
                col_price=args.col_price,
                col_brand=args.col_brand,
                col_llm_cat_id=args.col_llm_cat_id,
                max_rows=max_rows,
            )
        if n == 0:
            print("[extract] no valid products found; aborting.")
            sys.exit(1)
    else:
        print("[stage 1] skipped (using existing text/id files)")

    # ---- Resume: filter existing embeddings if --resume_emb_dir ----
    if resume_emb_dirs:
        print("-" * 80)
        print(f"Resume: filter existing embeddings from {len(resume_emb_dirs)} dirs")
        print("-" * 80)

        # Load target IDs
        with open(paths["id_file"], "r", encoding="utf-8") as f:
            target_ids = set(line.strip() for line in f if line.strip())
        print(f"[resume] target IDs: {len(target_ids):,}")

        # Scan each resume dir in order, collecting found embeddings
        total_kept = 0
        remaining_ids = set(target_ids)
        first_dir = True
        for ri, resume_dir in enumerate(resume_emb_dirs):
            if not remaining_ids:
                break
            src_final_emb = os.path.join(
                resume_dir, f"{args.output_prefix}_final_embeddings.tsv")
            print(f"\n[resume] [{ri+1}/{len(resume_emb_dirs)}] "
                  f"scanning {src_final_emb} ...")

            t0 = time.time()
            found_ids = set()
            kept = 0
            with open(src_final_emb, "r", encoding="utf-8") as fin:
                header = fin.readline()
                if first_dir:
                    # First dir: create output file with header
                    with open(paths["final_emb"], "w", encoding="utf-8") as fout:
                        fout.write(header)
                    first_dir = False
                with open(paths["final_emb"], "a", encoding="utf-8") as fout:
                    for line in tqdm(fin, desc=f"resume-{ri+1}", mininterval=10):
                        goid = line.split("\t", 1)[0]
                        if goid in remaining_ids:
                            fout.write(line)
                            found_ids.add(goid)
                            kept += 1
            remaining_ids -= found_ids
            total_kept += kept
            print(f"[resume] [{ri+1}] kept {kept:,} embeddings "
                  f"(remaining: {len(remaining_ids):,}) "
                  f"in {time.time()-t0:.1f}s")

        missing_ids = remaining_ids
        print(f"\n[resume] total kept: {total_kept:,}/{len(target_ids):,}, "
              f"missing: {len(missing_ids):,}")
        del target_ids, remaining_ids
        gc.collect()

        if total_kept == 0 and not missing_ids:
            print("[resume] no embeddings matched and no IDs found; aborting.")
            sys.exit(1)

        # If there are missing IDs, run inference + merge for just those
        if missing_ids:
            print(f"\n[resume] running inference for {len(missing_ids):,} "
                  f"missing items...")
            missing_text = os.path.join(args.work_dir, "_missing_text.tsv")
            missing_emb = os.path.join(args.work_dir, "_missing_emb.tsv")
            missing_final = os.path.join(args.work_dir, "_missing_final_emb.tsv")
            missing_temp = os.path.join(args.work_dir, "_missing_temp")

            # Write text file for missing items only
            write_missing_text_file(paths["text_file"], missing_text,
                                    missing_ids)

            # Run inference on missing items
            run_inference(
                text_file=missing_text,
                emb_output=missing_emb,
                temp_folder=missing_temp,
                gpu_ids=args.gpu_ids,
                num_sessions_per_gpu=args.num_sessions_per_gpu,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )

            # Merge missing items
            merge_results(
                text_file=missing_text,
                emb_output=missing_emb,
                final_file=missing_final,
            )

            # Append to the filtered final embeddings
            append_to_final_embeddings(paths["final_emb"], missing_final)

            # Cleanup temp files
            for f in (missing_text, missing_emb, missing_final):
                if os.path.isfile(f):
                    os.remove(f)
            print(f"[resume] done: {total_kept:,} reused + "
                  f"{len(missing_ids):,} newly inferred")
        del missing_ids
        gc.collect()

    # ---- Stage 2: run inference ----
    if not args.skip_inference:
        print("-" * 80)
        print("Stage 2/4: run embedding inference")
        print("-" * 80)
        run_inference(
            text_file=paths["text_file"],
            emb_output=paths["emb_inference"],
            temp_folder=paths["temp_folder"],
            gpu_ids=args.gpu_ids,
            num_sessions_per_gpu=args.num_sessions_per_gpu,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    else:
        print("[stage 2] skipped (using existing embeddings)")

    # Free cat_mapping if it was loaded (only needed for Stage 1 extract)
    try:
        del cat_mapping
    except NameError:
        pass
    gc.collect()

    # ---- Stage 3: merge ----
    if not args.skip_merge:
        if not os.path.isfile(paths["emb_inference"]):
            print(f"[merge] embedding output not found: {paths['emb_inference']}")
            sys.exit(1)
        print("-" * 80)
        print("Stage 3/4: merge id + text + embedding")
        print("-" * 80)
        merge_results(
            text_file=paths["text_file"],
            emb_output=paths["emb_inference"],
            final_file=paths["final_emb"],
        )
    else:
        print("[stage 3] skipped (using existing final embeddings)")

    # ---- Stage 4: build ANN index ----
    if not args.skip_index:
        if not os.path.isfile(paths["final_emb"]):
            print(f"[index] final embedding file not found: {paths['final_emb']}")
            sys.exit(1)
        print("-" * 80)
        print("Stage 4/4: build FAISS ANN index")
        print("-" * 80)
        ids, embeddings = load_final_embeddings(paths["final_emb"], args.emb_dim)
        index = build_ann_index(
            embeddings=embeddings,
            index_type=args.index_type,
            nlist=args.nlist,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            use_all_gpus=args.use_all_gpus,
        )
        faiss.write_index(index, paths["index_file"])
        print(f"[index] FAISS index saved -> {paths['index_file']}")
        save_id_mapping(ids, paths["id_map_file"])
        # Stage 4 done: index + embeddings stay in memory for Stage 5
        gc.collect()
    else:
        print("[stage 4] skipped")

    # ---- Stage 5: compute item-to-item similarities ----
    if not args.skip_similarity:
        print("-" * 80)
        print("Stage 5/5: compute top-k item similarities")
        print("-" * 80)
        sim_output = args.sim_output or paths["sim_file"]

        # Reuse in-memory data from Stage 4 if available;
        # otherwise load from disk (e.g., when Stage 4 was skipped).
        try:
            _have_inmemory = ids is not None and index is not None
        except NameError:
            _have_inmemory = False

        if _have_inmemory:
            print(f"[sim] reusing in-memory index + embeddings")
        else:
            print(f"[sim] loading index and embeddings from disk ...")
            ids, embeddings = load_final_embeddings(
                paths["final_emb"], args.emb_dim)
            index = faiss.read_index(paths["index_file"])
            print(f"[sim] loaded index: ntotal={index.ntotal:,}")

        # For HNSW, set efSearch for better recall during similarity search
        if args.index_type == "hnsw":
            index.hnsw.efSearch = max(64, args.sim_top_k * 4)
            print(f"[sim] HNSW efSearch set to {index.hnsw.efSearch}")

        compute_similarities(
            index=index,
            ids=ids,
            embeddings=embeddings,
            top_k=args.sim_top_k,
            output_path=sim_output,
        )
        # Free after similarity is written
        del ids, embeddings, index
        gc.collect()
    else:
        print("[stage 5] skipped (similarity computation)")

    # ---- Summary ----
    print("=" * 80)
    print("  Done.")
    print("=" * 80)
    for k, v in paths.items():
        exists = "ok" if os.path.exists(v) else "missing"
        print(f"  {k:<14}: {v}  [{exists}]")


if __name__ == "__main__":
    main()
