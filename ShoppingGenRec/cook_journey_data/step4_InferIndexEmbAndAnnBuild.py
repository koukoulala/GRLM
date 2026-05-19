"""
step4_InferIndexEmbAndAnnBuild.py
=================================

Pipeline step 4 — read item.json (from step0), infer item embeddings via
MatadorEmb ONNX, and build a FAISS ANN index for downstream query search.

Stages
------
1. extract  : read ``item.json`` (output of step0_combine_item_data.py),
              build a text per product (Title, Brand, Seller, Category,
              Gender, Price), write ``<work_dir>/<prefix>_text.tsv``
              (id\\ttext) and ``<work_dir>/<prefix>_ids.tsv``.
2. inference: run MatadorEmb ONNX inference, write
              ``<work_dir>/<prefix>_text_embeddings.tsv``.
3. merge    : join id+text+embedding, write
              ``<work_dir>/<prefix>_final_embeddings.tsv``.
4. index    : build a FAISS index, write
              ``<work_dir>/<prefix>_ann_<type>.index`` and
              ``<work_dir>/<prefix>_ann_ids.txt``.

Examples
--------
    python step4_InferIndexEmbAndAnnBuild.py \\
        --item_json  ./raw_data/item.json \\
        --work_dir   ./data/Index_2026_05_04 \\
        --output_prefix EnUs_Product \\
        --gpu_ids 0,1,2,3 \\
        --index_type hnsw

    # only rebuild ANN index from existing final embeddings
    python step4_InferIndexEmbAndAnnBuild.py \\
        --work_dir     ./data/Index_2026_05_04 \\
        --output_prefix EnUs_Product \\
        --only_index --index_type hnsw

    # debug
    python step4_InferIndexEmbAndAnnBuild.py ... --debug --debug_rows 100 --gpu_ids 0
"""

from __future__ import annotations

import argparse
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


# ============================================================================ #
# Helpers                                                                      #
# ============================================================================ #
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


# ============================================================================ #
# Stage 1 — extract product text from item.json                                #
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
# CLI                                                                          #
# ============================================================================ #
DEFAULT_ITEM_JSON = os.path.join(SCRIPT_DIR, "raw_data", "item.json")
DEFAULT_WORK_DIR = os.path.join(SCRIPT_DIR, "data", "Index_debug")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 4: infer item embeddings from item.json + build FAISS ANN index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O ---------------------------------------------------------------------
    g_io = p.add_argument_group("I/O")
    g_io.add_argument("--item_json", 
                      default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/item.json",
                      help="Input item.json (output of step0_combine_item_data.py).")
    g_io.add_argument("--work_dir", 
                      default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/MatadorEmb_Index",
                      help="Directory for all intermediate + final artefacts.")
    g_io.add_argument("--output_prefix", default="Items_full",
                      help="Filename prefix used for all output files.")

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

    # Debug -------------------------------------------------------------------
    g_dbg = p.add_argument_group("Debug")
    g_dbg.add_argument("--debug", action="store_true",
                       help="Limit text extraction to --debug_rows lines.")
    g_dbg.add_argument("--debug_rows", type=int, default=100)

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
    }


def main() -> None:
    args = parse_args()
    if args.only_index:
        args.skip_extract = True
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
    print(f"  item_json:      {args.item_json}")
    print(f"  work_dir:       {args.work_dir}")
    print(f"  output_prefix:  {args.output_prefix}")
    print(f"  inference_dir:  {INFERENCE_DIR}")
    print(f"  index_type:     {args.index_type}  (emb_dim={args.emb_dim})")
    print(f"  GPUs:           {args.gpu_ids}")
    print(f"  debug:          {args.debug} (rows={args.debug_rows})")
    print()

    # ---- Stage 1: extract product text from item.json ----
    if not args.skip_extract:
        print("-" * 80)
        print("Stage 1/4: extract product text from item.json")
        print("-" * 80)
        n = extract_product_text_from_json(
            item_json_file=args.item_json,
            text_file=paths["text_file"],
            id_file=paths["id_file"],
            max_rows=(args.debug_rows if args.debug else 0),
        )
        if n == 0:
            print("[extract] no valid products found; aborting.")
            sys.exit(1)
    else:
        print("[stage 1] skipped (using existing text/id files)")

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
    else:
        print("[stage 4] skipped")

    # ---- Summary ----
    print("=" * 80)
    print("  Done.")
    print("=" * 80)
    for k, v in paths.items():
        exists = "ok" if os.path.exists(v) else "missing"
        print(f"  {k:<14}: {v}  [{exists}]")


if __name__ == "__main__":
    main()
