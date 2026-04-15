"""Step 7: Assign TID to New Products via Embedding Similarity

For new products, generates embeddings using the same model as s0, then
searches against a FAISS index built from existing product embeddings.

- If nearest-neighbor similarity >= threshold (default 0.9):
    Inherits the TID from the matched existing product via item_id2tid.json.
- If similarity < threshold:
    Falls back to a pre-existing tid2item_id.json mapping. If the new
    product's GID is found there, uses that TID.
- Otherwise: logged as unmatched.

The FAISS index is saved to disk after first build for subsequent reuse.

Inputs:
    - New product metadata JSONL (one JSON object per line with 'id' field)
    - Existing product embeddings dir (embeddings.npy, item_ids.json from s0)
    - Existing item_id2tid.json  (GID -> 7-word TID list)
    - Fallback tid2item_id.json  (TID string -> [GIDs])

Outputs (new products only):
    - tid2item_id.json        : comma-joined TID -> list of new GIDs
    - item_id2tid.json        : new GID -> 7-word TID list
    - id2words.tsv            : one JSON per line {GID: [words]}
    - new_embeddings.npy      : embeddings for new products
    - new_item_ids.json       : item IDs aligned with new_embeddings
    - faiss_index.bin         : saved FAISS index (in existing_emb_dir)
    - s7_statistics.json      : detailed match/fallback/unmatched stats
    - s7_unmatched_items.json : items that could not be assigned a TID

Usage:
    python s7_assign_tid_by_similarity.py \\
        --new_item_file /path/to/new/diff_1m_item.jsonl \\
        --existing_emb_dir /path/to/existing/processed/ \\
        --existing_id2tid_file /path/to/item_id2tid.json \\
        --fallback_tid2item_file /path/to/tid2item_id.json \\
        --embedding_model /path/to/Qwen3-Embedding-0.6B \\
        --output_dir /path/to/output/ \\
        --similarity_threshold 0.9 \\
        --num_gpus 2 --batch_size 256 --max_length 1024
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import time
from typing import List, Dict, Tuple
from collections import Counter
import warnings
import faiss

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
# Data Loading (consistent with s0_init_emb.py)
# =============================================================================

def load_data(file_path: str) -> List[Dict]:
    """Load item data from JSON or JSONL file.

    Supports two formats:
    - JSON dict: {"item_id": {"title": ..., ...}, ...}
    - JSONL: one JSON object per line with an "id" field

    Auto-detects format based on file extension (.jsonl vs .json).
    """
    if file_path.endswith(".jsonl"):
        result_list = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Ensure 'id' field exists
                    if "id" not in item:
                        continue
                    item["id"] = str(item["id"])
                    result_list.append(item)
        return result_list
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result_list = []
        for key, value in data.items():
            new_item = {"id": key}
            new_item.update(value)
            result_list.append(new_item)
        return result_list


def prepare_text_for_embedding(item: Dict, max_field_len: int = 500) -> str:
    """Prepare text for embedding generation (same logic as s0).

    Includes title, description, categories, and structured attributes.
    Each field value is truncated to max_field_len characters.
    """
    text_parts = []
    for field in ["title", "description", "categories"]:
        val = item.get(field, "")
        if val:
            if len(val) > max_field_len:
                val = val[:max_field_len] + "..."
            text_parts.append(f"{field.capitalize()}: {val}")

    # Append structured attributes
    '''
    attributes = item.get("attributes", {})
    if isinstance(attributes, dict):
        for attr_name in ["Brand", "Seller", "Color", "Size", "Gender", "AgeGroup"]:
            attr_val = attributes.get(attr_name, "")
            if isinstance(attr_val, str):
                attr_val = attr_val.strip()
            if attr_val:
                if len(str(attr_val)) > max_field_len:
                    attr_val = str(attr_val)[:max_field_len] + "..."
                text_parts.append(f"{attr_name}: {attr_val}")
    '''
    return " | ".join(text_parts)


# =============================================================================
# Embedding Generation (consistent with s0_init_emb.py)
# =============================================================================

def _encode_batch(batch_texts, tokenizer, model, device, max_length):
    """Encode a single batch of texts into L2-normalized embeddings."""
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        attention_mask = inputs["attention_mask"]
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)


def _save_emb_checkpoint(sorted_embeddings, next_batch_idx, failed_indices,
                         ckpt_file, ckpt_meta_file, batch_size):
    """Save embedding checkpoint: concatenated embeddings + metadata."""
    all_emb = np.vstack(sorted_embeddings) if sorted_embeddings else np.array([])
    np.save(ckpt_file, all_emb)
    with open(ckpt_meta_file, "w") as f:
        json.dump({
            "next_batch_idx": next_batch_idx,
            "batch_size": batch_size,
            "rows_done": all_emb.shape[0] if len(all_emb.shape) > 1 else 0,
            "failed_indices": failed_indices,
        }, f)


def process_batch_on_gpu(
    rank: int,
    data_slice: List[Dict],
    tmp_dir: str,
    model_name: str,
    batch_size: int = 16,
    max_length: int = 512,
):
    """Process data slice on specific GPU (or CPU fallback), save results to disk."""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        print(f"Rank {rank}: Loading model on {device}...")
        model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=f"cuda:{rank}",
            trust_remote_code=True,
        )
    else:
        device = torch.device("cpu")
        print(f"Rank {rank}: No GPU available, loading model on CPU...")
        model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    texts = [prepare_text_for_embedding(item) for item in data_slice]
    item_ids = [item["id"] for item in data_slice]

    # Sort by text length to minimize padding waste within each batch
    sorted_indices = sorted(range(len(texts)), key=lambda j: len(texts[j]))
    texts_sorted = [texts[j] for j in sorted_indices]
    ids_sorted = [item_ids[j] for j in sorted_indices]

    # Collect embeddings in sorted order, then unsort at the end
    sorted_embeddings = []
    failed_indices = []

    # Resume: check for partial checkpoint
    ckpt_file = os.path.join(tmp_dir, f"ckpt_rank{rank}.npy")
    ckpt_meta_file = os.path.join(tmp_dir, f"ckpt_rank{rank}_meta.json")
    start_batch = 0
    if os.path.exists(ckpt_file) and os.path.exists(ckpt_meta_file):
        with open(ckpt_meta_file, "r") as f:
            ckpt_meta = json.load(f)
        ckpt_emb = np.load(ckpt_file)
        start_batch = ckpt_meta["next_batch_idx"]
        failed_indices = ckpt_meta.get("failed_indices", [])
        rows_done = start_batch * batch_size
        sorted_embeddings = [ckpt_emb[:rows_done]] if rows_done > 0 else []
        print(f"Rank {rank}: Resuming from batch {start_batch} "
              f"({rows_done:,} items done, {len(failed_indices)} failed)")

    CHECKPOINT_EVERY = 500

    print(f"Rank {rank}: Generating embeddings for {len(texts)} items "
          f"(sorted by text length)...")
    start_time = time.time()

    total_batches = (len(texts_sorted) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(
        tqdm(range(0, len(texts_sorted), batch_size), desc=f"Rank {rank}",
             mininterval=30, dynamic_ncols=True)
    ):
        if batch_idx < start_batch:
            continue

        batch_texts = texts_sorted[i : i + batch_size]
        actual_bs = len(batch_texts)

        try:
            embeddings_np = _encode_batch(
                batch_texts, tokenizer, model, device, max_length
            )
            sorted_embeddings.append(embeddings_np)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            torch.cuda.empty_cache()
            print(f"\nRank {rank}: Batch {batch_idx} OOM ({actual_bs} items), "
                  f"retrying with half batch...")
            half = actual_bs // 2
            try:
                emb1 = _encode_batch(
                    batch_texts[:half], tokenizer, model, device, max_length
                )
                emb2 = _encode_batch(
                    batch_texts[half:], tokenizer, model, device, max_length
                )
                sorted_embeddings.append(np.vstack([emb1, emb2]))
            except Exception as e2:
                print(f"Rank {rank}: Batch {batch_idx} FAILED even after "
                      f"retry: {e2}. Filling with zeros.")
                torch.cuda.empty_cache()
                dim = sorted_embeddings[-1].shape[1] if sorted_embeddings else 896
                sorted_embeddings.append(np.zeros((actual_bs, dim), dtype=np.float32))
                failed_indices.extend(list(range(i, i + actual_bs)))

        if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
            _save_emb_checkpoint(
                sorted_embeddings, batch_idx + 1, failed_indices,
                ckpt_file, ckpt_meta_file, batch_size
            )

    sorted_all = np.vstack(sorted_embeddings) if sorted_embeddings else np.array([])

    # Unsort: restore original order so embeddings align with item_ids
    embeddings_array = np.empty_like(sorted_all)
    for new_pos, orig_pos in enumerate(sorted_indices):
        embeddings_array[orig_pos] = sorted_all[new_pos]
    elapsed = time.time() - start_time
    print(f"Rank {rank}: Done in {elapsed:.2f}s")
    if failed_indices:
        print(f"Rank {rank}: WARNING {len(failed_indices)} items failed, "
              f"filled with zero embeddings")

    np.save(os.path.join(tmp_dir, f"embeddings_rank{rank}.npy"), embeddings_array)
    with open(os.path.join(tmp_dir, f"item_ids_rank{rank}.json"), "w") as f:
        json.dump(item_ids, f)
    if failed_indices:
        with open(os.path.join(tmp_dir, f"failed_rank{rank}.json"), "w") as f:
            json.dump(failed_indices, f)
    for cf in [ckpt_file, ckpt_meta_file]:
        if os.path.exists(cf):
            os.remove(cf)


def generate_embeddings_multi_gpu(
    data: List[Dict],
    model_name: str,
    num_gpus: int,
    batch_size: int = 16,
    max_length: int = 512,
    tmp_dir: str = "/tmp/emb_tmp",
) -> Tuple[List[str], np.ndarray]:
    """Generate embeddings using multiple GPUs, transfer via disk."""
    print(f"Using {num_gpus} GPUs, total items: {len(data)}")
    os.makedirs(tmp_dir, exist_ok=True)

    chunk_size = len(data) // num_gpus
    data_chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = len(data) if i == num_gpus - 1 else start + chunk_size
        data_chunks.append(data[start:end])

    processes = []
    start_time = time.time()

    for rank in range(num_gpus):
        p = mp.Process(
            target=process_batch_on_gpu,
            args=(rank, data_chunks[rank], tmp_dir, model_name, batch_size, max_length),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Collect results from disk in rank order
    all_embeddings = []
    all_item_ids = []
    for rank in range(num_gpus):
        emb = np.load(os.path.join(tmp_dir, f"embeddings_rank{rank}.npy"))
        with open(os.path.join(tmp_dir, f"item_ids_rank{rank}.json"), "r") as f:
            ids = json.load(f)
        print(f"Loaded {len(ids)} items from Rank {rank}")
        all_embeddings.append(emb)
        all_item_ids.extend(ids)
        os.remove(os.path.join(tmp_dir, f"embeddings_rank{rank}.npy"))
        os.remove(os.path.join(tmp_dir, f"item_ids_rank{rank}.json"))

    combined_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    print(f"Multi-GPU processing completed in {time.time() - start_time:.2f}s")
    return all_item_ids, combined_embeddings


# =============================================================================
# FAISS Index Management
# =============================================================================

def build_faiss_index(
    embeddings: np.ndarray,
    nlist: int = 4096,
    nprobe: int = 64,
    num_threads: int = 0,
) -> faiss.Index:
    """Build and train IVFFlat index on CPU.

    Since embeddings are L2-normalized, inner product equals cosine similarity.

    Args:
        embeddings: L2-normalized float32 array of shape (N, dim).
        nlist: Number of IVF clusters.
        nprobe: Number of clusters to probe during search.
        num_threads: CPU threads for FAISS OpenMP (0 = all cores).

    Returns:
        Trained FAISS CPU index.
    """
    n, dim = embeddings.shape
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    if num_threads > 0:
        faiss.omp_set_num_threads(num_threads)
        print(f"  FAISS OpenMP threads set to {num_threads}")
    else:
        cpu_count = os.cpu_count() or 1
        faiss.omp_set_num_threads(cpu_count)
        print(f"  FAISS OpenMP threads set to {cpu_count} (all cores)")

    effective_nlist = min(nlist, n // 40) if n > 0 else 1
    effective_nlist = max(effective_nlist, 1)

    print(f"  Building FAISS IVFFlat index: n={n:,}, dim={dim}, "
          f"nlist={effective_nlist}, nprobe={nprobe}")
    start_time = time.time()

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(
        quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT
    )

    print(f"  Training index on {n:,} vectors ...")
    index.train(embeddings)
    print(f"  Adding {n:,} vectors to index ...")
    index.add(embeddings)
    index.nprobe = min(nprobe, effective_nlist)

    elapsed = time.time() - start_time
    print(f"  FAISS index built in {elapsed:.2f}s")
    return index


def save_faiss_index(index: faiss.Index, path: str):
    """Save FAISS index to disk (converts GPU index to CPU if needed)."""
    try:
        cpu_index = faiss.index_gpu_to_cpu(index)
    except Exception:
        cpu_index = index
    faiss.write_index(cpu_index, path)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  FAISS index saved to: {path} ({file_size_mb:.1f} MB)")


def load_faiss_index(path: str) -> faiss.Index:
    """Load FAISS index from disk (returns CPU index)."""
    print(f"  Loading FAISS index from: {path}")
    start_time = time.time()
    index = faiss.read_index(path)
    elapsed = time.time() - start_time
    print(f"  Loaded in {elapsed:.2f}s, ntotal={index.ntotal:,}")
    return index


def move_index_to_gpu(index: faiss.Index, gpu_id: int = 0) -> faiss.Index:
    """Move FAISS index to GPU if available, otherwise keep on CPU."""
    use_gpu = hasattr(faiss, "StandardGpuResources")
    num_gpus = faiss.get_num_gpus() if use_gpu else 0

    if num_gpus > 0:
        if num_gpus > 1:
            print(f"  Moving FAISS index to all {num_gpus} GPUs")
            index = faiss.index_cpu_to_all_gpus(index)
        else:
            print(f"  Moving FAISS index to GPU {gpu_id}")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    else:
        print("  No GPU available for FAISS, using CPU search")

    return index


def search_nearest_neighbors(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    k: int = 1,
    batch_size: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search for k nearest neighbors in batches.

    Returns:
        D: similarity scores, shape (n_queries, k)
        I: indices into the index, shape (n_queries, k)
    """
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    n = len(query_embeddings)

    all_distances = []
    all_indices = []
    num_batches = (n + batch_size - 1) // batch_size

    for start in tqdm(range(0, n, batch_size), total=num_batches,
                      desc="  FAISS search", dynamic_ncols=True):
        end = min(start + batch_size, n)
        D_batch, I_batch = index.search(query_embeddings[start:end], k)
        all_distances.append(D_batch)
        all_indices.append(I_batch)

    D = np.vstack(all_distances)
    I = np.vstack(all_indices)
    return D, I


# =============================================================================
# TID Mapping Utilities
# =============================================================================

def load_id2tid(file_path: str) -> Dict[str, List[str]]:
    """Load item_id2tid.json: GID -> [word1, word2, ..., word7]."""
    print(f"  Loading item_id2tid from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"    {len(data):,} GID -> TID mappings")
    return data


def load_and_invert_tid2item(file_path: str) -> Dict[str, List[str]]:
    """Load tid2item_id.json and invert to GID -> TID word list.

    Input format:  {"word1,word2,...,word7": ["GID1", "GID2", ...], ...}
    Output format: {"GID1": ["word1", "word2", ..., "word7"], ...}
    """
    print(f"  Loading fallback tid2item_id from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        tid2item = json.load(f)
    print(f"    {len(tid2item):,} TID entries")

    gid2tid = {}
    for tid_str, gid_list in tid2item.items():
        words = [w.strip() for w in tid_str.split(",")]
        for gid in gid_list:
            gid2tid[str(gid)] = words

    print(f"    Inverted to {len(gid2tid):,} GID -> TID mappings")
    return gid2tid


def tid_words_to_key(words: List[str]) -> str:
    """Convert TID word list to comma-joined key string."""
    return ",".join(words)


# =============================================================================
# CLI Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign TID to new products via embedding similarity"
    )
    # --- Input files ---
    parser.add_argument(
        "--new_item_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/EvalData/diff_1m_item.jsonl",
        help="Path to new product metadata file (JSON or JSONL format)",
    )
    parser.add_argument(
        "--existing_emb_dir",
        type=str,
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed_v4",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed",
        help="Directory with existing embeddings (embeddings.npy, item_ids.json)",
    )
    parser.add_argument(
        "--existing_id2tid_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/sft_data_v4/item_id2tid/item_id2tid.json",
        help="Path to existing item_id2tid.json (GID -> 7-word TID)",
    )
    parser.add_argument(
        "--fallback_tid2item_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/EvalData/tid2item_id.json",
        help="Path to fallback tid2item_id.json (TID -> [GIDs]) for "
             "products below the similarity threshold",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/eval_new_products/",
        help="Directory to save all output files",
    )
    # --- Embedding model ---
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3-Embedding-0.6B",
        help="Path to embedding model (same as s0)",
    )
    # --- Similarity threshold ---
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        nargs="+",
        default=[0.8, 0.85, 0.9],
        help="One or more cosine similarity thresholds. Outputs are saved "
             "to separate subdirectories (e.g., threshold_0.90/). "
             "Default: 0.8 0.85 0.9",
    )
    # --- GPU and batch settings ---
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs for embedding generation (default: all available)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size per GPU for embedding generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum token length for tokenizer truncation",
    )
    # --- FAISS settings ---
    parser.add_argument(
        "--faiss_index_file",
        type=str,
        default=None,
        help="Path to save/load FAISS index. "
             "Default: <existing_emb_dir>/faiss_index.bin",
    )
    parser.add_argument(
        "--faiss_nlist",
        type=int,
        default=4096,
        help="Number of IVF clusters for FAISS index",
    )
    parser.add_argument(
        "--faiss_nprobe",
        type=int,
        default=128,
        help="Number of clusters to probe during FAISS search "
             "(higher=more accurate, recommended 128+ for threshold-based matching)",
    )
    parser.add_argument(
        "--faiss_threads",
        type=int,
        default=0,
        help="Number of CPU threads for FAISS (0=all cores)",
    )
    parser.add_argument(
        "--rebuild_index",
        action="store_true",
        default=False,
        help="Force rebuild FAISS index even if saved index file exists",
    )
    # --- Resume support ---
    parser.add_argument(
        "--resume_new_emb_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/eval_new_products/",
        help="If set and files exist, reuse new product embeddings from "
             "this directory (new_embeddings.npy, new_item_ids.json) "
             "instead of regenerating them.",
    )
    # --- Output filtering ---
    parser.add_argument(
        "--output_source",
        type=str,
        default="similarity",
        choices=["similarity", "all"],
        help="Which items to include in output files: "
             "'similarity' = only items matched via embedding similarity "
             "(above threshold); 'all' = include both similarity and "
             "fallback matches (default: similarity)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    num_gpus = args.num_gpus or torch.cuda.device_count()

    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if num_gpus == 0:
        print("Warning: No GPU available, will use CPU (slow)")
        num_gpus = 1

    faiss_index_file = args.faiss_index_file or os.path.join(
        args.existing_emb_dir, "faiss_index.bin"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Stage 1: Load or Build FAISS Index from Existing Products
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 1: Existing Product FAISS Index")
    print("=" * 60)

    # Always load existing item IDs (needed for mapping indices -> GIDs)
    # Prefer compact .npy format if available, fall back to .json
    existing_ids_npy = os.path.join(args.existing_emb_dir, "item_ids.npy")
    existing_ids_json = os.path.join(args.existing_emb_dir, "item_ids.json")
    if os.path.exists(existing_ids_npy):
        print(f"Loading existing item IDs from: {existing_ids_npy} (fast npy)")
        existing_item_ids_arr = np.load(existing_ids_npy)
        existing_item_ids = [str(x) for x in existing_item_ids_arr]
    else:
        print(f"Loading existing item IDs from: {existing_ids_json} (json)")
        with open(existing_ids_json, "r", encoding="utf-8") as f:
            existing_item_ids = json.load(f)
        # Save compact .npy for future fast loading
        try:
            ids_int = np.array([int(x) for x in existing_item_ids], dtype=np.int64)
            np.save(existing_ids_npy, ids_int)
            print(f"  Saved compact item_ids.npy ({ids_int.nbytes / 1e6:.1f} MB)")
        except (ValueError, OverflowError):
            pass  # non-numeric IDs, skip npy conversion
    print(f"  Existing products in index: {len(existing_item_ids):,}")

    if not args.rebuild_index and os.path.exists(faiss_index_file):
        # Load previously saved index
        index = load_faiss_index(faiss_index_file)
    else:
        # Build index from existing embeddings
        existing_emb_file = os.path.join(args.existing_emb_dir, "embeddings.npy")
        print(f"Loading existing embeddings from: {existing_emb_file}")
        existing_embeddings = np.load(existing_emb_file)
        print(f"  Shape: {existing_embeddings.shape}, "
              f"dtype: {existing_embeddings.dtype}")

        index = build_faiss_index(
            existing_embeddings,
            nlist=args.faiss_nlist,
            nprobe=args.faiss_nprobe,
            num_threads=args.faiss_threads,
        )

        # Save CPU index for reuse
        save_faiss_index(index, faiss_index_file)
        del existing_embeddings  # free memory

    # NOTE: Do NOT move index to GPU here. Wait until after embedding
    # generation (Stage 3) to avoid GPU memory competition.
    print(f"  FAISS index ready on CPU (ntotal={index.ntotal:,}). "
          f"Will move to GPU after embedding generation.")

    # =========================================================================
    # Stage 2: Load TID Mappings
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 2: Load TID Mappings")
    print("=" * 60)

    # Existing GID -> TID (for similarity-matched products)
    existing_id2tid = load_id2tid(args.existing_id2tid_file)

    # Fallback: invert tid2item_id to get GID -> TID (for unmatched products)
    fallback_gid2tid = load_and_invert_tid2item(args.fallback_tid2item_file)

    # =========================================================================
    # Stage 3: Load New Products and Generate Embeddings
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 3: New Product Embeddings")
    print("=" * 60)

    print(f"Loading new products from: {args.new_item_file}")
    new_data = load_data(args.new_item_file)
    print(f"  New products: {len(new_data):,}")

    if len(new_data) == 0:
        print("Error: No new products loaded!")
        return

    # Check for resume: reuse existing embeddings if available
    resume_dir = args.resume_new_emb_dir
    resume_emb_file = os.path.join(resume_dir, "new_embeddings.npy") if resume_dir else ""
    resume_ids_file = os.path.join(resume_dir, "new_item_ids.json") if resume_dir else ""

    if resume_dir and os.path.exists(resume_emb_file) and os.path.exists(resume_ids_file):
        print(f"\n[RESUME] Reusing new product embeddings from: {resume_dir}")
        new_embeddings = np.load(resume_emb_file)
        with open(resume_ids_file, "r", encoding="utf-8") as f:
            new_item_ids = json.load(f)
        print(f"  Loaded embeddings: shape={new_embeddings.shape}, "
              f"items={len(new_item_ids):,}")

        # Check for failed items that need re-embedding
        failed_sorted_indices = set()
        resume_tmp_dir = os.path.join(resume_dir, "_emb_tmp") if resume_dir else ""
        for fname in (os.listdir(resume_tmp_dir) if os.path.isdir(resume_tmp_dir) else []):
            if fname.startswith("failed_rank") and fname.endswith(".json"):
                fpath = os.path.join(resume_tmp_dir, fname)
                with open(fpath, "r") as f:
                    failed_sorted_indices.update(json.load(f))

        if failed_sorted_indices:
            print(f"  [RESUME-FAILED] Found {len(failed_sorted_indices):,} "
                  f"failed sorted indices from previous run")
            # Map sorted indices back to original item IDs via new_item_ids
            # Failed indices are indices into the sorted text list of the
            # previous run; we need the actual item IDs. Since we saved
            # item_ids in original order and embeddings are unsorted back,
            # zero-vector embeddings mark the failed items.
            zero_mask = np.all(new_embeddings == 0, axis=1)
            failed_gids = [new_item_ids[i] for i in range(len(new_item_ids))
                           if zero_mask[i]]
            print(f"  Zero-vector items (failed): {len(failed_gids):,}")

            if failed_gids:
                failed_gid_set = set(failed_gids)
                failed_items_data = [item for item in new_data
                                     if item["id"] in failed_gid_set]
                print(f"  Re-generating embeddings for {len(failed_items_data):,} "
                      f"failed items...")
                tmp_dir = os.path.join(args.output_dir, "_emb_tmp_retry")
                retry_ids, retry_embs = generate_embeddings_multi_gpu(
                    failed_items_data,
                    args.embedding_model,
                    num_gpus=num_gpus,
                    batch_size=max(args.batch_size // 2, 32),  # smaller batch
                    max_length=args.max_length,
                    tmp_dir=tmp_dir,
                )
                if os.path.isdir(tmp_dir):
                    try:
                        os.rmdir(tmp_dir)
                    except OSError:
                        pass

                # Merge retried embeddings back
                retry_id_to_idx = {rid: i for i, rid in enumerate(retry_ids)}
                patched = 0
                for i, gid in enumerate(new_item_ids):
                    if gid in retry_id_to_idx:
                        new_embeddings[i] = retry_embs[retry_id_to_idx[gid]]
                        patched += 1
                still_zero = int(np.all(new_embeddings == 0, axis=1).sum())
                print(f"  Patched {patched:,} embeddings, "
                      f"still zero: {still_zero:,}")

        # Validate alignment with new_data
        new_data_ids = {item["id"] for item in new_data}
        loaded_ids_set = set(new_item_ids)
        if loaded_ids_set != new_data_ids:
            missing = new_data_ids - loaded_ids_set
            extra = loaded_ids_set - new_data_ids
            print(f"  WARNING: ID mismatch! missing={len(missing):,}, "
                  f"extra={len(extra):,}")
            if missing:
                print(f"  Will regenerate embeddings for all items.")
                resume_dir = None  # force regeneration
            else:
                # Filter to only IDs in new_data, preserving new_data order
                id_to_emb_idx = {nid: i for i, nid in enumerate(new_item_ids)}
                reordered_ids = []
                reordered_embs = []
                for item in new_data:
                    if item["id"] in id_to_emb_idx:
                        reordered_ids.append(item["id"])
                        reordered_embs.append(
                            new_embeddings[id_to_emb_idx[item["id"]]]
                        )
                new_item_ids = reordered_ids
                new_embeddings = np.vstack(reordered_embs)
                new_data = [item for item in new_data
                            if item["id"] in loaded_ids_set]
                print(f"  Filtered and reordered to {len(new_item_ids):,} items")

    if not resume_dir or not os.path.exists(resume_emb_file):
        # Generate embeddings for all new products
        tmp_dir = os.path.join(args.output_dir, "_emb_tmp")
        new_item_ids, new_embeddings = generate_embeddings_multi_gpu(
            new_data,
            args.embedding_model,
            num_gpus=num_gpus,
            batch_size=args.batch_size,
            max_length=args.max_length,
            tmp_dir=tmp_dir,
        )

        # Clean up temp dir
        if os.path.isdir(tmp_dir):
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    print(f"New embeddings shape: {new_embeddings.shape}, "
          f"dtype: {new_embeddings.dtype}")

    # Save new embeddings for potential reuse
    new_emb_file = os.path.join(args.output_dir, "new_embeddings.npy")
    new_ids_file = os.path.join(args.output_dir, "new_item_ids.json")
    np.save(new_emb_file, new_embeddings)
    with open(new_ids_file, "w", encoding="utf-8") as f:
        json.dump(new_item_ids, f, ensure_ascii=False)
    print(f"Saved new embeddings to: {new_emb_file} "
          f"({new_embeddings.nbytes / 1e9:.2f} GB)")

    # =========================================================================
    # Stage 4: ANN Search — Find Nearest Existing Product
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 4: ANN Search (Top-1 Nearest Existing Product)")
    print("=" * 60)

    # FAISS search on CPU — index is already IVFFlat on CPU, just search directly.
    # For 6.2M index + 1M queries with 40 CPU cores, takes ~2-5 minutes.
    print(f"  Using CPU FAISS search (index ntotal={index.ntotal:,})")

    search_start = time.time()
    D, I = search_nearest_neighbors(index, new_embeddings, k=1)
    search_elapsed = time.time() - search_start

    # D[:, 0] = cosine similarity, I[:, 0] = index in existing_item_ids
    similarities = D[:, 0]
    matched_indices = I[:, 0]

    print(f"  Search completed in {search_elapsed:.2f}s")
    print(f"  Similarity statistics (all {len(similarities):,} queries):")
    print(f"    Mean:   {np.mean(similarities):.4f}")
    print(f"    Median: {np.median(similarities):.4f}")
    print(f"    Min:    {np.min(similarities):.4f}")
    print(f"    Max:    {np.max(similarities):.4f}")
    print(f"    Std:    {np.std(similarities):.4f}")

    # Histogram of similarity scores
    bins = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
    hist, _ = np.histogram(similarities, bins=bins)
    print(f"\n  Similarity distribution:")
    for i in range(len(bins) - 1):
        label = f"    [{bins[i]:.2f}, {bins[i+1]:.2f})"
        print(f"{label}: {hist[i]:>10,}")

    # =========================================================================
    # Stage 5+6: Assign TIDs & Save — for each threshold
    # =========================================================================
    # Pre-compute shared data
    new_data_dict = {item["id"]: item for item in new_data}
    total = len(new_item_ids)

    # Compute attribute presence stats (shared across thresholds)
    all_attr_names = set()
    for item in new_data:
        attrs = item.get("attributes", {})
        if isinstance(attrs, dict):
            all_attr_names.update(attrs.keys())

    field_names = ["title", "description", "categories", "attributes"]
    field_presence = {}
    for field in field_names:
        count = sum(1 for item in new_data if item.get(field))
        field_presence[field] = {
            "count": count,
            "pct": round(count / total * 100, 2) if total else 0,
        }

    attr_presence = {}
    for attr_name in sorted(all_attr_names):
        count = sum(
            1 for item in new_data
            if isinstance(item.get("attributes"), dict)
            and item["attributes"].get(attr_name)
        )
        attr_presence[attr_name] = {
            "count": count,
            "pct": round(count / total * 100, 2) if total else 0,
        }

    thresholds = sorted(args.similarity_threshold)
    print(f"\n  Will evaluate {len(thresholds)} threshold(s): {thresholds}")

    for thresh in thresholds:
        print("\n" + "=" * 60)
        print(f"Stage 5: Assign TIDs (threshold={thresh})")
        print("=" * 60)

        thresh_dir = os.path.join(args.output_dir, f"threshold_{thresh:.2f}")
        os.makedirs(thresh_dir, exist_ok=True)

        # Result mappings
        new_item_id2tid = {}
        new_tid2item_id = {}
        item_source = {}

        matched_by_similarity = 0
        matched_by_fallback = 0
        unmatched_count = 0
        sim_above_but_no_tid = 0

        unmatched_items = []
        similarity_records = {
            "above_threshold": [],
            "below_threshold_fallback": [],
            "below_threshold_unmatched": [],
        }

        for i in range(total):
            new_gid = new_item_ids[i]
            sim_score = float(similarities[i])
            match_idx = int(matched_indices[i])
            matched_existing_gid = (
                existing_item_ids[match_idx] if match_idx >= 0 else None
            )

            tid_words = None
            source = None

            if sim_score >= thresh and matched_existing_gid:
                if matched_existing_gid in existing_id2tid:
                    tid_words = existing_id2tid[matched_existing_gid]
                    source = "similarity"
                    matched_by_similarity += 1
                    similarity_records["above_threshold"].append(sim_score)
                else:
                    sim_above_but_no_tid += 1

            if tid_words is None:
                if new_gid in fallback_gid2tid:
                    tid_words = fallback_gid2tid[new_gid]
                    source = "fallback"
                    matched_by_fallback += 1
                    similarity_records["below_threshold_fallback"].append(sim_score)
                else:
                    unmatched_count += 1
                    unmatched_items.append({
                        "id": new_gid,
                        "nearest_existing_gid": matched_existing_gid,
                        "similarity": sim_score,
                        "title": new_data_dict.get(new_gid, {}).get("title", ""),
                    })
                    similarity_records["below_threshold_unmatched"].append(sim_score)
                    continue

            new_item_id2tid[new_gid] = tid_words
            item_source[new_gid] = source
            tid_key = tid_words_to_key(tid_words)
            if tid_key not in new_tid2item_id:
                new_tid2item_id[tid_key] = []
            new_tid2item_id[tid_key].append(new_gid)

        # --- Print summary ---
        print(f"\n  Assignment Summary (threshold={thresh}):")
        print(f"    Total new products:          {total:>10,}")
        print(f"    Matched by similarity:       {matched_by_similarity:>10,} "
              f"({matched_by_similarity / total * 100:.2f}%)")
        print(f"    Matched by fallback:         {matched_by_fallback:>10,} "
              f"({matched_by_fallback / total * 100:.2f}%)")
        print(f"    Unmatched:                   {unmatched_count:>10,} "
              f"({unmatched_count / total * 100:.2f}%)")
        print(f"    Sim>=threshold but no TID:   {sim_above_but_no_tid:>10,}")
        total_assigned = matched_by_similarity + matched_by_fallback
        print(f"    Total assigned:              {total_assigned:>10,} "
              f"({total_assigned / total * 100:.2f}%)")
        print(f"    Unique TIDs assigned:        {len(new_tid2item_id):>10,}")

        tid_sizes = [len(gids) for gids in new_tid2item_id.values()]
        if tid_sizes:
            print(f"\n  TID -> GID list sizes:")
            print(f"    Mean:          {np.mean(tid_sizes):.2f}")
            print(f"    Max:           {max(tid_sizes)}")
            print(f"    Singletons:    {sum(1 for s in tid_sizes if s == 1):,}")
            print(f"    Multi-mapped:  {sum(1 for s in tid_sizes if s > 1):,}")

        # --- Filter for tid2item_id / item_id2tid ---
        if args.output_source == "similarity":
            out_item_id2tid = {k: v for k, v in new_item_id2tid.items()
                              if item_source.get(k) == "similarity"}
            print(f"\n  Output filter: similarity only -> "
                  f"{len(out_item_id2tid):,} of {len(new_item_id2tid):,} items")
        else:
            out_item_id2tid = dict(new_item_id2tid)

        out_tid2item_id = {}
        for gid, words in out_item_id2tid.items():
            tid_key = tid_words_to_key(words)
            if tid_key not in out_tid2item_id:
                out_tid2item_id[tid_key] = []
            out_tid2item_id[tid_key].append(gid)

        # --- Save tid2item_id.json (filtered) ---
        tid2item_file = os.path.join(thresh_dir, "tid2item_id.json")
        with open(tid2item_file, "w", encoding="utf-8") as f:
            json.dump(out_tid2item_id, f, ensure_ascii=False, indent=2)
        print(f"  Saved tid2item_id.json: {tid2item_file} "
              f"({len(out_tid2item_id):,} TIDs, {len(out_item_id2tid):,} GIDs)")

        # --- Save item_id2tid.json (filtered) ---
        id2tid_file = os.path.join(thresh_dir, "item_id2tid.json")
        with open(id2tid_file, "w", encoding="utf-8") as f:
            json.dump(out_item_id2tid, f, ensure_ascii=False, indent=2)
        print(f"  Saved item_id2tid.json: {id2tid_file} "
              f"({len(out_item_id2tid):,} GIDs)")

        # --- Save id2words.tsv (ALL assigned items, not filtered) ---
        id2words_file = os.path.join(thresh_dir, "id2words.tsv")
        with open(id2words_file, "w", encoding="utf-8") as f:
            for item_id, words in new_item_id2tid.items():
                f.write(json.dumps({item_id: words}, ensure_ascii=False) + "\n")
        print(f"  Saved id2words.tsv: {id2words_file} "
              f"({len(new_item_id2tid):,} items, ALL sources)")

        # --- Save statistics ---
        def _safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        similarity_hist = {}
        for bi in range(len(bins) - 1):
            key = f"[{bins[bi]:.2f},{bins[bi+1]:.2f})"
            similarity_hist[key] = int(hist[bi])

        statistics = {
            "total_new_products": total,
            "similarity_threshold": thresh,
            "matched_by_similarity": matched_by_similarity,
            "matched_by_similarity_pct": round(
                matched_by_similarity / total * 100, 4) if total else 0,
            "matched_by_fallback": matched_by_fallback,
            "matched_by_fallback_pct": round(
                matched_by_fallback / total * 100, 4) if total else 0,
            "unmatched": unmatched_count,
            "unmatched_pct": round(
                unmatched_count / total * 100, 4) if total else 0,
            "total_assigned": total_assigned,
            "total_assigned_pct": round(
                total_assigned / total * 100, 4) if total else 0,
            "sim_above_threshold_but_no_tid": sim_above_but_no_tid,
            "unique_tids_assigned": len(new_tid2item_id),
            "output_source": args.output_source,
            "output_filtered_gids": len(out_item_id2tid),
            "similarity_overall": {
                "mean": float(np.mean(similarities)),
                "median": float(np.median(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities)),
                "std": float(np.std(similarities)),
            },
            "similarity_histogram": similarity_hist,
            "similarity_by_group": {
                "above_threshold": {
                    "count": len(similarity_records["above_threshold"]),
                    "mean": _safe_mean(similarity_records["above_threshold"]),
                },
                "below_threshold_fallback": {
                    "count": len(similarity_records["below_threshold_fallback"]),
                    "mean": _safe_mean(similarity_records["below_threshold_fallback"]),
                },
                "below_threshold_unmatched": {
                    "count": len(similarity_records["below_threshold_unmatched"]),
                    "mean": _safe_mean(similarity_records["below_threshold_unmatched"]),
                },
            },
            "tid_collision_stats": {
                "total_unique_tids": len(new_tid2item_id),
                "singletons": sum(1 for s in tid_sizes if s == 1) if tid_sizes else 0,
                "multi_mapped": sum(1 for s in tid_sizes if s > 1) if tid_sizes else 0,
                "max_gids_per_tid": max(tid_sizes) if tid_sizes else 0,
                "mean_gids_per_tid": float(np.mean(tid_sizes)) if tid_sizes else 0,
            },
            "input_stats": {
                "existing_products_in_index": len(existing_item_ids),
                "existing_id2tid_entries": len(existing_id2tid),
                "fallback_gid2tid_entries": len(fallback_gid2tid),
            },
            "new_product_field_presence": field_presence,
            "new_product_attribute_presence": attr_presence,
        }

        stats_file = os.path.join(thresh_dir, "s7_statistics.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"  Saved statistics: {stats_file}")

        # --- Save unmatched items ---
        unmatched_file = os.path.join(thresh_dir, "s7_unmatched_items.json")
        with open(unmatched_file, "w", encoding="utf-8") as f:
            json.dump(unmatched_items, f, ensure_ascii=False, indent=2)
        print(f"  Saved unmatched items: {unmatched_file} "
              f"({len(unmatched_items):,} items)")

    # Print field/attribute stats once at the end
    print(f"\n  New product field presence:")
    for field, info in field_presence.items():
        print(f"    {field:20s}: {info['count']:>10,} ({info['pct']:.2f}%)")
    print(f"  Attribute presence:")
    for attr_name, info in sorted(attr_presence.items(),
                                   key=lambda x: -x[1]["count"]):
        print(f"    {attr_name:20s}: {info['count']:>10,} ({info['pct']:.2f}%)")

    print(f"\nProcessing completed! Thresholds: {thresholds}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
