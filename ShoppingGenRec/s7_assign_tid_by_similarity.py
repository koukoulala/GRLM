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
    for field in ["categories", "title", "description"]:
        val = item.get(field, "")
        if val:
            if len(val) > max_field_len:
                val = val[:max_field_len] + "..."
            text_parts.append(f"{field.capitalize()}: {val}")

    # Append structured attributes
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


def _encode_batch_with_retry(batch_texts, tokenizer, model, device, max_length,
                             rank=0):
    """Encode texts with recursive batch-halving on OOM.

    Keeps halving the batch until size=1. Only records failure for
    individual items that truly cannot be encoded.

    Returns:
        embeddings: np.ndarray of shape (N, dim) — successful items.
            Failed items are filled with NaN so they can be identified later.
        failed_local_indices: list of indices (0-based within batch_texts)
            that failed even at batch_size=1.
    """
    try:
        return _encode_batch(batch_texts, tokenizer, model, device, max_length), []
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    n = len(batch_texts)
    if n == 1:
        print(f"Rank {rank}: Single item FAILED, marking as NaN")
        return None, [0]

    mid = n // 2
    print(f"Rank {rank}: OOM with {n} items, splitting to {mid} + {n - mid}")

    emb1, fail1 = _encode_batch_with_retry(
        batch_texts[:mid], tokenizer, model, device, max_length, rank)
    emb2, fail2 = _encode_batch_with_retry(
        batch_texts[mid:], tokenizer, model, device, max_length, rank)

    fail2_shifted = [idx + mid for idx in fail2]
    all_failed = fail1 + fail2_shifted

    if emb1 is not None and emb2 is not None:
        combined = np.vstack([emb1, emb2])
    elif emb1 is not None:
        dim = emb1.shape[1]
        placeholder = np.full((n - mid, dim), np.nan, dtype=np.float32)
        combined = np.vstack([emb1, placeholder])
    elif emb2 is not None:
        dim = emb2.shape[1]
        placeholder = np.full((mid, dim), np.nan, dtype=np.float32)
        combined = np.vstack([placeholder, emb2])
    else:
        return None, all_failed

    return combined, all_failed


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

        emb_result, batch_failed = _encode_batch_with_retry(
            batch_texts, tokenizer, model, device, max_length, rank
        )
        if emb_result is not None:
            sorted_embeddings.append(emb_result)
        else:
            dim = sorted_embeddings[-1].shape[1] if sorted_embeddings else 896
            sorted_embeddings.append(
                np.full((actual_bs, dim), np.nan, dtype=np.float32))
        if batch_failed:
            failed_indices.extend([i + fi for fi in batch_failed])
            print(f"Rank {rank}: Batch {batch_idx}: "
                  f"{len(batch_failed)} items failed")

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

    # Filter out failed items (NaN rows) — exclude them entirely
    if failed_indices:
        nan_mask = np.any(np.isnan(embeddings_array), axis=1)
        n_failed = int(nan_mask.sum())
        if n_failed > 0:
            failed_item_ids = [item_ids[j] for j in range(len(item_ids))
                               if nan_mask[j]]
            success_mask = ~nan_mask
            embeddings_array = embeddings_array[success_mask]
            item_ids = [item_ids[j] for j in range(len(item_ids))
                        if not nan_mask[j]]
            print(f"Rank {rank}: Excluded {n_failed} failed items, "
                  f"keeping {len(item_ids):,} successful items")
            with open(os.path.join(tmp_dir, f"failed_ids_rank{rank}.json"), "w") as f:
                json.dump(failed_item_ids, f)

    np.save(os.path.join(tmp_dir, f"embeddings_rank{rank}.npy"), embeddings_array)
    with open(os.path.join(tmp_dir, f"item_ids_rank{rank}.json"), "w") as f:
        json.dump(item_ids, f)
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


def extract_top_category(cat_str) -> str:
    """Extract top-level category from a hierarchical category string.

    Supports '|', ' > ', and '/' separators (checked in that order).
    Returns empty string if no category.
    """
    if not cat_str or not str(cat_str).strip():
        return ""
    cat_str = str(cat_str).strip()
    if "|" in cat_str:
        return cat_str.split("|")[0].strip()
    elif " > " in cat_str:
        return cat_str.split(" > ")[0].strip()
    elif "/" in cat_str:
        return cat_str.split("/")[0].strip()
    return cat_str.strip()


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
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/EvalData/diff_1m_item.jsonl",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/EvalData/s2_ckpt1425_0408Index_item.json",
        help="Path to new product metadata file (JSON or JSONL format)",
    )
    parser.add_argument(
        "--existing_emb_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed_v4",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed",
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
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/EvalData/s2_ckpt1425_0408Index_tid2item_id.json",
        help="Path to fallback tid2item_id.json (TID -> [GIDs]) for "
             "products below the similarity threshold",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/eval_new_products_full/",
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
        default=[0.8, 0.85, 0.9, 0.95],
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
        default=256,
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
        "--faiss_top_k",
        type=int,
        default=3,
        help="Number of nearest neighbors to retrieve from FAISS for "
             "reranking (default: 3)",
    )
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
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/eval_new_products_2/",
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
    # --- Case study ---
    parser.add_argument(
        "--case_study_samples",
        type=int,
        default=1000,
        help="If > 0, randomly sample this many new products and print "
             "detailed case study of their mapping results. Set to e.g. 20.",
    )
    parser.add_argument(
        "--case_study_only",
        action="store_true",
        default=False,
        help="Skip all computation. Read mapping_details.npz and output files "
             "from --output_dir to print case study only. Requires a previous "
             "full run to have saved mapping_details.npz.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # =========================================================================
    # Case Study Only mode: no FAISS, no embedding, just read saved files
    # =========================================================================
    if args.case_study_only:
        import random as _random
        _random.seed(SEED)
        n_samples = args.case_study_samples if args.case_study_samples > 0 else 20

        print("=" * 60)
        print("CASE STUDY ONLY MODE")
        print("=" * 60)
        print(f"  Output dir:  {args.output_dir}")
        print(f"  Samples:     {n_samples}")

        # Try to load similarity scores and matched GIDs (optional, independent)
        mapping_file = os.path.join(args.output_dir, "mapping_details.npz")
        matched_gids_file = os.path.join(args.output_dir, "matched_existing_gids.json")
        new_ids_file = os.path.join(args.output_dir, "new_item_ids.json")

        similarities = None
        matched_existing_gids = None
        new_item_ids = None

        # Load matched GIDs (can exist even without mapping_details.npz)
        if os.path.exists(matched_gids_file) and os.path.exists(new_ids_file):
            print(f"  Loading matched GIDs and new item IDs...")
            with open(matched_gids_file, "r", encoding="utf-8") as f:
                matched_existing_gids = json.load(f)
            with open(new_ids_file, "r", encoding="utf-8") as f:
                new_item_ids = json.load(f)
            print(f"  Loaded {len(new_item_ids):,} items with matched GIDs")

        # Load similarity scores (optional, on top of matched GIDs)
        has_sim_scores = False
        if os.path.exists(mapping_file) and new_item_ids is not None:
            print(f"  Loading similarity scores...")
            data = np.load(mapping_file)
            similarities = data["similarities"]
            has_sim_scores = True
            print(f"  Loaded similarity scores for {len(similarities):,} items")
        elif new_item_ids is None:
            print(f"  No matched_existing_gids.json or new_item_ids.json found")

        # Load TID mappings
        print(f"  Loading TID mappings...")
        existing_id2tid = load_id2tid(args.existing_id2tid_file)
        fallback_gid2tid = load_and_invert_tid2item(args.fallback_tid2item_file)

        # Load threshold output for assigned TIDs
        case_thresh = min(args.similarity_threshold)
        thresh_dir = os.path.join(args.output_dir, f"threshold_{case_thresh:.2f}")
        thresh_id2tid = {}
        thresh_id2tid_file = os.path.join(thresh_dir, "item_id2tid.json")
        if os.path.exists(thresh_id2tid_file):
            with open(thresh_id2tid_file, "r", encoding="utf-8") as f:
                thresh_id2tid = json.load(f)
            print(f"  Loaded threshold {case_thresh} item_id2tid: "
                  f"{len(thresh_id2tid):,} items")

        # Load new item metadata
        print(f"  Loading new item metadata...")
        new_data = load_data(args.new_item_file)
        new_data_by_id = {item["id"]: item for item in new_data}

        # Load existing item metadata (full, not just titles)
        existing_items = {}
        item_file_path = os.path.join(
            os.path.dirname(args.existing_emb_dir.rstrip("/")),
            "raw_data", "item.json"
        )
        if os.path.exists(item_file_path):
            print(f"  Loading existing item metadata...")
            with open(item_file_path, "r", encoding="utf-8") as f:
                existing_items = json.load(f)
            print(f"    {len(existing_items):,} existing items loaded")

        # Determine sample pool: use new_item_ids if available, else thresh_id2tid keys
        if new_item_ids:
            sample_pool = new_item_ids
        elif thresh_id2tid:
            sample_pool = list(thresh_id2tid.keys())
        else:
            sample_pool = [item["id"] for item in new_data]

        # Build gid -> matched_gid lookup
        gid2matched = {}
        if new_item_ids and matched_existing_gids:
            gid2matched = {new_item_ids[i]: matched_existing_gids[i]
                           for i in range(len(new_item_ids))}
        gid2sim = {}
        if new_item_ids and similarities is not None:
            gid2sim = {new_item_ids[i]: float(similarities[i])
                       for i in range(len(new_item_ids))}

        n_samples = min(n_samples, len(sample_pool))
        sample_gids = sorted(_random.sample(sample_pool, n_samples))

        print(f"\n{'#'*80}")
        print(f"  CASE STUDY: {n_samples} sampled new products")
        if has_sim_scores:
            print(f"  (with similarity scores, threshold={case_thresh})")
        else:
            print(f"  (without similarity scores)")
        print(f"{'#'*80}")

        records = []
        for rank, gid in enumerate(sample_gids, 1):
            sim = gid2sim.get(gid)
            matched_gid = gid2matched.get(gid, "")

            # New product info
            ni = new_data_by_id.get(gid, {})
            na = ni.get("attributes", {}) if isinstance(ni.get("attributes"), dict) else {}

            # Existing product info
            ei = existing_items.get(str(matched_gid), {}) if matched_gid else {}
            ea = ei.get("attributes", {}) if isinstance(ei.get("attributes"), dict) else {}
            ex_tid = existing_id2tid.get(str(matched_gid), [])

            # Assigned TID (from threshold output or computed)
            assigned_tid = thresh_id2tid.get(gid)
            if assigned_tid:
                source = "similarity"
            elif gid in fallback_gid2tid:
                assigned_tid = fallback_gid2tid[gid]
                source = "fallback"
            else:
                source = "unmatched"

            print(f"\n  [{rank}/{n_samples}] Assigned TID: "
                  f"{','.join(assigned_tid) if assigned_tid else 'NONE'} "
                  f"({source})")
            print(f"  NEW  GID: {gid}")
            print(f"    Title:       {ni.get('title', 'N/A')[:200]}")
            print(f"    Description: {str(ni.get('description', 'N/A'))[:200]}")
            print(f"    Categories:  {ni.get('categories', 'N/A')[:120]}")
            print(f"    Brand: {na.get('Brand', 'N/A')}  |  "
                  f"Seller: {na.get('Seller', 'N/A')}  |  "
                  f"Color: {na.get('Color', '-')}  |  "
                  f"Size: {na.get('Size', '-')}")
            print(f"  EXISTING GID: {matched_gid or 'N/A'}")
            if sim is not None:
                print(f"    Cosine similarity: {sim:.4f}")
            print(f"    Title:       {ei.get('title', 'N/A')[:200] if ei else 'N/A'}")
            print(f"    Description: {str(ei.get('description', 'N/A'))[:200] if ei else 'N/A'}")
            print(f"    Categories:  {ei.get('categories', 'N/A')[:120] if ei else 'N/A'}")
            print(f"    Brand: {ea.get('Brand', 'N/A')}  |  "
                  f"Seller: {ea.get('Seller', 'N/A')}")
            print(f"    ExistTID:    {','.join(ex_tid) if ex_tid else 'N/A'}")

            new_desc = str(ni.get("description", ""))
            ex_desc = str(ei.get("description", "")) if ei else ""

            records.append({
                "assigned_tid": assigned_tid or [],
                "assigned_source": source,
                "cosine_similarity": round(sim, 4) if sim is not None else None,
                "new_product": {
                    "gid": gid,
                    "title": ni.get("title", ""),
                    "description": new_desc[:500],
                    "categories": ni.get("categories", ""),
                    "brand": na.get("Brand", ""),
                    "seller": na.get("Seller", ""),
                    "color": na.get("Color", ""),
                    "size": na.get("Size", ""),
                },
                "existing_product": {
                    "gid": str(matched_gid) if matched_gid else "",
                    "title": ei.get("title", "") if ei else "",
                    "description": ex_desc[:500],
                    "categories": ei.get("categories", "") if ei else "",
                    "brand": ea.get("Brand", ""),
                    "seller": ea.get("Seller", ""),
                    "tid": ex_tid,
                },
            })

        # Save to file (pretty JSON, one record per block)
        case_study_file = os.path.join(args.output_dir, "case_study.jsonl")
        with open(case_study_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, indent=2) + "\n")
        print(f"\n  Case study saved to: {case_study_file}")
        print(f"\nDone!")
        return

    # =========================================================================
    # Normal mode: full pipeline
    # =========================================================================
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
    print("Stage 1: Load TID Mappings & Build FAISS Index")
    print("=" * 60)

    # Load TID mappings FIRST (needed to filter index)
    existing_id2tid = load_id2tid(args.existing_id2tid_file)
    fallback_gid2tid = load_and_invert_tid2item(args.fallback_tid2item_file)

    # Load existing item IDs
    existing_ids_npy = os.path.join(args.existing_emb_dir, "item_ids.npy")
    existing_ids_json = os.path.join(args.existing_emb_dir, "item_ids.json")
    if os.path.exists(existing_ids_npy):
        print(f"Loading existing item IDs from: {existing_ids_npy} (fast npy)")
        existing_item_ids_arr = np.load(existing_ids_npy)
        all_existing_ids = [str(x) for x in existing_item_ids_arr]
    else:
        print(f"Loading existing item IDs from: {existing_ids_json} (json)")
        with open(existing_ids_json, "r", encoding="utf-8") as f:
            all_existing_ids = json.load(f)
        try:
            ids_int = np.array([int(x) for x in all_existing_ids], dtype=np.int64)
            np.save(existing_ids_npy, ids_int)
            print(f"  Saved compact item_ids.npy ({ids_int.nbytes / 1e6:.1f} MB)")
        except (ValueError, OverflowError):
            pass
    print(f"  All existing products: {len(all_existing_ids):,}")

    # Filter to only products with TID mappings (so nearest neighbor always has TID)
    tid_set = set(existing_id2tid.keys())
    tid_indices = [i for i, gid in enumerate(all_existing_ids) if gid in tid_set]
    existing_item_ids = [all_existing_ids[i] for i in tid_indices]
    print(f"  Products with TID mapping: {len(existing_item_ids):,} "
          f"(filtered out {len(all_existing_ids) - len(existing_item_ids):,} without TID)")

    # Use filtered index file (different from full index)
    faiss_index_file_filtered = faiss_index_file.replace(".bin", "_tid_filtered.bin")

    if not args.rebuild_index and os.path.exists(faiss_index_file_filtered):
        index = load_faiss_index(faiss_index_file_filtered)
    else:
        # Build index from filtered embeddings only
        existing_emb_file = os.path.join(args.existing_emb_dir, "embeddings.npy")
        print(f"Loading existing embeddings from: {existing_emb_file}")
        all_embeddings = np.load(existing_emb_file)
        print(f"  Full shape: {all_embeddings.shape}")

        # Select only rows with TID
        tid_indices_arr = np.array(tid_indices, dtype=np.int64)
        filtered_embeddings = all_embeddings[tid_indices_arr]
        print(f"  Filtered shape: {filtered_embeddings.shape} "
              f"(only products with TID)")
        del all_embeddings

        index = build_faiss_index(
            filtered_embeddings,
            nlist=args.faiss_nlist,
            nprobe=args.faiss_nprobe,
            num_threads=args.faiss_threads,
        )

        save_faiss_index(index, faiss_index_file_filtered)
        del filtered_embeddings

    print(f"  FAISS index ready on CPU (ntotal={index.ntotal:,}, TID-filtered). "
          f"Will move to GPU after embedding generation.")

    # =========================================================================
    # Stage 2: Load New Products and Generate Embeddings
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 2: New Product Embeddings")
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

        # Validate alignment with new_data — missing items = failed in previous run
        new_data_ids = {item["id"] for item in new_data}
        loaded_ids_set = set(new_item_ids)
        missing = new_data_ids - loaded_ids_set
        extra = loaded_ids_set - new_data_ids

        if missing:
            print(f"  {len(missing):,} items missing from previous run, "
                  f"regenerating only those...")
            missing_items = [item for item in new_data
                            if item["id"] in missing]
            tmp_dir = os.path.join(args.output_dir, "_emb_tmp_retry")
            retry_ids, retry_embs = generate_embeddings_multi_gpu(
                missing_items,
                args.embedding_model,
                num_gpus=num_gpus,
                batch_size=max(args.batch_size // 2, 32),
                max_length=args.max_length,
                tmp_dir=tmp_dir,
            )
            if os.path.isdir(tmp_dir):
                try:
                    os.rmdir(tmp_dir)
                except OSError:
                    pass

            # Merge: combine loaded + retried, reorder to new_data order
            id_to_emb = {}
            for j, nid in enumerate(new_item_ids):
                id_to_emb[nid] = new_embeddings[j]
            for j, rid in enumerate(retry_ids):
                id_to_emb[rid] = retry_embs[j]

            new_item_ids = [item["id"] for item in new_data
                           if item["id"] in id_to_emb]
            new_embeddings = np.vstack(
                [id_to_emb[nid] for nid in new_item_ids])
            new_data = [item for item in new_data
                        if item["id"] in id_to_emb]
            still_missing = new_data_ids - set(new_item_ids)
            print(f"  Merged to {len(new_item_ids):,} items"
                  f"{f', still missing: {len(still_missing):,}' if still_missing else ''}")

        elif extra:
            print(f"  {len(extra):,} extra IDs not in new_data, filtering...")
            id_to_emb_idx = {nid: i for i, nid in enumerate(new_item_ids)}
            reordered_ids = []
            reordered_embs = []
            for item in new_data:
                if item["id"] in id_to_emb_idx:
                    reordered_ids.append(item["id"])
                    reordered_embs.append(
                        new_embeddings[id_to_emb_idx[item["id"]]])
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
    # Stage 3: ANN Search — Find Nearest Existing Product
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"Stage 3: ANN Search (Top-{args.faiss_top_k} Nearest Existing Products)")
    print("=" * 60)

    # FAISS search: try GPU first (much faster), fallback to CPU if OOM
    use_gpu_search = hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0
    if use_gpu_search:
        try:
            print(f"  Trying GPU FAISS search (float16, single GPU)...")
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            res.setTempMemory(512 * 1024 * 1024)  # 512MB temp
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
            print(f"  Moved to GPU 0 with float16, starting search...")
            search_start = time.time()
            D, I = search_nearest_neighbors(gpu_index, new_embeddings, k=args.faiss_top_k,
                                            batch_size=50000)
            search_elapsed = time.time() - search_start
            del gpu_index
            print(f"  GPU search completed in {search_elapsed:.2f}s")
        except Exception as e:
            print(f"  GPU FAISS search failed: {e}")
            print(f"  Falling back to CPU search...")
            use_gpu_search = False

    if not use_gpu_search:
        print(f"  Using CPU FAISS search (index ntotal={index.ntotal:,})")
        search_start = time.time()
        D, I = search_nearest_neighbors(index, new_embeddings, k=args.faiss_top_k)
        search_elapsed = time.time() - search_start

    # Print raw FAISS top-1 statistics
    top1_sims = D[:, 0]
    print(f"  Search completed in {search_elapsed:.2f}s")
    print(f"  FAISS Top-1 similarity statistics ({len(top1_sims):,} queries):")
    print(f"    Mean:   {np.mean(top1_sims):.4f}")
    print(f"    Median: {np.median(top1_sims):.4f}")
    print(f"    Min:    {np.min(top1_sims):.4f}")
    print(f"    Max:    {np.max(top1_sims):.4f}")
    print(f"    Std:    {np.std(top1_sims):.4f}")

    bins = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
    hist_top1, _ = np.histogram(top1_sims, bins=bins)
    print(f"\n  FAISS Top-1 similarity distribution:")
    for bi in range(len(bins) - 1):
        label = f"    [{bins[bi]:.2f}, {bins[bi+1]:.2f})"
        print(f"{label}: {hist_top1[bi]:>10,}")

    # =========================================================================
    # Stage 3.5: Reranking — Select Best from Top-5 (Plan E + B)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 3.5: Reranking Top-5 Candidates (Plan E + B)")
    print("=" * 60)

    # Load existing item metadata for attribute comparison
    item_file_path = os.path.join(
        os.path.dirname(args.existing_emb_dir.rstrip("/")),
        "raw_data", "item.json"
    )
    print(f"  Loading existing item metadata: {item_file_path}")
    existing_meta = {}  # gid -> (brand, seller, top_cat)
    with open(item_file_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    for _gid, _item in raw_items.items():
        _attrs = _item.get("attributes", {})
        _brand = ""
        _seller = ""
        if isinstance(_attrs, dict):
            _brand = str(_attrs.get("Brand", "")).strip().lower()
            _seller = str(_attrs.get("Seller", "")).strip().lower()
        _cat = extract_top_category(_item.get("categories", "")).lower()
        existing_meta[_gid] = (_brand, _seller, _cat)
    del raw_items
    print(f"    Loaded metadata for {len(existing_meta):,} existing items")

    # Pre-compute shared data
    new_data_dict = {item["id"]: item for item in new_data}
    total = len(new_item_ids)

    # Result arrays for reranked selection
    final_similarities = np.zeros(total, dtype=np.float32)
    final_matched_gids = [""] * total
    final_selection_methods = [""] * total
    final_original_ranks = np.full(total, -1, dtype=np.int32)

    # Reranking stats
    rerank_stats = Counter()
    rerank_position_counter = Counter()

    print(f"  Reranking {total:,} items (top-5 -> best candidate)...")
    print(f"    Shortcut: top-1 if any attr match & no category conflict")
    print(f"    Rerank bonuses: category +0.04, brand +0.02, seller +0.02")
    print(f"    Post-filter (B): reject if both have category but mismatch")
    rerank_start = time.time()

    for i in range(total):
        new_gid = new_item_ids[i]
        _ni = new_data_dict.get(new_gid, {})
        _na = _ni.get("attributes", {})
        if not isinstance(_na, dict):
            _na = {}
        new_brand = str(_na.get("Brand", "")).strip().lower()
        new_seller = str(_na.get("Seller", "")).strip().lower()
        new_top_cat = extract_top_category(_ni.get("categories", "")).lower()

        # Build candidate list from top-5 FAISS results
        candidates = []
        for j in range(D.shape[1]):
            idx = int(I[i, j])
            if idx < 0:
                continue
            cand_gid = existing_item_ids[idx]
            cand_sim = float(D[i, j])
            cand_brand, cand_seller, cand_top_cat = existing_meta.get(
                cand_gid, ("", "", ""))

            # Match: both non-empty and equal
            # Conflict: both non-empty and different
            # Neutral: either side is empty -> no match, no conflict, no bonus
            cat_match = bool(new_top_cat and cand_top_cat
                             and new_top_cat == cand_top_cat)
            brand_match = bool(new_brand and cand_brand
                               and new_brand == cand_brand)
            seller_match = bool(new_seller and cand_seller
                                and new_seller == cand_seller)
            cat_conflict = bool(new_top_cat and cand_top_cat
                                and new_top_cat != cand_top_cat)

            adjusted_sim = cand_sim
            if cat_match:
                adjusted_sim += 0.04
            if brand_match:
                adjusted_sim += 0.02
            if seller_match:
                adjusted_sim += 0.02

            candidates.append({
                "gid": cand_gid, "sim": cand_sim,
                "adjusted_sim": adjusted_sim, "original_rank": j,
                "cat_match": cat_match, "brand_match": brand_match,
                "seller_match": seller_match, "cat_conflict": cat_conflict,
            })

        if not candidates:
            final_selection_methods[i] = "no_candidate"
            rerank_stats["no_candidate"] += 1
            continue

        top1 = candidates[0]
        top1_any_match = (top1["cat_match"] or top1["brand_match"]
                          or top1["seller_match"])

        # Shortcut: top-1 has attribute match AND no category conflict
        if top1_any_match and not top1["cat_conflict"]:
            final_similarities[i] = top1["sim"]
            final_matched_gids[i] = top1["gid"]
            final_selection_methods[i] = "top1_shortcut"
            final_original_ranks[i] = 0
            rerank_stats["top1_shortcut"] += 1
            continue

        # Rerank: sort by adjusted score, pick first without cat conflict
        candidates.sort(key=lambda c: c["adjusted_sim"], reverse=True)
        selected = None
        for c in candidates:
            if not c["cat_conflict"]:
                selected = c
                break

        if selected is None:
            # All 5 candidates have category conflict -> cat_filtered
            final_similarities[i] = top1["sim"]
            final_matched_gids[i] = top1["gid"]
            final_selection_methods[i] = "cat_filtered"
            final_original_ranks[i] = top1["original_rank"]
            rerank_stats["cat_filtered"] += 1
            continue

        final_similarities[i] = selected["sim"]
        final_matched_gids[i] = selected["gid"]
        final_original_ranks[i] = selected["original_rank"]

        if selected["original_rank"] == 0:
            final_selection_methods[i] = "top1_reranked"
            rerank_stats["top1_reranked"] += 1
        else:
            final_selection_methods[i] = "reranked"
            rerank_stats["reranked"] += 1
            rerank_position_counter[selected["original_rank"]] += 1

    rerank_elapsed = time.time() - rerank_start
    print(f"\n  Reranking completed in {rerank_elapsed:.2f}s")
    print(f"\n  Selection method breakdown:")
    for method in ["top1_shortcut", "top1_reranked", "reranked",
                    "cat_filtered", "no_candidate"]:
        count = rerank_stats.get(method, 0)
        print(f"    {method:20s}: {count:>10,} ({count / total * 100:.2f}%)")
    if rerank_position_counter:
        print(f"\n  Reranked items - original FAISS rank of selected:")
        for rank_val in sorted(rerank_position_counter.keys()):
            count = rerank_position_counter[rank_val]
            pct = count / max(rerank_stats.get("reranked", 1), 1) * 100
            print(f"    Rank {rank_val}: {count:>10,} ({pct:.1f}% of reranked)")

    # Final similarity stats (after reranking)
    print(f"\n  Final similarity statistics (after reranking):")
    print(f"    Mean:   {np.mean(final_similarities):.4f}")
    print(f"    Median: {np.median(final_similarities):.4f}")
    print(f"    Min:    {np.min(final_similarities):.4f}")
    print(f"    Max:    {np.max(final_similarities):.4f}")

    hist, _ = np.histogram(final_similarities, bins=bins)
    print(f"\n  Final similarity distribution:")
    for bi in range(len(bins) - 1):
        label = f"    [{bins[bi]:.2f}, {bins[bi+1]:.2f})"
        print(f"{label}: {hist[bi]:>10,}")

    # Save mapping details for case study reuse
    mapping_file = os.path.join(args.output_dir, "mapping_details.npz")
    np.savez_compressed(
        mapping_file,
        similarities=final_similarities,
        top5_D=D,
        top5_I=I,
        final_original_ranks=final_original_ranks,
    )
    matched_gids_file = os.path.join(args.output_dir, "matched_existing_gids.json")
    with open(matched_gids_file, "w", encoding="utf-8") as f:
        json.dump(final_matched_gids, f)
    print(f"  Saved mapping details to: {mapping_file}")

    # =========================================================================
    # Stage 4: Assign TIDs & Save — for each threshold
    # =========================================================================

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
        print(f"Stage 4: Assign TIDs (threshold={thresh})")
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
        cat_filter_reject_count = 0

        unmatched_items = []
        similarity_records = {
            "above_threshold": [],
            "below_threshold_fallback": [],
            "below_threshold_unmatched": [],
        }

        for i in range(total):
            new_gid = new_item_ids[i]
            sim_score = float(final_similarities[i])
            matched_existing_gid = final_matched_gids[i] or None
            selection_method = final_selection_methods[i]

            tid_words = None
            source = None

            if selection_method == "cat_filtered":
                cat_filter_reject_count += 1

            # Category-filtered items skip similarity matching -> fallback
            if (selection_method != "cat_filtered"
                    and sim_score >= thresh and matched_existing_gid):
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
        print(f"    Category filter rejects:     {cat_filter_reject_count:>10,} "
              f"({cat_filter_reject_count / total * 100:.2f}%)")
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
                "mean": float(np.mean(final_similarities)),
                "median": float(np.median(final_similarities)),
                "min": float(np.min(final_similarities)),
                "max": float(np.max(final_similarities)),
                "std": float(np.std(final_similarities)),
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
            "reranking_stats": {
                "top1_shortcut": rerank_stats.get("top1_shortcut", 0),
                "top1_reranked": rerank_stats.get("top1_reranked", 0),
                "reranked": rerank_stats.get("reranked", 0),
                "cat_filtered": rerank_stats.get("cat_filtered", 0),
                "no_candidate": rerank_stats.get("no_candidate", 0),
                "rerank_position_distribution": dict(rerank_position_counter),
            },
            "cat_filter_reject_count": cat_filter_reject_count,
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

    # =========================================================================
    # Case Study: print detailed mapping examples
    # =========================================================================
    if args.case_study_samples > 0:
        import random as _random
        _random.seed(SEED)
        n_samples = min(args.case_study_samples, total)
        sample_indices = sorted(_random.sample(range(total), n_samples))

        # Build new_data lookup by id
        new_data_by_id = {item["id"]: item for item in new_data}
        # Build existing item lookup (try to load item file for titles)
        existing_item_titles = {}
        # Try to get titles from the item file used for existing products
        item_file_path = os.path.join(
            os.path.dirname(args.existing_emb_dir.rstrip("/")),
            "raw_data", "item.json"
        )
        if os.path.exists(item_file_path):
            print(f"\n  Loading existing item titles for case study...")
            with open(item_file_path, "r", encoding="utf-8") as f:
                _item_data = json.load(f)
            for _id, _v in _item_data.items():
                if isinstance(_v, dict) and "title" in _v:
                    existing_item_titles[str(_id)] = _v["title"]
            del _item_data
            print(f"    Loaded {len(existing_item_titles):,} titles")

        # Use the smallest threshold for case study
        case_thresh = thresholds[0]

        print(f"\n{'#'*80}")
        print(f"  CASE STUDY: {n_samples} sampled new products")
        print(f"  (threshold={case_thresh} for similarity/fallback label)")
        print(f"{'#'*80}")

        for rank, idx in enumerate(sample_indices, 1):
            gid = new_item_ids[idx]
            sim = float(final_similarities[idx])
            matched_gid = final_matched_gids[idx] or "N/A"

            # New product info
            new_item = new_data_by_id.get(gid, {})
            new_title = new_item.get("title", "N/A")
            new_cats = new_item.get("categories", "N/A")
            new_brand = new_item.get("attributes", {}).get("Brand", "N/A") if isinstance(new_item.get("attributes"), dict) else "N/A"

            # Existing product info
            ex_title = existing_item_titles.get(str(matched_gid), "N/A")
            ex_tid = existing_id2tid.get(str(matched_gid), [])

            # What TID was assigned?
            assigned_tid = None
            source = "unmatched"
            if sim >= case_thresh and str(matched_gid) in existing_id2tid:
                assigned_tid = existing_id2tid[str(matched_gid)]
                source = "similarity"
            elif gid in fallback_gid2tid:
                assigned_tid = fallback_gid2tid[gid]
                source = "fallback"

            print(f"\n  [{rank}/{n_samples}] New GID: {gid}")
            print(f"    Title:      {new_title[:120]}")
            print(f"    Categories: {new_cats[:100]}")
            print(f"    Brand:      {new_brand}")
            print(f"    ---")
            print(f"    Nearest existing GID: {matched_gid}")
            print(f"    Cosine similarity:    {sim:.4f}")
            print(f"    Existing title:       {ex_title[:120]}")
            print(f"    Existing TID:         {','.join(ex_tid) if ex_tid else 'N/A'}")
            print(f"    ---")
            print(f"    Assigned source:      {source}")
            print(f"    Assigned TID:         {','.join(assigned_tid) if assigned_tid else 'NONE'}")

        # Save case study to file
        case_study_file = os.path.join(args.output_dir, "case_study.jsonl")
        with open(case_study_file, "w", encoding="utf-8") as f:
            for idx in sample_indices:
                gid = new_item_ids[idx]
                sim = float(final_similarities[idx])
                matched_gid = final_matched_gids[idx] or None
                new_item = new_data_by_id.get(gid, {})

                assigned_tid = None
                source = "unmatched"
                if sim >= case_thresh and matched_gid and str(matched_gid) in existing_id2tid:
                    assigned_tid = existing_id2tid[str(matched_gid)]
                    source = "similarity"
                elif gid in fallback_gid2tid:
                    assigned_tid = fallback_gid2tid[gid]
                    source = "fallback"

                record = {
                    "new_gid": gid,
                    "new_title": new_item.get("title", ""),
                    "new_categories": new_item.get("categories", ""),
                    "new_brand": new_item.get("attributes", {}).get("Brand", "") if isinstance(new_item.get("attributes"), dict) else "",
                    "nearest_existing_gid": str(matched_gid) if matched_gid else "",
                    "cosine_similarity": round(sim, 4),
                    "existing_title": existing_item_titles.get(str(matched_gid), "") if matched_gid else "",
                    "existing_tid": existing_id2tid.get(str(matched_gid), []) if matched_gid else [],
                    "assigned_source": source,
                    "assigned_tid": assigned_tid or [],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n  Case study saved to: {case_study_file}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
