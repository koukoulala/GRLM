"""Step 0: Generate Item Embeddings & Compute Similarities

Uses a pre-trained embedding model (e.g., Qwen3-Embedding-0.6B) to generate
item embeddings via multi-GPU parallel processing, then computes top-k
cosine similarities using FAISS ANN (Approximate Nearest Neighbor) with GPU.

Usage:
    python s0_init_emb.py \
        --item_file ./raw_data/item.json \
        --output_dir ./processed/sum_data \
        --embedding_model /path/to/Qwen3-Embedding-0.6B \
        --num_gpus 2 --batch_size 64 --top_k 20 --max_length 512
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
import warnings
import tempfile
import ctypes

# Pre-load libfaiss.so from the local ROCm build so `import faiss` can find it
_FAISS_LIB_PATHS = [
    "/scratch/workspaceblobstore/users/xiaoyukou/faiss-gpu-rocm/build/faiss/libfaiss.so",
    "/home/aiscuser/.local/lib/python3.12/site-packages/faiss/libfaiss_python_callbacks.so",
]
for _lib in _FAISS_LIB_PATHS:
    if os.path.exists(_lib):
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)

import faiss

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_data(file_path: str) -> List[Dict]:
    """Load JSON file and convert to list of dictionaries.

    Uses orjson (Rust-based, ~5-10x faster than stdlib json) with a
    progress bar on the file read. Falls back to stdlib json.
    """
    file_size = os.path.getsize(file_path)
    print(f"  File size: {file_size / 1e9:.2f} GB")

    # Read raw bytes with progress bar
    print(f"  Reading file into memory ...")
    buf = bytearray()
    with open(file_path, "rb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True,
                  desc="  Reading", mininterval=2) as pbar:
            while True:
                chunk = f.read(64 * 1024 * 1024)  # 64MB chunks
                if not chunk:
                    break
                buf.extend(chunk)
                pbar.update(len(chunk))

    # Parse JSON
    print(f"  Parsing JSON ({len(buf) / 1e9:.2f} GB) ...")
    parse_start = time.time()
    try:
        import orjson
        data = orjson.loads(buf)
    except ImportError:
        data = json.loads(buf)
    del buf  # free memory
    parse_elapsed = time.time() - parse_start
    print(f"  Parsed in {parse_elapsed:.1f}s")

    # Convert {id: {...}} dict to [{"id": id, ...}] list
    result_list = []
    for key, value in data.items():
        value["id"] = key
        result_list.append(value)
    del data
    return result_list


def _fast_json_load(file_path: str):
    """Load JSON using orjson (fast) with fallback to stdlib json."""
    t0 = time.time()
    with open(file_path, "rb") as f:
        raw = f.read()
    try:
        import orjson
        result = orjson.loads(raw)
    except ImportError:
        result = json.loads(raw)
    print(f"  Loaded {file_path} ({len(raw) / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
    return result


def _fast_json_save(obj, file_path: str):
    """Save JSON using orjson (fast, compact) with fallback to stdlib json."""
    t0 = time.time()
    try:
        import orjson
        raw = orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)
    except ImportError:
        raw = json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode("utf-8")
    with open(file_path, "wb") as f:
        f.write(raw)
    print(f"  Saved {file_path} ({len(raw) / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


def prepare_text_for_embedding(item: Dict, max_field_len: int = 500) -> str:
    """Prepare text for embedding generation.

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
        # Single item still fails — record as failed
        print(f"Rank {rank}: Single item FAILED, marking as NaN")
        return None, [0]

    # Halve and recurse
    mid = n // 2
    print(f"Rank {rank}: OOM with {n} items, splitting to {mid} + {n - mid}")

    emb1, fail1 = _encode_batch_with_retry(
        batch_texts[:mid], tokenizer, model, device, max_length, rank)
    emb2, fail2 = _encode_batch_with_retry(
        batch_texts[mid:], tokenizer, model, device, max_length, rank)

    # Shift fail2 indices by mid
    fail2_shifted = [idx + mid for idx in fail2]
    all_failed = fail1 + fail2_shifted

    # Build combined result
    if emb1 is not None and emb2 is not None:
        combined = np.vstack([emb1, emb2])
    elif emb1 is not None:
        # emb2 is None (single-item failure) — fill NaN placeholder
        dim = emb1.shape[1]
        placeholder = np.full((n - mid, dim), np.nan, dtype=np.float32)
        combined = np.vstack([emb1, placeholder])
    elif emb2 is not None:
        dim = emb2.shape[1]
        placeholder = np.full((mid, dim), np.nan, dtype=np.float32)
        combined = np.vstack([placeholder, emb2])
    else:
        # Both halves fully failed
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
    failed_indices = []  # track failed batch ranges (in sorted order)

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
        # Rebuild sorted_embeddings from checkpoint chunks
        rows_done = start_batch * batch_size
        sorted_embeddings = [ckpt_emb[:rows_done]] if rows_done > 0 else []
        print(f"Rank {rank}: Resuming from batch {start_batch} "
              f"({rows_done:,} items done, {len(failed_indices)} failed)")

    CHECKPOINT_EVERY = 500  # save checkpoint every N batches

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
            # Entire batch failed (all items) — use NaN placeholder
            dim = sorted_embeddings[-1].shape[1] if sorted_embeddings else 896
            sorted_embeddings.append(
                np.full((actual_bs, dim), np.nan, dtype=np.float32))
        if batch_failed:
            # Map local batch indices to global sorted indices
            failed_indices.extend([i + fi for fi in batch_failed])
            print(f"Rank {rank}: Batch {batch_idx}: "
                  f"{len(batch_failed)} items failed")

        # Periodic checkpoint
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
    # Save failed indices for downstream handling
    if failed_indices:
        with open(os.path.join(tmp_dir, f"failed_rank{rank}.json"), "w") as f:
            json.dump(failed_indices, f)
    # Clean up checkpoint after successful completion
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

    # Collect results from disk in rank order (P3)
    all_embeddings = []
    all_item_ids = []
    for rank in range(num_gpus):
        emb = np.load(os.path.join(tmp_dir, f"embeddings_rank{rank}.npy"))
        with open(os.path.join(tmp_dir, f"item_ids_rank{rank}.json"), "r") as f:
            ids = json.load(f)
        print(f"Loaded {len(ids)} items from Rank {rank}")
        all_embeddings.append(emb)
        all_item_ids.extend(ids)
        # Clean up temp files
        os.remove(os.path.join(tmp_dir, f"embeddings_rank{rank}.npy"))
        os.remove(os.path.join(tmp_dir, f"item_ids_rank{rank}.json"))

    combined_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    print(f"Multi-GPU processing completed in {time.time() - start_time:.2f}s")
    return all_item_ids, combined_embeddings


def compute_similarities_faiss(
    embeddings: np.ndarray,
    item_ids: List[str],
    k: int = 20,
    faiss_gpu_id: int = 0,
    nlist: int = 4096,
    nprobe: int = 128,
    num_threads: int = 0,
    query_indices: np.ndarray = None,
    index_output_path: str = None,
) -> Dict:
    """Compute top-k cosine similarities using FAISS ANN.

    Automatically uses GPU if faiss-gpu is available, otherwise falls back to CPU.

    FAISS IVFFlat works in 3 stages:
    1. Train: K-means clusters the embedding space into `nlist` Voronoi cells.
       Each embedding is assigned to its nearest centroid.
    2. Add: All embeddings are inserted into the index, stored within their
       assigned cluster.
    3. Search: For each query, only `nprobe` nearest clusters are searched
       instead of all N items, reducing complexity from O(N) to O(N/nlist * nprobe).

    Since embeddings are L2-normalized, inner product (IP) equals cosine similarity.

    Args:
        embeddings: L2-normalized float32 array of shape (N, dim).
        item_ids: List of item ID strings, aligned with embeddings.
        k: Number of top similar items to return per item.
        faiss_gpu_id: Which GPU to use for FAISS search (ignored if GPU unavailable).
        nlist: Number of IVF clusters (higher = faster but less accurate).
        nprobe: Number of clusters to search (higher = more accurate but slower).
        num_threads: Number of CPU threads for FAISS OpenMP parallelism.
            0 means use all available cores. Only effective on CPU.
        query_indices: Optional numpy array of indices into embeddings/item_ids
            to search for. If None, searches for all items. The index is
            always built on ALL embeddings so neighbors come from the full set.
        index_output_path: If set, save the trained FAISS CPU index to this
            path for later reuse (e.g., querying with new items).
    """
    n, dim = embeddings.shape

    # Set CPU thread parallelism (helps both CPU-only and GPU pre/post-processing)
    if num_threads > 0:
        faiss.omp_set_num_threads(num_threads)
        print(f"  FAISS OpenMP threads set to {num_threads}")
    else:
        # Use all available cores
        import os as _os
        cpu_count = _os.cpu_count() or 1
        faiss.omp_set_num_threads(cpu_count)
        print(f"  FAISS OpenMP threads set to {cpu_count} (all cores)")

    # Check if FAISS GPU is available
    use_gpu = hasattr(faiss, "StandardGpuResources")
    num_faiss_gpus = faiss.get_num_gpus() if use_gpu else 0
    if num_faiss_gpus > 0:
        mode_str = f"FAISS GPU (x{num_faiss_gpus})"
    else:
        mode_str = "FAISS CPU"
        use_gpu = False

    # Determine query set
    if query_indices is not None:
        query_embeddings = embeddings[query_indices]
        query_global_indices = query_indices  # maps local query idx -> global idx
        num_queries = len(query_indices)
        print(f"Computing Top-{k} similarities for {num_queries} query items "
              f"(index built on {n} items) using {mode_str} ...")
    else:
        query_embeddings = embeddings
        query_global_indices = np.arange(n)
        num_queries = n
        print(f"Computing Top-{k} similarities for {n} items using {mode_str} ...")
    print(f"  Index: IVFFlat, nlist={nlist}, nprobe={nprobe}, dim={dim}")
    start_time = time.time()

    # Ensure contiguous float32
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Adjust nlist for small datasets
    effective_nlist = min(nlist, n // 40) if n > 0 else 1
    effective_nlist = max(effective_nlist, 1)

    # Build IVFFlat index for inner product (cosine sim on normalized vecs)
    quantizer = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT)

    # Try to train/add/search entirely on GPU for maximum speed
    index = cpu_index
    if use_gpu:
        try:
            if num_faiss_gpus > 1:
                print(f"  Cloning index to all {num_faiss_gpus} GPUs for train+add+search...")
                index = faiss.index_cpu_to_all_gpus(cpu_index)
            else:
                gpu_res = faiss.StandardGpuResources()
                print(f"  Moving index to GPU {faiss_gpu_id} for train+add+search...")
                index = faiss.index_cpu_to_gpu(gpu_res, faiss_gpu_id, cpu_index)
            print(f"  Training index on GPU with {n} vectors ...")
            index.train(embeddings)
            print(f"  Adding {n} vectors to GPU index ...")
            index.add(embeddings)
        except Exception as e:
            print(f"  GPU FAISS train/add failed: {e}")
            print(f"  Falling back to CPU train+add, then GPU search")
            index = cpu_index
            use_gpu = False
            print(f"  Training index on CPU with {n} vectors ...")
            cpu_index.train(embeddings)
            print(f"  Adding {n} vectors to CPU index ...")
            cpu_index.add(embeddings)
    else:
        print(f"  Training index on CPU with {n} vectors ...")
        cpu_index.train(embeddings)
        print(f"  Adding {n} vectors to CPU index ...")
        cpu_index.add(embeddings)

    cpu_index.nprobe = min(nprobe, effective_nlist)

    # Save the trained CPU index to disk for later reuse
    if index_output_path:
        try:
            if use_gpu and index is not cpu_index:
                # GPU->CPU transfer can fail for very large indices
                # (std::vector max_size limit). Fall back to rebuilding
                # a CPU index from scratch.
                try:
                    cpu_index = faiss.index_gpu_to_cpu(index)
                except RuntimeError:
                    print(f"  index_gpu_to_cpu failed (index too large), "
                          f"rebuilding CPU index for saving...")
                    quantizer_cpu = faiss.IndexFlatIP(dim)
                    cpu_index = faiss.IndexIVFFlat(
                        quantizer_cpu, dim, effective_nlist,
                        faiss.METRIC_INNER_PRODUCT)
                    cpu_index.train(embeddings)
                    cpu_index.add(embeddings)
                cpu_index.nprobe = min(nprobe, effective_nlist)
            faiss.write_index(cpu_index, index_output_path)
            print(f"  Saved FAISS index to: {index_output_path}")
        except Exception as e:
            print(f"  WARNING: Failed to save FAISS index: {e}")
            print(f"  Continuing with search (index saving is optional)...")

    # Batch search: search k+1 to exclude self, then filter
    print(f"  Searching top-{k+1} neighbors for {num_queries} queries ...")
    search_k = min(k + 1, n)
    search_batch_size = 100000
    all_distances = []
    all_indices = []
    num_batches = (num_queries + search_batch_size - 1) // search_batch_size

    search_failed_gpu = False
    for start in tqdm(range(0, num_queries, search_batch_size),
                      total=num_batches, desc="  FAISS search",
                      dynamic_ncols=True):
        end = min(start + search_batch_size, num_queries)
        try:
            D_batch, I_batch = index.search(query_embeddings[start:end], search_k)
        except Exception as e:
            if use_gpu and not search_failed_gpu:
                print(f"\n  GPU search failed at batch {start}: {e}")
                print(f"  Falling back to CPU for remaining batches...")
                index = cpu_index
                search_failed_gpu = True
                D_batch, I_batch = index.search(
                    query_embeddings[start:end], search_k)
            else:
                raise
        all_distances.append(D_batch)
        all_indices.append(I_batch)

    if search_failed_gpu:
        # Re-search the earlier GPU batches on CPU for consistency
        print(f"  Re-searching first {len(all_distances)-1} batches on CPU "
              f"for consistency...")
        all_distances = []
        all_indices = []
        for start in range(0, num_queries, search_batch_size):
            end = min(start + search_batch_size, num_queries)
            D_batch, I_batch = cpu_index.search(
                query_embeddings[start:end], search_k)
            all_distances.append(D_batch)
            all_indices.append(I_batch)

    D = np.vstack(all_distances)
    I = np.vstack(all_indices)

    # Build results dict, excluding self-matches
    results = {}
    for qi in tqdm(range(num_queries), desc="  Building results", dynamic_ncols=True):
        global_idx = int(query_global_indices[qi])
        similar_items = []
        for j in range(search_k):
            idx = int(I[qi, j])
            if idx != global_idx and idx >= 0:
                similar_items.append(
                    {
                        "item_id": item_ids[idx],
                        "similarity": float(D[qi, j]),
                    }
                )
            if len(similar_items) >= k:
                break
        results[item_ids[global_idx]] = similar_items

    elapsed = time.time() - start_time
    print(f"FAISS similarity search finished in {elapsed:.2f}s")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate item embeddings and compute similarities"
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/item.json",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/item.json",
        help="Path to item metadata JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/processed/",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/processed/",
        help="Directory to save similarity results",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3-Embedding-0.6B",
        help="Path to embedding model",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs (default: all available)",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size per GPU")
    parser.add_argument(
        "--top_k", type=int, default=8, help="Top-k similar items to compute"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum token length for tokenizer truncation",
    )
    parser.add_argument(
        "--faiss_nlist",
        type=int,
        default=4096,
        help="Number of IVF clusters for FAISS (higher=faster, less accurate)",
    )
    parser.add_argument(
        "--faiss_nprobe",
        type=int,
        default=64,
        help="Number of clusters to probe during FAISS search (higher=more accurate)",
    )
    parser.add_argument(
        "--faiss_threads",
        type=int,
        default=0,
        help="Number of CPU threads for FAISS (0=all cores, default: 0)",
    )
    parser.add_argument(
        "--resume_from_dir",
        type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/processed/",
        help="If set and files exist, reuses existing embeddings and only generates embeddings for new items."
    )
    return parser.parse_args()


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

    print(f"Loading data: {args.item_file}")
    data = load_data(args.item_file)
    print(f"Loaded {len(data)} items")

    if len(data) == 0:
        print("Error: No data loaded!")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    emb_file = os.path.join(args.output_dir, "embeddings.npy")
    ids_file = os.path.join(args.output_dir, "item_ids.json")
    index_file = os.path.join(args.output_dir, "faiss_index.bin")
    output_file = os.path.join(args.output_dir, "similarities.json")

    current_ids = [item["id"] for item in data]
    current_id_set = set(current_ids)

    # =========================================================================
    # Stage 1: Embedding generation (with resume support)
    # =========================================================================
    resume_dir = args.resume_from_dir.strip() if args.resume_from_dir else ""
    resume_emb_file = os.path.join(resume_dir, "embeddings.npy") if resume_dir else ""
    resume_ids_file = os.path.join(resume_dir, "item_ids.json") if resume_dir else ""
    resume_sim_file = os.path.join(resume_dir, "similarities.json") if resume_dir else ""

    new_id_set = set()  # track new items for incremental similarity

    if resume_dir and os.path.exists(resume_emb_file) and os.path.exists(resume_ids_file):
        print(f"\n[RESUME] Loading existing embeddings from: {resume_dir}")
        prev_embeddings = np.load(resume_emb_file)
        with open(resume_ids_file, "r", encoding="utf-8") as f:
            prev_ids = json.load(f)
        print(f"  Previous embeddings: {len(prev_ids):,} items, "
              f"shape={prev_embeddings.shape}")

        # Build mapping: prev_id -> index in prev_embeddings
        prev_id_set = set(prev_ids)
        prev_id_to_idx = {pid: i for i, pid in enumerate(prev_ids)}

        # Find new items that need embedding
        new_items = [item for item in data if item["id"] not in prev_id_set]
        # Find items to keep from previous (still in current data)
        kept_prev_ids = [pid for pid in prev_ids if pid in current_id_set]
        removed_count = len(prev_ids) - len(kept_prev_ids)

        print(f"  Items in current input:   {len(data):>10,}")
        print(f"  Already have embeddings:  {len(kept_prev_ids):>10,}")
        print(f"  Removed (not in input):   {removed_count:>10,}")
        print(f"  New items to process:     {len(new_items):>10,}")

        if new_items:
            # Generate embeddings only for new items
            print(f"\n  Generating embeddings for {len(new_items):,} new items...")
            tmp_dir = os.path.join(args.output_dir, "_emb_tmp")
            new_item_ids, new_embeddings = generate_embeddings_multi_gpu(
                new_items,
                args.embedding_model,
                num_gpus=num_gpus,
                batch_size=args.batch_size,
                max_length=args.max_length,
                tmp_dir=tmp_dir,
            )
            print(f"  New embeddings shape: {new_embeddings.shape}")
            new_id_set = set(new_item_ids)

            # Merge: build arrays in current data order
            new_id_to_idx = {nid: i for i, nid in enumerate(new_item_ids)}
            dim = prev_embeddings.shape[1]
            merged_embeddings = np.zeros(
                (len(data), dim), dtype=np.float32
            )
            merged_ids = []
            for i, item in enumerate(data):
                item_id = item["id"]
                merged_ids.append(item_id)
                if item_id in prev_id_to_idx:
                    merged_embeddings[i] = prev_embeddings[
                        prev_id_to_idx[item_id]
                    ]
                elif item_id in new_id_to_idx:
                    merged_embeddings[i] = new_embeddings[
                        new_id_to_idx[item_id]
                    ]

            item_ids = merged_ids
            embeddings = merged_embeddings
        else:
            # All items already have embeddings, just reorder to match
            # current data order
            print(f"  All items already have embeddings, reordering...")
            dim = prev_embeddings.shape[1]
            embeddings = np.zeros((len(data), dim), dtype=np.float32)
            item_ids = []
            for i, item in enumerate(data):
                item_id = item["id"]
                item_ids.append(item_id)
                if item_id in prev_id_to_idx:
                    embeddings[i] = prev_embeddings[
                        prev_id_to_idx[item_id]
                    ]
    else:
        # Full run: generate all embeddings
        tmp_dir = os.path.join(args.output_dir, "_emb_tmp")
        item_ids, embeddings = generate_embeddings_multi_gpu(
            data,
            args.embedding_model,
            num_gpus=num_gpus,
            batch_size=args.batch_size,
            max_length=args.max_length,
            tmp_dir=tmp_dir,
        )

    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Save embeddings
    np.save(emb_file, embeddings)
    print(f"  Saving item IDs to: {ids_file}")
    _fast_json_save(item_ids, ids_file)
    print(f"Saved embeddings to: {emb_file} ({embeddings.nbytes / 1e9:.2f} GB)")
    print(f"Saved item IDs to: {ids_file}")

    # =========================================================================
    # Stage 2: FAISS similarity search
    # =========================================================================
    # If resuming and we have previous similarities, only compute for new items
    # and merge with previous results.
    prev_similarities = {}
    if resume_dir and new_id_set and os.path.exists(resume_sim_file):
        print(f"\n[RESUME] Loading previous similarities from: {resume_sim_file}")
        prev_similarities = _fast_json_load(resume_sim_file)
        print(f"  Previous similarity entries: {len(prev_similarities):,}")

        # Filter: keep only entries for items still in current data
        prev_similarities = {
            k: v for k, v in prev_similarities.items() if k in current_id_set
        }
        # Also filter out neighbors that no longer exist
        for k in prev_similarities:
            prev_similarities[k] = [
                nb for nb in prev_similarities[k]
                if nb["item_id"] in current_id_set
            ]
        print(f"  Kept similarity entries (in current data): {len(prev_similarities):,}")

    if new_id_set:
        # Build query_indices: indices of new items in the merged arrays
        id_to_global_idx = {iid: i for i, iid in enumerate(item_ids)}
        query_indices = np.array(
            [id_to_global_idx[nid] for nid in new_id_set if nid in id_to_global_idx],
            dtype=np.int64,
        )
        print(f"\n  Computing similarities for {len(query_indices):,} new items only...")

        new_similarity_results = compute_similarities_faiss(
            embeddings,
            item_ids,
            k=args.top_k,
            faiss_gpu_id=0,
            nlist=args.faiss_nlist,
            nprobe=args.faiss_nprobe,
            num_threads=args.faiss_threads,
            query_indices=query_indices,
            index_output_path=index_file,
        )

        # Merge: previous similarities for old items + new results for new items
        similarity_results = prev_similarities
        similarity_results.update(new_similarity_results)
        print(f"  Merged similarities: {len(prev_similarities):,} old + "
              f"{len(new_similarity_results):,} new = {len(similarity_results):,} total")
    elif prev_similarities:
        # No new items, but we loaded previous similarities — just use them
        similarity_results = prev_similarities
        print(f"  No new items, using {len(similarity_results):,} previous similarities")
    else:
        # Full run: compute similarities for all items
        similarity_results = compute_similarities_faiss(
            embeddings,
            item_ids,
            k=args.top_k,
            faiss_gpu_id=0,
            nlist=args.faiss_nlist,
            nprobe=args.faiss_nprobe,
            num_threads=args.faiss_threads,
            index_output_path=index_file,
        )

    print(f"Saving similarity results to: {output_file}")
    _fast_json_save(similarity_results, output_file)

    # Clean up temp dir
    tmp_dir = os.path.join(args.output_dir, "_emb_tmp")
    if os.path.isdir(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)

    print("Processing completed!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
                                                                                                                                                                                                                                                                                                                                                                                                    