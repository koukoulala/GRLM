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
import faiss
import tempfile

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_data(file_path: str) -> List[Dict]:
    """Load JSON file and convert to list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result_list = []
    for key, value in data.items():
        new_item = {"id": key}
        new_item.update(value)
        result_list.append(new_item)
    return result_list


def prepare_text_for_embedding(item: Dict) -> str:
    """Prepare text for embedding generation."""
    text_parts = []
    for field in ["title", "description", "categories", "related_queries"]:
        val = item.get(field, "")
        if val:
            text_parts.append(f"{field.capitalize()}: {val}")
    return " | ".join(text_parts)


def process_batch_on_gpu(
    rank: int,
    data_slice: List[Dict],
    tmp_dir: str,
    model_name: str,
    batch_size: int = 16,
    max_length: int = 512,
):
    """Process data slice on specific GPU, save results to disk."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}: Loading model on {device}...")

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{rank}",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    texts = [prepare_text_for_embedding(item) for item in data_slice]
    item_ids = [item["id"] for item in data_slice]
    all_embeddings = []

    print(f"Rank {rank}: Generating embeddings for {len(texts)} items...")
    start_time = time.time()

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Rank {rank}"):
        batch_texts = texts[i : i + batch_size]

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

            # Mean pooling
            attention_mask = inputs["attention_mask"]
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

            # L2 normalization
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_np = embeddings.cpu().numpy().astype(np.float32)
            all_embeddings.append(embeddings_np)

    embeddings_array = np.vstack(all_embeddings) if all_embeddings else np.array([])
    elapsed = time.time() - start_time
    print(f"Rank {rank}: Done in {elapsed:.2f}s")

    # Save to disk instead of passing through Queue (P3)
    np.save(os.path.join(tmp_dir, f"embeddings_rank{rank}.npy"), embeddings_array)
    with open(os.path.join(tmp_dir, f"item_ids_rank{rank}.json"), "w") as f:
        json.dump(item_ids, f)


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
) -> Dict:
    """Compute top-k cosine similarities using FAISS ANN with GPU acceleration.

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
        faiss_gpu_id: Which GPU to use for FAISS search.
        nlist: Number of IVF clusters (higher = faster but less accurate).
        nprobe: Number of clusters to search (higher = more accurate but slower).
    """
    n, dim = embeddings.shape
    print(f"Computing Top-{k} similarities for {n} items using FAISS GPU ...")
    print(f"  Index: IVFFlat, nlist={nlist}, nprobe={nprobe}, dim={dim}")
    start_time = time.time()

    # Ensure contiguous float32
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Adjust nlist for small datasets
    effective_nlist = min(nlist, n // 40) if n > 0 else 1
    effective_nlist = max(effective_nlist, 1)

    # Build IVFFlat index on GPU for inner product (cosine sim on normalized vecs)
    quantizer = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT)

    # Move index to GPU for faster training and search
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, faiss_gpu_id, cpu_index)

    print(f"  Training index on {n} vectors ...")
    gpu_index.train(embeddings)
    print(f"  Adding {n} vectors to index ...")
    gpu_index.add(embeddings)
    gpu_index.nprobe = min(nprobe, effective_nlist)

    # Batch search: search k+1 to exclude self, then filter
    print(f"  Searching top-{k+1} neighbors ...")
    search_k = min(k + 1, n)
    # Search in batches to avoid GPU OOM on very large datasets
    search_batch_size = 100000
    all_distances = []
    all_indices = []
    for start in range(0, n, search_batch_size):
        end = min(start + search_batch_size, n)
        D_batch, I_batch = gpu_index.search(embeddings[start:end], search_k)
        all_distances.append(D_batch)
        all_indices.append(I_batch)
    D = np.vstack(all_distances)
    I = np.vstack(all_indices)

    # Build results dict, excluding self-matches
    results = {}
    for i in range(n):
        similar_items = []
        for j in range(search_k):
            idx = int(I[i, j])
            if idx != i and idx >= 0:
                similar_items.append(
                    {
                        "item_id": item_ids[idx],
                        "similarity": float(D[i, j]),
                    }
                )
            if len(similar_items) >= k:
                break
        results[item_ids[i]] = similar_items

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
        default="./raw_data/merged_clean_item.json",
        help="Path to item metadata JSON file (e.g., ./raw_data/item.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed/",
        help="Directory to save similarity results (e.g., ./processed/sum_data)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        #default="/data/xiaoyukou/ckpts/Qwen3-Embedding-0.6B",
        default="/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3-Embedding-0.6B",
        help="Path to embedding model",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs (default: all available)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument(
        "--top_k", type=int, default=10, help="Top-k similar items to compute"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
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
        default=128,
        help="Number of clusters to probe during FAISS search (higher=more accurate)",
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

    # Create temp dir for inter-process data transfer
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

    # Save embeddings as .npy for efficient storage (P2)
    os.makedirs(args.output_dir, exist_ok=True)
    emb_file = os.path.join(args.output_dir, "embeddings.npy")
    ids_file = os.path.join(args.output_dir, "item_ids.json")
    np.save(emb_file, embeddings)
    with open(ids_file, "w", encoding="utf-8") as f:
        json.dump(item_ids, f, ensure_ascii=False)
    print(f"Saved embeddings to: {emb_file} ({embeddings.nbytes / 1e9:.2f} GB)")
    print(f"Saved item IDs to: {ids_file}")

    # FAISS ANN similarity search with GPU (P0)
    similarity_results = compute_similarities_faiss(
        embeddings,
        item_ids,
        k=args.top_k,
        faiss_gpu_id=0,
        nlist=args.faiss_nlist,
        nprobe=args.faiss_nprobe,
    )

    output_file = os.path.join(args.output_dir, "similarities.json")
    print(f"Saving similarity results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(similarity_results, f, ensure_ascii=False, indent=2)

    # Clean up temp dir
    if os.path.isdir(tmp_dir):
        os.rmdir(tmp_dir)

    print("Processing completed!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
