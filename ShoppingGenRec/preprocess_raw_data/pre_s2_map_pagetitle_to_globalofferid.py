"""Step 1b: Map PageTitle Items to GlobalOfferIds via Embedding Similarity

Uses a pre-trained embedding model to generate embeddings for both
GlobalOfferId items (from item.json) and PageTitle items (from
page_title_item.json), then uses FAISS to find the closest GlobalOfferId
for each PageTitle item.

PageTitle items that match a GlobalOfferId above the similarity threshold
are mapped to that GlobalOfferId. This mapping is used downstream by
pre_s2 to consolidate item identifiers in user sequences.

Embeddings for all items are saved to raw_data/ for reuse by s0_init_emb.py
(via --precomputed_emb_dir), avoiding redundant embedding generation.

Input:
    - item.json             : GlobalOfferId items (from pre_s0)
    - page_title_item.json  : PageTitle items (from pre_s1)

Output:
    - pagetitle_to_globalofferid.json : {P-index: GlobalOfferId} mapping
    - embeddings.npy + item_ids.json  : all embeddings (for s0 resume)

Usage:
    python pre_s1b_map_pagetitle_to_globalofferid.py \\
        --item_file ./raw_data/item.json \\
        --page_title_item_file ./raw_data/page_title_item.json \\
        --embedding_model /path/to/Qwen3-Embedding-0.6B \\
        --similarity_threshold 0.85 \\
        --output_dir ./raw_data
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import faiss

# Import embedding generation functions from s0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
from s0_init_emb import (
    generate_embeddings_multi_gpu,
    prepare_text_for_embedding,
)


# =============================================================================
# Data Loading
# =============================================================================

def load_items(item_file, page_title_item_file):
    """Load and combine GlobalOfferId items and PageTitle items.

    Returns:
        Tuple of (combined_data_list, gid_ids_set, p_ids_set) where
        combined_data_list is a list of dicts with at least {"id", "title"}.
    """
    # Load GlobalOfferId items
    with open(item_file, "r", encoding="utf-8") as f:
        item_data = json.load(f)
    gid_items = []
    for key, value in item_data.items():
        new_item = {"id": key}
        new_item.update(value)
        gid_items.append(new_item)
    gid_ids = set(item_data.keys())
    print(f"  GlobalOfferId items: {len(gid_items):,}")

    # Load PageTitle items
    with open(page_title_item_file, "r", encoding="utf-8") as f:
        pt_data = json.load(f)
    p_items = []
    for key, value in pt_data.items():
        new_item = {"id": key}
        new_item.update(value)
        p_items.append(new_item)
    p_ids = set(pt_data.keys())
    print(f"  PageTitle items: {len(p_items):,}")

    # Combine: GlobalOfferId items first, then P-items
    combined = gid_items + p_items
    print(f"  Total combined: {len(combined):,}")

    return combined, gid_ids, p_ids


# =============================================================================
# Cross-Set FAISS Search
# =============================================================================

def search_p_to_gid(
    all_embeddings,
    all_item_ids,
    gid_ids,
    p_ids,
    similarity_threshold=0.85,
    faiss_nlist=4096,
    faiss_nprobe=64,
    num_threads=0,
):
    """Search for the closest GlobalOfferId for each P-item.

    Builds a FAISS index from GlobalOfferId embeddings, then searches
    P-item embeddings against it.

    Args:
        all_embeddings: (N, dim) float32 array of all item embeddings.
        all_item_ids: List of item ID strings aligned with embeddings.
        gid_ids: Set of GlobalOfferId strings.
        p_ids: Set of P-item ID strings.
        similarity_threshold: Minimum cosine similarity for mapping.
        faiss_nlist: Number of IVF clusters.
        faiss_nprobe: Number of clusters to search.
        num_threads: CPU threads for FAISS.

    Returns:
        Tuple of (mapping_dict, stats_dict) where mapping_dict maps
        P-index -> GlobalOfferId for items above threshold.
    """
    n, dim = all_embeddings.shape

    # Set FAISS threads
    if num_threads > 0:
        faiss.omp_set_num_threads(num_threads)
    else:
        cpu_count = os.cpu_count() or 1
        faiss.omp_set_num_threads(cpu_count)
        print(f"  FAISS threads: {cpu_count}")

    # Split embeddings by type
    id_to_idx = {item_id: i for i, item_id in enumerate(all_item_ids)}

    gid_indices = [id_to_idx[gid] for gid in all_item_ids if gid in gid_ids]
    p_indices = [id_to_idx[pid] for pid in all_item_ids if pid in p_ids]

    gid_embeddings = all_embeddings[gid_indices]
    p_embeddings = all_embeddings[p_indices]
    gid_id_list = [all_item_ids[i] for i in gid_indices]
    p_id_list = [all_item_ids[i] for i in p_indices]

    print(f"  GlobalOfferId embeddings: {gid_embeddings.shape}")
    print(f"  P-item embeddings: {p_embeddings.shape}")

    # Build FAISS index on GlobalOfferId embeddings
    n_gid = len(gid_id_list)
    effective_nlist = min(faiss_nlist, max(n_gid // 40, 1))
    effective_nlist = max(effective_nlist, 1)

    print(f"  Building FAISS index (IVFFlat, nlist={effective_nlist})...")
    gid_embeddings = np.ascontiguousarray(gid_embeddings, dtype=np.float32)
    p_embeddings = np.ascontiguousarray(p_embeddings, dtype=np.float32)

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(
        quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT
    )
    index.train(gid_embeddings)
    index.add(gid_embeddings)
    index.nprobe = min(faiss_nprobe, effective_nlist)
    print(f"  Index trained and built ({n_gid:,} vectors, nprobe={index.nprobe})")

    # Search P-items against the index (top-1)
    print(f"  Searching {len(p_id_list):,} P-items against index...")
    start_time = time.time()

    search_batch_size = 100000
    all_distances = []
    all_indices_arr = []
    for start in range(0, len(p_id_list), search_batch_size):
        end = min(start + search_batch_size, len(p_id_list))
        D_batch, I_batch = index.search(p_embeddings[start:end], 1)
        all_distances.append(D_batch)
        all_indices_arr.append(I_batch)

    D = np.vstack(all_distances).flatten()
    I = np.vstack(all_indices_arr).flatten()

    elapsed = time.time() - start_time
    print(f"  Search done in {elapsed:.1f}s")

    # Build mapping with threshold
    mapping = {}
    above_threshold = 0
    below_threshold = 0
    similarity_values = []

    for i, (p_id, dist, idx) in enumerate(zip(p_id_list, D, I)):
        idx = int(idx)
        sim = float(dist)
        similarity_values.append(sim)

        if idx >= 0 and sim >= similarity_threshold:
            mapping[p_id] = gid_id_list[idx]
            above_threshold += 1
        else:
            below_threshold += 1

    # Statistics
    sim_arr = np.array(similarity_values)
    stats = {
        "total_p_items": len(p_id_list),
        "mapped_above_threshold": above_threshold,
        "unmapped_below_threshold": below_threshold,
        "threshold": similarity_threshold,
        "sim_mean": float(sim_arr.mean()),
        "sim_median": float(np.median(sim_arr)),
        "sim_p25": float(np.percentile(sim_arr, 25)),
        "sim_p75": float(np.percentile(sim_arr, 75)),
        "sim_p90": float(np.percentile(sim_arr, 90)),
        "sim_min": float(sim_arr.min()),
        "sim_max": float(sim_arr.max()),
    }

    return mapping, stats


# =============================================================================
# Resume helpers
# =============================================================================

def load_partial_tmp_embeddings(tmp_dir):
    """Load any completed rank files from tmp_dir.

    Returns (item_ids, embeddings) merged from all available rank files,
    or ([], None) if nothing found.
    """
    import glob
    pattern = os.path.join(tmp_dir, "embeddings_rank*.npy")
    emb_files = sorted(glob.glob(pattern))
    if not emb_files:
        return [], None

    all_ids = []
    all_embs = []
    for emb_path in emb_files:
        # Extract rank number and find matching ids file
        basename = os.path.basename(emb_path)  # embeddings_rank1.npy
        rank_str = basename.replace("embeddings_rank", "").replace(".npy", "")
        ids_path = os.path.join(tmp_dir, f"item_ids_rank{rank_str}.json")
        if not os.path.exists(ids_path):
            print(f"  [TMP] Warning: found {basename} but no matching "
                  f"item_ids_rank{rank_str}.json, skipping")
            continue
        print(f"  [TMP] Loading rank {rank_str} from {tmp_dir}...")
        emb = np.load(emb_path)
        with open(ids_path, "r", encoding="utf-8") as f:
            ids = json.load(f)
        print(f"  [TMP]   rank {rank_str}: {len(ids):,} items, "
              f"shape={emb.shape}")
        all_ids.extend(ids)
        all_embs.append(emb)

    if not all_embs:
        return [], None
    combined = np.vstack(all_embs)
    print(f"  [TMP] Total recovered from tmp: {len(all_ids):,} items, "
          f"shape={combined.shape}")
    return all_ids, combined


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Map PageTitle items to GlobalOfferIds via embedding "
                    "similarity"
    )
    parser.add_argument(
        "--item_file", type=str,
        default="./raw_data/item.json",
        help="Path to item.json with GlobalOfferId items (from pre_s0)",
    )
    parser.add_argument(
        "--page_title_item_file", type=str,
        default="./raw_data/page_title_item.json",
        help="Path to page_title_item.json with P-items (from pre_s1)",
    )
    parser.add_argument(
        "--embedding_model", type=str,
        default="/scratch/workspaceblobstore/users/xiaoyukou/ckpts/"
                "Qwen3-Embedding-0.6B",
        help="Path to embedding model",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./raw_data",
        help="Directory to save outputs (default: ./raw_data)",
    )
    parser.add_argument(
        "--similarity_threshold", type=float, default=0.85,
        help="Minimum cosine similarity to map P-item to GlobalOfferId "
             "(default: 0.85)",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="Number of GPUs for embedding generation (default: all)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size per GPU for embedding generation",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048,
        help="Max token length for embedding model",
    )
    parser.add_argument(
        "--faiss_nlist", type=int, default=4096,
        help="FAISS IVF clusters",
    )
    parser.add_argument(
        "--faiss_nprobe", type=int, default=128,
        help="FAISS clusters to probe",
    )
    parser.add_argument(
        "--faiss_threads", type=int, default=0,
        help="FAISS CPU threads (0=all)",
    )
    parser.add_argument(
        "--resume_from_dir", type=str, default="./processed/",
        help="Directory containing embeddings.npy and item_ids.json from a "
             "previous run. If set and files exist, reuses existing embeddings and only generates embeddings for new items."
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    num_gpus = args.num_gpus or torch.cuda.device_count() or 1

    print("=" * 70)
    print("Step 1b: Map PageTitle Items to GlobalOfferIds")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  GPUs: {num_gpus}")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load items
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 1: Loading items")
    print("=" * 70)

    combined_data, gid_ids, p_ids = load_items(
        args.item_file, args.page_title_item_file
    )

    # =========================================================================
    # Step 2: Generate or load embeddings
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 2: Generating embeddings")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    emb_file = os.path.join(args.output_dir, "embeddings.npy")
    ids_file = os.path.join(args.output_dir, "item_ids.json")

    current_set = {item["id"] for item in combined_data}
    resume_dir = args.resume_from_dir.strip() if args.resume_from_dir else ""
    resume_emb = os.path.join(resume_dir, "embeddings.npy") if resume_dir else ""
    resume_ids = os.path.join(resume_dir, "item_ids.json") if resume_dir else ""

    # -----------------------------------------------------------------
    # Collect all previously-computed embeddings (resume + tmp)
    # -----------------------------------------------------------------
    prev_ids = []
    prev_embeddings = None

    # Source 1: resume dir (e.g. processed/)
    #   Also check output_dir for a checkpoint saved by a previous
    #   crashed run (so re-running the same command picks it up).
    resume_sources = []
    if resume_dir:
        resume_sources.append(resume_dir)
    if os.path.abspath(args.output_dir) != os.path.abspath(resume_dir or ""):
        resume_sources.append(args.output_dir)

    for src_dir in resume_sources:
        src_emb = os.path.join(src_dir, "embeddings.npy")
        src_ids = os.path.join(src_dir, "item_ids.json")
        if os.path.exists(src_emb) and os.path.exists(src_ids):
            print(f"  [RESUME] Loading existing embeddings from: {src_dir}")
            loaded_emb = np.load(src_emb)
            with open(src_ids, "r", encoding="utf-8") as f:
                loaded_ids = json.load(f)
            print(f"  Loaded {len(loaded_ids):,} embeddings, "
                  f"shape={loaded_emb.shape}")
            if prev_embeddings is not None:
                # Merge: add only items not already loaded
                existing_set = set(prev_ids)
                new_idx = [i for i, lid in enumerate(loaded_ids)
                           if lid not in existing_set]
                if new_idx:
                    prev_ids = prev_ids + [loaded_ids[i] for i in new_idx]
                    prev_embeddings = np.vstack(
                        [prev_embeddings, loaded_emb[new_idx]])
            else:
                prev_ids = loaded_ids
                prev_embeddings = loaded_emb

    # Source 2: partial rank files left in _emb_tmp from a crashed run
    tmp_dir = os.path.join(args.output_dir, "_emb_tmp")
    tmp_ids, tmp_embeddings = load_partial_tmp_embeddings(tmp_dir)
    if tmp_ids:
        if prev_embeddings is not None:
            # Merge: add only tmp items not already in prev
            prev_id_set_tmp = set(prev_ids)
            new_from_tmp_idx = [i for i, tid in enumerate(tmp_ids)
                                if tid not in prev_id_set_tmp]
            if new_from_tmp_idx:
                extra_ids = [tmp_ids[i] for i in new_from_tmp_idx]
                extra_embs = tmp_embeddings[new_from_tmp_idx]
                prev_ids = prev_ids + extra_ids
                prev_embeddings = np.vstack([prev_embeddings, extra_embs])
                print(f"  [TMP] Merged {len(extra_ids):,} new items from "
                      f"tmp (skipped {len(tmp_ids)-len(extra_ids):,} "
                      f"duplicates)")
        else:
            prev_ids = tmp_ids
            prev_embeddings = tmp_embeddings

    # -----------------------------------------------------------------
    # Determine what still needs to be generated
    # -----------------------------------------------------------------
    if prev_ids:
        prev_id_set = set(prev_ids)
        prev_id_to_idx = {pid: i for i, pid in enumerate(prev_ids)}

        missing_items = [item for item in combined_data
                         if item["id"] not in prev_id_set]
        kept_count = sum(1 for pid in prev_ids if pid in current_set)
        removed_count = len(prev_ids) - kept_count

        print(f"  Items in current input:   {len(combined_data):>10,}")
        print(f"  Already have embeddings:  {kept_count:>10,}")
        print(f"  Removed (not in input):   {removed_count:>10,}")
        print(f"  New items to process:     {len(missing_items):>10,}")

        if missing_items:
            # Save intermediate checkpoint so tmp data is not lost if
            # the next generation also crashes.  On re-run the user can
            # pass --resume_from_dir ./raw_data/ to pick this up.
            print(f"\n  Saving intermediate checkpoint "
                  f"({len(prev_ids):,} items) to {args.output_dir} ...")
            np.save(emb_file, prev_embeddings)
            with open(ids_file, "w", encoding="utf-8") as f:
                json.dump(prev_ids, f, ensure_ascii=False)

            print(f"\n  Generating embeddings for {len(missing_items):,} "
                  f"new items...")
            new_item_ids, new_embeddings = generate_embeddings_multi_gpu(
                missing_items,
                args.embedding_model,
                num_gpus=num_gpus,
                batch_size=args.batch_size,
                max_length=args.max_length,
                tmp_dir=tmp_dir,
            )
            print(f"  New embeddings shape: {new_embeddings.shape}")

            # Merge in current data order
            new_id_to_idx = {nid: i for i, nid in enumerate(new_item_ids)}
            dim = prev_embeddings.shape[1]
            all_embeddings = np.zeros(
                (len(combined_data), dim), dtype=np.float32)
            all_item_ids = []
            for i, item in enumerate(combined_data):
                item_id = item["id"]
                all_item_ids.append(item_id)
                if item_id in prev_id_to_idx:
                    all_embeddings[i] = prev_embeddings[
                        prev_id_to_idx[item_id]]
                elif item_id in new_id_to_idx:
                    all_embeddings[i] = new_embeddings[
                        new_id_to_idx[item_id]]
        else:
            # All items already have embeddings, reorder
            print(f"  All items already have embeddings, reordering...")
            dim = prev_embeddings.shape[1]
            all_embeddings = np.zeros(
                (len(combined_data), dim), dtype=np.float32)
            all_item_ids = []
            for i, item in enumerate(combined_data):
                item_id = item["id"]
                all_item_ids.append(item_id)
                if item_id in prev_id_to_idx:
                    all_embeddings[i] = prev_embeddings[
                        prev_id_to_idx[item_id]]
    else:
        # Full run: generate all embeddings
        all_item_ids, all_embeddings = generate_embeddings_multi_gpu(
            combined_data,
            args.embedding_model,
            num_gpus=num_gpus,
            batch_size=args.batch_size,
            max_length=args.max_length,
            tmp_dir=tmp_dir,
        )

    # Save embeddings for s0 reuse
    np.save(emb_file, all_embeddings)
    with open(ids_file, "w", encoding="utf-8") as f:
        json.dump(all_item_ids, f, ensure_ascii=False)
    print(f"  Saved embeddings to: {emb_file} "
          f"({all_embeddings.nbytes / 1e9:.2f} GB)")
    print(f"  Saved item IDs to: {ids_file}")

    # =========================================================================
    # Step 3: Cross-set FAISS search
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 3: Cross-set FAISS search (P-items -> GlobalOfferIds)")
    print("=" * 70)

    mapping, stats = search_p_to_gid(
        all_embeddings,
        all_item_ids,
        gid_ids,
        p_ids,
        similarity_threshold=args.similarity_threshold,
        faiss_nlist=args.faiss_nlist,
        faiss_nprobe=args.faiss_nprobe,
        num_threads=args.faiss_threads,
    )

    # =========================================================================
    # Step 4: Save results
    # =========================================================================
    print()
    print("=" * 70)
    print("Step 4: Results and output")
    print("=" * 70)

    print(f"  Total P-items:           {stats['total_p_items']:>10,}")
    print(f"  Mapped (>= {args.similarity_threshold}):      "
          f"{stats['mapped_above_threshold']:>10,} "
          f"({stats['mapped_above_threshold']/max(stats['total_p_items'],1)*100:.1f}%)")
    print(f"  Unmapped (< {args.similarity_threshold}):     "
          f"{stats['unmapped_below_threshold']:>10,}")
    print(f"\n  Similarity distribution:")
    print(f"    Min:    {stats['sim_min']:.4f}")
    print(f"    P25:    {stats['sim_p25']:.4f}")
    print(f"    Median: {stats['sim_median']:.4f}")
    print(f"    Mean:   {stats['sim_mean']:.4f}")
    print(f"    P75:    {stats['sim_p75']:.4f}")
    print(f"    P90:    {stats['sim_p90']:.4f}")
    print(f"    Max:    {stats['sim_max']:.4f}")

    # How many distinct GlobalOfferIds are mapped to
    mapped_gids = set(mapping.values())
    print(f"\n  Distinct GlobalOfferIds mapped to: {len(mapped_gids):,}")

    # Save mapping
    mapping_file = os.path.join(args.output_dir, "pagetitle_to_globalofferid.json")
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  Mapping saved to: {mapping_file}")

    # Save stats
    stats_file = os.path.join(args.output_dir, "pagetitle_mapping_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Stats saved to: {stats_file}")

    # Show samples
    print(f"\n  Sample mappings (first 5):")
    for i, (p_id, gid) in enumerate(list(mapping.items())[:5]):
        # Find titles
        p_title = ""
        gid_title = ""
        for item in combined_data:
            if item["id"] == p_id:
                p_title = item.get("title", "")[:80]
            elif item["id"] == gid:
                gid_title = item.get("title", "")[:80]
            if p_title and gid_title:
                break
        print(f"    {p_id} -> {gid}")
        print(f"      P-title:   {p_title}")
        print(f"      GID-title: {gid_title}")

    print(f"\nDone!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
