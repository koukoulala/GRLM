"""
Step 2: ANN Search and OUTPUT Generation.

Input:
- Journey Results TSV from step0 (with OUTPUT column)
- Query embeddings from Step 1 ({base}_query_embeddings.tsv)
- Product ANN index

Output:
- TSV with columns: UserId, ReadableUserEvents, RequestTime, UserHistory,
  ShoppingJourney, JourneyWithAllProducts, JourneyWithProducts

Usage:
    python step2_ann_search_and_output.py --input_file /path/to/Journey_Results.tsv [--debug]
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import faiss
from tqdm import tqdm

csv.field_size_limit(1 << 31 - 1)

# ANN index paths (fixed infrastructure)
DEFAULT_WORK_DIR = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/"
DEFAULT_INDEX_DIR = "/vc_data/users/wangying/OneRec/ShoppingJourney/CookData/data/"
INDEX_PATH = os.path.join(DEFAULT_INDEX_DIR, "0307_EnUs_Product_ann_hnsw.index")
ID_MAP_PATH = os.path.join(DEFAULT_INDEX_DIR, "0307_EnUs_Product_ann_ids.txt")
PRODUCT_PATH = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/0307_EnUs_Product.tsv"

PRODUCT_COLUMNS = ["GlobalOfferId", "Title", "Embedding", "Seller", "Gender",
                   "OriginalPrice", "LLMCatId", "CategoryName", "AgeGroup",
                   "Brand", "Description", "OfferUrl", "ImageUrl"]
OUTPUT_FIELDS = ["Title", "Seller", "Gender", "OriginalPrice", "LLMCatId",
                 "CategoryName", "AgeGroup", "Brand"]


# ========== Load Embeddings ==========
def load_embeddings(emb_path: str) -> Dict[str, np.ndarray]:
    """Load query embeddings from TSV (query \\t space-separated-floats)."""
    print(f"Loading embeddings from {emb_path}...")
    query_embs = {}
    with open(emb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                query = parts[0]
                emb = np.array([float(x) for x in parts[1].split()], dtype=np.float32)
                norm = np.linalg.norm(emb, ord=2)
                if norm > 0:
                    emb = emb / norm
                query_embs[query] = emb
    print(f"Loaded {len(query_embs):,} query embeddings")
    return query_embs


# ========== ANN Search ==========
def run_ann_search(query_embs: Dict[str, np.ndarray],
                   ann_output: str, top_k: int = 20) -> Dict[str, List[dict]]:
    """Run ANN search and return query -> products mapping."""
    # Load index
    print(f"Loading FAISS index from {INDEX_PATH}...")
    t0 = time.time()
    index = faiss.read_index(INDEX_PATH)
    print(f"Index loaded in {time.time()-t0:.1f}s ({index.ntotal:,} vectors)")

    # Load ID mapping
    with open(ID_MAP_PATH, "r") as f:
        id_mapping = [line.strip() for line in f]
    print(f"Loaded {len(id_mapping):,} IDs")

    # Load product metadata
    print(f"Loading product metadata from {PRODUCT_PATH}...")
    product_meta = {}
    with open(PRODUCT_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading products", mininterval=30, maxinterval=60):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= len(PRODUCT_COLUMNS):
                meta = {}
                for i, col in enumerate(PRODUCT_COLUMNS):
                    if col in OUTPUT_FIELDS:
                        meta[col] = parts[i]
                product_meta[parts[0]] = meta
    print(f"Loaded metadata for {len(product_meta):,} products")

    # Build query matrix
    queries = list(query_embs.keys())
    query_matrix = np.array([query_embs[q] for q in queries], dtype=np.float32)

    # Search
    print(f"Searching {len(queries):,} queries (top_k={top_k})...")
    index.hnsw.efSearch = 128
    t0 = time.time()
    distances, indices = index.search(query_matrix, top_k)
    elapsed = time.time() - t0
    qps = len(queries) / elapsed if elapsed > 0 else 0
    print(f"Search done in {elapsed:.1f}s ({qps:.0f} QPS)")

    # Build results
    query_to_products = {}
    for i, query in enumerate(queries):
        products = []
        for j in range(top_k):
            idx = indices[i][j]
            if idx < 0:
                continue
            pid = id_mapping[idx]
            meta = product_meta.get(pid, {})
            products.append({
                "rank": j + 1,
                "product_id": pid,
                "score": float(distances[i][j]),
                **meta
            })
        query_to_products[query] = products

    # Save ANN results
    print(f"Saving ANN results to {ann_output}...")
    with open(ann_output, "w", encoding="utf-8") as f:
        for query, products in query_to_products.items():
            f.write(f"{query}\t{json.dumps(products, ensure_ascii=False)}\n")
    print(f"Saved {len(query_to_products):,} query results")

    return query_to_products


# ========== Generate Output ==========
def generate_output(query_to_products: Dict[str, List[dict]],
                    input_file: str, final_output: str,
                    score_threshold: float = 0.90,
                    max_rows: int = 0,
                    jwp_output: str = None):
    """
    Read original journey file and generate output with product columns.

    Output columns:
    - UserId
    - ReadableUserEvents
    - RequestTime
    - UserHistory
    - ShoppingJourney: original OUTPUT (raw LLM journeys + queries, no products)
    - JourneyWithAllProducts: original journey + ALL products info (no score filter)
    - JourneyWithProducts: filtered journeys with products (score >= threshold, simplified fields)
    """
    print(f"\nGenerating output from {input_file}...")
    if max_rows > 0:
        print(f"  [DEBUG] Limiting to first {max_rows} valid rows")

    output_header = ["UserId", "ReadableUserEvents", "RequestTime", "UserHistory",
                     "ShoppingJourney", "JourneyWithAllProducts", "JourneyWithProducts"]

    # Statistics
    total_users = 0
    total_journeys = 0
    journeys_with_products = 0
    total_queries = 0
    queries_with_products = 0
    total_products = 0
    journey_product_counts = []   # products per journey (kept)
    selected_scores = []          # scores of kept products (score >= threshold)

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(final_output, "w", encoding="utf-8") as f_out, \
         (open(jwp_output, "w", encoding="utf-8") if jwp_output else open(os.devnull, "w")) as f_jwp:

        reader = csv.reader(f_in, delimiter="\t")
        raw_header = next(reader)
        col_map = {name.strip(): idx for idx, name in enumerate(raw_header)}
        idx_uid = col_map["UserId"]
        idx_events = col_map["ReadableUserEvents"]
        idx_time = col_map.get("RequestTime")
        idx_history = col_map.get("UserHistory")
        idx_output = col_map["OUTPUT"]

        f_out.write("\t".join(output_header) + "\n")
        if jwp_output:
            jwp_header = ["UserId", "ReadableUserEvents", "RequestTime",
                          "UserHistory", "JourneyWithProducts"]
            f_jwp.write("\t".join(jwp_header) + "\n")

        line_count = 0
        for parts in tqdm(reader, desc="Generating output",
                          mininterval=60, maxinterval=90):
            if len(parts) <= idx_output:
                continue

            try:
                output_json = json.loads(parts[idx_output])
            except json.JSONDecodeError:
                continue

            journeys = output_json.get("ContinuedJourneys", [])
            total_q = sum(len(j.get("Queries", [])) for j in journeys)
            if total_q == 0:
                continue

            total_users += 1

            # Build JourneyWithAllProducts (original journey + all products, no filter)
            enriched = {"ContinuedJourneys": []}
            # Build JourneyWithProducts (filtered, score >= threshold, simplified fields)
            filtered = {"ContinuedJourneys": []}

            for journey in journeys:
                total_journeys += 1

                # Enriched journey (all products)
                ej = {
                    "Title": journey.get("Title", ""),
                    "Reason": journey.get("Reason", ""),
                    "SourceEventIds": journey.get("SourceEventIds", []),
                    "ConfidenceLevel": journey.get("ConfidenceLevel", 0),
                    "Queries": []
                }

                # Filtered journey (score >= threshold)
                fj = {
                    "Title": journey.get("Title", ""),
                    "Reason": journey.get("Reason", ""),
                    "Queries": []
                }

                for q in journey.get("Queries", []): 
                    total_queries += 1
                    qt = q.get("Query", "")
                    products = query_to_products.get(qt, [])

                    # All products for enriched
                    ej["Queries"].append({
                        "Query": qt,
                        "Products": products
                    })

                    # Filtered products
                    fp = []
                    for p in products:
                        s = p.get("score", 0)
                        if s >= score_threshold:
                            fp.append({
                                "OfferId": p.get("product_id", ""),
                                "Title": p.get("Title", ""),
                                "Seller": p.get("Seller", ""),
                                "Price": p.get("OriginalPrice", "")
                            })
                            selected_scores.append(s)

                    total_products += len(fp)
                    if fp:
                        queries_with_products += 1
                        fj["Queries"].append({"Query": qt, "Products": fp})

                enriched["ContinuedJourneys"].append(ej)

                if fj["Queries"]:
                    journeys_with_products += 1
                    # Count total products in this filtered journey
                    j_prod_count = sum(len(q["Products"]) for q in fj["Queries"])
                    journey_product_counts.append(j_prod_count)
                    filtered["ContinuedJourneys"].append(fj)

            # Write row
            row = [
                parts[idx_uid],
                parts[idx_events],
                parts[idx_time] if idx_time is not None and len(parts) > idx_time else "",
                parts[idx_history] if idx_history is not None and len(parts) > idx_history else "",
                json.dumps(output_json, ensure_ascii=False),     # ShoppingJourney (original)
                json.dumps(enriched, ensure_ascii=False),         # JourneyWithAllProducts (all products)
                json.dumps(filtered, ensure_ascii=False)          # JourneyWithProducts (filtered)
            ]
            f_out.write("\t".join(row) + "\n")
            if jwp_output:
                # Write JWP row: cols 0,1,2,3,6 (UserId, ReadableUserEvents,
                # RequestTime, UserHistory, JourneyWithProducts)
                f_jwp.write("\t".join([row[0], row[1], row[2], row[3], row[6]]) + "\n")

            line_count += 1
            if max_rows > 0 and line_count >= max_rows:
                break

    print(f"\nOutput saved to: {final_output}")
    if jwp_output:
        print(f"JWP output saved to: {jwp_output}")
    print(f"\n{'='*80}")
    print(f"Statistics (Score Threshold: {score_threshold})")
    print(f"{'='*80}")
    print(f"  Users: {total_users:,}")
    print(f"  Journeys: {total_journeys:,} (with products: {journeys_with_products:,}, "
          f"{journeys_with_products/max(total_journeys,1)*100:.1f}%)")
    print(f"  Queries: {total_queries:,} (with products: {queries_with_products:,}, "
          f"{queries_with_products/max(total_queries,1)*100:.1f}%)")
    print(f"  Products (after filter): {total_products:,}")
    print(f"  Avg products/query: {total_products/max(queries_with_products,1):.2f}")

    # Per-journey product count stats
    if journey_product_counts:
        arr = np.array(journey_product_counts)
        print(f"\n  --- Products per Journey (filtered) ---")
        print(f"    Min: {arr.min():>6}  P25: {int(np.percentile(arr, 25)):>6}  "
              f"P50: {int(np.percentile(arr, 50)):>6}  P75: {int(np.percentile(arr, 75)):>6}  "
              f"P90: {int(np.percentile(arr, 90)):>6}  Max: {arr.max():>6}  "
              f"Mean: {arr.mean():.1f}")

    # Selected product score stats
    if selected_scores:
        arr = np.array(selected_scores)
        print(f"\n  --- Selected Product Scores (score >= {score_threshold}) ---")
        print(f"    Min: {arr.min():.4f}  P10: {np.percentile(arr, 10):.4f}  "
              f"P25: {np.percentile(arr, 25):.4f}  P50: {np.percentile(arr, 50):.4f}  "
              f"P75: {np.percentile(arr, 75):.4f}  P90: {np.percentile(arr, 90):.4f}  "
              f"Max: {arr.max():.4f}")
        print(f"    Mean: {arr.mean():.4f}  Std: {arr.std():.4f}")

    print(f"{'='*80}")


# ========== Process Single File ==========
def process_file(input_file, work_dir, skip_ann, top_k, score_threshold,
                 max_rows=0):
    """Process a single input file: ANN search + output generation.

    Returns True on success.
    """
    base = os.path.splitext(os.path.basename(input_file))[0]
    embedding_file = os.path.join(work_dir, f"{base}_query_embeddings.tsv")
    ann_output = os.path.join(work_dir, f"{base}_ann_results.tsv")
    final_output = os.path.join(work_dir, f"{base}_with_products.tsv")
    jwp_output = os.path.join(work_dir, f"{base}_JWP.tsv")

    print(f"  Input file:     {input_file}")
    print(f"  Work dir:       {work_dir}")
    print(f"  Embedding file: {embedding_file}")
    print(f"  ANN output:     {ann_output}")
    print(f"  Final output:   {final_output}")
    print(f"  JWP output:     {jwp_output}")

    if skip_ann:
        # Load existing ANN results
        print("=" * 80)
        print("Loading existing ANN results")
        print("=" * 80)
        print(f"Loading from {ann_output}...")
        query_to_products = {}
        with open(ann_output, "r", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    query_to_products[p[0]] = json.loads(p[1])
        print(f"Loaded {len(query_to_products):,} query results")
    else:
        # Step 1: Load embeddings
        print("=" * 80)
        print("Step 1: Load query embeddings")
        print("=" * 80)
        query_embs = load_embeddings(embedding_file)

        if not query_embs:
            print("No embeddings found! Run step1 first.")
            return False

        # Step 2: ANN search
        print("\n" + "=" * 80)
        print("Step 2: ANN search")
        print("=" * 80)
        query_to_products = run_ann_search(query_embs, ann_output, top_k=top_k)

    # Step 3: Generate output
    print("\n" + "=" * 80)
    print("Step 3: Generate output with JourneyWithProducts")
    print("=" * 80)
    generate_output(query_to_products, input_file, final_output,
                    score_threshold, max_rows, jwp_output=jwp_output)

    print("\nDone!")
    return True


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Step 2: ANN search and OUTPUT generation")
    parser.add_argument("--input_file", type=str,
                        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/ShoppingJourney_Input_80K_1_results.tsv",
                        help="Path to Journey_Results TSV from step0")
    parser.add_argument("--input_folder", type=str, 
                        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/CookData/",
                        help="Path to a folder; processes all *_results.tsv / *_Results.tsv files inside it")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Working directory (default: same dir as input_file)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only first N rows")
    parser.add_argument("--debug_rows", type=int, default=5, help="Number of rows in debug mode (default: 5)")
    parser.add_argument("--skip_ann", action="store_true", help="Skip ANN search, load existing results")
    parser.add_argument("--top_k", type=int, default=20, help="Number of nearest neighbors per query")
    parser.add_argument("--score_threshold", type=float, default=0.85, help="Score threshold (default: 0.85)")
    args = parser.parse_args()

    max_rows = args.debug_rows if args.debug else 0

    if args.input_folder and os.path.isdir(args.input_folder):
        # Folder mode: find all *_results.tsv / *_Results.tsv files
        all_files = sorted(
            glob.glob(os.path.join(args.input_folder, "*_results.tsv"))
            + glob.glob(os.path.join(args.input_folder, "*_Results.tsv"))
        )
        all_files = sorted(set(all_files))
        # Exclude files that are themselves outputs from step1/step2/step3
        all_files = [f for f in all_files
                     if not any(f.endswith(s) for s in (
                         "_query_embeddings.tsv", "_queries.tsv",
                         "_ann_results.tsv", "_with_products.tsv",
                     ))]

        work_dir = args.work_dir or args.input_folder

        # Check which files already have JWP output
        files_to_process = []
        files_skipped = []
        for f in all_files:
            base = os.path.splitext(os.path.basename(f))[0]
            jwp_file = os.path.join(work_dir, f"{base}_JWP.tsv")
            if os.path.isfile(jwp_file):
                files_skipped.append(f)
            else:
                files_to_process.append(f)

        print(f"Folder: {args.input_folder}")
        print(f"Found {len(all_files)} *_results.tsv file(s)")
        print(f"  To process: {len(files_to_process)}")
        print(f"  Skipped (already have JWP): {len(files_skipped)}")
        if files_skipped:
            for f in files_skipped:
                print(f"    SKIP: {os.path.basename(f)}")
        print()

        if not files_to_process:
            print("Nothing to process!")
            return

        os.makedirs(work_dir, exist_ok=True)
        files_succeeded = []
        files_failed = []
        for i, f in enumerate(files_to_process, 1):
            print(f"{'#' * 70}")
            print(f"Processing file {i}/{len(files_to_process)}: {os.path.basename(f)}")
            print(f"{'#' * 70}")
            ok = process_file(f, work_dir, args.skip_ann,
                              args.top_k, args.score_threshold,
                              max_rows=max_rows)
            if ok:
                files_succeeded.append(f)
            else:
                files_failed.append(f)
            print()

        # Final summary
        print()
        print("=" * 70)
        print("FOLDER PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Total *_results.tsv found:       {len(all_files)}")
        print(f"  Skipped (already done):        {len(files_skipped)}")
        print(f"  Processed successfully:        {len(files_succeeded)}")
        print(f"  Failed:                        {len(files_failed)}")
        if files_succeeded:
            print("\nProcessed:")
            for f in files_succeeded:
                print(f"  OK:   {os.path.basename(f)}")
        if files_failed:
            print("\nFailed:")
            for f in files_failed:
                print(f"  FAIL: {os.path.basename(f)}")

    elif args.input_file:
        # Single-file mode
        work_dir = args.work_dir or os.path.dirname(args.input_file)
        os.makedirs(work_dir, exist_ok=True)
        process_file(args.input_file, work_dir, args.skip_ann,
                     args.top_k, args.score_threshold, max_rows=max_rows)

    else:
        print("ERROR: Please specify --input_file or --input_folder")
        sys.exit(1)


if __name__ == "__main__":
    main()
