"""
Step 2: ANN Search and OUTPUT Generation.

Input:
- Query embeddings from Step 1 (200K_query_embeddings.tsv)
- Original journey data (ShoppingJourney_Input_200K_Output_KeepHis50Results.tsv)
- Product ANN index

Output:
- TSV with columns: UserId, ReadableUserEvents, RequestTime, UserHistory, ShoppingJourney, JourneyWithProducts
  - ShoppingJourney: original OUTPUT (raw LLM output with journeys and queries)
  - JourneyWithProducts: filtered OUTPUT with products (score >= threshold, OfferId/Title/Seller/Price)

Usage:
    python step2_ann_search_and_output.py [--debug]
"""

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import faiss
from tqdm import tqdm

# ========== Paths ==========
INPUT_FILE = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/TestData/ShoppingJourney_TestData_50K_Journey_Results.tsv"
WORK_DIR = "/vc_data/users/wangying/OneRec/ShoppingJourney/CookData/data/testdata/"
EMBEDDING_FILE = os.path.join(WORK_DIR, "50K_query_embeddings.tsv")
ANN_OUTPUT = os.path.join(WORK_DIR, "50K_query_ann_results.tsv")
FINAL_OUTPUT = os.path.join(WORK_DIR,  "50K_journey_with_products.tsv")

# ANN index paths
INDEX_PATH = os.path.join(WORK_DIR + "../", "0307_EnUs_Product_ann_hnsw.index")
ID_MAP_PATH = os.path.join(WORK_DIR + "../", "0307_EnUs_Product_ann_ids.txt")
PRODUCT_PATH = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/0307_EnUs_Product.tsv"

# Column indices (0-based) for the input file
COL_USER_ID = 7
COL_READABLE_EVENTS = 8
COL_REQUEST_TIME = 9
COL_USER_HISTORY = 10
COL_OUTPUT = 12

# ANN settings
TOP_K = 10
SCORE_THRESHOLD = 0.90

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
def run_ann_search(query_embs: Dict[str, np.ndarray]) -> Dict[str, List[dict]]:
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
        for line in tqdm(f, desc="Loading products"):
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
    print(f"Searching {len(queries):,} queries...")
    index.hnsw.efSearch = 128
    t0 = time.time()
    distances, indices = index.search(query_matrix, TOP_K)
    elapsed = time.time() - t0
    qps = len(queries) / elapsed if elapsed > 0 else 0
    print(f"Search done in {elapsed:.1f}s ({qps:.0f} QPS)")

    # Build results
    query_to_products = {}
    for i, query in enumerate(queries):
        products = []
        for j in range(TOP_K):
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
    print(f"Saving ANN results to {ANN_OUTPUT}...")
    with open(ANN_OUTPUT, "w", encoding="utf-8") as f:
        for query, products in query_to_products.items():
            f.write(f"{query}\t{json.dumps(products, ensure_ascii=False)}\n")
    print(f"Saved {len(query_to_products):,} query results")

    return query_to_products


# ========== Generate Output ==========
def generate_output(query_to_products: Dict[str, List[dict]],
                    score_threshold: float = SCORE_THRESHOLD,
                    max_rows: int = 0):
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
    print(f"\nGenerating output from {INPUT_FILE}...")
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

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(FINAL_OUTPUT, "w", encoding="utf-8") as f_out:

        f_in.readline()  # skip header
        f_out.write("\t".join(output_header) + "\n")

        line_count = 0
        for line in tqdm(f_in, desc="Generating output"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= COL_OUTPUT:
                continue

            try:
                output_json = json.loads(parts[COL_OUTPUT])
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
                        if p.get("score", 0) >= score_threshold:
                            fp.append({
                                "OfferId": p.get("product_id", ""),
                                "Title": p.get("Title", ""),
                                "Seller": p.get("Seller", ""),
                                "Price": p.get("OriginalPrice", "")
                            })

                    total_products += len(fp)
                    if fp:
                        queries_with_products += 1
                        fj["Queries"].append({"Query": qt, "Products": fp})

                enriched["ContinuedJourneys"].append(ej)

                if fj["Queries"]:
                    journeys_with_products += 1
                    filtered["ContinuedJourneys"].append(fj)

            # Write row
            row = [
                parts[COL_USER_ID],
                parts[COL_READABLE_EVENTS],
                parts[COL_REQUEST_TIME],
                parts[COL_USER_HISTORY],
                json.dumps(output_json, ensure_ascii=False),     # ShoppingJourney (original)
                json.dumps(enriched, ensure_ascii=False),         # JourneyWithAllProducts (all products)
                json.dumps(filtered, ensure_ascii=False)          # JourneyWithProducts (filtered)
            ]
            f_out.write("\t".join(row) + "\n")

            line_count += 1
            if max_rows > 0 and line_count >= max_rows:
                break

    print(f"\nOutput saved to: {FINAL_OUTPUT}")
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
    print(f"{'='*80}")


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Step 2: ANN search and OUTPUT generation")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only first N rows")
    parser.add_argument("--debug_rows", type=int, default=5, help="Number of rows in debug mode (default: 5)")
    parser.add_argument("--skip_ann", action="store_true", help="Skip ANN search, load existing results")
    parser.add_argument("--score_threshold", type=float, default=0.90, help="Score threshold (default: 0.90)")
    args = parser.parse_args()

    max_rows = args.debug_rows if args.debug else 0

    if args.skip_ann:
        # Load existing ANN results
        print("=" * 80)
        print("Loading existing ANN results")
        print("=" * 80)
        print(f"Loading from {ANN_OUTPUT}...")
        query_to_products = {}
        with open(ANN_OUTPUT, "r", encoding="utf-8") as f:
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
        query_embs = load_embeddings(EMBEDDING_FILE)

        if not query_embs:
            print("No embeddings found! Run step1 first.")
            return

        # Step 2: ANN search
        print("\n" + "=" * 80)
        print("Step 2: ANN search")
        print("=" * 80)
        query_to_products = run_ann_search(query_embs)

    # Step 3: Generate output
    print("\n" + "=" * 80)
    print("Step 3: Generate output with JourneyWithProducts")
    print("=" * 80)
    generate_output(query_to_products, args.score_threshold, max_rows)

    print("\nStep 2 Done!")


if __name__ == "__main__":
    main()
