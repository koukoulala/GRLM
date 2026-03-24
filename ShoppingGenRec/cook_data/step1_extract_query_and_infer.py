"""
Step 1: Extract queries from ShoppingJourney OUTPUT column and generate embeddings.

Steps:
1. Extract all unique queries from OUTPUT column
2. Save queries to a TSV file (one query per line, no header)
3. Call inference_onnx_distributed_entry.py to generate embeddings

Usage:
    python step1_extract_query_and_infer.py [--debug]
"""

import argparse
import json
import os
import subprocess
import sys
from tqdm import tqdm

# ========== Paths ==========
INPUT_FILE = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/TestData/ShoppingJourney_TestData_50K_Journey_Results.tsv"
WORK_DIR = "/vc_data/users/wangying/OneRec/ShoppingJourney/CookData/data/testdata/"
if os.path.exists(WORK_DIR):
    print(f"Using existing work directory: {WORK_DIR}")
else:
    os.makedirs(WORK_DIR, exist_ok=True)
    print(f"Created work directory: {WORK_DIR}")

QUERY_FILE = os.path.join(WORK_DIR, "50K_queries.tsv")
EMBEDDING_OUTPUT = os.path.join(WORK_DIR, "50K_query_embeddings.tsv")

# Inference script paths
INFERENCE_DIR = "/vc_data/users/wangying/OneRec/common/run_matador_emb"
INFERENCE_SCRIPT = os.path.join(INFERENCE_DIR, "inference_onnx_distributed_entry.py")
TOKENIZER_PATH = os.path.join(INFERENCE_DIR, "simiaozuo_dense_retrieval_url_data_20250415_checkpoints_model_1_checkpoint-keyword")
MODEL_PATH = os.path.join(TOKENIZER_PATH, "model_dynamic.onnx")
TEMP_FOLDER = os.path.join(WORK_DIR, "temp_emb")

# Column index for OUTPUT (0-based)
COL_OUTPUT = 12


# ========== Step 1: Extract Queries ==========
def extract_queries(max_rows: int = 0):
    """Extract all unique queries from the OUTPUT column."""
    print(f"Extracting queries from {INPUT_FILE}...")
    if max_rows > 0:
        print(f"  [DEBUG] Limiting to first {max_rows} rows")

    queries = set()
    line_count = 0
    valid_count = 0
    error_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in tqdm(f, desc="Extracting queries"):
            line_count += 1
            if max_rows > 0 and line_count > max_rows:
                break

            parts = line.rstrip("\n").split("\t")
            if len(parts) <= COL_OUTPUT:
                error_count += 1
                continue

            try:
                output = json.loads(parts[COL_OUTPUT])
                journeys = output.get("ContinuedJourneys", [])
                total_q = sum(len(j.get("Queries", [])) for j in journeys)
                if total_q == 0:
                    continue

                valid_count += 1
                for journey in journeys:
                    for q in journey.get("Queries", []):
                        query_text = q.get("Query", "").strip()
                        if query_text:
                            queries.add(query_text)
            except json.JSONDecodeError:
                error_count += 1

            if max_rows > 0 and valid_count >= max_rows:
                break

    print(f"Processed {line_count:,} rows (errors: {error_count})")
    print(f"Extracted {len(queries):,} unique queries from {valid_count} valid rows")
    return queries


# ========== Step 2: Save Queries ==========
def save_queries(queries):
    """Save queries to TSV file (one query per line, no header, no id)."""
    print(f"Saving {len(queries):,} queries to {QUERY_FILE}...")
    with open(QUERY_FILE, "w", encoding="utf-8") as f:
        for q in sorted(queries):
            f.write(q + "\n")
    print(f"Saved to {QUERY_FILE}")


# ========== Step 3: Run Embedding Inference ==========
def run_inference(gpu_count=4):
    """Run MatadorEmb inference to generate embeddings."""
    print(f"\nRunning embedding inference...")
    print(f"  Input: {QUERY_FILE}")
    print(f"  Output: {EMBEDDING_OUTPUT}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  GPUs: {gpu_count}")

    gpu_ids = ",".join(str(i) for i in range(gpu_count))

    cmd = [
        sys.executable,
        INFERENCE_SCRIPT,
        "--tokenizer_path", TOKENIZER_PATH,
        "--model_path", MODEL_PATH,
        "--data_file", QUERY_FILE,
        "--output_file", EMBEDDING_OUTPUT,
        "--temp_folder", TEMP_FOLDER,
        "--inference_type", "keyword",
        "--compute_file", "v0",
        "--normalize_and_quantize", "0",
        "--num_gpus", str(gpu_count),
        "--num_sessions_per_gpu", "1",
        "--max_length", "512",
        "--max_length_entity", "-1",
        "--batch_size", "256",
        "--include_id_num", "0",
    ]

    print(f"Command: {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    result = subprocess.run(cmd, cwd=INFERENCE_DIR, env=env)

    if result.returncode != 0:
        print(f"Inference failed with return code {result.returncode}")
        sys.exit(1)

    # Verify output
    line_count = 0
    with open(EMBEDDING_OUTPUT, "r") as f:
        for _ in f:
            line_count += 1
    print(f"\nEmbedding output: {line_count:,} lines in {EMBEDDING_OUTPUT}")


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract queries and generate embeddings")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only first N rows")
    parser.add_argument("--debug_rows", type=int, default=5, help="Number of rows in debug mode (default: 5)")
    parser.add_argument("--skip_inference", action="store_true", help="Skip embedding inference (use existing embeddings)")
    parser.add_argument("--gpu_count", type=int, default=2, help="Number of GPUs for inference (default: 2)")
    args = parser.parse_args()

    max_rows = args.debug_rows if args.debug else 0

    # Step 1: Extract queries
    print("=" * 80)
    print("Step 1: Extract queries")
    print("=" * 80)
    queries = extract_queries(max_rows=max_rows)

    if not queries:
        print("No queries found!")
        return

    # Step 2: Save queries
    print("\n" + "=" * 80)
    print("Step 2: Save queries")
    print("=" * 80)
    save_queries(queries)

    # Step 3: Run embedding inference
    if not args.skip_inference:
        print("\n" + "=" * 80)
        print("Step 3: Run embedding inference")
        print("=" * 80)
        run_inference(gpu_count=args.gpu_count)
    else:
        print("\n[Skipping inference, using existing embeddings]")

    print("\nStep 1 Done!")
    print(f"Query file: {QUERY_FILE}")
    print(f"Embedding file: {EMBEDDING_OUTPUT}")


if __name__ == "__main__":
    main()
