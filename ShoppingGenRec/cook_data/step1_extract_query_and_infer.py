"""
Step 1: Extract queries from ShoppingJourney OUTPUT column and generate embeddings.

Steps:
1. Extract all unique queries from OUTPUT column
2. Save queries to a TSV file (one query per line, no header)
3. Call inference_onnx_distributed_entry.py to generate embeddings

Usage:
    python step1_extract_query_and_infer.py --input_file /path/to/Journey_Results.tsv [--debug]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

# Inference script paths (fixed infrastructure)
INFERENCE_DIR = "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/common/run_matador_emb"
INFERENCE_SCRIPT = os.path.join(INFERENCE_DIR, "inference_onnx_distributed_entry.py")
TOKENIZER_PATH = os.path.join(INFERENCE_DIR, "simiaozuo_dense_retrieval_url_data_20250415_checkpoints_model_1_checkpoint-keyword")
MODEL_PATH = os.path.join(TOKENIZER_PATH, "model_dynamic.onnx")


# ========== Step 1: Extract Queries ==========
def extract_queries(input_file, col_output, max_rows=0):
    """Extract all unique queries from the OUTPUT column."""
    print(f"Extracting queries from {input_file}...")
    if max_rows > 0:
        print(f"  [DEBUG] Limiting to first {max_rows} rows")

    queries = set()
    line_count = 0
    valid_count = 0
    error_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for parts in tqdm(reader, desc="Extracting queries",
                          mininterval=60, maxinterval=90):
            line_count += 1
            if max_rows > 0 and line_count > max_rows:
                break

            if len(parts) <= col_output:
                error_count += 1
                continue

            try:
                output = json.loads(parts[col_output])
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
def save_queries(queries, query_file):
    """Save queries to TSV file (one query per line, no header, no id)."""
    print(f"Saving {len(queries):,} queries to {query_file}...")
    with open(query_file, "w", encoding="utf-8") as f:
        for q in sorted(queries):
            f.write(q + "\n")
    print(f"Saved to {query_file}")


# ========== Step 3: Run Embedding Inference ==========
def run_inference(query_file, embedding_output, work_dir, gpu_count=4, gpu_ids=None):
    """Run MatadorEmb inference to generate embeddings.

    Args:
        gpu_ids: Comma-separated physical GPU IDs to use (e.g. "1" or "0,1").
                 If None, uses 0..gpu_count-1.
    """
    temp_folder = os.path.join(work_dir, "temp_emb")
    print(f"\nRunning embedding inference...")
    print(f"  Input: {query_file}")
    print(f"  Output: {embedding_output}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  GPUs: {gpu_count}")

    # Determine which physical GPU IDs to use
    if gpu_ids is None:
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES",
                                 ",".join(str(i) for i in range(gpu_count)))
    gpu_id_list = [g.strip() for g in gpu_ids.split(",")]
    print(f"  Physical GPU IDs: {gpu_id_list}")

    # The external inference_onnx_distributed_entry.py assigns
    # device = 0,1,2... and the child process does:
    #   os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    # This maps device index directly to physical GPU ID.
    # So we bypass the entry script and directly call the child
    # inference script with the correct physical GPU IDs.

    # Split data into shards (clean old splits first)
    os.makedirs(temp_folder, exist_ok=True)
    for old_file in os.listdir(temp_folder):
        if old_file.startswith("data_split_"):
            os.remove(os.path.join(temp_folder, old_file))
    num_shards = gpu_count
    split_cmd = f"split -d -n l/{num_shards} {query_file} {temp_folder}/data_split_"
    print(f"  Splitting data: {split_cmd}")
    os.system(split_cmd)

    # Find split files
    split_files = sorted([
        os.path.join(temp_folder, f)
        for f in os.listdir(temp_folder)
        if f.startswith("data_split_") and os.path.isfile(os.path.join(temp_folder, f))
    ])
    print(f"  Split into {len(split_files)} shards")

    # Compute file
    compute_script = os.path.join(INFERENCE_DIR, "inference_simcse_onnx_v0.py")

    # Launch one process per shard, assigning physical GPU IDs
    results_dir = os.path.join(temp_folder, "results")
    os.makedirs(results_dir, exist_ok=True)

    processes = []
    result_files = []
    for i, split_file in enumerate(split_files):
        physical_gpu = gpu_id_list[i % len(gpu_id_list)]
        show_bar = int(i == len(split_files) - 1)
        result_file = os.path.join(results_dir,
                                   os.path.basename(split_file) + "_results.tsv")
        result_files.append(result_file)

        cmd = [
            sys.executable, compute_script,
            "--data_file", split_file,
            "--output_file", result_file,
            "--inference_type", "keyword",
            "--normalize_and_quantize", "0",
            "--device", physical_gpu,
            "--tokenizer_path", TOKENIZER_PATH,
            "--model_path", MODEL_PATH,
            "--max_length", "512",
            "--max_length_entity", "-1",
            "--batch_size", "256",
            "--show_bar", str(show_bar),
            "--include_id_num", "0",
        ]
        print(f"  Shard {i}: GPU {physical_gpu} -> {os.path.basename(split_file)}")
        p = subprocess.Popen(cmd, cwd=INFERENCE_DIR)
        processes.append(p)

    # Wait for all
    for p in processes:
        p.wait()

    # Check exit codes
    failed = [i for i, p in enumerate(processes) if p.returncode != 0]
    if failed:
        print(f"Inference failed on shards: {failed}")
        sys.exit(1)

    # Merge results
    with open(embedding_output, "w") as out_f:
        for rf in result_files:
            with open(rf, "r") as in_f:
                for line in in_f:
                    out_f.write(line)

    # Verify output
    line_count = 0
    with open(embedding_output, "r") as f:
        for _ in f:
            line_count += 1
    print(f"\nEmbedding output: {line_count:,} lines in {embedding_output}")


def find_output_column(input_file):
    """Read header to find the OUTPUT column index."""
    with open(input_file, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    for i, name in enumerate(header):
        if name.strip() == "OUTPUT":
            return i
    raise ValueError(f"'OUTPUT' column not found in header: {header}")


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract queries and generate embeddings")
    parser.add_argument("--input_file", type=str,
                        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/ShoppingJourney_Input_80K_1_results.tsv",
                        help="Path to Journey_Results TSV from step0")
    parser.add_argument("--work_dir", type=str, 
                        default=None,
                        help="Working directory for outputs")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only first N rows")
    parser.add_argument("--debug_rows", type=int, default=5, help="Number of rows in debug mode (default: 5)")
    parser.add_argument("--skip_inference", action="store_true", help="Skip embedding inference (use existing embeddings)")
    parser.add_argument("--gpu_count", type=int, default=2, help="Number of GPUs for inference (default: 2)")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated physical GPU IDs to use (e.g. '1' or '0,1'). "
                             "Overrides CUDA_VISIBLE_DEVICES. If not set, uses env or 0..gpu_count-1.")
    args = parser.parse_args()

    # Derive paths from input file name
    work_dir = args.work_dir or os.path.dirname(args.input_file)
    os.makedirs(work_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input_file))[0]
    query_file = os.path.join(work_dir, f"{base}_queries.tsv")
    embedding_output = os.path.join(work_dir, f"{base}_query_embeddings.tsv")

    col_output = find_output_column(args.input_file)
    print(f"  Input file:    {args.input_file}")
    print(f"  OUTPUT column: index {col_output}")
    print(f"  Query file:    {query_file}")
    print(f"  Embedding out: {embedding_output}")

    max_rows = args.debug_rows if args.debug else 0

    # Step 1: Extract queries
    print("=" * 80)
    print("Step 1: Extract queries")
    print("=" * 80)
    queries = extract_queries(args.input_file, col_output, max_rows=max_rows)

    if not queries:
        print("No queries found!")
        return

    # Step 2: Save queries
    print("\n" + "=" * 80)
    print("Step 2: Save queries")
    print("=" * 80)
    save_queries(queries, query_file)

    # Step 3: Run embedding inference
    if not args.skip_inference:
        print("\n" + "=" * 80)
        print("Step 3: Run embedding inference")
        print("=" * 80)
        run_inference(query_file, embedding_output, work_dir,
                      gpu_count=args.gpu_count, gpu_ids=args.gpu_ids)
    else:
        print("\n[Skipping inference, using existing embeddings]")

    print("\nStep 1 Done!")
    print(f"Query file: {query_file}")
    print(f"Embedding file: {embedding_output}")


if __name__ == "__main__":
    main()
