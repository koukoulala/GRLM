#!/bin/bash
# GRLM RL Data Preparation Script
# Follows OpenOneRec style: random sampling for test set
# Implements Leave-One-Out strategy alignment with SFT evaluation

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parameters
TEST_SIZE=${TEST_SIZE:-1000}
SEED=${SEED:-42}
OUTPUT_FORMAT=${OUTPUT_FORMAT:-"both"}

echo "=============================================="
echo "GRLM RL Data Preparation"
echo "=============================================="
echo "Test size per dataset: $TEST_SIZE"
echo "Random seed: $SEED"
echo "Output format: $OUTPUT_FORMAT"
echo "=============================================="

# Process each dataset
DATASETS=("beauty" "sports" "toys")

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Processing dataset: $dataset"
    echo "----------------------------------------------"
    
    python s7_build_rl_data_v2.py \
        --dataset "$dataset" \
        --test_size "$TEST_SIZE" \
        --seed "$SEED" \
        --output_format "$OUTPUT_FORMAT"
    
    echo "Done: $dataset"
done

echo ""
echo "=============================================="
echo "All datasets processed!"
echo "=============================================="

# Print summary
echo ""
echo "Output directories:"
for dataset in "${DATASETS[@]}"; do
    echo "  ./${dataset}/rl_data/"
    ls -la "./${dataset}/rl_data/" 2>/dev/null | grep -E "\.parquet|\.json" | awk '{print "    " $NF " (" $5 " bytes)"}'
done

echo ""
echo "Data is ready for GRPO training!"
echo "Update run_grlm_grpo.sh with the correct paths:"
echo "  export DATA_DIR=/path/to/GRLM/in_domain/{dataset}/rl_data"
