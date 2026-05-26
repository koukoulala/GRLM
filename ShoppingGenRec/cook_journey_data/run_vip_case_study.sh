#!/bin/bash
# =============================================================================
#  VIP Case Study Pipeline: vip_users.tsv → step3 → step5 → step6 → step8
# =============================================================================
#
#  Runs the full shopping journey pipeline for VIP users, producing
#  an interactive HTML visualization with product images and links.
#
#  Usage:
#      bash run_vip_case_study.sh                              # date=20260516, all steps
#      bash run_vip_case_study.sh --date 20260513              # use 20260513 data
#      bash run_vip_case_study.sh --date 20260513 --from step6 # 20260513, from step6
#      bash run_vip_case_study.sh --from step8                 # date=20260516, step8 only
#
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives in cook_journey_data/, project root is one level up
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COOK_DIR="${SCRIPT_DIR}"

# ── Parse arguments ──────────────────────────────────────────────────────────

DATA_DATE="20260516"
START_FROM="step3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)  DATA_DATE="$2"; shift 2 ;;
        --from)  START_FROM="$2"; shift 2 ;;
        *)       echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Paths derived from DATA_DATE ─────────────────────────────────────────────

DATA_ROOT="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/${DATA_DATE}"
VIP_DIR="${DATA_ROOT}/vip_case_study"

# Input
VIP_INPUT="${PROJECT_DIR}/resources/vip_users.tsv"

# Shared resources — 20260513 uses raw_data_v2/ folder
if [[ "${DATA_DATE}" == "20260513" ]]; then
    RAW_DIR="${DATA_ROOT}/raw_data_v2"
else
    RAW_DIR="${DATA_ROOT}/raw_data"
fi
ITEM_JSON="${RAW_DIR}/item.json"
INDEX_DIR="${RAW_DIR}/MatadorEmb_Index"

# ORIGINAL_INDEX only exists for 20260516
ORIGINAL_INDEX=""
if [[ "${DATA_DATE}" == "20260516" ]]; then
    ORIGINAL_INDEX="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/ProductGroup/20260515_ProductBestOffer_Sampled.tsv"
fi

# Intermediate file names (all under VIP_DIR)
STEP3_OUTPUT="${VIP_DIR}/vip_users_Journey_Results.tsv"
STEP5_PREFIX="vip_users"
STEP5_OUTPUT="${VIP_DIR}/${STEP5_PREFIX}_journey_with_products.tsv"
STEP6_OUTPUT_DIR="${VIP_DIR}/ranker_output"
STEP6_MERGED=""  # auto-detected after step6 merge
STEP8_OUTPUT_DIR="${VIP_DIR}"

# LLM settings
COPILOT_MODEL="gpt-5.2"
STEP6_COPILOT_MODEL="gpt-5.2"
GPU_IDS="1"

echo "=================================================================="
echo "  VIP Case Study Pipeline"
echo "=================================================================="
echo "  Script dir:    ${SCRIPT_DIR}"
echo "  VIP output:    ${VIP_DIR}"
echo "  VIP input:     ${VIP_INPUT}"
echo "  Data date:     ${DATA_DATE}"
echo "  Start from:    ${START_FROM}"
echo "  Item JSON:     ${ITEM_JSON}"
echo "  Index dir:     ${INDEX_DIR}"
if [[ -n "${ORIGINAL_INDEX}" ]]; then
    echo "  Orig index:    ${ORIGINAL_INDEX}"
fi
echo "=================================================================="
echo ""

mkdir -p "${VIP_DIR}"

# ── Helper ───────────────────────────────────────────────────────────────────

step_should_run() {
    local step="$1"
    case "${START_FROM}" in
        step3) return 0 ;;
        step5) [[ "$step" != "step3" ]] && return 0 || return 1 ;;
        step6) [[ "$step" == "step6" || "$step" == "step8" ]] && return 0 || return 1 ;;
        step8) [[ "$step" == "step8" ]] && return 0 || return 1 ;;
        *)     echo "ERROR: Unknown --from value: ${START_FROM}"; exit 1 ;;
    esac
}

# ── Clean up stale outputs from the starting step onward ─────────────────────
echo "  Cleaning stale outputs from ${START_FROM} onward..."

# step3 onward: remove journey results + everything downstream
if [[ "${START_FROM}" == "step3" ]]; then
    rm -rf "${VIP_DIR}"/_journey_checkpoint_vip_users 2>/dev/null
    rm -f  "${VIP_DIR}"/vip*_Results.tsv 2>/dev/null
    rm -f  "${STEP5_OUTPUT}" 2>/dev/null
    rm -rf "${VIP_DIR}"/vip_users_*.npz "${VIP_DIR}"/vip_users_query_*.tsv 2>/dev/null
fi

# step5 onward: remove step5 output + everything downstream
if [[ "${START_FROM}" == "step3" || "${START_FROM}" == "step5" ]]; then
    rm -f  "${STEP5_OUTPUT}" 2>/dev/null
    rm -rf "${STEP6_OUTPUT_DIR}" 2>/dev/null
    rm -f  "${VIP_DIR}"/*_Ranked.tsv 2>/dev/null
    rm -f  "${VIP_DIR}"/*_L3*.html "${VIP_DIR}"/*_L3*.jsonl 2>/dev/null
fi

# step6 onward: remove ranker output + HTML
if [[ "${START_FROM}" == "step6" ]]; then
    rm -rf "${STEP6_OUTPUT_DIR}"/_ranker_ckpt_* 2>/dev/null
    rm -f  "${STEP6_OUTPUT_DIR}"/*_Results.tsv 2>/dev/null
    rm -f  "${VIP_DIR}"/*_Ranked.tsv 2>/dev/null
    rm -f  "${VIP_DIR}"/*_L3*.html "${VIP_DIR}"/*_L3*.jsonl 2>/dev/null
fi

# step8 onward: remove HTML only
if [[ "${START_FROM}" == "step8" ]]; then
    rm -f "${VIP_DIR}"/*_L3*.html "${VIP_DIR}"/*_L3*.jsonl 2>/dev/null
fi

echo "  Done."
echo ""

# =============================================================================
#  Step 3: Generate shopping journeys via LLM
# =============================================================================
if step_should_run "step3"; then
    echo "────────────────────────────────────────────────────────────────────"
    echo "  Step 3: Generate shopping journeys (LLM)"
    echo "────────────────────────────────────────────────────────────────────"

    python3 "${COOK_DIR}/step3_generate_journey_query.py" \
        --input_file "${VIP_INPUT}" \
        --output_dir "${VIP_DIR}" \
        --copilot_model "${COPILOT_MODEL}" \
        --no-random_event_window \
        --resume_checkpoint_dir \
        --combine_file "" \
        --num_workers 20 \
        --max_events 500 \
        --chunk_size 100

    # Find the actual output file (step3 appends _Journey_Results.tsv)
    STEP3_ACTUAL=$(ls -t "${VIP_DIR}"/vip*_Results.tsv 2>/dev/null | head -1)
    if [[ -z "${STEP3_ACTUAL}" ]]; then
        echo "ERROR: Step 3 output not found in ${VIP_DIR}"
        exit 1
    fi
    STEP3_OUTPUT="${STEP3_ACTUAL}"
    echo ""
    echo "  Step 3 output: ${STEP3_OUTPUT}"
    echo "  Rows: $(wc -l < "${STEP3_OUTPUT}")"
    echo ""
else
    echo "[SKIP] Step 3 (--from ${START_FROM})"
    # Try to find existing step3 output
    if [[ ! -f "${STEP3_OUTPUT}" ]]; then
        STEP3_ACTUAL=$(ls -t "${VIP_DIR}"/vip*_Results.tsv 2>/dev/null | head -1)
        if [[ -n "${STEP3_ACTUAL}" ]]; then
            STEP3_OUTPUT="${STEP3_ACTUAL}"
        fi
    fi
    echo "  Using: ${STEP3_OUTPUT}"
fi

# =============================================================================
#  Step 5: Query embedding + ANN search → JourneyWithProducts
# =============================================================================
if step_should_run "step5"; then
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  Step 5: Query embedding + ANN search"
    echo "────────────────────────────────────────────────────────────────────"

    if [[ ! -f "${STEP3_OUTPUT}" ]]; then
        echo "ERROR: Step 3 output not found: ${STEP3_OUTPUT}"
        exit 1
    fi

    python3 "${COOK_DIR}/step5_InferQueryEmbAndAnnSearch.py" \
        --input_tsv "${STEP3_OUTPUT}" \
        --item_json "${ITEM_JSON}" \
        --work_dir "${VIP_DIR}" \
        --output_prefix "${STEP5_PREFIX}" \
        --index_dir "${INDEX_DIR}" \
        --gpu_ids "${GPU_IDS}" \
        --top_k 20 \
        --keep_chunks

    echo ""
    echo "  Step 5 output: ${STEP5_OUTPUT}"
    echo "  Size: $(du -h "${STEP5_OUTPUT}" | cut -f1)"
    echo ""
else
    echo "[SKIP] Step 5 (--from ${START_FROM})"
    echo "  Using: ${STEP5_OUTPUT}"
fi

# =============================================================================
#  Step 6: LLM Ranker (inference → merge)
# =============================================================================
if step_should_run "step6"; then
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  Step 6: LLM Ranker"
    echo "────────────────────────────────────────────────────────────────────"

    if [[ ! -f "${STEP5_OUTPUT}" ]]; then
        echo "ERROR: Step 5 output not found: ${STEP5_OUTPUT}"
        exit 1
    fi

    mkdir -p "${STEP6_OUTPUT_DIR}"

    # For 22 VIP users (likely <100 journeys), run inference directly (no split)
    echo "  [step6] Running inference..."
    python3 "${COOK_DIR}/step6_call_LLM_ranker.py" \
        --input_file "${STEP5_OUTPUT}" \
        --output_dir "${STEP6_OUTPUT_DIR}" \
        --copilot_model "${STEP6_COPILOT_MODEL}" \
        --num_workers 20 \
        --max_tokens 10000

    # Merge results
    echo ""
    echo "  [step6] Merging results..."
    python3 "${COOK_DIR}/step6_call_LLM_ranker.py" \
        --input_file "${STEP5_OUTPUT}" \
        --output_dir "${VIP_DIR}" \
        --merge_dir "${STEP6_OUTPUT_DIR}"

    # Find merged output
    STEP6_ACTUAL=$(ls -t "${VIP_DIR}"/*_Ranked.tsv 2>/dev/null | head -1)
    if [[ -n "${STEP6_ACTUAL}" ]]; then
        STEP6_MERGED="${STEP6_ACTUAL}"
    fi

    echo ""
    echo "  Step 6 output: ${STEP6_MERGED}"
    echo "  Rows: $(wc -l < "${STEP6_MERGED}")"
    echo ""
else
    echo "[SKIP] Step 6 (--from ${START_FROM})"
    # Try to find existing step6 output
    STEP6_ACTUAL=$(ls -t "${VIP_DIR}"/*_Ranked.tsv 2>/dev/null | head -1)
    if [[ -n "${STEP6_ACTUAL}" ]]; then
        STEP6_MERGED="${STEP6_ACTUAL}"
    fi
    echo "  Using: ${STEP6_MERGED}"
fi

# =============================================================================
#  Step 8: L3 Reranker + HTML Visualization
# =============================================================================
if step_should_run "step8"; then
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  Step 8: HTML Visualization"
    echo "────────────────────────────────────────────────────────────────────"

    if [[ ! -f "${STEP6_MERGED}" ]]; then
        echo "ERROR: Step 6 output not found: ${STEP6_MERGED}"
        exit 1
    fi

    # Build step8 command with conditional --original_index_file
    STEP8_CMD=(python3 "${COOK_DIR}/step8_generate_html.py"
        --input "${STEP6_MERGED}"
        --results_dir "${STEP8_OUTPUT_DIR}"
        --item_json "${ITEM_JSON}"
        --top_k 12
    )
    if [[ -n "${ORIGINAL_INDEX}" && -f "${ORIGINAL_INDEX}" ]]; then
        STEP8_CMD+=(--original_index_file "${ORIGINAL_INDEX}")
    fi
    "${STEP8_CMD[@]}"

    echo ""
fi

# =============================================================================
#  Summary
# =============================================================================
echo ""
echo "=================================================================="
echo "  VIP Case Study Pipeline — Complete!"
echo "=================================================================="
echo ""
echo "  Output directory: ${VIP_DIR}"
echo ""
echo "  Files:"
ls -lhS "${VIP_DIR}"/*.tsv "${VIP_DIR}"/*.html "${VIP_DIR}"/*.jsonl 2>/dev/null | \
    awk '{printf "    %-6s  %s\n", $5, $NF}'
echo ""
echo "  To view the HTML, download:"
HTML_FILE=$(ls -t "${VIP_DIR}"/*_L3.html 2>/dev/null | head -1)
if [[ -n "${HTML_FILE}" ]]; then
    echo "    ${HTML_FILE}"
fi
echo "=================================================================="
