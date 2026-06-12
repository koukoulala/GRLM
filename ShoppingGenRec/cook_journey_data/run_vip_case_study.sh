#!/bin/bash
# =============================================================================
#  VIP Case Study Pipeline: vip_users.tsv → step3 → step5 → step6 → step7 → step8
# =============================================================================
#
#  Runs the full shopping journey pipeline for VIP users, producing
#  an interactive HTML visualization with product images and links.
#
#  Usage:
#      bash run_vip_case_study.sh                                          # date=20260528, all steps
#      bash run_vip_case_study.sh --date 20260528 --source IDB             # raw_data_IDB, vip_case_study_IDB
#      bash run_vip_case_study.sh --date 20260528 --source PG --from step6 # from step6
#      bash run_vip_case_study.sh --from step7                             # step7+step8 only
#      bash run_vip_case_study.sh --from step8                             # step8 only
#      bash run_vip_case_study.sh --model gpt-5.2                          # use gpt-5.2 (default: gpt-5.4)
#      bash run_vip_case_study.sh --tag new                               # vip_case_study_IDB_new
#      bash run_vip_case_study.sh --do_sft                                # also run s3 SFT + vis HTML
#
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives in cook_journey_data/, project root is one level up
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COOK_DIR="${SCRIPT_DIR}"

# ── Parse arguments ──────────────────────────────────────────────────────────

DATA_DATE="20260528"
START_FROM="step3"
SOURCE="IDB"
MODEL="gpt-5.4"
TAG=""
DO_SFT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)   DATA_DATE="$2"; shift 2 ;;
        --from)   START_FROM="$2"; shift 2 ;;
        --source) SOURCE="$2";    shift 2 ;;
        --model)  MODEL="$2";     shift 2 ;;
        --tag)    TAG="$2";       shift 2 ;;
        --do_sft) DO_SFT=true;    shift ;;
        *)        echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Paths derived from DATA_DATE + SOURCE ────────────────────────────────────

DATA_ROOT="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/${DATA_DATE}"

# SOURCE suffix for directory names (e.g., "_IDB", "_PG", or "")
if [[ -n "${SOURCE}" ]]; then
    SOURCE_SUFFIX="_${SOURCE}"
else
    SOURCE_SUFFIX=""
fi

VIP_DIR="${DATA_ROOT}/vip_case_study${SOURCE_SUFFIX}${TAG:+_${TAG}}_${MODEL}"
RAW_DIR="${DATA_ROOT}/raw_data${SOURCE_SUFFIX}"

# Input
VIP_INPUT="${PROJECT_DIR}/resources/vip_users.tsv"

ITEM_JSON="${RAW_DIR}/item.json"
INDEX_DIR="${RAW_DIR}/MatadorEmb_Index"

# Intermediate file names (all under VIP_DIR)
STEP3_OUTPUT="${VIP_DIR}/vip_users_Journey_Results.tsv"
STEP5_PREFIX="vip_users"
STEP5_OUTPUT="${VIP_DIR}/${STEP5_PREFIX}_journey_with_products.tsv"
STEP6_OUTPUT_DIR="${VIP_DIR}/ranker_output"
STEP6_MERGED=""  # auto-detected after step6 merge
STEP8_RESULTS_DIR="${VIP_DIR}/html"

# LLM settings — all LLM steps use the same model
COPILOT_MODEL="${MODEL}"
GPU_IDS="1"

echo "=================================================================="
echo "  VIP Case Study Pipeline"
echo "=================================================================="
echo "  Script dir:    ${SCRIPT_DIR}"
echo "  VIP output:    ${VIP_DIR}"
echo "  VIP input:     ${VIP_INPUT}"
echo "  Data date:     ${DATA_DATE}"
echo "  Source:        ${SOURCE:-"(default)"}"
echo "  Start from:    ${START_FROM}"
echo "  Model:         ${COPILOT_MODEL}"
echo "  Tag:           ${TAG:-"(none)"}"
echo "  Do SFT:        ${DO_SFT}"
echo "  RAW dir:       ${RAW_DIR}"
echo "  Item JSON:     ${ITEM_JSON}"
echo "  Index dir:     ${INDEX_DIR}"
echo "=================================================================="
echo ""

mkdir -p "${VIP_DIR}"

# ── Helper ───────────────────────────────────────────────────────────────────

# Ordered step list for comparison
declare -A STEP_ORDER=( [step3]=1 [step5]=2 [step6]=3 [step7]=4 [step8]=5 )

step_should_run() {
    local step="$1"
    local start_ord="${STEP_ORDER[${START_FROM}]:-}"
    local step_ord="${STEP_ORDER[${step}]:-}"
    if [[ -z "${start_ord}" ]]; then
        echo "ERROR: Unknown --from value: ${START_FROM}"; exit 1
    fi
    if [[ -z "${step_ord}" ]]; then
        echo "ERROR: Unknown step: ${step}"; exit 1
    fi
    [[ "${step_ord}" -ge "${start_ord}" ]]
}

# ── Clean up stale outputs from the starting step onward ─────────────────────
echo "  Cleaning stale outputs from ${START_FROM} onward..."

if step_should_run "step3"; then
    rm -rf "${VIP_DIR}"/_journey_checkpoint_vip_users 2>/dev/null || true
    rm -f  "${VIP_DIR}"/vip*_Results.tsv 2>/dev/null || true
    rm -rf "${VIP_DIR}"/vip_users_*.npz "${VIP_DIR}"/vip_users_query_*.tsv 2>/dev/null || true
fi

if step_should_run "step5"; then
    rm -f  "${STEP5_OUTPUT}" 2>/dev/null || true
fi

if step_should_run "step6"; then
    rm -rf "${STEP6_OUTPUT_DIR}"/_ranker_ckpt_* 2>/dev/null || true
    rm -f  "${STEP6_OUTPUT_DIR}"/*_Results.tsv 2>/dev/null || true
    rm -f  "${VIP_DIR}"/*_Ranked.tsv 2>/dev/null || true
fi

if step_should_run "step7"; then
    : # step7 is read-only stats, nothing to clean
fi

if step_should_run "step8"; then
    rm -rf "${STEP8_RESULTS_DIR}" 2>/dev/null || true
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

    # For VIP users (small set), run inference directly (no split)
    echo "  [step6] Running inference..."
    python3 "${COOK_DIR}/step6_call_LLM_ranker.py" \
        --input_file "${STEP5_OUTPUT}" \
        --output_dir "${STEP6_OUTPUT_DIR}" \
        --copilot_model "${COPILOT_MODEL}" \
        --num_workers 20 \
        --max_tokens 10000

    # Merge results (output to same model-specific dir)
    echo ""
    echo "  [step6] Merging results..."
    python3 "${COOK_DIR}/step6_call_LLM_ranker.py" \
        --input_file "${STEP5_OUTPUT}" \
        --output_dir "${STEP6_OUTPUT_DIR}" \
        --merge_dir "${STEP6_OUTPUT_DIR}"

    # Find merged output
    STEP6_ACTUAL=$(ls -t "${STEP6_OUTPUT_DIR}"/*_Ranked.tsv 2>/dev/null | head -1)
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
    STEP6_ACTUAL=$(ls -t "${STEP6_OUTPUT_DIR}"/*_Ranked.tsv 2>/dev/null | head -1)
    if [[ -n "${STEP6_ACTUAL}" ]]; then
        STEP6_MERGED="${STEP6_ACTUAL}"
    fi
    echo "  Using: ${STEP6_MERGED}"
fi

# =============================================================================
#  Step 7: Statistics on ranked output
# =============================================================================
if step_should_run "step7"; then
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  Step 7: Data statistics"
    echo "────────────────────────────────────────────────────────────────────"

    if [[ -z "${STEP6_MERGED}" ]]; then
        STEP6_ACTUAL=$(ls -t "${STEP6_OUTPUT_DIR}"/*_Ranked.tsv 2>/dev/null | head -1)
        if [[ -n "${STEP6_ACTUAL}" ]]; then
            STEP6_MERGED="${STEP6_ACTUAL}"
        fi
    fi

    if [[ -n "${STEP6_MERGED}" && -f "${STEP6_MERGED}" ]]; then
        python3 "${COOK_DIR}/step7_stats.py" \
            --input_file "${STEP6_MERGED}"
    else
        echo "  WARNING: No ranked TSV found, skipping step7"
    fi
    echo ""
else
    echo "[SKIP] Step 7 (--from ${START_FROM})"
fi

# =============================================================================
#  Step 8: L3 Reranker + HTML Visualization
# =============================================================================
if step_should_run "step8"; then
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  Step 8: HTML Visualization"
    echo "────────────────────────────────────────────────────────────────────"

    if [[ -z "${STEP6_MERGED}" ]]; then
        STEP6_ACTUAL=$(ls -t "${STEP6_OUTPUT_DIR}"/*_Ranked.tsv 2>/dev/null | head -1)
        if [[ -n "${STEP6_ACTUAL}" ]]; then
            STEP6_MERGED="${STEP6_ACTUAL}"
        fi
    fi

    if [[ -z "${STEP6_MERGED}" || ! -f "${STEP6_MERGED}" ]]; then
        echo "ERROR: Step 6 output not found in ${STEP6_OUTPUT_DIR}"
        exit 1
    fi

    mkdir -p "${STEP8_RESULTS_DIR}"
    python3 "${COOK_DIR}/step8_generate_html.py" \
        --input "${STEP6_MERGED}" \
        --results_dir "${STEP8_RESULTS_DIR}" \
        --item_json "${ITEM_JSON}" \
        --top_k 12

    echo ""
else
    echo "[SKIP] Step 8 (--from ${START_FROM})"
fi

# =============================================================================
#  SFT Data Generation (optional: --do_sft)
# =============================================================================
if [[ "${DO_SFT}" == true ]]; then
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  SFT: s3 Journey SFT Data + Visualization HTML"
    echo "────────────────────────────────────────────────────────────────────"

    if [[ -z "${STEP6_MERGED}" ]]; then
        STEP6_ACTUAL=$(ls -t "${STEP6_OUTPUT_DIR}"/*_Ranked.tsv 2>/dev/null | head -1)
        if [[ -n "${STEP6_ACTUAL}" ]]; then
            STEP6_MERGED="${STEP6_ACTUAL}"
        fi
    fi

    if [[ -z "${STEP6_MERGED}" || ! -f "${STEP6_MERGED}" ]]; then
        echo "  WARNING: No ranked TSV found, skipping SFT"
    else
        ID2META="${DATA_ROOT}/processed${SOURCE_SUFFIX}/id2meta_with_norm.json"
        SFT_DIR="${VIP_DIR}/sft_data"

        if [[ ! -f "${ID2META}" ]]; then
            echo "  WARNING: id2meta not found: ${ID2META}, skipping SFT"
        else
            # Run s3 for profile2journey
            echo ""
            echo "  [SFT] Running s3 (profile2journey)..."
            python3 "${PROJECT_DIR}/s3_build_journey_sft_data.py" \
                --task profile2journey \
                --ranked_journey_file "${STEP6_MERGED}" \
                --id2meta_file "${ID2META}" \
                --output_dir "${SFT_DIR}"

            # Generate vis HTML from the profile2journey vis JSONL
            VIS_JSONL="${SFT_DIR}/profile2journey_sft_for_vis.jsonl"
            if [[ -f "${VIS_JSONL}" ]]; then
                echo ""
                echo "  [SFT] Generating visualization HTML..."
                python3 "${COOK_DIR}/step8_generate_html.py" \
                    --input "${VIS_JSONL}" \
                    --results_dir "${SFT_DIR}" \
                    --item_json "${ITEM_JSON}" \
                    --skip_rerank
            fi

            echo ""
            echo "  SFT output: ${SFT_DIR}"
            ls -lhS "${SFT_DIR}"/*.json "${SFT_DIR}"/*.jsonl "${SFT_DIR}"/*.html 2>/dev/null | \
                awk '{printf "    %-6s  %s\n", $5, $NF}'
        fi
    fi
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
echo "  Ranker output:   ${STEP6_OUTPUT_DIR}"
echo "  HTML output:     ${STEP8_RESULTS_DIR}"
echo ""
echo "  Files:"
ls -lhS "${VIP_DIR}"/*.tsv "${STEP6_OUTPUT_DIR}"/*.tsv "${STEP8_RESULTS_DIR}"/*.html "${STEP8_RESULTS_DIR}"/*.jsonl 2>/dev/null | \
    awk '{printf "    %-6s  %s\n", $5, $NF}'
echo ""
echo "  To view the HTML, download:"
HTML_FILE=$(ls -t "${STEP8_RESULTS_DIR}"/*_L3.html 2>/dev/null | head -1)
if [[ -n "${HTML_FILE}" ]]; then
    echo "    ${HTML_FILE}"
fi
echo "=================================================================="
