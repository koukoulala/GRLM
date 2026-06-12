#!/bin/bash
# =============================================================================
#  Term ID Pipeline (single version, sequential execution)
#
#  Usage:
#      bash run_tid_ab_test.sh --ver v4                        # full: step9→s1→s2→s3→step8
#      bash run_tid_ab_test.sh --ver v4 --from s1              # skip step9_0
#      bash run_tid_ab_test.sh --ver v4 --from s1 --until s2   # s1 + s2 only (no HTML)
#      bash run_tid_ab_test.sh --ver v4 --runs 3               # 3× (s1→s2) + consistency + s3→step8
#      bash run_tid_ab_test.sh --ver v4 --runs 3 --until s2    # 3× (s1→s2) + consistency, skip HTML
#      bash run_tid_ab_test.sh --ver v4 --suffix new            # output → processed_v4_new
#      bash run_tid_ab_test.sh --ver v4 --suffix new --runs 3   # output → processed_v4_new_run{1,2,3}
#
#  A/B test multiple versions in parallel:
#      nohup bash run_tid_ab_test.sh --ver v3 > logs/pipeline_v3.out 2>&1 &
#      nohup bash run_tid_ab_test.sh --ver v4 > logs/pipeline_v4.out 2>&1 &
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────

DATA_ROOT="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528"
VIP_DIR="${DATA_ROOT}/vip_case_study_IDB_new"
ITEM_FILE="${DATA_ROOT}/raw_data_IDB/item.json"
SIM_FILE="${DATA_ROOT}/raw_data_IDB/MatadorEmb_Index/similarities.json"
ITEM_JSON="${ITEM_FILE}"

FILTER_FILE="${VIP_DIR}/filter_offer_ids.txt"

VERSION=""
SUFFIX=""
START_FROM="step9"
UNTIL="step8"
RUNS=1
COPILOT_MODEL="gpt-5.4"
COPILOT_WORKERS=40

# ── Parse arguments ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ver)     VERSION="$2";         shift 2 ;;
        --suffix)  SUFFIX="$2";          shift 2 ;;
        --from)    START_FROM="$2";      shift 2 ;;
        --until)   UNTIL="$2";           shift 2 ;;
        --runs)    RUNS="$2";            shift 2 ;;
        --model)   COPILOT_MODEL="$2";   shift 2 ;;
        --workers) COPILOT_WORKERS="$2"; shift 2 ;;
        --vip_dir) VIP_DIR="$2";         shift 2 ;;
        *)         echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${VERSION}" ]] && { echo "ERROR: --ver is required (e.g., --ver v4)"; exit 1; }

# ── Derived paths ────────────────────────────────────────────────────────────

PROMPT_FILE="prompts/term_generation${VERSION^}.md"
VER_TAG="${VERSION}${SUFFIX:+_${SUFFIX}}"
OUTPUT_DIR="${VIP_DIR}/processed_${VER_TAG}"

# ── Step ordering ────────────────────────────────────────────────────────────

declare -A STEP_ORDER=( [step9]=1 [s1]=2 [s2]=3 [s3]=4 [step8]=5 )

should_run() {
    local ord="${STEP_ORDER[$1]:-}"
    [[ -z "${ord}" ]] && { echo "ERROR: Unknown step: $1"; exit 1; }
    local from_ord="${STEP_ORDER[${START_FROM}]:-}"
    local until_ord="${STEP_ORDER[${UNTIL}]:-}"
    [[ -z "${from_ord}" ]] && { echo "ERROR: Unknown --from: ${START_FROM}"; exit 1; }
    [[ -z "${until_ord}" ]] && { echo "ERROR: Unknown --until: ${UNTIL}"; exit 1; }
    [[ "${ord}" -ge "${from_ord}" && "${ord}" -le "${until_ord}" ]]
}

# ── Auto-detect ranked TSV ───────────────────────────────────────────────────

find_ranked_tsv() {
    local d f
    for d in "${VIP_DIR}/ranker_output" "${VIP_DIR}/ranker_output_gpt-5.2" "${VIP_DIR}"; do
        f=$(ls -t "${d}"/*_Ranked.tsv 2>/dev/null | head -1) || true
        [[ -n "${f}" ]] && { echo "${f}"; return; }
    done
    echo ""
}

# ── Print config ─────────────────────────────────────────────────────────────

echo "=================================================================="
echo "  Term ID Pipeline"
echo "=================================================================="
echo "  Version:     ${VERSION}"
[[ -n "${SUFFIX}" ]] && echo "  Suffix:      ${SUFFIX}"
echo "  Prompt:      ${PROMPT_FILE}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Range:       ${START_FROM} → ${UNTIL}"
echo "  Runs:        ${RUNS}"
echo "  Model:       ${COPILOT_MODEL}"
echo "  Workers:     ${COPILOT_WORKERS}"
echo "=================================================================="
echo ""

cd "${PROJECT_DIR}"

# =============================================================================
#  Step 9.0: Extract OfferIds
# =============================================================================
if should_run "step9"; then
    echo "── step9_0: Extract OfferIds ───────────────────────────────────────"
    RANKED_TSV=$(find_ranked_tsv)
    [[ -z "${RANKED_TSV}" ]] && { echo "  ERROR: Ranked TSV not found"; exit 1; }
    python3 cook_journey_data/step9_0_extract_items.py \
        --input "${RANKED_TSV}" \
        --output "${FILTER_FILE}"
    echo ""
fi

# =============================================================================
#  s1: Generate TIDs (with optional multi-run for consistency check)
# =============================================================================
if should_run "s1"; then
    echo "── s1: Generate TIDs ───────────────────────────────────────────────"

    [[ ! -f "${PROMPT_FILE}" ]] && { echo "  ERROR: Prompt not found: ${PROMPT_FILE}"; exit 1; }

    if [[ ${RUNS} -eq 1 ]]; then
        # ── Single run: s1 → s2 sequentially ──
        mkdir -p "${OUTPUT_DIR}"
        python3 -u s1_generate_tid.py \
            --prompt_file "${PROMPT_FILE}" \
            --filter_items_file "${FILTER_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --item_file "${ITEM_FILE}" \
            --similarity_file "${SIM_FILE}" \
            --copilot_model "${COPILOT_MODEL}" \
            --copilot_workers "${COPILOT_WORKERS}" \
            --resume_from_multi_path
        echo ""
    else
        # ── Multi-run: each run does s1 → s2, then compare ──
        COMPARE_FILES=()
        for ((r=1; r<=RUNS; r++)); do
            RUN_DIR="${VIP_DIR}/processed_${VER_TAG}_run${r}"
            mkdir -p "${RUN_DIR}"
            echo ""
            echo "  ══ Run ${r}/${RUNS} ══════════════════════════════════════════"
            echo "  ── s1: Generate TIDs → ${RUN_DIR}"
            python3 -u s1_generate_tid.py \
                --prompt_file "${PROMPT_FILE}" \
                --filter_items_file "${FILTER_FILE}" \
                --output_dir "${RUN_DIR}" \
                --item_file "${ITEM_FILE}" \
                --similarity_file "${SIM_FILE}" \
                --copilot_model "${COPILOT_MODEL}" \
                --copilot_workers "${COPILOT_WORKERS}"
            COMPARE_FILES+=("${RUN_DIR}/id2meta_with_norm.json")

            if should_run "s2"; then
                echo "  ── s2: Evaluate Run ${r}"
                python3 -u s2_0_evaluate_tid.py \
                    --id2meta_file "${RUN_DIR}/id2meta_with_norm.json" \
                    --copilot_model "${COPILOT_MODEL}" \
                    --copilot_workers "${COPILOT_WORKERS}"
            fi
        done

        echo ""
        echo "  ── Consistency Check (${RUNS} runs) ──"
        python3 -u s2_0_evaluate_tid.py --compare_runs "${COMPARE_FILES[@]}"

        # Copy run1 as the canonical output for downstream steps
        mkdir -p "${OUTPUT_DIR}"
        cp "${VIP_DIR}/processed_${VER_TAG}_run1/id2meta_with_norm.json" \
           "${OUTPUT_DIR}/id2meta_with_norm.json"
        echo "  Copied run1 → ${OUTPUT_DIR}/id2meta_with_norm.json"
        echo ""
    fi
fi

# =============================================================================
#  s2: Evaluate TID quality (single-run mode only; multi-run already did s2 above)
# =============================================================================
if should_run "s2" && [[ ${RUNS} -eq 1 ]]; then
    echo "── s2: Evaluate TID quality ────────────────────────────────────────"
    ID2META="${OUTPUT_DIR}/id2meta_with_norm.json"
    [[ ! -f "${ID2META}" ]] && { echo "  ERROR: ${ID2META} not found"; exit 1; }
    python3 -u s2_0_evaluate_tid.py \
        --id2meta_file "${ID2META}" \
        --copilot_model "${COPILOT_MODEL}" \
        --copilot_workers "${COPILOT_WORKERS}"
    echo ""
fi

# =============================================================================
#  s3: Build SFT data
# =============================================================================
if should_run "s3"; then
    echo "── s3: Build SFT data ──────────────────────────────────────────────"
    RANKED_TSV=$(find_ranked_tsv)
    [[ -z "${RANKED_TSV}" || ! -f "${RANKED_TSV}" ]] && { echo "  ERROR: Ranked TSV not found"; exit 1; }

    ID2META="${OUTPUT_DIR}/id2meta_with_norm.json"
    SFT_DIR="${VIP_DIR}/sft_data_${VER_TAG}"
    [[ ! -f "${ID2META}" ]] && { echo "  ERROR: ${ID2META} not found"; exit 1; }

    python3 -u s3_build_journey_sft_data.py \
        --task profile2journey \
        --ranked_journey_file "${RANKED_TSV}" \
        --id2meta_file "${ID2META}" \
        --output_dir "${SFT_DIR}"
    echo ""
fi

# =============================================================================
#  step8: Generate visualization HTML
# =============================================================================
if should_run "step8"; then
    echo "── step8: Visualization HTML ───────────────────────────────────────"
    SFT_DIR="${VIP_DIR}/sft_data_${VER_TAG}"
    VIS_JSONL="${SFT_DIR}/profile2journey_sft_for_vis.jsonl"

    if [[ -f "${VIS_JSONL}" ]]; then
        python3 -u cook_journey_data/step8_generate_html.py \
            --input "${VIS_JSONL}" \
            --results_dir "${SFT_DIR}" \
            --item_json "${ITEM_JSON}" \
            --skip_rerank
    else
        echo "  WARNING: ${VIS_JSONL} not found, skipping"
    fi
    echo ""
fi

# =============================================================================
#  Summary
# =============================================================================
echo "=================================================================="
echo "  Pipeline Complete! (${VERSION})"
echo "=================================================================="
echo "  TIDs:   ${OUTPUT_DIR}/id2meta_with_norm.json"
EVAL_DIR="${OUTPUT_DIR}/eval"
EVAL_JSON=$(ls -t "${EVAL_DIR}"/eval_statistics.json 2>/dev/null | head -1) || true
echo "  Eval:   ${EVAL_JSON:-"(not run)"}"
SFT_DIR="${VIP_DIR}/sft_data_${VER_TAG}"
if [[ -d "${SFT_DIR}" ]]; then
    echo "  SFT:    ${SFT_DIR}/"
    HTML=$(ls -t "${SFT_DIR}"/*_L3.html 2>/dev/null | head -1) || true
    echo "  HTML:   ${HTML:-"(not generated)"}"
fi
if [[ ${RUNS} -gt 1 ]]; then
    echo "  Consistency: ${VIP_DIR}/processed_${VER_TAG}_run1/consistency_report.json"
    for ((r=1; r<=RUNS; r++)); do
        RUN_EVAL="${VIP_DIR}/processed_${VER_TAG}_run${r}/eval/eval_statistics.json"
        [[ -f "${RUN_EVAL}" ]] && echo "  Eval(run${r}): ${RUN_EVAL}"
    done
fi
echo "=================================================================="
