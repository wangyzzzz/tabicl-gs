#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/server/code/git/TabICLv2-test"
PYTHON_BIN="/data/yes/envs/TabICLv2-GS/bin/python"
OUT_ROOT="outputs/5.4-marker_count-decoupled"
LOG_DIR="${REPO_DIR}/${OUT_ROOT}/logs"
FULL_MAIN_RESULTS="${REPO_DIR}/outputs/5.4-duli-liudang/main_results_non_pig_fixed.csv"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

TRAIT_SPECS=(
  "cotton1245|cotton_fiblen_17_18_cotton_fiblen_17_18"
  "cotton1245|cotton_fibelo_17_18_cotton_fibelo_17_18"
  "rice529|grain_weight"
  "rice529|grain_width"
  "soybean951|lw_beijing_2013_lw_beijing_2013"
  "soybean951|bbd_beijing_2013_bbd_beijing_2013"
  "wheat406|sl_e1"
  "wheat406|sl_e2"
)

run_trait() {
  local gpu="$1"
  local dataset="$2"
  local trait="$3"
  local trait_slug
  trait_slug="$(echo "${trait}" | tr '[:upper:]' '[:lower:]' | tr '/ ' '__')"
  local log_path="${LOG_DIR}/${dataset}_${trait_slug}_gpu${gpu}.log"

  echo "[gpu${gpu}] $(date '+%F %T') ${dataset} :: ${trait} start" | tee -a "${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTHONPATH=src "${PYTHON_BIN}" \
    scripts/run_54_marker_count_decoupled_trait.py \
    --dataset "${dataset}" \
    --trait-col "${trait}" \
    --output-root "${OUT_ROOT}" \
    --marker-counts 2000 50000 \
    --fold-ids 1 2 3 4 5 \
    --seed 2026 \
    --block-trials 10 \
    --block-inner-folds 3 >> "${log_path}" 2>&1
  echo "[gpu${gpu}] $(date '+%F %T') ${dataset} :: ${trait} done" | tee -a "${log_path}"
}

run_worker() {
  local gpu="$1"
  shift
  local specs=("$@")
  local spec
  for spec in "${specs[@]}"; do
    IFS="|" read -r dataset trait <<< "${spec}"
    run_trait "${gpu}" "${dataset}" "${trait}"
  done
}

WORKER0_SPECS=()
WORKER1_SPECS=()
for idx in "${!TRAIT_SPECS[@]}"; do
  if (( idx % 2 == 0 )); then
    WORKER0_SPECS+=("${TRAIT_SPECS[idx]}")
  else
    WORKER1_SPECS+=("${TRAIT_SPECS[idx]}")
  fi
done

run_worker 0 "${WORKER0_SPECS[@]}" &
PID0=$!
run_worker 1 "${WORKER1_SPECS[@]}" &
PID1=$!

echo "PID0=${PID0}" | tee "${LOG_DIR}/pipeline_pids.txt"
echo "PID1=${PID1}" | tee -a "${LOG_DIR}/pipeline_pids.txt"

wait "${PID0}"
wait "${PID1}"

PYTHONPATH=src "${PYTHON_BIN}" scripts/summarize_54_marker_count_decoupled.py \
  --marker-root "${OUT_ROOT}" \
  --full-main-results-csv "${FULL_MAIN_RESULTS}" \
  --output-csv "${OUT_ROOT}/marker_count_main_results.csv" \
  --output-json "${OUT_ROOT}/marker_count_main_results.json" | tee "${LOG_DIR}/summary.log"
