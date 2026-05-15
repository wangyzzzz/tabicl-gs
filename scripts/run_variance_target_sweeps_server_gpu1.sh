#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/server/code/git/TabICLv2-test"
PYTHON_BIN="/data/yes/envs/TabICLv2-GS/bin/python"
BASE_CONFIG="configs/tabicl_block/window_tabicl_optuna20.yaml"
OUTPUT_ROOT="outputs/variance_target_sweeps_2traits"
LOG_DIR="${REPO_DIR}/${OUTPUT_ROOT}/logs"
TARGETS=(0.90 0.95 0.97 0.99)

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

run_trait() {
  local gpu="$1"
  local trait="$2"
  local group_size="$3"
  local include_scalar="$4"
  local slug="$5"
  local log_path="${LOG_DIR}/${slug}_gpu${gpu}.log"
  echo "[gpu${gpu}] $(date '+%F %T') start ${trait}" | tee -a "${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
    scripts/run_tabicl_variance_target_sweep.py \
    --config "${BASE_CONFIG}" \
    --trait-col "${trait}" \
    --group-size "${group_size}" \
    --include-block-scalar "${include_scalar}" \
    --output-root "${OUTPUT_ROOT}/${slug}" \
    --variance-targets "${TARGETS[@]}" >> "${log_path}" 2>&1
  echo "[gpu${gpu}] $(date '+%F %T') done ${trait}" | tee -a "${log_path}"
}

run_trait 0 "Heading_date" 500 false "heading_date" &
PID0=$!
run_trait 1 "Num_panicles" 200 false "num_panicles" &
PID1=$!

echo "PID0=${PID0}" | tee "${LOG_DIR}/pipeline_pids.txt"
echo "PID1=${PID1}" | tee -a "${LOG_DIR}/pipeline_pids.txt"

wait "${PID0}"
wait "${PID1}"
