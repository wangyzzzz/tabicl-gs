#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/server/code/git/TabICLv2-test"
LOG_ROOT="${REPO_DIR}/outputs/5.4-duli-liudang/logs"

mkdir -p "${LOG_ROOT}"
cd "${REPO_DIR}"

echo "[full] $(date '+%F %T') start no-prior + baseline_3models" | tee -a "${LOG_ROOT}/full_pipeline.log"

bash scripts/run_54_duli_liudang_noprior_server_gpu1.sh >> "${LOG_ROOT}/full_noprior.log" 2>&1 &
PID_NOPRIOR=$!

bash scripts/run_54_duli_liudang_baseline_server_gpu1.sh >> "${LOG_ROOT}/full_baseline.log" 2>&1 &
PID_BASELINE=$!

echo "PID_NOPRIOR=${PID_NOPRIOR}" | tee -a "${LOG_ROOT}/full_pipeline.log"
echo "PID_BASELINE=${PID_BASELINE}" | tee -a "${LOG_ROOT}/full_pipeline.log"

wait "${PID_NOPRIOR}"
wait "${PID_BASELINE}"

echo "[full] $(date '+%F %T') base artifacts done; start fusion" | tee -a "${LOG_ROOT}/full_pipeline.log"
bash scripts/run_54_duli_liudang_fusion_server_gpu1.sh >> "${LOG_ROOT}/full_fusion.log" 2>&1

echo "[full] $(date '+%F %T') all done" | tee -a "${LOG_ROOT}/full_pipeline.log"
