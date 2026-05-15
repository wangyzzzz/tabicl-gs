#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/server/code/git/TabICLv2-test"
PYTHON_BIN="/data/yes/envs/TabICLv2-GS/bin/python"
NO_PRIOR_ROOT="outputs/5.4-duli-liudang/no_prior"
BASELINE_ROOT="outputs/5.4-duli-liudang/baseline_3models"
FUSION_ROOT="outputs/5.4-duli-liudang/fusion"
LOG_DIR="${REPO_DIR}/outputs/5.4-duli-liudang/logs/fusion"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

trait_complete() {
  local path="$1"
  local kind="$2"
  PYTHONPATH=src "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
from pathlib import Path
import pandas as pd

path = Path("${path}")
kind = "${kind}"
metrics_path = path / "fold_metrics.csv"
if not metrics_path.exists():
    raise SystemExit(1)
frame = pd.read_csv(metrics_path)
folds = set(frame["fold"].dropna().astype(int))
if folds != {1, 2, 3, 4, 5}:
    raise SystemExit(1)
if kind == "no_prior":
    required = [
        path / "fold_1" / "tabicl_inner_oof_predictions.npy",
        path / "fold_1" / "tabicl_inner_oof_targets.npy",
        path / "fold_1" / "tabicl_inner_oof_summary.json",
    ]
else:
    required = []
    for model_name in ("GBLUP", "BayesB", "RKHS"):
        required.extend([
            path / "fold_1" / model_name / "inner_oof_predictions.npy",
            path / "fold_1" / model_name / "inner_oof_targets.npy",
            path / "fold_1" / model_name / "inner_oof_summary.json",
        ])
for item in required:
    if not item.exists():
        raise SystemExit(1)
raise SystemExit(0)
PY
}

fusion_complete() {
  local path="$1"
  PYTHONPATH=src "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
from pathlib import Path

path = Path("${path}")
required = [
    path / "tabicl_bayesb_single_prior" / "fold_metrics.csv",
    path / "tabicl_gblup_single_prior" / "fold_metrics.csv",
    path / "tabicl_rkhs_single_prior" / "fold_metrics.csv",
    path / "tabicl_tabicl_dual_prior" / "fold_metrics.csv",
    path / "tabicl_tabicl_triple_prior" / "fold_metrics.csv",
    path / "prior_only_bayesb" / "fold_metrics.csv",
    path / "prior_only_gblup" / "fold_metrics.csv",
    path / "prior_only_rkhs" / "fold_metrics.csv",
    path / "prior_only_bayesb_gblup" / "fold_metrics.csv",
    path / "prior_only_triple" / "fold_metrics.csv",
    path / "decoupled_fusion_summary.json",
]
raise SystemExit(0 if all(item.exists() for item in required) else 1)
PY
}

find "${REPO_DIR}/${NO_PRIOR_ROOT}" -mindepth 2 -maxdepth 2 -type d | sort | while read -r trait_dir; do
  dataset_slug="$(basename "$(dirname "${trait_dir}")")"
  trait_slug="$(basename "${trait_dir}")"
  baseline_dir="${REPO_DIR}/${BASELINE_ROOT}/${dataset_slug}/${trait_slug}"
  fusion_dir="${REPO_DIR}/${FUSION_ROOT}/${dataset_slug}/${trait_slug}"
  log_path="${LOG_DIR}/${dataset_slug}_${trait_slug}.log"

  if ! trait_complete "${trait_dir}" "no_prior"; then
    echo "[skip] $(date '+%F %T') ${dataset_slug}/${trait_slug} no-prior incomplete" | tee -a "${log_path}"
    continue
  fi
  if ! trait_complete "${baseline_dir}" "baseline"; then
    echo "[skip] $(date '+%F %T') ${dataset_slug}/${trait_slug} baseline incomplete" | tee -a "${log_path}"
    continue
  fi
  if fusion_complete "${fusion_dir}"; then
    echo "[resume] $(date '+%F %T') ${dataset_slug}/${trait_slug} fusion already complete" | tee -a "${log_path}"
    continue
  fi

  echo "[run] $(date '+%F %T') ${dataset_slug}/${trait_slug} fusion start" | tee -a "${log_path}"
  PYTHONPATH=src "${PYTHON_BIN}" scripts/run_decoupled_prior_fusion_from_archives.py \
    --tabicl-root "${NO_PRIOR_ROOT}/${dataset_slug}/${trait_slug}" \
    --baseline-root "${BASELINE_ROOT}/${dataset_slug}/${trait_slug}" \
    --output-root "${FUSION_ROOT}/${dataset_slug}/${trait_slug}" >> "${log_path}" 2>&1
  echo "[run] $(date '+%F %T') ${dataset_slug}/${trait_slug} fusion done" | tee -a "${log_path}"
done
