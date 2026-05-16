#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/server/code/git/TabICLv2-test"
PYTHON_BIN="/data/yes/envs/TabICLv2-GS/bin/python"
CONFIG_NO_PRIOR="configs/tabicl_block/window_tabicl_dynamic99_traitscan.yaml"
CONFIG_BASELINE="configs/tabicl_block/window_baseline_only_3models_liudang.yaml"
OUTPUT_ROOT="${1:-outputs/repro_git_20260515}"
LOG_DIR="${REPO_DIR}/${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

CASES=(
  "rice529|Grain_weight|0"
  "rice529|Grain_width|1"
  "Soybean951|LW_BeiJing_2013_LW_BeiJing_2013|0"
)

dataset_slug_of() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

trait_slug_of() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '/ ' '__'
}

prepare_dataset_if_needed() {
  local dataset="$1"
  if [[ "${dataset}" == "rice529" ]]; then
    PYTHONPATH=src "${PYTHON_BIN}" scripts/prepare_rice529_plink_cache.py \
      --genotype-csv genome/rice529/rice529_gen.csv \
      --phenotype-csv genome/rice529/rice529_phe.csv \
      --plink-prefix genome/rice529/plink/rice529 \
      --sample-id-col sample_id \
      --max-snps 10000 \
      --seed 2026
    return 0
  fi

  PYTHONPATH=src "${PYTHON_BIN}" scripts/prepare_multi_dataset_plink_cache.py \
    --dataset "${dataset}" \
    --max-snps 10000 \
    --seed 2026
}

dataset_field() {
  local dataset="$1"
  local field="$2"
  PYTHONPATH=src "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

dataset = "${dataset}"
if dataset == "rice529":
    summary = Path("genome/rice529/plink/rice529_cache_summary.json")
elif dataset in {"pig3534", "wheat406"}:
    summary = Path("genome") / dataset / "plink" / f"{dataset.lower()}_cache_summary.json"
else:
    summary = Path("genome") / dataset / f"{dataset.lower()}_cache_summary.json"

data = json.loads(summary.read_text(encoding="utf-8"))
value = data["${field}"]
if isinstance(value, list):
    for item in value:
        print(item)
else:
    print(value)
PY
}

best_block_of() {
  local dataset="$1"
  local trait_slug="$2"
  local block_root
  if [[ "${dataset}" == "rice529" ]]; then
    block_root="outputs/rice529_10traits_tabicl_tabicl_tabiclxgb_dualprior"
  else
    block_root="outputs/multidataset_alltraits_dualprior/$(dataset_slug_of "${dataset}")"
  fi

  PYTHONPATH=src "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

path = Path("${block_root}") / "${trait_slug}" / "fold1_tabicl_block_search" / "best_block.json"
payload = json.loads(path.read_text(encoding="utf-8"))
print(int(payload["group_size"]))
PY
}

noprior_complete() {
  local trait_root="$1"
  PYTHONPATH=src "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
from pathlib import Path
import pandas as pd

root = Path("${trait_root}")
required = [
    root / "fold_metrics.csv",
    root / "fold_1" / "tabicl_inner_oof_predictions.npy",
    root / "fold_1" / "tabicl_inner_oof_targets.npy",
    root / "fold_1" / "tabicl_inner_oof_summary.json",
]
if not all(path.exists() for path in required):
    raise SystemExit(1)
frame = pd.read_csv(root / "fold_metrics.csv")
raise SystemExit(0 if set(frame["fold"].astype(int)) == {1, 2, 3, 4, 5} else 1)
PY
}

baseline_complete() {
  local trait_root="$1"
  PYTHONPATH=src "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
from pathlib import Path
import pandas as pd

root = Path("${trait_root}")
if not (root / "fold_metrics.csv").exists():
    raise SystemExit(1)
for model in ("BayesB", "GBLUP", "RKHS"):
    for name in ("inner_oof_predictions.npy", "inner_oof_targets.npy", "inner_oof_summary.json"):
        if not (root / "fold_1" / model / name).exists():
            raise SystemExit(1)
frame = pd.read_csv(root / "fold_metrics.csv")
models = set(frame["model"].astype(str))
folds = set(frame["fold"].astype(int))
raise SystemExit(0 if {"BayesB", "GBLUP", "RKHS"}.issubset(models) and folds == {1, 2, 3, 4, 5} else 1)
PY
}

fusion_complete() {
  local trait_root="$1"
  [[ -f "${trait_root}/decoupled_fusion_summary.json" ]] && \
  [[ -f "${trait_root}/tabicl_tabicl_triple_prior/fold_metrics.csv" ]] && \
  [[ -f "${trait_root}/prior_only_triple/fold_metrics.csv" ]]
}

compare_complete() {
  local compare_root="$1"
  [[ -f "${compare_root}/compare_main.csv" ]] && \
  [[ -f "${compare_root}/compare_main.json" ]]
}

run_noprior() {
  local dataset="$1"
  local trait="$2"
  local gpu="$3"
  local dataset_slug trait_slug out_dir log_path plink_prefix phenotype_csv sample_id_col best_block
  dataset_slug="$(dataset_slug_of "${dataset}")"
  trait_slug="$(trait_slug_of "${trait}")"
  out_dir="${OUTPUT_ROOT}/no_prior/${dataset_slug}/${trait_slug}"
  log_path="${LOG_DIR}/${dataset_slug}_${trait_slug}_noprior.log"

  if noprior_complete "${out_dir}"; then
    echo "[skip] $(date '+%F %T') no_prior ${dataset}/${trait}" | tee -a "${log_path}"
    return 0
  fi

  prepare_dataset_if_needed "${dataset}" >> "${log_path}" 2>&1
  plink_prefix="$(dataset_field "${dataset}" "subset_plink_prefix" | tail -n 1)"
  phenotype_csv="$(dataset_field "${dataset}" "prepared_phenotype_csv" | tail -n 1)"
  sample_id_col="$(dataset_field "${dataset}" "sample_id_col" | tail -n 1)"
  best_block="$(best_block_of "${dataset}" "${trait_slug}")"

  echo "[run] $(date '+%F %T') no_prior ${dataset}/${trait} gpu=${gpu} best_block=${best_block}" | tee -a "${log_path}"
  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTHONPATH=src "${PYTHON_BIN}" \
    scripts/run_tabicl_fixed_block_folds.py \
    --config "${CONFIG_NO_PRIOR}" \
    --output-root "${out_dir}" \
    --trait-col "${trait}" \
    --plink-prefix "${plink_prefix}" \
    --phenotype-csv "${phenotype_csv}" \
    --phenotype-sample-id-col "${sample_id_col}" \
    --group-size "${best_block}" \
    --fold-ids 1 2 3 4 5 >> "${log_path}" 2>&1
}

run_baseline() {
  local dataset="$1"
  local trait="$2"
  local dataset_slug trait_slug out_dir log_path plink_prefix phenotype_csv sample_id_col
  dataset_slug="$(dataset_slug_of "${dataset}")"
  trait_slug="$(trait_slug_of "${trait}")"
  out_dir="${OUTPUT_ROOT}/baseline_3models/${dataset_slug}/${trait_slug}"
  log_path="${LOG_DIR}/${dataset_slug}_${trait_slug}_baseline.log"

  if baseline_complete "${out_dir}"; then
    echo "[skip] $(date '+%F %T') baseline ${dataset}/${trait}" | tee -a "${log_path}"
    return 0
  fi

  prepare_dataset_if_needed "${dataset}" >> "${log_path}" 2>&1
  plink_prefix="$(dataset_field "${dataset}" "subset_plink_prefix" | tail -n 1)"
  phenotype_csv="$(dataset_field "${dataset}" "prepared_phenotype_csv" | tail -n 1)"
  sample_id_col="$(dataset_field "${dataset}" "sample_id_col" | tail -n 1)"

  echo "[run] $(date '+%F %T') baseline ${dataset}/${trait}" | tee -a "${log_path}"
  env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 PYTHONPATH=src "${PYTHON_BIN}" \
    scripts/run_tabicl_gs_experiment.py \
    --config "${CONFIG_BASELINE}" \
    --strategy window \
    --trait-col "${trait}" \
    --plink-prefix "${plink_prefix}" \
    --phenotype-csv "${phenotype_csv}" \
    --phenotype-sample-id-col "${sample_id_col}" \
    --output-dir "${out_dir}" >> "${log_path}" 2>&1
}

run_fusion() {
  local dataset="$1"
  local trait="$2"
  local dataset_slug trait_slug out_dir log_path
  dataset_slug="$(dataset_slug_of "${dataset}")"
  trait_slug="$(trait_slug_of "${trait}")"
  out_dir="${OUTPUT_ROOT}/fusion/${dataset_slug}/${trait_slug}"
  log_path="${LOG_DIR}/${dataset_slug}_${trait_slug}_fusion.log"

  if fusion_complete "${out_dir}"; then
    echo "[skip] $(date '+%F %T') fusion ${dataset}/${trait}" | tee -a "${log_path}"
    return 0
  fi

  echo "[run] $(date '+%F %T') fusion ${dataset}/${trait}" | tee -a "${log_path}"
  PYTHONPATH=src "${PYTHON_BIN}" scripts/run_decoupled_prior_fusion_from_archives.py \
    --tabicl-root "${OUTPUT_ROOT}/no_prior/${dataset_slug}/${trait_slug}" \
    --baseline-root "${OUTPUT_ROOT}/baseline_3models/${dataset_slug}/${trait_slug}" \
    --output-root "${out_dir}" >> "${log_path}" 2>&1
}

run_compare() {
  local dataset="$1"
  local trait="$2"
  local dataset_slug trait_slug compare_dir log_path
  dataset_slug="$(dataset_slug_of "${dataset}")"
  trait_slug="$(trait_slug_of "${trait}")"
  compare_dir="${OUTPUT_ROOT}/compare/${dataset_slug}/${trait_slug}"
  log_path="${LOG_DIR}/${dataset_slug}_${trait_slug}_compare.log"

  if compare_complete "${compare_dir}"; then
    echo "[skip] $(date '+%F %T') compare ${dataset}/${trait}" | tee -a "${log_path}"
    return 0
  fi

  mkdir -p "${compare_dir}"
  echo "[run] $(date '+%F %T') compare ${dataset}/${trait}" | tee -a "${log_path}"
  PYTHONPATH=src "${PYTHON_BIN}" scripts/compare_decoupled_weight_schemes.py \
    --trait-no-prior-root "${OUTPUT_ROOT}/no_prior/${dataset_slug}/${trait_slug}" \
    --trait-baseline-root "${OUTPUT_ROOT}/baseline_3models/${dataset_slug}/${trait_slug}" \
    --dataset "${dataset_slug}" \
    --trait-slug "${trait_slug}" \
    --output-csv "${compare_dir}/compare_main.csv" \
    --output-json "${compare_dir}/compare_main.json" >> "${log_path}" 2>&1
}

write_compare_summary() {
  PYTHONPATH=src "${PYTHON_BIN}" - <<PY
from __future__ import annotations

from pathlib import Path
import pandas as pd

root = Path("${OUTPUT_ROOT}")
frames = []
for path in sorted((root / "compare").glob("*/*/compare_main.csv")):
    frames.append(pd.read_csv(path))
if not frames:
    raise SystemExit("No compare_main.csv files found.")
frame = pd.concat(frames, ignore_index=True).sort_values(["dataset", "trait_slug"]).reset_index(drop=True)
cols = [
    "dataset",
    "trait_slug",
    "no_prior_tabicl",
    "BayesB",
    "GBLUP",
    "RKHS",
    "only_triple",
    "triple_two_step_clip",
    "triple_two_step_ls",
    "triple_all_ls",
    "single_bayesb_two_step_ls",
    "single_gblup_two_step_ls",
    "single_rkhs_two_step_ls",
]
frame[cols].to_csv(root / "repro_compare_summary.csv", index=False)
print(frame[cols].round(6).to_string(index=False))
PY
}

declare -A prepared=()
for case_spec in "${CASES[@]}"; do
  IFS="|" read -r dataset trait gpu <<< "${case_spec}"
  if [[ -z "${prepared[${dataset}]:-}" ]]; then
    prepare_dataset_if_needed "${dataset}"
    prepared["${dataset}"]=1
  fi
  run_noprior "${dataset}" "${trait}" "${gpu}" &
  noprior_pid=$!
  run_baseline "${dataset}" "${trait}" &
  baseline_pid=$!
  if ! wait "${noprior_pid}"; then
    echo "[error] no_prior failed for ${dataset}/${trait}; see ${LOG_DIR}" >&2
    exit 1
  fi
  if ! wait "${baseline_pid}"; then
    echo "[error] baseline failed for ${dataset}/${trait}; see ${LOG_DIR}" >&2
    exit 1
  fi
  run_fusion "${dataset}" "${trait}"
  run_compare "${dataset}" "${trait}"
done

write_compare_summary
echo "[done] $(date '+%F %T') reproducibility check outputs: ${OUTPUT_ROOT}"
