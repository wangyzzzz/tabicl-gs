#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/server/code/git/TabICLv2-test"
PYTHON_BIN="/data/yes/envs/TabICLv2-GS/bin/python"
RICE_PREP_SCRIPT="scripts/prepare_rice529_plink_cache.py"
MULTI_PREP_SCRIPT="scripts/prepare_multi_dataset_plink_cache.py"
CONFIG_PATH="configs/tabicl_block/window_baseline_only_3models_liudang.yaml"
OUTPUT_ROOT="outputs/5.4-duli-liudang/baseline_3models"
LOG_DIR="${REPO_DIR}/outputs/5.4-duli-liudang/logs/baseline_3models"
DATASETS=("rice529" "Cotton1245" "Soybean951" "pig3534" "wheat406")

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

dataset_slug_of() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

trait_slug_of() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '/ ' '__'
}

prepare_dataset_if_needed() {
  local dataset="$1"
  local dataset_slug
  dataset_slug="$(dataset_slug_of "${dataset}")"
  local prep_log="${LOG_DIR}/${dataset_slug}_prepare.log"
  if [[ "${dataset}" == "rice529" ]]; then
    echo "[prepare] $(date '+%F %T') ${dataset}" | tee -a "${prep_log}"
    PYTHONPATH=src "${PYTHON_BIN}" "${RICE_PREP_SCRIPT}" \
      --genotype-csv genome/rice529/rice529_gen.csv \
      --phenotype-csv genome/rice529/rice529_phe.csv \
      --plink-prefix genome/rice529/plink/rice529 \
      --sample-id-col sample_id \
      --max-snps 10000 \
      --seed 2026 >> "${prep_log}" 2>&1
    return 0
  fi

  echo "[prepare] $(date '+%F %T') ${dataset}" | tee -a "${prep_log}"
  PYTHONPATH=src "${PYTHON_BIN}" "${MULTI_PREP_SCRIPT}" \
    --dataset "${dataset}" \
    --max-snps 10000 \
    --seed 2026 >> "${prep_log}" 2>&1
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

baseline_complete() {
  local trait_root="$1"
  PYTHONPATH=src "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
from pathlib import Path
import pandas as pd

trait_root = Path("${trait_root}")
metrics_path = trait_root / "fold_metrics.csv"
required_models = {"GBLUP", "BayesB", "RKHS"}
if not metrics_path.exists():
    raise SystemExit(1)
for model_name in required_models:
    for suffix in ("inner_oof_predictions.npy", "inner_oof_targets.npy", "inner_oof_summary.json"):
        if not (trait_root / "fold_1" / model_name / suffix).exists():
            raise SystemExit(1)
frame = pd.read_csv(metrics_path)
folds = set(frame["fold"].dropna().astype(int))
models = set(frame["model"].dropna().astype(str))
raise SystemExit(0 if folds == {1, 2, 3, 4, 5} and required_models.issubset(models) else 1)
PY
}

write_trait_metadata() {
  local trait_root="$1"
  local dataset="$2"
  local trait="$3"
  PYTHONPATH=src "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

trait_root = Path("${trait_root}")
trait_root.mkdir(parents=True, exist_ok=True)
payload = {
    "dataset": "${dataset}",
    "trait_col": "${trait}",
    "baseline_models": ["GBLUP", "BayesB", "RKHS"],
    "config_path": "${CONFIG_PATH}",
    "seed": 2026,
    "max_snps": 10000,
    "outer_cv_folds": 5,
    "inner_oof_fold": 1,
    "inner_oof_n_splits": 3,
}
(trait_root / "liudang_run_metadata.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY
}

run_trait() {
  local worker_id="$1"
  local dataset="$2"
  local trait="$3"
  local dataset_slug
  dataset_slug="$(dataset_slug_of "${dataset}")"
  local trait_slug
  trait_slug="$(trait_slug_of "${trait}")"
  local log_path="${LOG_DIR}/${dataset_slug}_${trait_slug}_w${worker_id}.log"
  local out_dir="${REPO_DIR}/${OUTPUT_ROOT}/${dataset_slug}/${trait_slug}"

  local plink_prefix
  plink_prefix="$(dataset_field "${dataset}" "subset_plink_prefix" | tail -n 1)"
  local phenotype_csv
  phenotype_csv="$(dataset_field "${dataset}" "prepared_phenotype_csv" | tail -n 1)"
  local sample_id_col
  sample_id_col="$(dataset_field "${dataset}" "sample_id_col" | tail -n 1)"

  echo "[worker${worker_id}] $(date '+%F %T') ${dataset} :: ${trait} start" | tee -a "${log_path}"
  if baseline_complete "${out_dir}"; then
    echo "[resume] $(date '+%F %T') skip complete ${dataset} :: ${trait}" | tee -a "${log_path}"
    return 0
  fi

  write_trait_metadata "${out_dir}" "${dataset}" "${trait}"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
    scripts/run_tabicl_gs_experiment.py \
    --config "${CONFIG_PATH}" \
    --strategy window \
    --trait-col "${trait}" \
    --plink-prefix "${plink_prefix}" \
    --phenotype-csv "${phenotype_csv}" \
    --phenotype-sample-id-col "${sample_id_col}" \
    --output-dir "${OUTPUT_ROOT}/${dataset_slug}/${trait_slug}" >> "${log_path}" 2>&1

  echo "[worker${worker_id}] $(date '+%F %T') ${dataset} :: ${trait} done" | tee -a "${log_path}"
}

run_dataset_worker() {
  local worker_id="$1"
  shift
  local worker_datasets=("$@")
  local dataset
  for dataset in "${worker_datasets[@]}"; do
    prepare_dataset_if_needed "${dataset}"
    local traits
    mapfile -t traits < <(dataset_field "${dataset}" "trait_cols")
    local trait
    for trait in "${traits[@]}"; do
      run_trait "${worker_id}" "${dataset}" "${trait}"
    done
  done
}

WORKER0_DATASETS=()
WORKER1_DATASETS=()
for idx in "${!DATASETS[@]}"; do
  if (( idx % 2 == 0 )); then
    WORKER0_DATASETS+=("${DATASETS[idx]}")
  else
    WORKER1_DATASETS+=("${DATASETS[idx]}")
  fi
done

run_dataset_worker 0 "${WORKER0_DATASETS[@]}" &
PID0=$!
run_dataset_worker 1 "${WORKER1_DATASETS[@]}" &
PID1=$!

echo "PID0=${PID0}" | tee "${LOG_DIR}/pipeline_pids.txt"
echo "PID1=${PID1}" | tee -a "${LOG_DIR}/pipeline_pids.txt"

wait "${PID0}"
wait "${PID1}"
