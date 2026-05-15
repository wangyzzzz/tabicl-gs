#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/server/code/git/TabICLv2-test"
PYTHON_BIN="/data/yes/envs/TabICLv2-GS/bin/python"
OUT_ROOT="${REPO_DIR}/outputs/sample_size_impact_fold1_screening"
LOG_DIR="${OUT_ROOT}/logs"
TRAIT_JSON="${OUT_ROOT}/selected_traits.json"
COMPARE_CSV="${OUT_ROOT}/current_compare.csv"
PROPORTIONS=(0.2 1.0)
BLOCK_SEARCH_PROPORTIONS=(0.2 1.0)
FOLD_IDS=(1)
DATASETS=("soybean951" "pig3534" "wheat406")

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

prepare_rice529() {
  echo "[prepare] $(date '+%F %T') rice529" | tee -a "${LOG_DIR}/rice529_prepare.log"
  PYTHONPATH=src "${PYTHON_BIN}" scripts/prepare_rice529_plink_cache.py \
    --genotype-csv genome/rice529/rice529_gen.csv \
    --phenotype-csv genome/rice529/rice529_phe.csv \
    --plink-prefix genome/rice529/plink/rice529 \
    --sample-id-col sample_id \
    --max-snps 10000 \
    --seed 2026 >> "${LOG_DIR}/rice529_prepare.log" 2>&1
}

prepare_other_datasets() {
  echo "[prepare] $(date '+%F %T') multi-dataset cache" | tee -a "${LOG_DIR}/multidataset_prepare.log"
  PYTHONPATH=src "${PYTHON_BIN}" scripts/prepare_multi_dataset_plink_cache.py \
    --dataset Cotton1245 Soybean951 pig3534 wheat406 \
    --max-snps 10000 \
    --seed 2026 >> "${LOG_DIR}/multidataset_prepare.log" 2>&1
}

contains_dataset() {
  local target="$1"
  local dataset
  for dataset in "${DATASETS[@]}"; do
    if [[ "${dataset}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

build_compare_csv() {
  PYTHONPATH=src "${PYTHON_BIN}" - <<'PY' > "${COMPARE_CSV}"
from pathlib import Path
import json
import pandas as pd
import csv

rows = []

def infer_trait_col(dataset_name: str, trait_slug: str) -> str:
    if dataset_name == "rice529":
        path = Path("genome/rice529/rice529_phe.csv")
        sep = ","
        sample_id_col = "sample_id"
    elif dataset_name == "cotton1245":
        path = Path("genome/Cotton1245/Cotton_all.txt")
        sep = "\t"
        sample_id_col = "Taxa"
    elif dataset_name == "soybean951":
        path = Path("genome/Soybean951/Soybean_all.txt")
        sep = "\t"
        sample_id_col = "Taxa"
    elif dataset_name == "pig3534":
        path = Path("genome/pig3534/phenotypes.txt")
        sep = ","
        sample_id_col = "ID"
    elif dataset_name == "wheat406":
        path = Path("genome/wheat406/wheat406_phe.csv")
        sep = ","
        sample_id_col = "sample_id"
    else:
        return trait_slug

    with path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f, delimiter=sep))
    candidates = [c for c in header if c != sample_id_col]
    slug_map = {c.lower().replace("/", "__").replace(" ", "__"): c for c in candidates}
    return slug_map.get(trait_slug, trait_slug)

def add_rows_from_root(dataset_name, dual_root, baseline_root):
    dual_root = Path(dual_root)
    baseline_root = Path(baseline_root)
    if not dual_root.exists() or not baseline_root.exists():
        return
    for trait_dir in sorted(dual_root.iterdir()):
        if not trait_dir.is_dir():
            continue
        dual_summary = trait_dir / "tabicl_tabicl_dual_prior" / "fold_summary.csv"
        noprior_summary = trait_dir / "tabicl_tabicl_no_prior" / "fold_summary.csv"
        baseline_fold = baseline_root / trait_dir.name / "fold_metrics.csv"
        prior_summary = trait_dir / "prior_only_bayesb_gblup" / "summary.json"
        if not (dual_summary.exists() and noprior_summary.exists() and baseline_fold.exists()):
            continue
        dual_df = pd.read_csv(dual_summary)
        noprior_df = pd.read_csv(noprior_summary)
        if len(dual_df) < 5 or len(noprior_df) < 5:
            continue
        if prior_summary.exists():
            prior = json.loads(prior_summary.read_text())
            prior_only = float(prior["pearson_mean"])
        else:
            prior_only = float("nan")
        base_df = pd.read_csv(baseline_fold)
        pearson_col = "pearson" if "pearson" in base_df.columns else "test_pearson"
        base_means = base_df.groupby("model", as_index=True)[pearson_col].mean().to_dict()
        row = {
            "dataset": dataset_name,
            "trait": trait_dir.name,
            "trait_col": infer_trait_col(dataset_name, trait_dir.name),
            "dual_prior": float(dual_df["pearson"].mean()),
            "prior_only": prior_only,
            "no_prior": float(noprior_df["pearson"].mean()),
            "GBLUP": float(base_means.get("GBLUP", float("nan"))),
            "BayesA": float(base_means.get("BayesA", float("nan"))),
            "BayesB": float(base_means.get("BayesB", float("nan"))),
            "BayesLasso": float(base_means.get("BayesLasso", float("nan"))),
        }
        row["best_baseline"] = max(row["GBLUP"], row["BayesA"], row["BayesB"], row["BayesLasso"])
        row["best_baseline_model"] = max(["GBLUP", "BayesA", "BayesB", "BayesLasso"], key=lambda m: row[m])
        row["dual_minus_prior_only"] = row["dual_prior"] - row["prior_only"] if pd.notna(row["prior_only"]) else float("nan")
        rows.append(row)

add_rows_from_root(
    "rice529",
    "outputs/rice529_10traits_tabicl_tabicl_tabiclxgb_dualprior",
    "outputs/rice529_10traits_baseline_only",
)

for dataset in ["cotton1245", "soybean951", "pig3534", "wheat406"]:
    add_rows_from_root(
        dataset,
        f"outputs/multidataset_alltraits_dualprior/{dataset}",
        f"outputs/multidataset_alltraits_baseline_only/{dataset}",
    )

pd.DataFrame(rows).sort_values(["dataset", "trait"]).to_csv("/dev/stdout", index=False)
PY
}

dataset_field() {
  local dataset="$1"
  local field="$2"
  PYTHONPATH=src "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
dataset = "${dataset}"
dataset_lower = dataset.lower()
if dataset_lower == "rice529":
    summary = Path("genome/rice529/plink/rice529_cache_summary.json")
elif dataset in {"pig3534", "wheat406"}:
    summary = Path("genome") / dataset / "plink" / f"{dataset_lower}_cache_summary.json"
else:
    title = dataset if dataset[0].isupper() else dataset.capitalize()
    summary = Path("genome") / title / f"{dataset_lower}_cache_summary.json"
data = json.loads(summary.read_text(encoding="utf-8"))
value = data["${field}"]
if isinstance(value, list):
    for item in value:
        print(item)
else:
    print(value)
PY
}

run_trait() {
  local gpu="$1"
  local dataset="$2"
  local selection_tag="$3"
  local trait_col="$4"
  local trait_slug_input="$5"
  local dataset_slug
  dataset_slug="$(echo "${dataset}" | tr '[:upper:]' '[:lower:]')"
  local trait_slug
  trait_slug="${trait_slug_input}"
  local trait_root="outputs/sample_size_impact_fold1_screening/${dataset_slug}/${trait_slug}"
  local log_path="${LOG_DIR}/${dataset_slug}_${trait_slug}_gpu${gpu}.log"

  local plink_prefix
  local phenotype_csv
  local sample_id_col
  plink_prefix="$(dataset_field "${dataset}" "subset_plink_prefix" | tail -n 1)"
  phenotype_csv="$(dataset_field "${dataset}" "prepared_phenotype_csv" | tail -n 1)"
  sample_id_col="$(dataset_field "${dataset}" "sample_id_col" | tail -n 1)"

  echo "[gpu${gpu}] $(date '+%F %T') ${dataset} :: ${trait_col} -> ${trait_slug} (${selection_tag}) start" | tee -a "${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTHONPATH=src "${PYTHON_BIN}" \
    scripts/run_sample_size_impact_experiment.py \
    --config configs/tabicl_block/window_tabicl_group_shared_gate_prior.yaml \
    --output-root "${trait_root}" \
    --trait-col "${trait_col}" \
    --plink-prefix "${plink_prefix}" \
    --phenotype-csv "${phenotype_csv}" \
    --phenotype-sample-id-col "${sample_id_col}" \
    --group-size 823 \
    --fold-ids "${FOLD_IDS[@]}" \
    --proportions "${PROPORTIONS[@]}" \
    --repeats 3 \
    --selection-tag "${selection_tag}" \
    --block-search-proportions "${BLOCK_SEARCH_PROPORTIONS[@]}" \
    --block-search-min 200 \
    --block-search-max 1500 \
    --block-search-trials 10 \
    --block-search-inner-folds 3 >> "${log_path}" 2>&1
  echo "[gpu${gpu}] $(date '+%F %T') ${dataset} :: ${trait_col} done" | tee -a "${log_path}"
}

run_worker() {
  local gpu="$1"
  shift
  local entries=("$@")
  local entry
  for entry in "${entries[@]}"; do
    IFS=$'\t' read -r dataset selection_tag trait_slug trait_col <<< "${entry}"
    run_trait "${gpu}" "${dataset}" "${selection_tag}" "${trait_col}" "${trait_slug}"
  done
}

if contains_dataset "rice529"; then
  prepare_rice529
fi
prepare_other_datasets
build_compare_csv

PYTHONPATH=src "${PYTHON_BIN}" scripts/select_sample_size_impact_traits.py \
  --compare-csv "${COMPARE_CSV}" \
  --output-json "${TRAIT_JSON}" \
  --datasets "${DATASETS[@]}"

mapfile -t TRAIT_ENTRIES < <(PYTHONPATH=src "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
data = json.loads(Path("${TRAIT_JSON}").read_text(encoding="utf-8"))
order = "${DATASETS[*]}".split()
for dataset in order:
    info = data[dataset]
    print(f"{dataset}\tgain_trait\t{info['gain_trait']['trait']}\t{info['gain_trait']['trait_col']}")
    print(f"{dataset}\tflat_trait\t{info['flat_trait']['trait']}\t{info['flat_trait']['trait_col']}")
PY
)

WORKER0_ENTRIES=()
WORKER1_ENTRIES=()
for idx in "${!TRAIT_ENTRIES[@]}"; do
  if (( idx % 2 == 0 )); then
    WORKER0_ENTRIES+=("${TRAIT_ENTRIES[idx]}")
  else
    WORKER1_ENTRIES+=("${TRAIT_ENTRIES[idx]}")
  fi
done

run_worker 0 "${WORKER0_ENTRIES[@]}" &
PID0=$!
run_worker 1 "${WORKER1_ENTRIES[@]}" &
PID1=$!

echo "PID0=${PID0}" | tee "${LOG_DIR}/pipeline_pids.txt"
echo "PID1=${PID1}" | tee -a "${LOG_DIR}/pipeline_pids.txt"

wait "${PID0}"
wait "${PID1}"
