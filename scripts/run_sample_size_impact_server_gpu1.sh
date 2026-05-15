#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/server/code/git/TabICLv2-test"
PYTHON_BIN="/data/yes/envs/TabICLv2-GS/bin/python"
DUAL_ROOT="${REPO_DIR}/outputs/multidataset_alltraits_dualprior"
BASELINE_ROOT="${REPO_DIR}/outputs/multidataset_alltraits_baseline_only"
OUT_ROOT="${REPO_DIR}/outputs/sample_size_impact"
LOG_DIR="${OUT_ROOT}/logs"
TRAIT_JSON="${OUT_ROOT}/selected_traits.json"
COMPARE_CSV="${OUT_ROOT}/current_compare.csv"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

PYTHONPATH=src "${PYTHON_BIN}" - <<'PY' > "${COMPARE_CSV}"
from pathlib import Path
import json
import pandas as pd

root_dual = Path("outputs/multidataset_alltraits_dualprior")
root_base = Path("outputs/multidataset_alltraits_baseline_only")
rows = []
for dataset_dir in sorted(root_dual.iterdir()):
    if not dataset_dir.is_dir() or dataset_dir.name == "logs":
        continue
    for trait_dir in sorted(dataset_dir.iterdir()):
        if not trait_dir.is_dir():
            continue
        dual_summary = trait_dir / "tabicl_tabicl_dual_prior" / "fold_summary.csv"
        noprior_summary = trait_dir / "tabicl_tabicl_no_prior" / "fold_summary.csv"
        prior_summary = trait_dir / "prior_only_bayesb_gblup" / "summary.json"
        baseline_fold = root_base / dataset_dir.name / trait_dir.name / "fold_metrics.csv"
        if not (dual_summary.exists() and noprior_summary.exists() and prior_summary.exists() and baseline_fold.exists()):
            continue
        dual_df = pd.read_csv(dual_summary)
        if len(dual_df) < 5:
            continue
        noprior_df = pd.read_csv(noprior_summary)
        prior = json.loads(prior_summary.read_text())
        base_df = pd.read_csv(baseline_fold)
        base_means = base_df.groupby("model", as_index=True)["pearson"].mean().to_dict()
        row = {
            "dataset": dataset_dir.name,
            "trait": trait_dir.name,
            "dual_prior": float(dual_df["pearson"].mean()),
            "prior_only": float(prior["pearson_mean"]),
            "no_prior": float(noprior_df["pearson"].mean()),
            "GBLUP": float(base_means.get("GBLUP", float("nan"))),
            "BayesA": float(base_means.get("BayesA", float("nan"))),
            "BayesB": float(base_means.get("BayesB", float("nan"))),
            "BayesLasso": float(base_means.get("BayesLasso", float("nan"))),
        }
        row["best_baseline"] = max(row["GBLUP"], row["BayesA"], row["BayesB"], row["BayesLasso"])
        row["best_baseline_model"] = max(["GBLUP","BayesA","BayesB","BayesLasso"], key=lambda m: row[m])
        row["dual_minus_prior_only"] = row["dual_prior"] - row["prior_only"]
        rows.append(row)
pd.DataFrame(rows).sort_values(["dataset","trait"]).to_csv("/dev/stdout", index=False)
PY

PYTHONPATH=src "${PYTHON_BIN}" scripts/select_sample_size_impact_traits.py \
  --compare-csv "${COMPARE_CSV}" \
  --output-json "${TRAIT_JSON}"

echo "Trait selections saved to ${TRAIT_JSON}"
