# tabicl-gs

TabICL for genomic selection, with decoupled fusion workflows, baseline models, and manuscript support scripts.

## What is included

- `src/tabicl_gs/`: core Python package
- `scripts/`: runnable pipelines, summarizers, and figure/table builders
- `tests/`: unit and integration tests
- `configs/`: YAML experiment configs
- `docs/`: manuscript drafts and notes
- `r/`: R helpers for baseline fitting

## What is intentionally excluded

- large raw genotype/phenotype data
- generated outputs and figures
- cache and temporary folders

## Main project line

- current main line: `5.4-duli-liudang`
- decoupled fusion entry point: `scripts/run_decoupled_prior_fusion_from_archives.py`
- formal results are produced on `server@GPU1`
- manuscript main results use the `two_step_ls` fusion columns from compare tables, especially
  `outputs/5.4-duli-liudang/main_results_non_pig_fixed.csv` and
  `outputs/5.4-duli-liudang/compare_all_41_traits.csv`
- the direct `fusion/*/tabicl_*_prior/fold_metrics.csv` artifacts are retained for historical
  compatibility and correspond to the `two_step_clip` fusion output, not the manuscript
  `two_step_ls` result

## Local setup

This repository uses a standard `src` layout and `pyproject.toml`.
For development, install the project in editable mode and run the test suite from the repo root.
