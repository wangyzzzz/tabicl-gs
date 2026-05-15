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

## Local setup

This repository uses a standard `src` layout and `pyproject.toml`.
For development, install the project in editable mode and run the test suite from the repo root.

