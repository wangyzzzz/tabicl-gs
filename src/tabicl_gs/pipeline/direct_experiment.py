from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.data.plink import (
    align_phenotype_to_sample_ids,
    impute_by_train_mean,
    load_plink_matrix,
    plink_num_snps,
    read_phenotype_table,
)
from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.eval.splits import make_outer_cv_splits
from tabicl_gs.models.tabicl import build_tabicl_regressor


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _runtime_device_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {"cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "")}
    try:
        import torch

        metadata["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metadata["process_cuda_device_index"] = int(torch.cuda.current_device())
            metadata["process_cuda_device_name"] = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception as exc:  # pragma: no cover
        metadata["torch_cuda_metadata_error"] = str(exc)
    return metadata


def run_direct_experiment(config: dict[str, Any], max_folds: int | None = None) -> pd.DataFrame:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    phenotype = read_phenotype_table(
        config["phenotype_csv"],
        sample_id_col=config.get("phenotype_sample_id_col", "sample_id"),
    )
    total_snps = plink_num_snps(config["plink_prefix"])
    selected_snp_indices = subsample_snp_indices(total_snps, config["max_snps"], config["seed"])
    plink_data = load_plink_matrix(config["plink_prefix"], snp_indices=selected_snp_indices)
    sample_id_col = config.get("phenotype_sample_id_col", "sample_id")
    phenotype, keep_indices = align_phenotype_to_sample_ids(
        phenotype,
        plink_data.sample_ids,
        sample_id_col=sample_id_col,
    )
    genotype = plink_data.matrix[np.asarray(keep_indices, dtype=np.int64)].astype(np.float32)
    target = phenotype[config["trait_col"]].to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(target)
    genotype = genotype[valid_mask]
    target = target[valid_mask]
    sample_ids = [sample_id for sample_id, keep in zip(plink_data.sample_ids, valid_mask.tolist()) if keep]
    splits = make_outer_cv_splits(genotype, config["outer_cv_folds"], config["seed"])
    if max_folds is not None:
        splits = splits[:max_folds]

    runtime_device = _runtime_device_metadata()
    metrics_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_train = genotype[train_idx]
        X_test = genotype[test_idx]
        y_train = target[train_idx]
        y_test = target[test_idx]
        X_train, X_test = impute_by_train_mean(X_train, X_test)

        model_kwargs = dict(config["model"])
        model_kwargs["random_state"] = config["seed"] + fold_id

        total_start = time.perf_counter()
        fit_start = time.perf_counter()
        model = build_tabicl_regressor(**model_kwargs)
        model.fit(X_train.astype(np.float32), y_train.astype(np.float32))
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        predictions = np.asarray(model.predict(X_test.astype(np.float32)), dtype=np.float32)
        predict_seconds = time.perf_counter() - predict_start
        total_seconds = time.perf_counter() - total_start

        metrics_rows.append(
            {
                "fold": fold_id,
                "model": "TabICLv2-direct-10000",
                "strategy": "direct",
                "n_snps": len(selected_snp_indices),
                "input_dim": genotype.shape[1],
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": total_seconds,
                "device": str(model.device_),
                "cuda_visible_devices": runtime_device.get("cuda_visible_devices", ""),
                **regression_metrics(y_test, predictions),
            }
        )

        pd.DataFrame(
            {
                "sample_id": [sample_ids[index] for index in test_idx],
                "y_true": y_test,
                "tabicl_direct_pred": predictions,
            }
        ).to_csv(fold_dir / "direct_predictions.csv", index=False)
        _save_json(
            fold_dir / "fold_metadata.json",
            {
                "fold": fold_id,
                "selected_snp_count": len(selected_snp_indices),
                "selected_snp_indices": selected_snp_indices,
                "selected_snp_ids": plink_data.snp_ids,
                "input_dim": genotype.shape[1],
                "runtime_device": runtime_device,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": total_seconds,
                "model_device": str(model.device_),
            },
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    return metrics_frame
