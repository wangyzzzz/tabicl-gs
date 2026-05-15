from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from tabicl_gs.eval.splits import make_outer_cv_splits


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "eval_prior_only_from_dual_outputs.py"
    spec = importlib.util.spec_from_file_location("eval_prior_only_from_dual_outputs", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_estimate_alpha_uses_fold1_inner_val_oof_prior_not_saved_alpha_group(tmp_path, monkeypatch):
    module = _load_script_module()

    X_train = np.arange(12, dtype=np.float32).reshape(6, 2)
    y_train = np.array([0.20, 0.60, 0.90, 0.10, 0.80, 0.40], dtype=np.float32)

    def fake_load_fold_data(config, fold_id):
        assert fold_id == 1
        X_test = np.zeros((2, 2), dtype=np.float32)
        y_test = np.zeros(2, dtype=np.float32)
        return X_train, y_train, X_test, y_test

    monkeypatch.setattr(module, "_load_fold_data_for_prior_eval", fake_load_fold_data)

    dual_dir = tmp_path / "trait_root" / "tabicl_tabicl_dual_prior"
    prior_cache = dual_dir / "fold_1" / "prior_cache"
    prior_cache.mkdir(parents=True, exist_ok=True)

    # 即使历史 summary 里有落盘的 alpha_group，这里也必须完全忽略它。
    saved_group_summary = dual_dir / "fold_1" / "group_shared_gate_group_summary.json"
    saved_group_summary.write_text(
        json.dumps({"alpha_group": [0.99], "w_group": [0.5]}, ensure_ascii=False),
        encoding="utf-8",
    )

    inner_splits = make_outer_cv_splits(X_train, 3, 2027)
    bayesb_oof = np.zeros_like(y_train, dtype=np.float32)
    gblup_oof = np.zeros_like(y_train, dtype=np.float32)

    for inner_id, (_, valid_idx) in enumerate(inner_splits, start=1):
        inner_dir = prior_cache / f"inner_{inner_id}"
        inner_dir.mkdir(parents=True, exist_ok=True)
        bayesb_valid = np.linspace(0.3, 0.3 + 0.1 * (len(valid_idx) - 1), len(valid_idx), dtype=np.float32)
        gblup_valid = np.linspace(0.1, 0.1 + 0.05 * (len(valid_idx) - 1), len(valid_idx), dtype=np.float32)
        np.save(inner_dir / "bayesb_valid.npy", bayesb_valid)
        np.save(inner_dir / "gblup_valid.npy", gblup_valid)
        bayesb_oof[np.asarray(valid_idx, dtype=np.int64)] = bayesb_valid
        gblup_oof[np.asarray(valid_idx, dtype=np.int64)] = gblup_valid

    expected_alpha = float(np.mean(module._clip_alpha_targets(y_train, bayesb_oof, gblup_oof)))
    observed_alpha = module._estimate_alpha_from_fold1_oof(
        dual_dir,
        {"seed": 2026},
    )

    assert np.isclose(observed_alpha, expected_alpha)
    assert not np.isclose(observed_alpha, 0.99)


def test_evaluate_prior_only_reports_alpha_source_metadata(tmp_path, monkeypatch):
    module = _load_script_module()

    monkeypatch.setattr(module, "_estimate_alpha_from_fold1_oof", lambda dual_dir, base_config: 0.25)

    def fake_load_fold_predictions(dual_fold_dir, base_config, fold_id):
        y_true = np.array([1.0, 2.0], dtype=np.float32)
        bayesb_test = np.array([2.0, 3.0], dtype=np.float32)
        gblup_test = np.array([0.0, 1.0], dtype=np.float32)
        return y_true, bayesb_test, gblup_test

    monkeypatch.setattr(module, "_load_fold_predictions", fake_load_fold_predictions)

    dual_root = tmp_path / "dual"
    fold_df, metadata = module.evaluate_prior_only(
        dual_root=dual_root,
        dataset="Cotton1245",
        trait_col="trait_x",
    )

    assert len(fold_df) == 5
    assert np.allclose(fold_df["alpha"].to_numpy(dtype=np.float32), 0.25)
    assert metadata["alpha"] == 0.25
    assert metadata["alpha_source"] == "fold1_inner_val_oof_prior"
    assert str(metadata["alpha_source_dir"]).endswith("tabicl_tabicl_dual_prior/fold_1/prior_cache")


def test_load_fold_predictions_reads_dual_prior_cache_test_arrays(tmp_path, monkeypatch):
    module = _load_script_module()

    dual_fold_dir = tmp_path / "tabicl_tabicl_dual_prior" / "fold_3"
    prior_cache = dual_fold_dir / "prior_cache"
    prior_cache.mkdir(parents=True, exist_ok=True)
    np.save(prior_cache / "bayesb_test.npy", np.array([0.1, 0.2], dtype=np.float32))
    np.save(prior_cache / "gblup_test.npy", np.array([0.3, 0.4], dtype=np.float32))

    def fake_load_fold_data(config, fold_id):
        assert fold_id == 3
        return (
            np.zeros((4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        )

    monkeypatch.setattr(module, "_load_fold_data_for_prior_eval", fake_load_fold_data)

    y_true, bayesb_test, gblup_test = module._load_fold_predictions(
        dual_fold_dir=dual_fold_dir,
        base_config={"seed": 2026},
        fold_id=3,
    )

    assert np.allclose(y_true, np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(bayesb_test, np.array([0.1, 0.2], dtype=np.float32))
    assert np.allclose(gblup_test, np.array([0.3, 0.4], dtype=np.float32))


def test_load_fold_predictions_falls_back_to_outer_prediction_files(tmp_path, monkeypatch):
    module = _load_script_module()

    dual_fold_dir = tmp_path / "tabicl_tabicl_dual_prior" / "fold_2"
    (dual_fold_dir / "prior_cache" / "bayesb_outer" / "eval_fit").mkdir(parents=True, exist_ok=True)
    (dual_fold_dir / "prior_cache" / "gblup_outer" / "_residual_target" / "GBLUP" / "test_fit").mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savetxt(
        dual_fold_dir / "prior_cache" / "bayesb_outer" / "eval_fit" / "predictions.csv",
        np.array([0.11, 0.22], dtype=np.float32),
        delimiter=",",
    )
    np.savetxt(
        dual_fold_dir / "prior_cache" / "gblup_outer" / "_residual_target" / "GBLUP" / "test_fit" / "predictions.csv",
        np.array([0.33, 0.44], dtype=np.float32),
        delimiter=",",
    )

    def fake_load_fold_data(config, fold_id):
        assert fold_id == 2
        return (
            np.zeros((4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        )

    monkeypatch.setattr(module, "_load_fold_data_for_prior_eval", fake_load_fold_data)

    y_true, bayesb_test, gblup_test = module._load_fold_predictions(
        dual_fold_dir=dual_fold_dir,
        base_config={"seed": 2026},
        fold_id=2,
    )

    assert np.allclose(y_true, np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(bayesb_test, np.array([0.11, 0.22], dtype=np.float32))
    assert np.allclose(gblup_test, np.array([0.33, 0.44], dtype=np.float32))


def test_build_base_config_supports_rice529():
    module = _load_script_module()

    config = module._build_base_config("rice529", "Heading_date")

    assert config["trait_col"] == "Heading_date"
    assert config["phenotype_sample_id_col"] == "sample_id"
    assert config["plink_prefix"].endswith("genome/rice529/plink/rice529_max10000_seed2026")
    assert config["phenotype_csv"].endswith("genome/rice529/rice529_phe.csv")
