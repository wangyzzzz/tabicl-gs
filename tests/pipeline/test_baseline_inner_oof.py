from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import tabicl_gs.pipeline.experiment as experiment


def test_run_experiment_saves_baseline_fold1_inner_oof(monkeypatch, tmp_path: Path):
    phenotype = np.array(
        [
            ("s0", 0.0),
            ("s1", 1.0),
            ("s2", 2.0),
            ("s3", 3.0),
            ("s4", 4.0),
            ("s5", 5.0),
        ],
        dtype=[("sample_id", "U8"), ("Trait", "f4")],
    )

    class FakePhenotypeTable:
        def __init__(self, arr):
            self._arr = arr

        def __getitem__(self, key):
            if key == "sample_id":
                return self._arr["sample_id"]
            if key == "Trait":
                return self._arr["Trait"]
            raise KeyError(key)

    class FakeSeries:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def to_numpy(self, dtype=None):
            if dtype is None:
                return np.asarray(self._arr)
            return np.asarray(self._arr, dtype=dtype)

    class FakeFrame:
        def __init__(self, rows):
            self.rows = list(rows)

        def to_csv(self, path, index=False):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not self.rows:
                path.write_text("", encoding="utf-8")
                return
            headers = list(self.rows[0].keys())
            lines = [",".join(headers)]
            for row in self.rows:
                lines.append(",".join("" if row[h] is None else str(row[h]) for h in headers))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(experiment, "read_phenotype_table", lambda *args, **kwargs: FakePhenotypeTable(phenotype))
    monkeypatch.setattr(experiment, "plink_num_snps", lambda *args, **kwargs: 4)
    monkeypatch.setattr(experiment, "subsample_snp_indices", lambda total, max_snps, seed: [0, 1, 2, 3])

    class FakePlink:
        sample_ids = [f"s{i}" for i in range(6)]
        snp_ids = [f"snp_{i}" for i in range(4)]
        matrix = np.arange(24, dtype=np.float32).reshape(6, 4)

    monkeypatch.setattr(experiment, "load_plink_matrix", lambda *args, **kwargs: FakePlink())

    def fake_align(phenotype_table, sample_ids, sample_id_col="sample_id"):
        return {"Trait": FakeSeries(phenotype["Trait"])}, np.arange(len(sample_ids), dtype=np.int64)

    monkeypatch.setattr(experiment, "align_phenotype_to_sample_ids", fake_align)
    monkeypatch.setattr(experiment, "impute_by_train_mean", lambda train, test: (np.asarray(train), np.asarray(test)))
    monkeypatch.setattr(
        experiment,
        "make_outer_cv_splits",
        lambda X, n_splits, seed: [
            (np.array([0, 1, 2, 3], dtype=np.int64), np.array([4, 5], dtype=np.int64)),
            (np.array([2, 3, 4, 5], dtype=np.int64), np.array([0, 1], dtype=np.int64)),
        ]
        if X.shape[0] == 6
        else [
            (np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)),
            (np.array([0, 2, 3], dtype=np.int64), np.array([1], dtype=np.int64)),
            (np.array([1, 2, 3], dtype=np.int64), np.array([0], dtype=np.int64)),
        ],
    )
    monkeypatch.setattr(experiment, "resolve_two_stage_model_specs", lambda config: [])

    def fake_run_r_baseline(
        model_name,
        X_train,
        y_train,
        X_test,
        output_dir,
        rscript_path,
        seed,
        sommer_method=None,
        keep_artifacts=True,
        return_beta=False,
        bandwidth_scale=None,
    ):
        class Result:
            predictions = np.full(X_test.shape[0], fill_value=float(len(model_name)), dtype=np.float32)
            metadata = {"device": "R"}
            command = ["fake", model_name]
            beta = None

        return Result()

    monkeypatch.setattr(experiment, "run_r_baseline", fake_run_r_baseline)
    monkeypatch.setattr(experiment.pd, "DataFrame", lambda rows: FakeFrame(rows))

    config = {
        "seed": 2026,
        "max_snps": 4,
        "grouping_strategy": "window",
        "outer_cv_folds": 2,
        "trait_col": "Trait",
        "plink_prefix": "fake",
        "phenotype_csv": "fake.csv",
        "phenotype_sample_id_col": "sample_id",
        "output_dir": str(tmp_path / "out"),
        "main_models": [],
        "baselines": {
            "rscript_path": "Rscript",
            "sommer_method": "mmer",
            "gblup": True,
            "bayesA": False,
            "bayesB": True,
            "bayesLasso": False,
            "rkhs": True,
        },
        "baseline_inner_oof": {
            "enabled": True,
            "fold": 1,
            "n_splits": 3,
            "models": ["GBLUP", "BayesB", "RKHS"],
        },
    }

    experiment.run_experiment(config)

    fold1_dir = tmp_path / "out" / "fold_1"
    for baseline_name in ("GBLUP", "BayesB", "RKHS"):
        baseline_dir = fold1_dir / baseline_name
        assert (baseline_dir / "inner_oof_predictions.npy").exists()
        assert (baseline_dir / "inner_oof_targets.npy").exists()
        summary = json.loads((baseline_dir / "inner_oof_summary.json").read_text(encoding="utf-8"))
        assert summary["baseline_model"] == baseline_name
        assert summary["source"] == "baseline_inner_oof"
        assert int(summary["fold"]) == 1

    fold2_dir = tmp_path / "out" / "fold_2"
    assert not (fold2_dir / "GBLUP" / "inner_oof_predictions.npy").exists()

    metadata = json.loads((fold1_dir / "fold_metadata.json").read_text(encoding="utf-8"))
    assert "baseline_runs" in metadata
    assert metadata["baseline_runs"]["GBLUP"]["inner_oof"]["prediction_path"].endswith(
        "/fold_1/GBLUP/inner_oof_predictions.npy"
    )
