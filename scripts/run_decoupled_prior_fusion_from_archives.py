from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.models.group_shared_gate import _clip_gate_targets, _fit_simplex_prior_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build single/dual/triple and only-prior results by reusing archived no-prior TabICL and baseline outputs."
    )
    parser.add_argument("--tabicl-root", required=True, help="Existing no-prior TabICL root, e.g. .../tabicl_tabicl_no_prior")
    parser.add_argument("--baseline-root", required=True, help="Existing baseline-only trait root")
    parser.add_argument("--output-root", required=True, help="Trait root where fusion outputs will be written")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tabicl_outer(fold_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(fold_dir / "tabicl_predictions.csv")
    pred_cols = [col for col in frame.columns if col.endswith("_pred")]
    if len(pred_cols) != 1:
        raise ValueError(f"Expected exactly one *_pred column in {fold_dir / 'tabicl_predictions.csv'}, got {pred_cols}")
    y_true = frame["y_true"].to_numpy(dtype=np.float32)
    y_pred = frame[pred_cols[0]].to_numpy(dtype=np.float32)
    return y_true, y_pred


def _load_tabicl_inner_oof(fold1_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.load(fold1_dir / "tabicl_inner_oof_targets.npy").astype(np.float32).reshape(-1)
    y_pred = np.load(fold1_dir / "tabicl_inner_oof_predictions.npy").astype(np.float32).reshape(-1)
    return y_true, y_pred


def _load_baseline_outer(baseline_root: Path, fold_id: int, model_name: str) -> np.ndarray:
    path = baseline_root / f"fold_{fold_id}" / model_name / "predictions.csv"
    return np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(-1)


def _load_baseline_inner_oof(baseline_root: Path, model_name: str) -> np.ndarray:
    path = baseline_root / "fold_1" / model_name / "inner_oof_predictions.npy"
    return np.load(path).astype(np.float32).reshape(-1)


def _build_only_prior(y_true_inner: np.ndarray, prior_inner: dict[str, np.ndarray], prior_outer: dict[str, list[np.ndarray]], prior_names: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
    prior_matrix = np.column_stack([prior_inner[name] for name in prior_names]).astype(np.float32)
    weights = _fit_simplex_prior_weights(y_true_inner, prior_matrix).astype(np.float32)
    fold_preds: list[np.ndarray] = []
    for fold_idx in range(len(next(iter(prior_outer.values())))):
        pred = np.zeros_like(prior_outer[prior_names[0]][fold_idx], dtype=np.float32)
        for j, name in enumerate(prior_names):
            pred = pred + float(weights[j]) * prior_outer[name][fold_idx]
        fold_preds.append(pred.astype(np.float32))
    return weights, fold_preds


def _build_tabicl_plus_prior(
    y_true_inner: np.ndarray,
    y_tabicl_inner: np.ndarray,
    prior_inner: dict[str, np.ndarray],
    y_tabicl_outer: list[np.ndarray],
    prior_outer: dict[str, list[np.ndarray]],
    prior_names: list[str],
) -> tuple[np.ndarray, float, np.ndarray, list[np.ndarray]]:
    prior_matrix = np.column_stack([prior_inner[name] for name in prior_names]).astype(np.float32)
    prior_weights = _fit_simplex_prior_weights(y_true_inner, prior_matrix).astype(np.float32)
    y_prior_inner = np.sum(prior_matrix * prior_weights.reshape(1, -1), axis=1).astype(np.float32)
    w_tabicl = float(np.mean(_clip_gate_targets(y_true_inner, y_tabicl_inner, y_prior_inner)))

    final_weights = np.zeros(len(prior_names) + 1, dtype=np.float32)
    final_weights[0] = np.float32(w_tabicl)
    final_weights[1:] = (1.0 - np.float32(w_tabicl)) * prior_weights

    fold_preds: list[np.ndarray] = []
    for fold_idx in range(len(y_tabicl_outer)):
        y_prior = np.zeros_like(y_tabicl_outer[fold_idx], dtype=np.float32)
        for j, name in enumerate(prior_names):
            y_prior = y_prior + float(prior_weights[j]) * prior_outer[name][fold_idx]
        pred = (y_prior + np.float32(w_tabicl) * (y_tabicl_outer[fold_idx] - y_prior)).astype(np.float32)
        fold_preds.append(pred)
    return prior_weights, w_tabicl, final_weights, fold_preds


def _write_fold_metrics(out_dir: Path, model_name: str, rows: list[dict[str, float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    if rows:
        summary = {
            "model": model_name,
            "pearson_mean": float(np.mean([row["pearson"] for row in rows])),
            "r2_mean": float(np.mean([row["r2"] for row in rows])),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_group_summary(
    out_dir: Path,
    prior_names: list[str],
    prior_weights: np.ndarray,
    w_tabicl: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prior_names": list(prior_names),
        "prior_weight_group": [prior_weights.astype(float).tolist()],
        "w_group": [float(w_tabicl)],
        "group_counts": [1],
        "group_probs_mean": [1.0],
        "group_mode": "decoupled_reuse",
        "assignment_mode": "fixed_single_group",
        "group_centroids": [[0.0]],
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
    }
    (out_dir / "group_shared_gate_group_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _single_slug(model_name: str) -> str:
    return str(model_name).strip().lower()


def main() -> None:
    args = parse_args()
    tabicl_root = Path(args.tabicl_root)
    baseline_root = Path(args.baseline_root)
    output_root = Path(args.output_root)
    fold_ids = [int(v) for v in args.folds]

    fold1_tabicl_dir = tabicl_root / "fold_1"
    y_true_inner, y_tabicl_inner = _load_tabicl_inner_oof(fold1_tabicl_dir)
    y_tabicl_outer: list[np.ndarray] = []
    y_true_outer: list[np.ndarray] = []
    for fold_id in fold_ids:
        y_true_fold, y_pred_fold = _load_tabicl_outer(tabicl_root / f"fold_{fold_id}")
        y_true_outer.append(y_true_fold)
        y_tabicl_outer.append(y_pred_fold)

    baseline_names = ["BayesB", "GBLUP", "RKHS"]
    baseline_inner = {name: _load_baseline_inner_oof(baseline_root, name) for name in baseline_names}
    baseline_outer = {
        name: [_load_baseline_outer(baseline_root, fold_id, name) for fold_id in fold_ids]
        for name in baseline_names
    }

    single_results: dict[str, dict[str, object]] = {}
    for single_name in baseline_names:
        only_weights, only_preds = _build_only_prior(
            y_true_inner=y_true_inner,
            prior_inner={single_name: baseline_inner[single_name]},
            prior_outer={single_name: baseline_outer[single_name]},
            prior_names=[single_name],
        )
        prior_weights, w_tabicl, _single_final, tabicl_preds = _build_tabicl_plus_prior(
            y_true_inner=y_true_inner,
            y_tabicl_inner=y_tabicl_inner,
            prior_inner={single_name: baseline_inner[single_name]},
            y_tabicl_outer=y_tabicl_outer,
            prior_outer={single_name: baseline_outer[single_name]},
            prior_names=[single_name],
        )
        single_results[single_name] = {
            "only_weights": only_weights,
            "only_preds": only_preds,
            "prior_weights": prior_weights,
            "w_tabicl": w_tabicl,
            "tabicl_preds": tabicl_preds,
        }

    dual_weights, dual_only_preds = _build_only_prior(
        y_true_inner=y_true_inner,
        prior_inner={"BayesB": baseline_inner["BayesB"], "GBLUP": baseline_inner["GBLUP"]},
        prior_outer={"BayesB": baseline_outer["BayesB"], "GBLUP": baseline_outer["GBLUP"]},
        prior_names=["BayesB", "GBLUP"],
    )
    dual_prior_weights, dual_w_tabicl, _dual_final, dual_tabicl_preds = _build_tabicl_plus_prior(
        y_true_inner=y_true_inner,
        y_tabicl_inner=y_tabicl_inner,
        prior_inner={"BayesB": baseline_inner["BayesB"], "GBLUP": baseline_inner["GBLUP"]},
        y_tabicl_outer=y_tabicl_outer,
        prior_outer={"BayesB": baseline_outer["BayesB"], "GBLUP": baseline_outer["GBLUP"]},
        prior_names=["BayesB", "GBLUP"],
    )

    triple_weights, triple_only_preds = _build_only_prior(
        y_true_inner=y_true_inner,
        prior_inner={name: baseline_inner[name] for name in baseline_names},
        prior_outer={name: baseline_outer[name] for name in baseline_names},
        prior_names=baseline_names,
    )
    triple_prior_weights, triple_w_tabicl, _triple_final, triple_tabicl_preds = _build_tabicl_plus_prior(
        y_true_inner=y_true_inner,
        y_tabicl_inner=y_tabicl_inner,
        prior_inner={name: baseline_inner[name] for name in baseline_names},
        y_tabicl_outer=y_tabicl_outer,
        prior_outer={name: baseline_outer[name] for name in baseline_names},
        prior_names=baseline_names,
    )

    single_rows_by_name = {name: [] for name in baseline_names}
    single_only_rows_by_name = {name: [] for name in baseline_names}
    dual_rows = []
    dual_only_rows = []
    triple_rows = []
    triple_only_rows = []
    for idx, fold_id in enumerate(fold_ids):
        y_true = y_true_outer[idx]
        for single_name in baseline_names:
            single_metric = regression_metrics(y_true, single_results[single_name]["tabicl_preds"][idx])
            single_only_metric = regression_metrics(y_true, single_results[single_name]["only_preds"][idx])
            single_rows_by_name[single_name].append({"fold": fold_id, **single_metric})
            single_only_rows_by_name[single_name].append(
                {
                    "fold": fold_id,
                    f"w_{_single_slug(single_name)}": float(single_results[single_name]["only_weights"][0]),
                    **single_only_metric,
                }
            )
        dual_metric = regression_metrics(y_true, dual_tabicl_preds[idx])
        dual_only_metric = regression_metrics(y_true, dual_only_preds[idx])
        triple_metric = regression_metrics(y_true, triple_tabicl_preds[idx])
        triple_only_metric = regression_metrics(y_true, triple_only_preds[idx])
        dual_rows.append({"fold": fold_id, **dual_metric})
        dual_only_rows.append(
            {
                "fold": fold_id,
                "alpha": float(dual_weights[0]),
                "w_bayesb": float(dual_weights[0]),
                "w_gblup": float(dual_weights[1]),
                **dual_only_metric,
            }
        )
        triple_rows.append({"fold": fold_id, **triple_metric})
        triple_only_rows.append(
            {
                "fold": fold_id,
                "w_bayesb": float(triple_weights[0]),
                "w_gblup": float(triple_weights[1]),
                "w_rkhs": float(triple_weights[2]),
                **triple_only_metric,
            }
        )

    for single_name in baseline_names:
        single_slug = _single_slug(single_name)
        single_out = output_root / f"tabicl_{single_slug}_single_prior"
        single_only_out = output_root / f"prior_only_{single_slug}"
        _write_fold_metrics(single_out, f"{single_slug}_single_prior_tabicl", single_rows_by_name[single_name])
        _write_fold_metrics(single_only_out, f"only_single_prior_{single_slug}", single_only_rows_by_name[single_name])
        _write_group_summary(
            single_out / "fold_1",
            [single_name],
            single_results[single_name]["prior_weights"],
            float(single_results[single_name]["w_tabicl"]),
        )

    dual_out = output_root / "tabicl_tabicl_dual_prior"
    triple_out = output_root / "tabicl_tabicl_triple_prior"
    _write_fold_metrics(dual_out, "dual_prior_tabicl", dual_rows)
    _write_fold_metrics(output_root / "prior_only_bayesb_gblup", "only_dual_prior", dual_only_rows)
    _write_fold_metrics(triple_out, "triple_prior_tabicl", triple_rows)
    _write_fold_metrics(output_root / "prior_only_triple", "only_triple_prior", triple_only_rows)

    _write_group_summary(dual_out / "fold_1", ["BayesB", "GBLUP"], dual_prior_weights, dual_w_tabicl)
    _write_group_summary(triple_out / "fold_1", baseline_names, triple_prior_weights, triple_w_tabicl)

    summary = {
        "tabicl_root": str(tabicl_root),
        "baseline_root": str(baseline_root),
        "output_root": str(output_root),
        "single": {
            single_name: {
                "prior_weights": {single_name: float(single_results[single_name]["prior_weights"][0])},
                "w_tabicl": float(single_results[single_name]["w_tabicl"]),
            }
            for single_name in baseline_names
        },
        "dual": {
            "prior_weights": {"BayesB": float(dual_prior_weights[0]), "GBLUP": float(dual_prior_weights[1])},
            "w_tabicl": float(dual_w_tabicl),
        },
        "triple": {
            "prior_weights": {
                "BayesB": float(triple_prior_weights[0]),
                "GBLUP": float(triple_prior_weights[1]),
                "RKHS": float(triple_prior_weights[2]),
            },
            "w_tabicl": float(triple_w_tabicl),
        },
    }
    (output_root / "decoupled_fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
