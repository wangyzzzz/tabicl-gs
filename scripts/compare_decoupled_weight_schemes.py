from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.models.group_shared_gate import _clip_gate_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single/dual/triple prior fusion under multiple weight schemes from decoupled archives."
    )
    parser.add_argument("--no-prior-root")
    parser.add_argument("--baseline-root")
    parser.add_argument("--traits", nargs="+", help="dataset_slug/trait_slug")
    parser.add_argument("--trait-no-prior-root")
    parser.add_argument("--trait-baseline-root")
    parser.add_argument("--dataset")
    parser.add_argument("--trait-slug")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _solve_sum1_nonnegative_ls(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    P = np.asarray(P, dtype=np.float64)
    n_cols = int(P.shape[1])
    if n_cols == 1:
        return np.ones(1, dtype=np.float64)
    best_w = None
    best_loss = None
    for mask in range(1, 1 << n_cols):
        active = [idx for idx in range(n_cols) if (mask >> idx) & 1]
        P_active = P[:, active]
        ones = np.ones((len(active), 1), dtype=np.float64)
        gram = 2.0 * (P_active.T @ P_active)
        rhs = np.concatenate([2.0 * (P_active.T @ y), np.array([1.0], dtype=np.float64)])
        kkt = np.block([[gram, ones], [ones.T, np.zeros((1, 1), dtype=np.float64)]])
        try:
            sol = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
        w_active = np.asarray(sol[: len(active)], dtype=np.float64)
        if np.any(w_active < -1e-8):
            continue
        w = np.zeros(n_cols, dtype=np.float64)
        w[active] = np.clip(w_active, 0.0, None)
        w_sum = float(w.sum())
        if w_sum <= 0.0:
            continue
        w /= w_sum
        loss = float(np.square(y - P @ w).sum())
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_w = w
    if best_w is None:
        raise RuntimeError("Failed to solve nonnegative sum-to-1 least squares.")
    return best_w.astype(np.float64)


def _load_no_prior_fold_predictions(fold_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(fold_dir / "tabicl_predictions.csv")
    pred_cols = [col for col in frame.columns if col.endswith("_pred")]
    if len(pred_cols) != 1:
        raise ValueError(f"Expected exactly one prediction column in {fold_dir / 'tabicl_predictions.csv'}")
    sample_ids = frame["sample_id"].astype(str).tolist()
    y_true = frame["y_true"].to_numpy(dtype=np.float32)
    y_pred = frame[pred_cols[0]].to_numpy(dtype=np.float32)
    return y_true, y_pred, sample_ids


def _load_baseline_fold_predictions(fold_dir: Path, model_name: str) -> np.ndarray:
    path = fold_dir / model_name / "predictions.csv"
    return np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(-1)


def _verify_fold_alignment(no_prior_fold_dir: Path, baseline_fold_dir: Path) -> None:
    no_prior = pd.read_csv(no_prior_fold_dir / "tabicl_predictions.csv")
    baseline = pd.read_csv(baseline_fold_dir / "tabicl_predictions.csv")
    same_sample = no_prior["sample_id"].astype(str).tolist() == baseline["sample_id"].astype(str).tolist()
    same_y = no_prior["y_true"].round(8).tolist() == baseline["y_true"].round(8).tolist()
    if not same_sample or not same_y:
        raise ValueError(
            f"Fold alignment mismatch between {no_prior_fold_dir} and {baseline_fold_dir}: "
            f"same_sample={same_sample}, same_y={same_y}"
        )


def _mean_metric(rows: list[dict[str, float]], metric_name: str) -> float:
    return float(np.mean([row[metric_name] for row in rows]))


def _compute_pool_metrics(
    y_true_inner: np.ndarray,
    y_tabicl_inner: np.ndarray,
    prior_inner: dict[str, np.ndarray],
    y_true_outer: list[np.ndarray],
    y_tabicl_outer: list[np.ndarray],
    prior_outer: dict[str, list[np.ndarray]],
    prior_names: list[str],
) -> dict[str, object]:
    prior_matrix = np.column_stack([prior_inner[name] for name in prior_names]).astype(np.float64)
    prior_weights = _solve_sum1_nonnegative_ls(y_true_inner, prior_matrix)
    y_prior_inner = (prior_matrix @ prior_weights).astype(np.float64)

    only_rows = []
    for fold_idx, y_true in enumerate(y_true_outer):
        pred = np.zeros_like(y_true, dtype=np.float32)
        for j, name in enumerate(prior_names):
            pred = pred + float(prior_weights[j]) * prior_outer[name][fold_idx]
        only_rows.append(regression_metrics(y_true, pred))

    w_clip = float(np.mean(_clip_gate_targets(y_true_inner, y_tabicl_inner, y_prior_inner)))
    clip_rows = []
    for fold_idx, y_true in enumerate(y_true_outer):
        y_prior = np.zeros_like(y_true, dtype=np.float32)
        for j, name in enumerate(prior_names):
            y_prior = y_prior + float(prior_weights[j]) * prior_outer[name][fold_idx]
        pred = (y_prior + np.float32(w_clip) * (y_tabicl_outer[fold_idx] - y_prior)).astype(np.float32)
        clip_rows.append(regression_metrics(y_true, pred))

    two_step_ls_weights = _solve_sum1_nonnegative_ls(
        y_true_inner,
        np.column_stack([y_prior_inner, y_tabicl_inner]).astype(np.float64),
    )
    w_prior_ls = float(two_step_ls_weights[0])
    w_tabicl_ls = float(two_step_ls_weights[1])
    two_step_ls_rows = []
    for fold_idx, y_true in enumerate(y_true_outer):
        y_prior = np.zeros_like(y_true, dtype=np.float32)
        for j, name in enumerate(prior_names):
            y_prior = y_prior + float(prior_weights[j]) * prior_outer[name][fold_idx]
        pred = (w_prior_ls * y_prior + w_tabicl_ls * y_tabicl_outer[fold_idx]).astype(np.float32)
        two_step_ls_rows.append(regression_metrics(y_true, pred))

    all_matrix = np.column_stack([y_tabicl_inner] + [prior_inner[name] for name in prior_names]).astype(np.float64)
    all_weights = _solve_sum1_nonnegative_ls(y_true_inner, all_matrix)
    all_ls_rows = []
    for fold_idx, y_true in enumerate(y_true_outer):
        pred = float(all_weights[0]) * y_tabicl_outer[fold_idx]
        for j, name in enumerate(prior_names):
            pred = pred + float(all_weights[j + 1]) * prior_outer[name][fold_idx]
        all_ls_rows.append(regression_metrics(y_true, pred))

    return {
        "only_pearson": _mean_metric(only_rows, "pearson"),
        "only_r2": _mean_metric(only_rows, "r2"),
        "two_step_clip_pearson": _mean_metric(clip_rows, "pearson"),
        "two_step_clip_r2": _mean_metric(clip_rows, "r2"),
        "two_step_ls_pearson": _mean_metric(two_step_ls_rows, "pearson"),
        "two_step_ls_r2": _mean_metric(two_step_ls_rows, "r2"),
        "all_ls_pearson": _mean_metric(all_ls_rows, "pearson"),
        "all_ls_r2": _mean_metric(all_ls_rows, "r2"),
        "weights": {
            "prior_only_ls": {name: float(prior_weights[idx]) for idx, name in enumerate(prior_names)},
            "two_step_clip_final": {
                "TabICL": float(w_clip),
                **{
                    name: float((1.0 - w_clip) * prior_weights[idx])
                    for idx, name in enumerate(prior_names)
                },
            },
            "two_step_ls_final": {
                "TabICL": float(w_tabicl_ls),
                **{
                    name: float(w_prior_ls * prior_weights[idx])
                    for idx, name in enumerate(prior_names)
                },
            },
            "all_ls_final": {
                "TabICL": float(all_weights[0]),
                **{
                    name: float(all_weights[idx + 1])
                    for idx, name in enumerate(prior_names)
                },
            },
        },
    }


def _compute_trait_outputs(
    *,
    dataset_slug: str,
    trait_slug: str,
    no_prior_trait_dir: Path,
    baseline_trait_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    y_true_inner = np.load(no_prior_trait_dir / "fold_1" / "tabicl_inner_oof_targets.npy").astype(np.float32).reshape(-1)
    y_tabicl_inner = np.load(no_prior_trait_dir / "fold_1" / "tabicl_inner_oof_predictions.npy").astype(np.float32).reshape(-1)

    prior_inner = {
        "BayesB": np.load(baseline_trait_dir / "fold_1" / "BayesB" / "inner_oof_predictions.npy").astype(np.float32).reshape(-1),
        "GBLUP": np.load(baseline_trait_dir / "fold_1" / "GBLUP" / "inner_oof_predictions.npy").astype(np.float32).reshape(-1),
        "RKHS": np.load(baseline_trait_dir / "fold_1" / "RKHS" / "inner_oof_predictions.npy").astype(np.float32).reshape(-1),
    }

    y_true_outer: list[np.ndarray] = []
    y_tabicl_outer: list[np.ndarray] = []
    prior_outer = {"BayesB": [], "GBLUP": [], "RKHS": []}
    for fold_id in range(1, 6):
        no_prior_fold_dir = no_prior_trait_dir / f"fold_{fold_id}"
        baseline_fold_dir = baseline_trait_dir / f"fold_{fold_id}"
        _verify_fold_alignment(no_prior_fold_dir, baseline_fold_dir)
        y_true, y_tabicl, _sample_ids = _load_no_prior_fold_predictions(no_prior_fold_dir)
        y_true_outer.append(y_true)
        y_tabicl_outer.append(y_tabicl)
        for model_name in ("BayesB", "GBLUP", "RKHS"):
            prior_outer[model_name].append(_load_baseline_fold_predictions(baseline_fold_dir, model_name))

    no_prior_rows = [regression_metrics(y_true_outer[idx], y_tabicl_outer[idx]) for idx in range(5)]
    bayesb_rows = [regression_metrics(y_true_outer[idx], prior_outer["BayesB"][idx]) for idx in range(5)]
    gblup_rows = [regression_metrics(y_true_outer[idx], prior_outer["GBLUP"][idx]) for idx in range(5)]
    rkhs_rows = [regression_metrics(y_true_outer[idx], prior_outer["RKHS"][idx]) for idx in range(5)]

    single_bayesb = _compute_pool_metrics(
        y_true_inner=y_true_inner,
        y_tabicl_inner=y_tabicl_inner,
        prior_inner={"BayesB": prior_inner["BayesB"]},
        y_true_outer=y_true_outer,
        y_tabicl_outer=y_tabicl_outer,
        prior_outer={"BayesB": prior_outer["BayesB"]},
        prior_names=["BayesB"],
    )
    single_gblup = _compute_pool_metrics(
        y_true_inner=y_true_inner,
        y_tabicl_inner=y_tabicl_inner,
        prior_inner={"GBLUP": prior_inner["GBLUP"]},
        y_true_outer=y_true_outer,
        y_tabicl_outer=y_tabicl_outer,
        prior_outer={"GBLUP": prior_outer["GBLUP"]},
        prior_names=["GBLUP"],
    )
    single_rkhs = _compute_pool_metrics(
        y_true_inner=y_true_inner,
        y_tabicl_inner=y_tabicl_inner,
        prior_inner={"RKHS": prior_inner["RKHS"]},
        y_true_outer=y_true_outer,
        y_tabicl_outer=y_tabicl_outer,
        prior_outer={"RKHS": prior_outer["RKHS"]},
        prior_names=["RKHS"],
    )
    dual = _compute_pool_metrics(
        y_true_inner=y_true_inner,
        y_tabicl_inner=y_tabicl_inner,
        prior_inner={"BayesB": prior_inner["BayesB"], "GBLUP": prior_inner["GBLUP"]},
        y_true_outer=y_true_outer,
        y_tabicl_outer=y_tabicl_outer,
        prior_outer={"BayesB": prior_outer["BayesB"], "GBLUP": prior_outer["GBLUP"]},
        prior_names=["BayesB", "GBLUP"],
    )
    triple = _compute_pool_metrics(
        y_true_inner=y_true_inner,
        y_tabicl_inner=y_tabicl_inner,
        prior_inner=prior_inner,
        y_true_outer=y_true_outer,
        y_tabicl_outer=y_tabicl_outer,
        prior_outer=prior_outer,
        prior_names=["BayesB", "GBLUP", "RKHS"],
    )

    row = {
        "dataset": dataset_slug,
        "trait_slug": trait_slug,
        "no_prior_tabicl": _mean_metric(no_prior_rows, "pearson"),
        "no_prior_tabicl_r2": _mean_metric(no_prior_rows, "r2"),
        "BayesB": _mean_metric(bayesb_rows, "pearson"),
        "BayesB_r2": _mean_metric(bayesb_rows, "r2"),
        "GBLUP": _mean_metric(gblup_rows, "pearson"),
        "GBLUP_r2": _mean_metric(gblup_rows, "r2"),
        "RKHS": _mean_metric(rkhs_rows, "pearson"),
        "RKHS_r2": _mean_metric(rkhs_rows, "r2"),
        "only_single_bayesb": float(single_bayesb["only_pearson"]),
        "only_single_bayesb_r2": float(single_bayesb["only_r2"]),
        "single_bayesb_two_step_clip": float(single_bayesb["two_step_clip_pearson"]),
        "single_bayesb_two_step_clip_r2": float(single_bayesb["two_step_clip_r2"]),
        "single_bayesb_two_step_ls": float(single_bayesb["two_step_ls_pearson"]),
        "single_bayesb_two_step_ls_r2": float(single_bayesb["two_step_ls_r2"]),
        "single_bayesb_all_ls": float(single_bayesb["all_ls_pearson"]),
        "single_bayesb_all_ls_r2": float(single_bayesb["all_ls_r2"]),
        "only_single_gblup": float(single_gblup["only_pearson"]),
        "only_single_gblup_r2": float(single_gblup["only_r2"]),
        "single_gblup_two_step_clip": float(single_gblup["two_step_clip_pearson"]),
        "single_gblup_two_step_clip_r2": float(single_gblup["two_step_clip_r2"]),
        "single_gblup_two_step_ls": float(single_gblup["two_step_ls_pearson"]),
        "single_gblup_two_step_ls_r2": float(single_gblup["two_step_ls_r2"]),
        "single_gblup_all_ls": float(single_gblup["all_ls_pearson"]),
        "single_gblup_all_ls_r2": float(single_gblup["all_ls_r2"]),
        "only_single_rkhs": float(single_rkhs["only_pearson"]),
        "only_single_rkhs_r2": float(single_rkhs["only_r2"]),
        "single_rkhs_two_step_clip": float(single_rkhs["two_step_clip_pearson"]),
        "single_rkhs_two_step_clip_r2": float(single_rkhs["two_step_clip_r2"]),
        "single_rkhs_two_step_ls": float(single_rkhs["two_step_ls_pearson"]),
        "single_rkhs_two_step_ls_r2": float(single_rkhs["two_step_ls_r2"]),
        "single_rkhs_all_ls": float(single_rkhs["all_ls_pearson"]),
        "single_rkhs_all_ls_r2": float(single_rkhs["all_ls_r2"]),
        "only_dual": float(dual["only_pearson"]),
        "only_dual_r2": float(dual["only_r2"]),
        "dual_two_step_clip": float(dual["two_step_clip_pearson"]),
        "dual_two_step_clip_r2": float(dual["two_step_clip_r2"]),
        "dual_two_step_ls": float(dual["two_step_ls_pearson"]),
        "dual_two_step_ls_r2": float(dual["two_step_ls_r2"]),
        "dual_all_ls": float(dual["all_ls_pearson"]),
        "dual_all_ls_r2": float(dual["all_ls_r2"]),
        "only_triple": float(triple["only_pearson"]),
        "only_triple_r2": float(triple["only_r2"]),
        "triple_two_step_clip": float(triple["two_step_clip_pearson"]),
        "triple_two_step_clip_r2": float(triple["two_step_clip_r2"]),
        "triple_two_step_ls": float(triple["two_step_ls_pearson"]),
        "triple_two_step_ls_r2": float(triple["two_step_ls_r2"]),
        "triple_all_ls": float(triple["all_ls_pearson"]),
        "triple_all_ls_r2": float(triple["all_ls_r2"]),
    }
    payload = {
        "metrics": row,
        "weights": {
            "single_bayesb": single_bayesb["weights"],
            "single_gblup": single_gblup["weights"],
            "single_rkhs": single_rkhs["weights"],
            "dual": dual["weights"],
            "triple": triple["weights"],
        },
    }
    return row, payload


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    payload: dict[str, object] = {}
    if args.trait_no_prior_root and args.trait_baseline_root and args.dataset and args.trait_slug:
        row, trait_payload = _compute_trait_outputs(
            dataset_slug=str(args.dataset),
            trait_slug=str(args.trait_slug),
            no_prior_trait_dir=Path(args.trait_no_prior_root),
            baseline_trait_dir=Path(args.trait_baseline_root),
        )
        rows.append(row)
        payload[f"{args.dataset}/{args.trait_slug}"] = trait_payload
    else:
        if not args.no_prior_root or not args.baseline_root or not args.traits:
            raise ValueError(
                "Either provide --no-prior-root/--baseline-root/--traits, or provide "
                "--trait-no-prior-root/--trait-baseline-root/--dataset/--trait-slug."
            )
        no_prior_root = Path(args.no_prior_root)
        baseline_root = Path(args.baseline_root)
        for trait_key in args.traits:
            dataset_slug, trait_slug = trait_key.split("/", 1)
            row, trait_payload = _compute_trait_outputs(
                dataset_slug=dataset_slug,
                trait_slug=trait_slug,
                no_prior_trait_dir=no_prior_root / dataset_slug / trait_slug,
                baseline_trait_dir=baseline_root / dataset_slug / trait_slug,
            )
            rows.append(row)
            payload[trait_key] = trait_payload

    frame = pd.DataFrame(rows).sort_values(["dataset", "trait_slug"]).reset_index(drop=True)
    frame.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
