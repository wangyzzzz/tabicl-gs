from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize single/dual/triple prior TabICL lines with only-prior and weights.")
    parser.add_argument("--single-root", required=True)
    parser.add_argument("--dual-rice-root", required=True)
    parser.add_argument("--dual-multi-root", required=True)
    parser.add_argument("--triple-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _mean_metric_from_fold_metrics(path: Path, model_name: str | None = None) -> dict[str, float]:
    if not path.exists():
        return {"pearson": float("nan"), "r2": float("nan")}
    frame = pd.read_csv(path)
    if model_name is not None:
        frame = frame[frame["model"].astype(str) == model_name].copy()
    if frame.empty:
        return {"pearson": float("nan"), "r2": float("nan")}
    return {
        "pearson": float(frame["pearson"].mean()),
        "r2": float(frame["r2"].mean()),
    }


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_list_value(values: list[float] | None, idx: int) -> float:
    if not values or idx >= len(values):
        return float("nan")
    return float(values[idx])


def _mean_nested_value(values: list[list[float]] | None, row_idx: int, col_idx: int) -> float:
    if not values or row_idx >= len(values):
        return float("nan")
    row = values[row_idx]
    if col_idx >= len(row):
        return float("nan")
    return float(row[col_idx])


def _load_single_metrics(trait_root: Path) -> dict[str, float]:
    return _mean_metric_from_fold_metrics(trait_root / "tabicl_gblup_single_prior" / "fold_metrics.csv")


def _load_single_only_metrics(trait_root: Path) -> dict[str, float]:
    return _mean_metric_from_fold_metrics(trait_root / "prior_only_gblup" / "fold_metrics.csv")


def _load_dual_metrics(trait_root: Path) -> dict[str, float]:
    fold_paths = sorted((trait_root / "tabicl_tabicl_dual_prior").glob("fold_*/fold_metrics.csv"))
    rows = [pd.read_csv(path) for path in fold_paths]
    if not rows:
        return {"pearson": float("nan"), "r2": float("nan")}
    frame = pd.concat(rows, ignore_index=True)
    return {
        "pearson": float(frame["pearson"].mean()),
        "r2": float(frame["r2"].mean()),
    }


def _load_dual_only_metrics(trait_root: Path) -> dict[str, float]:
    return _mean_metric_from_fold_metrics(trait_root / "prior_only_bayesb_gblup" / "fold_metrics.csv")


def _load_triple_metrics(trait_root: Path) -> dict[str, float]:
    fold_paths = sorted((trait_root / "tabicl_tabicl_triple_prior").glob("fold_*/fold_metrics.csv"))
    rows = [pd.read_csv(path) for path in fold_paths]
    if not rows:
        return {"pearson": float("nan"), "r2": float("nan")}
    frame = pd.concat(rows, ignore_index=True)
    return {
        "pearson": float(frame["pearson"].mean()),
        "r2": float(frame["r2"].mean()),
    }


def _load_triple_only_metrics(trait_root: Path) -> dict[str, float]:
    return _mean_metric_from_fold_metrics(trait_root / "prior_only_triple" / "fold_metrics.csv")


def _load_single_weights(trait_root: Path) -> dict[str, float]:
    fold_meta_paths = sorted((trait_root / "tabicl_gblup_single_prior").glob("fold_*/fold_metadata.json"))
    if not fold_meta_paths:
        return {
            "single_w_tabicl": float("nan"),
            "single_w_gblup": float("nan"),
        }
    tabicl_weights: list[float] = []
    gblup_weights: list[float] = []
    for meta_path in fold_meta_paths:
        meta = _load_json_if_exists(meta_path)
        if meta is None:
            continue
        model_runs = meta.get("model_runs", {})
        if not model_runs:
            continue
        run = next(iter(model_runs.values()))
        summary = run.get("group_summary") or {}
        w_group = summary.get("w_group")
        prior_weight_group = summary.get("prior_weight_group")
        w_tabicl = _mean_list_value(w_group, 0)
        if pd.isna(w_tabicl):
            continue
        prior_gblup = _mean_nested_value(prior_weight_group, 0, 0)
        tabicl_weights.append(float(w_tabicl))
        gblup_weights.append(float((1.0 - w_tabicl) * prior_gblup))
    if not tabicl_weights:
        return {
            "single_w_tabicl": float("nan"),
            "single_w_gblup": float("nan"),
        }
    return {
        "single_w_tabicl": float(sum(tabicl_weights) / len(tabicl_weights)),
        "single_w_gblup": float(sum(gblup_weights) / len(gblup_weights)),
    }


def _load_dual_weights(trait_root: Path) -> dict[str, float]:
    summary = _load_json_if_exists(trait_root / "tabicl_tabicl_dual_prior" / "fold_1" / "group_shared_gate_group_summary.json")
    if summary is None:
        return {
            "dual_w_tabicl": float("nan"),
            "dual_w_bayesb": float("nan"),
            "dual_w_gblup": float("nan"),
        }
    w_group = summary.get("w_group")
    prior_weight_group = summary.get("prior_weight_group")
    w_tabicl = _mean_list_value(w_group, 0)
    prior_bayesb = _mean_nested_value(prior_weight_group, 0, 0)
    prior_gblup = _mean_nested_value(prior_weight_group, 0, 1)
    return {
        "dual_w_tabicl": w_tabicl,
        "dual_w_bayesb": (1.0 - w_tabicl) * prior_bayesb if pd.notna(w_tabicl) else float("nan"),
        "dual_w_gblup": (1.0 - w_tabicl) * prior_gblup if pd.notna(w_tabicl) else float("nan"),
    }


def _load_triple_weights(trait_root: Path) -> dict[str, float]:
    summary = _load_json_if_exists(trait_root / "tabicl_tabicl_triple_prior" / "fold_1" / "group_shared_gate_group_summary.json")
    if summary is None:
        return {
            "triple_w_tabicl": float("nan"),
            "triple_w_bayesb": float("nan"),
            "triple_w_gblup": float("nan"),
            "triple_w_rkhs": float("nan"),
        }
    w_group = summary.get("w_group")
    prior_weight_group = summary.get("prior_weight_group")
    w_tabicl = _mean_list_value(w_group, 0)
    prior_bayesb = _mean_nested_value(prior_weight_group, 0, 0)
    prior_gblup = _mean_nested_value(prior_weight_group, 0, 1)
    prior_rkhs = _mean_nested_value(prior_weight_group, 0, 2)
    return {
        "triple_w_tabicl": w_tabicl,
        "triple_w_bayesb": (1.0 - w_tabicl) * prior_bayesb if pd.notna(w_tabicl) else float("nan"),
        "triple_w_gblup": (1.0 - w_tabicl) * prior_gblup if pd.notna(w_tabicl) else float("nan"),
        "triple_w_rkhs": (1.0 - w_tabicl) * prior_rkhs if pd.notna(w_tabicl) else float("nan"),
    }


def _dual_trait_root(dataset_slug: str, trait_slug: str, rice_root: Path, multi_root: Path) -> Path:
    if dataset_slug == "rice529":
        return rice_root / trait_slug
    return multi_root / dataset_slug / trait_slug


def build_compare(
    single_root: Path,
    dual_rice_root: Path,
    dual_multi_root: Path,
    triple_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_dir in sorted([p for p in triple_root.iterdir() if p.is_dir() and p.name != "logs"]):
        dataset_slug = dataset_dir.name
        for trait_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            trait_slug = trait_dir.name
            row: dict[str, object] = {
                "dataset": dataset_slug,
                "trait_slug": trait_slug,
            }
            single_trait_root = single_root / dataset_slug / trait_slug
            dual_trait_root = _dual_trait_root(dataset_slug, trait_slug, dual_rice_root, dual_multi_root)
            triple_trait_root = trait_dir

            single = _load_single_metrics(single_trait_root)
            single_only = _load_single_only_metrics(single_trait_root)
            dual = _load_dual_metrics(dual_trait_root)
            dual_only = _load_dual_only_metrics(dual_trait_root)
            triple = _load_triple_metrics(triple_trait_root)
            triple_only = _load_triple_only_metrics(triple_trait_root)

            row["single_prior_tabicl_pearson"] = single["pearson"]
            row["single_prior_tabicl_r2"] = single["r2"]
            row["only_single_prior_pearson"] = single_only["pearson"]
            row["only_single_prior_r2"] = single_only["r2"]
            row["dual_prior_tabicl_pearson"] = dual["pearson"]
            row["dual_prior_tabicl_r2"] = dual["r2"]
            row["only_dual_prior_pearson"] = dual_only["pearson"]
            row["only_dual_prior_r2"] = dual_only["r2"]
            row["triple_prior_tabicl_pearson"] = triple["pearson"]
            row["triple_prior_tabicl_r2"] = triple["r2"]
            row["only_triple_prior_pearson"] = triple_only["pearson"]
            row["only_triple_prior_r2"] = triple_only["r2"]
            row.update(_load_single_weights(single_trait_root))
            row.update(_load_dual_weights(dual_trait_root))
            row.update(_load_triple_weights(triple_trait_root))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "trait_slug"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    single_root = Path(args.single_root)
    dual_rice_root = Path(args.dual_rice_root)
    dual_multi_root = Path(args.dual_multi_root)
    triple_root = Path(args.triple_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_df = build_compare(
        single_root=single_root,
        dual_rice_root=dual_rice_root,
        dual_multi_root=dual_multi_root,
        triple_root=triple_root,
    )
    compare_df.to_csv(output_dir / "trait_compare.csv", index=False)

    summary = {
        "traits_completed": int(len(compare_df)),
        "datasets_completed": int(compare_df["dataset"].nunique()) if not compare_df.empty else 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
