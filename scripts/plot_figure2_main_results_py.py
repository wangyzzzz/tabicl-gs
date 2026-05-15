from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "figure2_inputs"
OUTPUT_DIR = ROOT / "outputs" / "figures_20260513_python"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MPLCONFIGDIR = ROOT / "outputs" / "matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "neutral_dark": "#272727",
    "neutral_mid": "#767676",
    "neutral_light": "#D8D8D8",
    "signal_blue": "#3182BD",
    "signal_teal": "#33B5A5",
    "accent_red": "#D24B40",
    "accent_orange": "#E28E2C",
    "accent_gold": "#C8A03A",
    "accent_green": "#4D8B31",
}

DATASET_PRETTY = {
    "cotton1245": "Cotton",
    "rice529": "Rice",
    "soybean951": "Soybean",
    "wheat406": "Wheat",
}


def short_trait_label(value: str) -> str:
    parts = value.split("_")
    if len(parts) % 2 == 0:
        half = len(parts) // 2
        if parts[:half] == parts[half:]:
            return "_".join(parts[:half])
    return value


def setup_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "figure.titlesize": 17,
        }
    )
    sns.set_style("white")


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom", ha="left")


def save_outputs(fig: plt.Figure, prefix: Path) -> None:
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")


def draw_panel_a(ax: plt.Axes, single_long: pd.DataFrame) -> None:
    order = ["Single-BayesB", "Single-GBLUP", "Single-RKHS"]
    palette = {
        "Single-BayesB": PALETTE["signal_blue"],
        "Single-GBLUP": PALETTE["signal_teal"],
        "Single-RKHS": PALETTE["accent_orange"],
    }

    sns.boxplot(
        data=single_long,
        x="prior_label",
        y="single_vs_own_prior_pct",
        order=order,
        hue="prior_label",
        palette=palette,
        dodge=False,
        width=0.62,
        linewidth=1.0,
        fliersize=0,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.stripplot(
        data=single_long,
        x="prior_label",
        y="single_vs_own_prior_pct",
        order=order,
        color=PALETTE["neutral_dark"],
        alpha=0.45,
        size=5,
        jitter=0.10,
        ax=ax,
    )
    ax.axhline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0, zorder=0)

    means = single_long.groupby("prior_label", as_index=False, observed=False)["single_vs_own_prior_pct"].mean()
    means = means.set_index("prior_label").loc[order].reset_index()
    for idx, row in means.iterrows():
        ax.scatter(idx, row["single_vs_own_prior_pct"], marker="D", s=80, facecolor="white", edgecolor="black", zorder=5)
        ax.text(idx, row["single_vs_own_prior_pct"] + 0.18, f'{row["single_vs_own_prior_pct"]:.2f}%', ha="center", va="bottom", fontsize=9)

    ax.set_title("Single-prior fusion improves matched priors", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Gain over matched prior (%)")
    ax.tick_params(axis="x", rotation=16)
    ax.set_ylim(-1.1, max(6.0, single_long["single_vs_own_prior_pct"].max() + 0.4))


def draw_panel_b(ax: plt.Axes, triple_trait: pd.DataFrame) -> None:
    records = []
    mapping = [
        ("BayesB", "triple_vs_BayesB_pct"),
        ("GBLUP", "triple_vs_GBLUP_pct"),
        ("RKHS", "triple_vs_RKHS_pct"),
        ("Best baseline", "triple_vs_best_baseline_pct"),
        ("No-prior TabICL", "triple_vs_no_prior_tabicl_pct"),
    ]
    for label, col in mapping:
        tmp = triple_trait[["dataset", "trait_label", col]].copy()
        tmp["comparator"] = label
        tmp["gain_pct"] = tmp[col]
        records.append(tmp[["dataset", "trait_label", "comparator", "gain_pct"]])
    triple_long = pd.concat(records, ignore_index=True)

    order = ["BayesB", "GBLUP", "RKHS", "Best baseline", "No-prior TabICL"]
    palette = {
        "BayesB": PALETTE["signal_blue"],
        "GBLUP": PALETTE["signal_teal"],
        "RKHS": PALETTE["accent_orange"],
        "Best baseline": PALETTE["accent_gold"],
        "No-prior TabICL": PALETTE["accent_red"],
    }
    ylim_upper = 12

    sns.boxplot(
        data=triple_long,
        x="comparator",
        y="gain_pct",
        order=order,
        hue="comparator",
        palette=palette,
        dodge=False,
        width=0.62,
        linewidth=1.0,
        fliersize=0,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    clipped = triple_long["gain_pct"].clip(upper=ylim_upper)
    plot_df = triple_long.copy()
    plot_df["plot_gain"] = clipped
    sns.stripplot(
        data=plot_df,
        x="comparator",
        y="plot_gain",
        order=order,
        color=PALETTE["neutral_dark"],
        alpha=0.45,
        size=5,
        jitter=0.10,
        ax=ax,
    )
    ax.axhline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0, zorder=0)

    means = triple_long.groupby("comparator", as_index=False)["gain_pct"].mean()
    means = means.set_index("comparator").loc[order].reset_index()
    for idx, row in means.iterrows():
        ax.scatter(idx, min(row["gain_pct"], ylim_upper - 0.2), marker="D", s=80, facecolor="white", edgecolor="black", zorder=5)
        ax.text(idx, min(row["gain_pct"], ylim_upper - 0.2) + 0.18, f'{row["gain_pct"]:.2f}%', ha="center", va="bottom", fontsize=9)

    outliers = triple_long[triple_long["gain_pct"] > ylim_upper]
    if not outliers.empty:
        x_pos = order.index("No-prior TabICL")
        ax.scatter([x_pos] * len(outliers), [ylim_upper - 0.25] * len(outliers), marker="^", s=90, facecolor=PALETTE["accent_red"], edgecolor="black", zorder=6)
        ax.text(x_pos - 0.8, ylim_upper - 0.1, f"{len(outliers)} traits clipped above {ylim_upper}%", fontsize=9, color=PALETTE["neutral_dark"])

    ax.set_title("Triple-prior fusion improves over baselines\nand no-prior TabICL", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Gain of triple-prior fusion (%)")
    ax.tick_params(axis="x", rotation=16)
    ax.set_ylim(-2.2, 12.7)


def draw_panel_c(ax: plt.Axes, triple_trait: pd.DataFrame, single_long: pd.DataFrame) -> None:
    heat_single = single_long[["dataset", "trait_label", "prior_label", "single_vs_own_prior_pct"]].copy()
    heat_single = heat_single.rename(columns={"prior_label": "method", "single_vs_own_prior_pct": "gain_pct"})

    heat_triple = triple_trait[["dataset", "trait_label", "triple_vs_only_triple_pct"]].copy()
    heat_triple["method"] = "Triple (vs only-triple)"
    heat_triple = heat_triple.rename(columns={"triple_vs_only_triple_pct": "gain_pct"})

    heat_df = pd.concat(
        [
            heat_single[["dataset", "trait_label", "method", "gain_pct"]],
            heat_triple[["dataset", "trait_label", "method", "gain_pct"]],
        ],
        ignore_index=True,
    )

    method_order = ["Single-BayesB", "Single-GBLUP", "Single-RKHS", "Triple (vs only-triple)"]
    trait_order_df = triple_trait.sort_values(["dataset", "triple_vs_only_triple_pct", "trait_label"], ascending=[True, False, True]).copy()
    trait_order = list(dict.fromkeys([f'{d}|||{t}' for d, t in zip(trait_order_df["dataset"], trait_order_df["trait_label"])]))

    heat_df["trait_id"] = [f"{d}|||{t}" for d, t in zip(heat_df["dataset"], heat_df["trait_label"])]
    pivot = heat_df.pivot_table(index="trait_id", columns="method", values="gain_pct", aggfunc="first")
    pivot = pivot.reindex(index=trait_order, columns=method_order)

    cmap = LinearSegmentedColormap.from_list("gain_map", [PALETTE["accent_red"], "#FFF9F0", PALETTE["signal_blue"]])
    vmax = max(5.0, np.nanmax(np.abs(pivot.values)))
    norm = TwoSlopeNorm(vmin=-max(1.0, vmax * 0.25), vcenter=0.0, vmax=vmax)

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels(method_order, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([idx.split("|||", 1)[1] for idx in pivot.index], fontsize=7)
    ax.set_title("Trait-level gains remain heterogeneous", pad=24)
    ax.text(
        0.0,
        1.005,
        "Single columns are measured against the matched prior-only model;\ntriple is measured against only-triple-prior",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
    )
    ax.tick_params(length=0)

    for i in range(pivot.shape[0] + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.0)
    for j in range(pivot.shape[1] + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.0)

    dataset_ranges = []
    start = 0
    for ds in ["cotton1245", "rice529", "soybean951", "wheat406"]:
        count = sum(idx.startswith(f"{ds}|||") for idx in pivot.index)
        if count:
            dataset_ranges.append((ds, start, start + count - 1))
            start += count
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for ds, y0, y1 in dataset_ranges:
        ax.text(
            -0.28,
            (y0 + y1) / 2,
            DATASET_PRETTY[ds],
            transform=trans,
            va="center",
            ha="right",
            fontsize=12,
            fontweight="bold",
        )

    cbar = plt.colorbar(im, ax=ax, fraction=0.032, pad=0.04)
    cbar.ax.set_title("Gain (%)", fontsize=10, pad=10)


def draw_panel_d(ax: plt.Axes, triple_vs_best: pd.DataFrame) -> None:
    df = triple_vs_best.sort_values(["triple_minus_best_pct", "dataset", "trait_label"], ascending=[False, True, True]).copy()
    df["y"] = np.arange(len(df))[::-1]
    df["direction"] = np.where(df["triple_minus_best_pct"] >= 0, "Positive", "Negative")
    ax.axvline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0, zorder=0)
    for _, row in df.iterrows():
        color = PALETTE["accent_green"] if row["triple_minus_best_pct"] >= 0 else PALETTE["accent_red"]
        ax.hlines(row["y"], 0, row["triple_minus_best_pct"], color=color, linewidth=1.6, alpha=0.95, zorder=2)
        ax.scatter(row["triple_minus_best_pct"], row["y"], s=38, facecolor=color, edgecolor="black", linewidth=0.6, zorder=4)
    ax.set_yticks(df["y"])
    ax.set_yticklabels(df["trait_label"], fontsize=7)
    ax.set_xlabel("Triple vs best baseline (%)")
    ax.set_ylabel("")
    wins = int((df["triple_minus_best_pct"] >= 0).sum())
    ax.set_title(f"Triple beats best baseline in {wins}/{len(df)} traits", pad=10)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.8)

    legend_elems = [
        Line2D([0], [0], color=PALETTE["accent_red"], lw=2, marker="o", markersize=7, markerfacecolor=PALETTE["accent_red"], markeredgecolor="black", label="<0%"),
        Line2D([0], [0], color=PALETTE["accent_green"], lw=2, marker="o", markersize=7, markerfacecolor=PALETTE["accent_green"], markeredgecolor="black", label=">=0%"),
    ]
    ax.legend(handles=legend_elems, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.995))

    dataset_ranges = []
    for ds in ["cotton1245", "rice529", "soybean951", "wheat406"]:
        sub = df[df["dataset"] == ds]
        if not sub.empty:
            dataset_ranges.append((ds, sub["y"].min(), sub["y"].max()))
    x_left = float(df["triple_minus_best_pct"].min()) - 0.25
    for ds, y0, y1 in dataset_ranges:
        ax.text(
            x_left,
            (y0 + y1) / 2,
            DATASET_PRETTY[ds],
            va="center",
            ha="right",
            fontsize=12,
            fontweight="bold",
        )


def main() -> None:
    setup_theme()

    single_long = pd.read_csv(INPUT_DIR / "figure2_single_long.csv")
    triple_trait = pd.read_csv(INPUT_DIR / "figure2_triple_trait.csv")
    triple_vs_best = pd.read_csv(INPUT_DIR / "figure2_triple_vs_best.csv")

    single_long["dataset"] = pd.Categorical(single_long["dataset"], ["cotton1245", "rice529", "soybean951", "wheat406"], ordered=True)
    single_long["prior_label"] = pd.Categorical(
        single_long["prior_name"].map(
            {"BayesB": "Single-BayesB", "GBLUP": "Single-GBLUP", "RKHS": "Single-RKHS"}
        ),
        ["Single-BayesB", "Single-GBLUP", "Single-RKHS"],
        ordered=True,
    )
    single_long["trait_label"] = single_long["trait_slug"].map(short_trait_label)

    triple_trait["dataset"] = pd.Categorical(triple_trait["dataset"], ["cotton1245", "rice529", "soybean951", "wheat406"], ordered=True)
    triple_trait["trait_label"] = triple_trait["trait_slug"].map(short_trait_label)

    triple_vs_best["dataset"] = pd.Categorical(triple_vs_best["dataset"], ["cotton1245", "rice529", "soybean951", "wheat406"], ordered=True)
    triple_vs_best["trait_label"] = triple_vs_best["trait_slug"].map(short_trait_label)

    fig = plt.figure(figsize=(14.4, 16.8), constrained_layout=False)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.98, 1.08],
        width_ratios=[1.14, 1.0],
        left=0.13,
        right=0.98,
        top=0.93,
        bottom=0.06,
        hspace=0.24,
        wspace=0.30,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a, single_long)
    draw_panel_b(ax_b, triple_trait)
    draw_panel_c(ax_c, triple_trait, single_long)
    draw_panel_d(ax_d, triple_vs_best)

    add_panel_letter(ax_a, "a")
    add_panel_letter(ax_b, "b")
    add_panel_letter(ax_c, "c")
    add_panel_letter(ax_d, "d")

    fig.suptitle(
        "Prior-integrated fusion consistently strengthens TabICL in genomic prediction",
        x=0.02,
        y=0.978,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.958,
        "Main results across 36 non-pig traits under the 5.4-duli-liudang pipeline",
        ha="left",
        fontsize=12,
    )

    prefix = OUTPUT_DIR / "figure2_main_results_py"
    save_outputs(fig, prefix)
    print(f"Saved Python Figure 2 to: {prefix}")


if __name__ == "__main__":
    main()
