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
XLSX = ROOT / "outputs" / "results_support_workbook" / "results_support_tables_20260512.xlsx"
OUTDIR = ROOT / "outputs" / "figures_20260513_python"
OUTDIR.mkdir(parents=True, exist_ok=True)

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

DATASET_ORDER = ["cotton1245", "rice529", "soybean951", "wheat406"]

DATASET_COLORS = {
    "cotton1245": PALETTE["signal_blue"],
    "rice529": PALETTE["signal_teal"],
    "soybean951": PALETTE["accent_orange"],
    "wheat406": PALETTE["accent_gold"],
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


def save_outputs(fig: plt.Figure, prefix: Path) -> None:
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom", ha="left")


def dataset_group_labels(ax: plt.Axes, ordered_ids: list[str], offset_axes: float = -0.22) -> None:
    groups: list[tuple[str, int, int]] = []
    start = 0
    for ds in DATASET_ORDER:
        count = sum(item.startswith(f"{ds}|||") for item in ordered_ids)
        if count:
            groups.append((ds, start, start + count - 1))
            start += count
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for ds, y0, y1 in groups:
        ax.text(
            offset_axes,
            (y0 + y1) / 2,
            DATASET_PRETTY[ds],
            transform=trans,
            va="center",
            ha="right",
            fontsize=12,
            fontweight="bold",
        )


def clean_trait_series(series: pd.Series) -> pd.Series:
    return series.map(short_trait_label)


def nice_boxplot_with_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: list[str],
    palette: dict[str, str],
    mean_label_fmt: str = "{:.2f}%",
    text_nudge: float = 0.18,
) -> None:
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        order=order,
        hue=x,
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
        data=data,
        x=x,
        y=y,
        order=order,
        color=PALETTE["neutral_dark"],
        alpha=0.45,
        size=5,
        jitter=0.10,
        ax=ax,
    )
    means = data.groupby(x, as_index=False, observed=False)[y].mean()
    means = means.set_index(x).loc[order].reset_index()
    for idx, row in means.iterrows():
        ax.scatter(idx, row[y], marker="D", s=70, facecolor="white", edgecolor="black", zorder=5)
        ax.text(idx, row[y] + text_nudge, mean_label_fmt.format(row[y]), ha="center", va="bottom", fontsize=9)


def load_sheet(name: str) -> pd.DataFrame:
    return pd.read_excel(XLSX, sheet_name=name, header=1)


def load_result1_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r1_main = load_sheet("R1_main36")
    r1_summary = load_sheet("R1_summary")
    r1_cases = load_sheet("R1_cases")

    r1_main["dataset"] = pd.Categorical(r1_main["dataset"], DATASET_ORDER, ordered=True)
    r1_main["trait_label"] = clean_trait_series(r1_main["trait_slug"])
    r1_cases["dataset"] = pd.Categorical(r1_cases["dataset"], DATASET_ORDER, ordered=True)
    r1_cases["trait_label"] = clean_trait_series(r1_cases["trait_slug"])
    return r1_main, r1_summary, r1_cases


def result1_summary_bar(r1_summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["No-prior TabICL", "BayesB", "GBLUP", "RKHS"],
            "value": [
                float(r1_summary["mean_no_prior_tabicl"].iloc[0]),
                float(r1_summary.loc[r1_summary["baseline"] == "BayesB", "mean_baseline"].iloc[0]),
                float(r1_summary.loc[r1_summary["baseline"] == "GBLUP", "mean_baseline"].iloc[0]),
                float(r1_summary.loc[r1_summary["baseline"] == "RKHS", "mean_baseline"].iloc[0]),
            ],
        }
    )


def figure1a() -> Path:
    _, r1_summary, _ = load_result1_tables()
    summary_bar = result1_summary_bar(r1_summary)

    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=0.18)

    colors = [PALETTE["accent_red"], PALETTE["signal_blue"], PALETTE["signal_teal"], PALETTE["accent_orange"]]
    ax.bar(summary_bar["method"], summary_bar["value"], color=colors, edgecolor="black", linewidth=0.85, width=0.72)
    for idx, row in summary_bar.iterrows():
        ax.text(idx, row["value"] + 0.0026, f'{row["value"]:.3f}', ha="center", va="bottom", fontsize=9.5)

    ymin = summary_bar["value"].min() - 0.02
    ymax = summary_bar["value"].max() + 0.03
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Mean Pearson correlation")
    ax.set_xlabel("")
    ax.set_title("Mean accuracy across 36 non-pig traits", pad=10)
    ax.tick_params(axis="x", rotation=16)
    ax.yaxis.grid(True, color="#ECECEC", linewidth=0.8)
    ax.set_axisbelow(True)
    add_panel_letter(ax, "a")

    prefix = OUTDIR / "fig1a_mean_pearson_bar_py"
    save_outputs(fig, prefix)
    plt.close(fig)
    return prefix


def figure1b() -> Path:
    r1_main, _, r1_cases = load_result1_tables()

    rel_long = r1_main[
        ["dataset", "trait_label", "no_vs_BayesB_pct", "no_vs_GBLUP_pct", "no_vs_RKHS_pct"]
    ].melt(
        id_vars=["dataset", "trait_label"],
        value_vars=["no_vs_BayesB_pct", "no_vs_GBLUP_pct", "no_vs_RKHS_pct"],
        var_name="baseline",
        value_name="pct",
    )
    rel_long["baseline"] = rel_long["baseline"].map(
        {
            "no_vs_BayesB_pct": "BayesB",
            "no_vs_GBLUP_pct": "GBLUP",
            "no_vs_RKHS_pct": "RKHS",
        }
    )
    rel_long["trait_id"] = [f"{d}|||{t}" for d, t in zip(rel_long["dataset"], rel_long["trait_label"])]

    order_df = (
        r1_main.assign(mean_pct=r1_main[["no_vs_BayesB_pct", "no_vs_GBLUP_pct", "no_vs_RKHS_pct"]].mean(axis=1))
        .sort_values(["dataset", "mean_pct", "trait_label"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    ordered_ids = [f"{d}|||{t}" for d, t in zip(order_df["dataset"], order_df["trait_label"])]
    y_map = {trait_id: idx for idx, trait_id in enumerate(ordered_ids)}
    rel_long["y_base"] = rel_long["trait_id"].map(y_map)

    fig, ax = plt.subplots(figsize=(9.2, 11.8), constrained_layout=False)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.95, bottom=0.08)

    marker_map = {"BayesB": "o", "GBLUP": "s", "RKHS": "D"}
    color_map = {
        "BayesB": PALETTE["signal_blue"],
        "GBLUP": PALETTE["signal_teal"],
        "RKHS": PALETTE["accent_orange"],
    }
    y_offset = {"BayesB": -0.20, "GBLUP": 0.00, "RKHS": 0.20}

    highlight_ids = set(f"{d}|||{t}" for d, t in zip(r1_cases["dataset"], r1_cases["trait_label"]))
    for trait_id in ordered_ids:
        y = y_map[trait_id]
        if trait_id in highlight_ids:
            ax.axhspan(y - 0.48, y + 0.48, color="#FFF3F0", zorder=0)
        row = rel_long[rel_long["trait_id"] == trait_id]
        ax.plot(
            [row["pct"].min(), row["pct"].max()],
            [y, y],
            color=PALETTE["neutral_light"],
            linewidth=1.0,
            zorder=1,
        )

    for baseline in ["BayesB", "GBLUP", "RKHS"]:
        sub = rel_long[rel_long["baseline"] == baseline].copy()
        ys = sub["y_base"] + y_offset[baseline]
        ax.scatter(
            sub["pct"],
            ys,
            s=42,
            marker=marker_map[baseline],
            color=color_map[baseline],
            edgecolor="black",
            linewidth=0.55,
            alpha=0.95,
            label=baseline,
            zorder=3,
        )

    xmax = float(np.nanmax(np.abs(rel_long["pct"]))) + 1.0
    ax.axvline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0, zorder=1)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(len(ordered_ids) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(ordered_ids)))
    ax.set_yticklabels([trait_id.split("|||", 1)[1] for trait_id in ordered_ids], fontsize=8.2)
    ax.set_xlabel("Relative difference of no-prior TabICL vs baseline (%)")
    ax.set_ylabel("")
    ax.set_title("Trait-level heterogeneity against BayesB, GBLUP, and RKHS", pad=10)
    ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.set_axisbelow(True)
    dataset_group_labels(ax, ordered_ids, offset_axes=-0.18)

    for idx in range(len(ordered_ids) + 1):
        ax.axhline(idx - 0.5, color="#F4F4F4", linewidth=0.6, zorder=0)

    for tick, trait_id in zip(ax.get_yticklabels(), ordered_ids):
        if trait_id in highlight_ids:
            tick.set_color(PALETTE["accent_red"])
            tick.set_fontweight("bold")

    ax.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.02), handletextpad=0.5, columnspacing=1.1)
    add_panel_letter(ax, "b")

    prefix = OUTDIR / "fig1b_trait_relative_diff_py"
    save_outputs(fig, prefix)
    plt.close(fig)
    return prefix


def figure1() -> Path:
    r1_main = load_sheet("R1_main36")
    r1_summary = load_sheet("R1_summary")
    r1_cases = load_sheet("R1_cases")

    r1_main["dataset"] = pd.Categorical(r1_main["dataset"], DATASET_ORDER, ordered=True)
    r1_main["trait_label"] = clean_trait_series(r1_main["trait_slug"])
    r1_cases["trait_label"] = clean_trait_series(r1_cases["trait_slug"])

    # panel a: mean Pearson summary
    summary_bar = pd.DataFrame(
        {
            "method": ["No-prior TabICL", "BayesB", "GBLUP", "RKHS"],
            "value": [
                float(r1_summary["mean_no_prior_tabicl"].iloc[0]),
                float(r1_summary.loc[r1_summary["baseline"] == "BayesB", "mean_baseline"].iloc[0]),
                float(r1_summary.loc[r1_summary["baseline"] == "GBLUP", "mean_baseline"].iloc[0]),
                float(r1_summary.loc[r1_summary["baseline"] == "RKHS", "mean_baseline"].iloc[0]),
            ],
        }
    )

    # panel b: relative diff distribution
    rel_long = pd.concat(
        [
            pd.DataFrame({"baseline": "BayesB", "pct": r1_main["no_vs_BayesB_pct"]}),
            pd.DataFrame({"baseline": "GBLUP", "pct": r1_main["no_vs_GBLUP_pct"]}),
            pd.DataFrame({"baseline": "RKHS", "pct": r1_main["no_vs_RKHS_pct"]}),
        ],
        ignore_index=True,
    )
    win_map = dict(zip(r1_summary["baseline"], r1_summary["no_prior_win_count"]))
    rel_long["baseline_label"] = rel_long["baseline"].map(lambda x: f"{x}\n{win_map[x]}/36 wins")

    # panel c: heatmap
    heat_df = r1_main[["dataset", "trait_label", "no_vs_BayesB_pct", "no_vs_GBLUP_pct", "no_vs_RKHS_pct"]].copy()
    heat_long = heat_df.melt(
        id_vars=["dataset", "trait_label"],
        var_name="baseline",
        value_name="pct",
    )
    heat_long["baseline"] = heat_long["baseline"].map(
        {
            "no_vs_BayesB_pct": "BayesB",
            "no_vs_GBLUP_pct": "GBLUP",
            "no_vs_RKHS_pct": "RKHS",
        }
    )
    order_df = (
        r1_main.assign(mean_pct=r1_main[["no_vs_BayesB_pct", "no_vs_GBLUP_pct", "no_vs_RKHS_pct"]].mean(axis=1))
        .sort_values(["dataset", "mean_pct", "trait_label"], ascending=[True, False, True])
    )
    trait_ids = [f"{d}|||{t}" for d, t in zip(order_df["dataset"], order_df["trait_label"])]
    heat_long["trait_id"] = [f"{d}|||{t}" for d, t in zip(heat_long["dataset"], heat_long["trait_label"])]
    heat = heat_long.pivot_table(index="trait_id", columns="baseline", values="pct", aggfunc="first")
    heat = heat.reindex(index=trait_ids, columns=["BayesB", "GBLUP", "RKHS"])

    # panel d: case examples
    methods = ["no_prior_tabicl", "BayesB", "GBLUP", "RKHS"]
    case_method_labels = {
        "no_prior_tabicl": "No-prior TabICL",
        "BayesB": "BayesB",
        "GBLUP": "GBLUP",
        "RKHS": "RKHS",
    }
    case_palette = {
        "No-prior TabICL": PALETTE["accent_red"],
        "BayesB": PALETTE["signal_blue"],
        "GBLUP": PALETTE["signal_teal"],
        "RKHS": PALETTE["accent_orange"],
    }
    case_long = r1_cases.melt(
        id_vars=["dataset", "trait_label"],
        value_vars=methods,
        var_name="method",
        value_name="pearson",
    )
    case_long["method_label"] = case_long["method"].map(case_method_labels)
    case_order = list(r1_cases["trait_label"])

    fig = plt.figure(figsize=(14.4, 15.8), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.11, right=0.98, top=0.93, bottom=0.06, wspace=0.28, hspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # a
    bar_colors = [PALETTE["accent_red"], PALETTE["signal_blue"], PALETTE["signal_teal"], PALETTE["accent_orange"]]
    ax1.bar(summary_bar["method"], summary_bar["value"], color=bar_colors, edgecolor="black", linewidth=0.8)
    for i, row in summary_bar.iterrows():
        ax1.text(i, row["value"] + 0.008, f'{row["value"]:.3f}', ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Mean Pearson correlation")
    ax1.set_title("No-prior TabICL is competitive but not the average best", pad=10)
    ax1.tick_params(axis="x", rotation=18)
    ax1.set_ylim(0.60, 0.70)

    # b
    nice_boxplot_with_points(
        ax2,
        rel_long,
        x="baseline_label",
        y="pct",
        order=[f"BayesB\n{win_map['BayesB']}/36 wins", f"GBLUP\n{win_map['GBLUP']}/36 wins", f"RKHS\n{win_map['RKHS']}/36 wins"],
        palette={
            f"BayesB\n{win_map['BayesB']}/36 wins": PALETTE["signal_blue"],
            f"GBLUP\n{win_map['GBLUP']}/36 wins": PALETTE["signal_teal"],
            f"RKHS\n{win_map['RKHS']}/36 wins": PALETTE["accent_orange"],
        },
    )
    ax2.axhline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0, zorder=0)
    ax2.set_title("Average deficits dominate, but local wins remain", pad=10)
    ax2.set_xlabel("")
    ax2.set_ylabel("No-prior vs baseline (%)")
    ax2.tick_params(axis="x", rotation=14)

    # c
    cmap = LinearSegmentedColormap.from_list("r1", [PALETTE["accent_red"], "#FFF9F0", PALETTE["signal_blue"]])
    vmax = max(6.0, np.nanmax(np.abs(heat.values)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax3.imshow(heat.values, aspect="auto", cmap=cmap, norm=norm)
    ax3.set_xticks(np.arange(heat.shape[1]))
    ax3.set_xticklabels(list(heat.columns), rotation=18, ha="right")
    ax3.set_yticks(np.arange(heat.shape[0]))
    ax3.set_yticklabels([idx.split("|||", 1)[1] for idx in heat.index], fontsize=6.8)
    ax3.set_title("Trait-level relative differences are strongly heterogeneous", pad=12)
    ax3.tick_params(length=0)
    for i in range(heat.shape[0] + 1):
        ax3.axhline(i - 0.5, color="white", linewidth=0.9)
    for j in range(heat.shape[1] + 1):
        ax3.axvline(j - 0.5, color="white", linewidth=0.9)
    dataset_group_labels(ax3, list(heat.index), offset_axes=-0.28)
    cbar = plt.colorbar(im, ax=ax3, fraction=0.036, pad=0.04)
    cbar.ax.set_title("Diff (%)", fontsize=10, pad=8)

    # d
    x = np.arange(len(case_order))
    width = 0.18
    case_method_order = ["No-prior TabICL", "BayesB", "GBLUP", "RKHS"]
    method_to_col = {
        "No-prior TabICL": "no_prior_tabicl",
        "BayesB": "BayesB",
        "GBLUP": "GBLUP",
        "RKHS": "RKHS",
    }
    offsets = {
        "No-prior TabICL": -1.5 * width,
        "BayesB": -0.5 * width,
        "GBLUP": 0.5 * width,
        "RKHS": 1.5 * width,
    }
    for method_label in case_method_order:
        values = r1_cases[method_to_col[method_label]].to_numpy()
        ax4.bar(
            x + offsets[method_label],
            values,
            width=width,
            color=case_palette[method_label],
            edgecolor="black",
            linewidth=0.8,
            label=method_label,
        )
    ax4.set_xticks(x)
    ax4.set_xticklabels(case_order, rotation=16, ha="right")
    ax4.set_ylabel("Pearson correlation")
    ax4.set_title("Some traits already favor no-prior TabICL", pad=10)
    ax4.set_ylim(0.54, 0.875)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=case_palette[k], markeredgecolor="black", markersize=8, label=k)
        for k in case_palette
    ]
    ax4.legend(handles=handles, frameon=False, ncol=2, loc="upper left")

    for ax, letter in zip([ax1, ax2, ax3, ax4], list("abcd")):
        add_panel_letter(ax, letter)

    fig.suptitle("Table foundation model enters GS but is not stably optimal alone", x=0.02, y=0.982, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.962, "Result 1 across 36 non-pig traits", ha="left", fontsize=12)

    prefix = OUTDIR / "figure1_noprior_vs_baselines_py"
    save_outputs(fig, prefix)
    plt.close(fig)
    return prefix


def figure3() -> Path:
    weights = load_sheet("R3_weights36")
    weights["dataset"] = pd.Categorical(weights["dataset"], DATASET_ORDER, ordered=True)
    weights["trait_label"] = clean_trait_series(weights["trait_slug"])
    weights["mean_single_w"] = weights[["w_single_bayesb", "w_single_gblup", "w_single_rkhs"]].mean(axis=1)

    # panel a
    w_long = weights.melt(
        id_vars=["dataset", "trait_label"],
        value_vars=["w_single_bayesb", "w_single_gblup", "w_single_rkhs", "w_triple"],
        var_name="method",
        value_name="w_value",
    )
    w_long["method"] = w_long["method"].map(
        {
            "w_single_bayesb": "Single-BayesB",
            "w_single_gblup": "Single-GBLUP",
            "w_single_rkhs": "Single-RKHS",
            "w_triple": "Triple",
        }
    )

    # panel c heatmap
    heat = weights[["dataset", "trait_label", "w_single_bayesb", "w_single_gblup", "w_single_rkhs", "w_triple"]].copy()
    order_df = weights.sort_values(["dataset", "mean_single_w", "trait_label"], ascending=[True, False, True])
    trait_ids = [f"{d}|||{t}" for d, t in zip(order_df["dataset"], order_df["trait_label"])]
    heat["trait_id"] = [f"{d}|||{t}" for d, t in zip(heat["dataset"], heat["trait_label"])]
    heat = heat.set_index("trait_id")[["w_single_bayesb", "w_single_gblup", "w_single_rkhs", "w_triple"]].reindex(trait_ids)
    heat.columns = ["Single-BayesB", "Single-GBLUP", "Single-RKHS", "Triple"]

    # panel d prior shares
    share_long = weights.melt(
        id_vars=["dataset", "trait_label"],
        value_vars=["bayesb_share", "gblup_share", "rkhs_share"],
        var_name="prior",
        value_name="share",
    )
    share_long["prior"] = share_long["prior"].map(
        {
            "bayesb_share": "BayesB share",
            "gblup_share": "GBLUP share",
            "rkhs_share": "RKHS share",
        }
    )

    fig = plt.figure(figsize=(14.4, 15.8), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.11, right=0.98, top=0.93, bottom=0.06, wspace=0.28, hspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    nice_boxplot_with_points(
        ax1,
        w_long,
        x="method",
        y="w_value",
        order=["Single-BayesB", "Single-GBLUP", "Single-RKHS", "Triple"],
        palette={
            "Single-BayesB": PALETTE["signal_blue"],
            "Single-GBLUP": PALETTE["signal_teal"],
            "Single-RKHS": PALETTE["accent_orange"],
            "Triple": PALETTE["accent_red"],
        },
        mean_label_fmt="{:.3f}",
        text_nudge=0.004,
    )
    ax1.set_title("TabICL keeps a stable but non-trivial weight", pad=10)
    ax1.set_xlabel("")
    ax1.set_ylabel("OLS weight of TabICL")
    ax1.set_ylim(0.40, 0.57)
    ax1.tick_params(axis="x", rotation=18)

    # b correlation
    x = weights["no_prior_vs_best_baseline_pct"].to_numpy()
    y = weights["mean_single_w"].to_numpy()
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax2.plot(xs, coef[0] * xs + coef[1], color=PALETTE["neutral_dark"], linewidth=1.3)
    for ds in DATASET_ORDER:
        sub = weights[weights["dataset"] == ds]
        ax2.scatter(
            sub["no_prior_vs_best_baseline_pct"],
            sub["mean_single_w"],
            s=52,
            color=DATASET_COLORS[ds],
            edgecolor="black",
            alpha=0.9,
            label=DATASET_PRETTY[ds],
        )
    corr = np.corrcoef(x, y)[0, 1]
    ax2.text(0.04, 0.95, f"r = {corr:.3f}", transform=ax2.transAxes, ha="left", va="top", fontsize=10)
    ax2.set_title("Higher single-fusion weight appears when no-prior is more competitive", pad=10)
    ax2.set_xlabel("No-prior TabICL vs best baseline (%)")
    ax2.set_ylabel("Mean single-prior weight of TabICL")
    ax2.legend(frameon=False, ncol=2, loc="lower right")

    # c heatmap
    cmap = LinearSegmentedColormap.from_list("weights", ["#FFF9F0", "#AFC8E4", PALETTE["signal_blue"]])
    im = ax3.imshow(heat.values, aspect="auto", cmap=cmap, vmin=0.42, vmax=0.56)
    ax3.set_xticks(np.arange(heat.shape[1]))
    ax3.set_xticklabels(list(heat.columns), rotation=18, ha="right")
    ax3.set_yticks(np.arange(heat.shape[0]))
    ax3.set_yticklabels([idx.split("|||", 1)[1] for idx in heat.index], fontsize=6.8)
    ax3.set_title("Trait-level TabICL weights remain heterogeneous", pad=12)
    ax3.tick_params(length=0)
    for i in range(heat.shape[0] + 1):
        ax3.axhline(i - 0.5, color="white", linewidth=0.9)
    for j in range(heat.shape[1] + 1):
        ax3.axvline(j - 0.5, color="white", linewidth=0.9)
    dataset_group_labels(ax3, list(heat.index), offset_axes=-0.28)
    cbar = plt.colorbar(im, ax=ax3, fraction=0.036, pad=0.04)
    cbar.ax.set_title("w", fontsize=10, pad=8)

    nice_boxplot_with_points(
        ax4,
        share_long,
        x="prior",
        y="share",
        order=["BayesB share", "GBLUP share", "RKHS share"],
        palette={
            "BayesB share": PALETTE["signal_blue"],
            "GBLUP share": PALETTE["signal_teal"],
            "RKHS share": PALETTE["accent_orange"],
        },
        mean_label_fmt="{:.3f}",
        text_nudge=0.0015,
    )
    ax4.set_title("Triple prior shares stay broadly balanced", pad=10)
    ax4.set_xlabel("")
    ax4.set_ylabel("Share inside prior aggregate")
    ax4.set_ylim(0.28, 0.38)
    ax4.tick_params(axis="x", rotation=18)

    for ax, letter in zip([ax1, ax2, ax3, ax4], list("abcd")):
        add_panel_letter(ax, letter)

    fig.suptitle("Fusion weights show stable yet trait-dependent contributions from TabICL", x=0.02, y=0.982, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.962, "Result 3 across 36 non-pig traits", ha="left", fontsize=12)

    prefix = OUTDIR / "figure3_weight_mechanism_py"
    save_outputs(fig, prefix)
    plt.close(fig)
    return prefix


def figure4() -> Path:
    trait = load_sheet("R4_sample8_trait")
    mean = load_sheet("R4_sample8_mean")
    cases = load_sheet("R4_sample8_cases")

    trait["dataset"] = pd.Categorical(trait["dataset"], DATASET_ORDER, ordered=True)
    trait["trait_label"] = clean_trait_series(trait["trait_slug"])
    cases["trait_label"] = clean_trait_series(cases["trait_slug"])

    x_order = ["20%", "60%", "100%"]
    x_map = {lab: i for i, lab in enumerate(x_order)}

    fig = plt.figure(figsize=(14.6, 15.8), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.10, right=0.98, top=0.93, bottom=0.06, wspace=0.30, hspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    subgs = gs[1, 1].subgridspec(2, 2, wspace=0.30, hspace=0.40)
    ax4s = [fig.add_subplot(subgs[i, j]) for i in range(2) for j in range(2)]

    # a mean Pearson
    methods = [
        ("mean_no_prior", "No-prior TabICL", PALETTE["accent_red"]),
        ("mean_BayesB", "BayesB", PALETTE["signal_blue"]),
        ("mean_GBLUP", "GBLUP", PALETTE["signal_teal"]),
        ("mean_RKHS", "RKHS", PALETTE["accent_orange"]),
        ("mean_triple", "Triple", PALETTE["accent_green"]),
    ]
    for col, label, color in methods:
        ax1.plot(mean["sample_label"], mean[col], marker="o", markersize=6, linewidth=2, color=color, label=label)
    ax1.set_title("All methods improve with larger sample size", pad=10)
    ax1.set_xlabel("Sample fraction")
    ax1.set_ylabel("Mean Pearson correlation")
    ax1.legend(frameon=False, ncol=2, loc="lower right")

    # b gain trajectories
    gain_methods = [
        ("mean_triple_vs_BayesB_pct", "vs BayesB", PALETTE["signal_blue"]),
        ("mean_triple_vs_GBLUP_pct", "vs GBLUP", PALETTE["signal_teal"]),
        ("mean_triple_vs_RKHS_pct", "vs RKHS", PALETTE["accent_orange"]),
        ("mean_triple_vs_best_pct", "vs best baseline", PALETTE["accent_green"]),
    ]
    for col, label, color in gain_methods:
        ax2.plot(mean["sample_label"], mean[col], marker="o", markersize=6, linewidth=2, color=color, label=label)
    ax2.axhline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0)
    ax2.set_title("Triple gains become more stable as sample size increases", pad=10)
    ax2.set_xlabel("Sample fraction")
    ax2.set_ylabel("Relative gain of triple (%)")
    ax2.legend(frameon=False, ncol=2, loc="upper left")
    for _, row in mean.iterrows():
        ax2.text(x_map[row["sample_label"]], row["mean_triple_vs_best_pct"] + 0.10, f'{int(row["triple_gt_best_count"])}/8', ha="center", fontsize=9)

    # c heatmap
    heat = trait.pivot_table(index=["dataset", "trait_label"], columns="sample_label", values="triple_vs_best_baseline_pct", aggfunc="first")
    heat = heat.reindex(columns=x_order)
    heat = heat.sort_index(level=[0, 1])
    ordered_ids = [f"{ds}|||{tr}" for ds, tr in heat.index]
    heat.index = ordered_ids
    cmap = LinearSegmentedColormap.from_list("sample_gain", [PALETTE["accent_red"], "#FFF9F0", PALETTE["signal_blue"]])
    vmax = max(5.5, np.nanmax(np.abs(heat.values)))
    norm = TwoSlopeNorm(vmin=-max(1.0, vmax * 0.25), vcenter=0.0, vmax=vmax)
    im = ax3.imshow(heat.values, aspect="auto", cmap=cmap, norm=norm)
    ax3.set_xticks(np.arange(len(x_order)))
    ax3.set_xticklabels(x_order)
    ax3.set_yticks(np.arange(heat.shape[0]))
    ax3.set_yticklabels([idx.split("|||", 1)[1] for idx in heat.index], fontsize=7)
    ax3.set_title("Trait-level triple gains remain sample-size dependent", pad=10)
    ax3.tick_params(length=0)
    for i in range(heat.shape[0] + 1):
        ax3.axhline(i - 0.5, color="white", linewidth=0.9)
    for j in range(heat.shape[1] + 1):
        ax3.axvline(j - 0.5, color="white", linewidth=0.9)
    dataset_group_labels(ax3, list(heat.index), offset_axes=-0.28)
    cbar = plt.colorbar(im, ax=ax3, fraction=0.036, pad=0.04)
    cbar.ax.set_title("Gain (%)", fontsize=10, pad=8)

    # d cases
    case_palette = {
        "No-prior": PALETTE["neutral_mid"],
        "Best baseline": PALETTE["accent_gold"],
        "Triple": PALETTE["accent_green"],
    }
    for ax, (trait_name, grp) in zip(ax4s, cases.groupby("trait_label", sort=False)):
        grp = grp.sort_values("sample_fraction")
        ax.plot(grp["sample_label"], grp["no_prior_tabicl"], marker="o", color=case_palette["No-prior"], linewidth=1.8, label="No-prior")
        ax.plot(grp["sample_label"], grp["best_baseline"], marker="o", color=case_palette["Best baseline"], linewidth=1.8, label="Best baseline")
        ax.plot(grp["sample_label"], grp["triple_two_step_ls"], marker="o", color=case_palette["Triple"], linewidth=2.0, label="Triple")
        ax.set_title(trait_name, fontsize=10, pad=6)
        ax.set_ylim(min(grp[["no_prior_tabicl", "best_baseline", "triple_two_step_ls"]].min()) - 0.03, max(grp[["no_prior_tabicl", "best_baseline", "triple_two_step_ls"]].max()) + 0.03)
        ax.tick_params(axis="x", rotation=0)
    ax4s[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.74, 0.49, "Representative trajectories", fontsize=12, fontweight="bold", ha="center")

    for ax, letter in zip([ax1, ax2, ax3, ax4s[0]], list("abcd")):
        add_panel_letter(ax, letter)

    fig.suptitle("Sample size reshapes the stability of prior-integrated fusion", x=0.02, y=0.982, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.962, "Result 4 across the completed 8-trait sample-size panel", ha="left", fontsize=12)

    prefix = OUTDIR / "figure4_sample_size_py"
    save_outputs(fig, prefix)
    plt.close(fig)
    return prefix


def figure5() -> Path:
    raw = load_sheet("R5_marker_raw")
    mean = load_sheet("R5_marker_mean")
    trait = load_sheet("R5_marker_trait")

    raw["dataset"] = pd.Categorical(raw["dataset"], DATASET_ORDER, ordered=True)
    raw["trait_label"] = clean_trait_series(raw["trait_slug"])
    trait["dataset"] = pd.Categorical(trait["dataset"], DATASET_ORDER, ordered=True)
    trait["trait_label"] = clean_trait_series(trait["trait_slug"])

    x_order = ["2K", "10K", "50K"]

    fig = plt.figure(figsize=(14.6, 15.8), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.10, right=0.98, top=0.93, bottom=0.06, wspace=0.30, hspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # a mean Pearson
    methods = [
        ("mean_no_prior", "No-prior TabICL", PALETTE["accent_red"]),
        ("mean_BayesB", "BayesB", PALETTE["signal_blue"]),
        ("mean_GBLUP", "GBLUP", PALETTE["signal_teal"]),
        ("mean_RKHS", "RKHS", PALETTE["accent_orange"]),
        ("mean_triple", "Triple", PALETTE["accent_green"]),
    ]
    for col, label, color in methods:
        ax1.plot(mean["marker_label"], mean[col], marker="o", markersize=6, linewidth=2, color=color, label=label)
    ax1.set_title("Higher SNP density raises the overall performance ceiling", pad=10)
    ax1.set_xlabel("Marker count")
    ax1.set_ylabel("Mean Pearson correlation")
    ax1.legend(frameon=False, ncol=2, loc="lower right")

    # b triple gains
    gain_methods = [
        ("mean_triple_vs_BayesB_pct", "vs BayesB", PALETTE["signal_blue"]),
        ("mean_triple_vs_GBLUP_pct", "vs GBLUP", PALETTE["signal_teal"]),
        ("mean_triple_vs_RKHS_pct", "vs RKHS", PALETTE["accent_orange"]),
    ]
    for col, label, color in gain_methods:
        ax2.plot(mean["marker_label"], mean[col], marker="o", markersize=6, linewidth=2, color=color, label=label)
    ax2.axhline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0)
    ax2.set_title("Triple gains remain positive but do not expand uniformly", pad=10)
    ax2.set_xlabel("Marker count")
    ax2.set_ylabel("Relative gain of triple (%)")
    ax2.legend(frameon=False, ncol=2, loc="upper left")

    # c heatmap trait x marker
    heat = raw.pivot_table(index=["dataset", "trait_label"], columns="marker_label", values="triple_two_step_ls_vs_best_baseline_pct", aggfunc="first")
    heat = heat.reindex(columns=x_order)
    heat = heat.sort_index(level=[0, 1])
    ordered_ids = [f"{ds}|||{tr}" for ds, tr in heat.index]
    heat.index = ordered_ids
    cmap = LinearSegmentedColormap.from_list("marker_gain", [PALETTE["accent_red"], "#FFF9F0", PALETTE["signal_blue"]])
    vmax = max(5.5, np.nanmax(np.abs(heat.values)))
    norm = TwoSlopeNorm(vmin=-max(1.0, vmax * 0.28), vcenter=0.0, vmax=vmax)
    im = ax3.imshow(heat.values, aspect="auto", cmap=cmap, norm=norm)
    ax3.set_xticks(np.arange(len(x_order)))
    ax3.set_xticklabels(x_order)
    ax3.set_yticks(np.arange(heat.shape[0]))
    ax3.set_yticklabels([idx.split("|||", 1)[1] for idx in heat.index], fontsize=7)
    ax3.set_title("Exceptions persist at the trait level", pad=10)
    ax3.tick_params(length=0)
    for i in range(heat.shape[0] + 1):
        ax3.axhline(i - 0.5, color="white", linewidth=0.9)
    for j in range(heat.shape[1] + 1):
        ax3.axvline(j - 0.5, color="white", linewidth=0.9)
    dataset_group_labels(ax3, list(heat.index), offset_axes=-0.28)
    cbar = plt.colorbar(im, ax=ax3, fraction=0.036, pad=0.04)
    cbar.ax.set_title("Gain (%)", fontsize=10, pad=8)

    # d single-prior gains
    single_cols = [
        ("mean_single_bayesb_vs_own_prior_pct", "Single-BayesB", PALETTE["signal_blue"]),
        ("mean_single_gblup_vs_own_prior_pct", "Single-GBLUP", PALETTE["signal_teal"]),
        ("mean_single_rkhs_vs_own_prior_pct", "Single-RKHS", PALETTE["accent_orange"]),
    ]
    for col, label, color in single_cols:
        ax4.plot(mean["marker_label"], mean[col], marker="o", markersize=6, linewidth=2, color=color, label=label)
    ax4.axhline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0)
    ax4.set_title("Single-prior fusion also benefits from richer markers", pad=10)
    ax4.set_xlabel("Marker count")
    ax4.set_ylabel("Gain over matched prior (%)")
    ax4.legend(frameon=False, ncol=1, loc="upper right")

    for ax, letter in zip([ax1, ax2, ax3, ax4], list("abcd")):
        add_panel_letter(ax, letter)

    fig.suptitle("Marker density improves absolute performance but not every relative gain equally", x=0.02, y=0.982, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.962, "Result 5 across the completed 8-trait marker-count panel", ha="left", fontsize=12)

    prefix = OUTDIR / "figure5_marker_count_py"
    save_outputs(fig, prefix)
    plt.close(fig)
    return prefix


def figure6() -> Path:
    raw = load_sheet("R6_tabpfn_raw")
    single = load_sheet("R6_tabpfn_single")
    single_sum = load_sheet("R6_tabpfn_single_sum")
    summary = load_sheet("R6_tabpfn_sum")

    raw = raw[(raw["dataset"] != "mean") & (raw["trait"] != "8_traits")].copy()
    raw["dataset"] = pd.Categorical(raw["dataset"], DATASET_ORDER, ordered=True)
    raw["trait_label"] = clean_trait_series(raw["trait"])
    single["dataset"] = pd.Categorical(single["dataset"], DATASET_ORDER, ordered=True)
    single["trait_label"] = clean_trait_series(single["trait"])

    fig = plt.figure(figsize=(14.4, 15.8), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.10, right=0.98, top=0.93, bottom=0.06, wspace=0.30, hspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # a summary means
    summary_map = dict(zip(summary["metric"], summary["value"]))
    mean_bar = pd.DataFrame(
        {
            "method": ["No-prior TabPFN", "BayesB", "GBLUP", "RKHS", "Only-triple", "Triple"],
            "value": [
                float(summary_map["mean_no_prior_tabpfn"]),
                float(summary_map["mean_BayesB"]),
                float(summary_map["mean_GBLUP"]),
                float(summary_map["mean_RKHS"]),
                float(summary_map["mean_only_triple_fusion"]),
                float(summary_map["mean_triple_fusion"]),
            ],
        }
    )
    bar_colors = [
        PALETTE["accent_red"],
        PALETTE["signal_blue"],
        PALETTE["signal_teal"],
        PALETTE["accent_orange"],
        PALETTE["accent_gold"],
        PALETTE["accent_green"],
    ]
    ax1.bar(mean_bar["method"], mean_bar["value"], color=bar_colors, edgecolor="black", linewidth=0.8)
    for i, row in mean_bar.iterrows():
        ax1.text(i, row["value"] + 0.006, f'{row["value"]:.3f}', ha="center", va="bottom", fontsize=9)
    ax1.set_title("TabPFN also benefits from fusion, but the baseline gap remains", pad=10)
    ax1.set_ylabel("Mean Pearson correlation")
    ax1.tick_params(axis="x", rotation=18)
    ax1.set_ylim(0.58, 0.66)

    # b single-prior gains
    single["prior_label"] = single["prior_name"].map(
        {"BayesB": "Single-BayesB", "GBLUP": "Single-GBLUP", "RKHS": "Single-RKHS"}
    )
    nice_boxplot_with_points(
        ax2,
        single,
        x="prior_label",
        y="single_vs_own_prior_pct",
        order=["Single-BayesB", "Single-GBLUP", "Single-RKHS"],
        palette={
            "Single-BayesB": PALETTE["signal_blue"],
            "Single-GBLUP": PALETTE["signal_teal"],
            "Single-RKHS": PALETTE["accent_orange"],
        },
    )
    ax2.axhline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0)
    ax2.set_title("Single-prior TabPFN often improves the matched prior", pad=10)
    ax2.set_xlabel("")
    ax2.set_ylabel("Gain over matched prior (%)")
    ax2.tick_params(axis="x", rotation=18)

    # c triple vs best baseline lollipop
    df = raw.sort_values(["dataset", "triple_vs_best_baseline_pct", "trait_label"], ascending=[True, False, True]).copy()
    df["trait_id"] = [f"{d}|||{t}" for d, t in zip(df["dataset"], df["trait_label"])]
    df["y"] = np.arange(len(df))[::-1]
    df["direction"] = np.where(df["triple_vs_best_baseline_pct"] >= 0, ">=0%", "<0%")
    for _, row in df.iterrows():
        color = PALETTE["accent_green"] if row["direction"] == ">=0%" else PALETTE["neutral_light"]
        ax3.plot([row["best_baseline"], row["triple_fusion"]], [row["y"], row["y"]], color=color, linewidth=1.4, alpha=0.9)
        ax3.scatter(row["best_baseline"], row["y"], s=48, facecolor="white", edgecolor=PALETTE["neutral_dark"], zorder=3)
        ax3.scatter(row["triple_fusion"], row["y"], s=58, facecolor=color, edgecolor="black", linewidth=0.7, zorder=4)
    ax3.set_yticks(df["y"])
    ax3.set_yticklabels(df["trait_label"], fontsize=8)
    ax3.set_xlabel("Pearson correlation")
    ax3.set_ylabel("")
    wins = int((df["triple_vs_best_baseline_pct"] >= 0).sum())
    ax3.set_title(f"Triple-TabPFN beats the best baseline in {wins}/8 traits", pad=10)
    legend_elems = [
        Line2D([0], [0], color=PALETTE["neutral_light"], lw=2, marker="o", markersize=7, markerfacecolor=PALETTE["neutral_light"], markeredgecolor="black", label="<0%"),
        Line2D([0], [0], color=PALETTE["accent_green"], lw=2, marker="o", markersize=7, markerfacecolor=PALETTE["accent_green"], markeredgecolor="black", label=">=0%"),
    ]
    ax3.legend(handles=legend_elems, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    trans = blended_transform_factory(ax3.transAxes, ax3.transData)
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        ax3.text(
            -0.26,
            sub["y"].mean(),
            DATASET_PRETTY[ds],
            transform=trans,
            va="center",
            ha="right",
            fontsize=12,
            fontweight="bold",
        )

    # d only vs triple relative gains
    only_vs_best_pct = ((raw["only_triple_fusion"] - raw["best_baseline"]) / raw["best_baseline"] * 100.0).mean()
    comp_bar = pd.DataFrame(
        {
            "comparison": ["BayesB", "GBLUP", "RKHS", "Best baseline"],
            "Only-triple": [
                float(summary_map["mean_only_vs_BayesB_pct"]),
                float(summary_map["mean_only_vs_GBLUP_pct"]),
                float(summary_map["mean_only_vs_RKHS_pct"]),
                float(only_vs_best_pct),
            ],
            "Triple": [
                float(summary_map["mean_triple_vs_BayesB_pct"]),
                float(summary_map["mean_triple_vs_GBLUP_pct"]),
                float(summary_map["mean_triple_vs_RKHS_pct"]),
                float(summary_map["mean_triple_vs_best_baseline_pct"]),
            ],
        }
    )
    x = np.arange(len(comp_bar))
    width = 0.34
    ax4.bar(
        x - width / 2,
        comp_bar["Only-triple"],
        width=width,
        color=PALETTE["accent_gold"],
        edgecolor="black",
        linewidth=0.8,
        label="Only-triple",
    )
    ax4.bar(
        x + width / 2,
        comp_bar["Triple"],
        width=width,
        color=PALETTE["accent_green"],
        edgecolor="black",
        linewidth=0.8,
        label="Triple",
    )
    ax4.axhline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0)
    for idx, row in comp_bar.iterrows():
        ax4.text(idx - width / 2, row["Only-triple"] + 0.08, f'{row["Only-triple"]:.2f}%', ha="center", va="bottom", fontsize=8.3, color=PALETTE["accent_gold"])
        ax4.text(idx + width / 2, row["Triple"] + 0.08, f'{row["Triple"]:.2f}%', ha="center", va="bottom", fontsize=8.3, color=PALETTE["accent_green"])
    ax4.set_xticks(x)
    ax4.set_xticklabels(comp_bar["comparison"])
    ax4.set_ylabel("Mean relative gain (%)")
    ax4.set_title("TabPFN fusion improves over priors, but not always over the best baseline", pad=10)
    ax4.legend(frameon=False, ncol=2, loc="upper left")

    for ax, letter in zip([ax1, ax2, ax3, ax4], list("abcd")):
        add_panel_letter(ax, letter)

    fig.suptitle("TabPFN supports framework extensibility but remains less stable than TabICL", x=0.02, y=0.982, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.962, "Result 6 under the 10K-SNP 8-trait supplementary validation line", ha="left", fontsize=12)

    prefix = OUTDIR / "figure6_tabpfn_extensibility_py"
    save_outputs(fig, prefix)
    plt.close(fig)
    return prefix


def main() -> None:
    setup_theme()
    outputs = [
        figure1a(),
        figure1b(),
        figure3(),
        figure4(),
        figure5(),
        figure6(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
