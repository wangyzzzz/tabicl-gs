from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "outputs" / "results_support_workbook" / "results_support_tables_20260512.xlsx"
FIG2_DIR = ROOT / "outputs" / "figure2_inputs"
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

DATASET_MARKERS = {
    "cotton1245": "o",
    "rice529": "s",
    "soybean951": "^",
    "wheat406": "D",
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
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 17,
        }
    )


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom", ha="left")


def save_outputs(fig: plt.Figure, prefix: Path) -> None:
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")


def draw_case_cleveland(ax: plt.Axes) -> None:
    cases = pd.read_excel(XLSX, sheet_name="R1_cases").copy()
    cases["trait_label"] = cases["trait_slug"].map(short_trait_label)
    case_order = list(cases["trait_label"])

    method_cols = [
        ("no_prior_tabicl", "No-prior TabICL", PALETTE["accent_red"]),
        ("BayesB", "BayesB", PALETTE["signal_blue"]),
        ("GBLUP", "GBLUP", PALETTE["signal_teal"]),
        ("RKHS", "RKHS", PALETTE["accent_orange"]),
    ]

    y_positions = np.arange(len(case_order))[::-1]
    for y, (_, row) in zip(y_positions, cases.iterrows()):
        values = [row[col] for col, _, _ in method_cols]
        ax.hlines(y, min(values), max(values), color=PALETTE["neutral_light"], linewidth=2.0, zorder=1)

    offsets = np.linspace(-0.18, 0.18, len(method_cols))
    for offset, (col, label, color) in zip(offsets, method_cols):
        ax.scatter(
            cases[col],
            y_positions + offset,
            s=70,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
            label=label,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(case_order)
    ax.set_xlabel("Pearson correlation")
    ax.set_title("Alt for Fig. 1d: horizontal Cleveland dot plot", pad=10)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="lower right")


def draw_gain_lollipop(ax: plt.Axes) -> None:
    df = pd.read_csv(FIG2_DIR / "figure2_triple_vs_best.csv").copy()
    df["trait_label"] = df["trait_slug"].map(short_trait_label)
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df = df.sort_values(["triple_minus_best_pct", "dataset", "trait_label"], ascending=[False, True, True]).reset_index(drop=True)
    df["y"] = np.arange(len(df))[::-1]
    colors = np.where(df["triple_minus_best_pct"] >= 0, PALETTE["accent_green"], PALETTE["accent_red"])

    ax.axvline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0, zorder=0)
    for _, row in df.iterrows():
        color = PALETTE["accent_green"] if row["triple_minus_best_pct"] >= 0 else PALETTE["accent_red"]
        ax.hlines(row["y"], 0, row["triple_minus_best_pct"], color=color, linewidth=1.6, alpha=0.9, zorder=2)
        ax.scatter(row["triple_minus_best_pct"], row["y"], s=36, color=color, edgecolor="black", linewidth=0.6, zorder=3)

    ax.set_yticks(df["y"])
    ax.set_yticklabels(df["trait_label"], fontsize=7.1)
    ax.set_xlabel("Triple vs best baseline (%)")
    wins = int((df["triple_minus_best_pct"] >= 0).sum())
    ax.set_title(f"Alt for Fig. 2d: sorted gain forest plot ({wins}/{len(df)} positive)", pad=10)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.8)

    x_left = float(df["triple_minus_best_pct"].min()) - 0.6
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        ax.text(
            x_left,
            sub["y"].mean(),
            DATASET_PRETTY[ds],
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )


def ternary_xy(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = g + 0.5 * r
    y = r * (np.sqrt(3) / 2.0)
    return x, y


def draw_ternary_preview(ax: plt.Axes) -> None:
    weights = pd.read_excel(XLSX, sheet_name="R3_weights36").copy()
    weights["dataset"] = pd.Categorical(weights["dataset"], DATASET_ORDER, ordered=True)
    b = weights["bayesb_share"].to_numpy()
    g = weights["gblup_share"].to_numpy()
    r = weights["rkhs_share"].to_numpy()
    x, y = ternary_xy(b, g, r)

    triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2.0],
            [0.0, 0.0],
        ]
    )
    ax.plot(triangle[:, 0], triangle[:, 1], color=PALETTE["neutral_dark"], linewidth=1.2)
    for frac in [0.25, 0.50, 0.75]:
        ax.plot([0.5 * frac, 1.0 - 0.5 * frac], [frac * np.sqrt(3) / 2.0, frac * np.sqrt(3) / 2.0], color="#EFEFEF", linewidth=0.8, zorder=0)

    cmap = LinearSegmentedColormap.from_list("ternary_gain", [PALETTE["accent_red"], "#FFF9F0", PALETTE["signal_blue"]])
    vmax = max(2.0, np.nanmax(np.abs(weights["triple_vs_best_baseline_pct"].to_numpy())))
    norm = TwoSlopeNorm(vmin=-max(0.6, vmax * 0.25), vcenter=0.0, vmax=vmax)

    for ds in DATASET_ORDER:
        sub = weights[weights["dataset"] == ds]
        xs, ys = ternary_xy(sub["bayesb_share"].to_numpy(), sub["gblup_share"].to_numpy(), sub["rkhs_share"].to_numpy())
        sc = ax.scatter(
            xs,
            ys,
            c=sub["triple_vs_best_baseline_pct"],
            cmap=cmap,
            norm=norm,
            s=72,
            marker=DATASET_MARKERS[ds],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.95,
            label=DATASET_PRETTY[ds],
            zorder=3,
        )

    cx, cy = ternary_xy(np.array([1 / 3]), np.array([1 / 3]), np.array([1 / 3]))
    ax.scatter(cx, cy, s=95, marker="P", color="white", edgecolor=PALETTE["neutral_dark"], linewidth=0.9, zorder=4)
    ax.text(cx[0], cy[0] - 0.05, "1/3 each", ha="center", va="top", fontsize=8.5)

    ax.text(-0.04, -0.04, "BayesB", ha="right", va="top", fontsize=10.5, fontweight="bold", color=PALETTE["signal_blue"])
    ax.text(1.04, -0.04, "GBLUP", ha="left", va="top", fontsize=10.5, fontweight="bold", color=PALETTE["signal_teal"])
    ax.text(0.50, np.sqrt(3) / 2.0 + 0.04, "RKHS", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=PALETTE["accent_orange"])

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, np.sqrt(3) / 2.0 + 0.10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Alt for Fig. 3d: ternary map of triple prior shares", pad=10)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_title("Triple\nvs best\n(%)", fontsize=8.5, pad=8)


def draw_tabpfn_dumbbell(ax: plt.Axes) -> None:
    raw = pd.read_excel(XLSX, sheet_name="R6_tabpfn_raw").copy()
    raw = raw[(raw["dataset"] != "mean") & (raw["trait"] != "8_traits")].copy()

    summary = pd.DataFrame(
        {
            "comparison": ["BayesB", "GBLUP", "RKHS", "Best baseline"],
            "only": [
                raw["only_vs_BayesB_pct"].mean(),
                raw["only_vs_GBLUP_pct"].mean(),
                raw["only_vs_RKHS_pct"].mean(),
                ((raw["only_triple_fusion"] - raw["best_baseline"]) / raw["best_baseline"] * 100.0).mean(),
            ],
            "triple": [
                raw["triple_vs_BayesB_pct"].mean(),
                raw["triple_vs_GBLUP_pct"].mean(),
                raw["triple_vs_RKHS_pct"].mean(),
                raw["triple_vs_best_baseline_pct"].mean(),
            ],
        }
    )
    summary["y"] = np.arange(len(summary))[::-1]

    ax.axvline(0, color=PALETTE["neutral_mid"], linestyle=(0, (4, 4)), linewidth=1.0, zorder=0)
    for _, row in summary.iterrows():
        ax.hlines(row["y"], row["only"], row["triple"], color=PALETTE["neutral_light"], linewidth=2.0, zorder=1)
        ax.scatter(row["only"], row["y"], s=68, color=PALETTE["accent_gold"], edgecolor="black", linewidth=0.7, zorder=3, label="Only-triple" if row["comparison"] == "BayesB" else None)
        ax.scatter(row["triple"], row["y"], s=68, color=PALETTE["accent_green"], edgecolor="black", linewidth=0.7, zorder=3, label="Triple" if row["comparison"] == "BayesB" else None)
        ax.text(row["only"] - 0.06, row["y"] + 0.12, f'{row["only"]:.2f}%', ha="right", va="bottom", fontsize=8.3, color=PALETTE["accent_gold"])
        ax.text(row["triple"] + 0.06, row["y"] + 0.12, f'{row["triple"]:.2f}%', ha="left", va="bottom", fontsize=8.3, color=PALETTE["accent_green"])

    ax.set_yticks(summary["y"])
    ax.set_yticklabels(summary["comparison"])
    ax.set_xlabel("Mean relative gain (%)")
    ax.set_title("Alt for Fig. 6d: paired dumbbell for Only-triple vs Triple", pad=10)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")


def main() -> None:
    setup_theme()

    fig = plt.figure(figsize=(14.8, 12.8), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, left=0.08, right=0.98, top=0.92, bottom=0.07, wspace=0.28, hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    draw_case_cleveland(ax1)
    draw_gain_lollipop(ax2)
    draw_ternary_preview(ax3)
    draw_tabpfn_dumbbell(ax4)

    for ax, letter in zip([ax1, ax2, ax3, ax4], list("abcd")):
        add_panel_letter(ax, letter)

    fig.suptitle("Alternative chart-type previews for the current Results figures", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.958, "These are preview-only replacements for visual comparison and do not overwrite the current main figures.", ha="left", fontsize=11)

    prefix = OUTDIR / "figure_alt_type_previews"
    save_outputs(fig, prefix)
    plt.close(fig)
    print(prefix)


if __name__ == "__main__":
    main()
