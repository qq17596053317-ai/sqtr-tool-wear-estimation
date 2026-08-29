"""Create the balanced v36 Figure 9.

Figure contract
---------------
Core conclusion: detector sensitivity and response action are separate; subset
routing fixes the original one-channel structural limit, but the abrupt-pair
comparisons do not identify it as the lowest-error deployable response.
Archetype: quantitative grid.
Target/output: double-column journal figure, 183 mm wide; SVG/PDF and 600-dpi
TIFF/PNG exports with editable vector text where supported.
Panel map: (a) detector threshold trade-off; (b) paired original-versus-subset
router unit means; (c) balanced comparator-minus-subset effects for four abrupt
response strategies, without exploratory p values in the graphic.
Evidence hierarchy: panel c is the balanced summary; panel b shows the router's
structural comparison; panel a shows detector operating-point sensitivity.
Statistics: n = 12 independent units; 30 randomisations are technical repeats;
panel c uses unadjusted 95% independent-unit bootstrap intervals.
Reviewer risk: avoid visually privileging the only nominal p < 0.05 contrast.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "targeted_reviewer_experiments" / "concurrent_pair_detector_audit" / "figure9"
OUT.mkdir(parents=True, exist_ok=True)

GRID = (
    ROOT
    / "targeted_reviewer_experiments"
    / "detector_threshold_sensitivity"
    / "compiled_results"
    / "detector_threshold_grid_summary.csv"
)
UNITS = (
    ROOT
    / "targeted_reviewer_experiments"
    / "concurrent_pair_detector_audit"
    / "abrupt_results"
    / "concurrent_pair_unit_means.csv"
)
EFFECTS = (
    ROOT
    / "targeted_reviewer_experiments"
    / "concurrent_pair_detector_audit"
    / "analysis"
    / "figure9c_balanced_effects.csv"
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#777777"
LIGHT_GREY = "#D9D9D9"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def build() -> plt.Figure:
    grid = pd.read_csv(GRID)
    grid = grid[grid["Dataset"] == "Combined"].copy()
    units = pd.read_csv(UNITS)
    effects = pd.read_csv(EFFECTS)

    fig = plt.figure(figsize=(183 / 25.4, 65 / 25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.14], wspace=0.72)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # a, detector threshold trade-off
    quantile_colors = {97.5: ORANGE, 99.0: BLUE, 99.5: GREEN}
    floor_markers = {3: "o", 4: "s", 5: "^"}
    for quantile, color in quantile_colors.items():
        subset = grid[np.isclose(grid["DetectorQuantile"], quantile)]
        for _, row in subset.iterrows():
            ax_a.scatter(
                100 * row["MeanCleanTriggerRate"],
                row["SQTRRelativeMAE"],
                s=34,
                marker=floor_markers[int(row["DetectorFloor"])],
                facecolors=color,
                edgecolors="white",
                linewidths=0.55,
                zorder=3,
            )
    chosen = grid[(grid["DetectorFloor"] == 4) & np.isclose(grid["DetectorQuantile"], 99.0)].iloc[0]
    ax_a.annotate(
        "Prespecified",
        xy=(100 * chosen["MeanCleanTriggerRate"], chosen["SQTRRelativeMAE"]),
        xytext=(7.0, 1.0778),
        arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.8},
        fontsize=6.4,
    )
    quantile_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, markeredgecolor="white", markersize=5, label=f"q{q:g}")
        for q, color in quantile_colors.items()
    ]
    floor_handles = [
        plt.Line2D([], [], marker=marker, linestyle="", color="black", markerfacecolor="white", markersize=5, label=f"floor {floor}")
        for floor, marker in floor_markers.items()
    ]
    ax_a.legend(handles=quantile_handles + floor_handles, ncol=2, loc="lower right", columnspacing=0.7, handletextpad=0.3)
    ax_a.set_title("Detector-threshold trade-off", loc="left", fontweight="bold")
    ax_a.set_xlabel("Clean-input trigger rate (%)")
    ax_a.set_ylabel("Relative MAE under distortion")
    ax_a.set_xlim(0, 18)
    ax_a.set_ylim(1.0660, 1.0836)
    ax_a.grid(axis="y", color=LIGHT_GREY, lw=0.5)
    panel_label(ax_a, "a")

    # b, paired unit effects
    for _, row in units.iterrows():
        ax_b.plot(
            [0, 1],
            [row["SingleSQTR_RelativeMAE"], row["SubsetRoutingRelativeMAE"]],
            color="#B8B8B8",
            lw=0.75,
            zorder=1,
        )
    for dataset, marker, color in [("NUAA", "o", BLUE), ("PHM2010", "^", ORANGE)]:
        subset = units[units["Dataset"] == dataset]
        ax_b.scatter(np.zeros(len(subset)), subset["SingleSQTR_RelativeMAE"], s=29, marker=marker, color=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax_b.scatter(np.ones(len(subset)), subset["SubsetRoutingRelativeMAE"], s=29, marker=marker, color=color, edgecolor="white", linewidth=0.45, zorder=3, label="PHM tools" if dataset == "PHM2010" else "NUAA experiments")
    ax_b.set_title("Concurrent abrupt distortions", loc="left", fontweight="bold")
    ax_b.set_ylabel("Unit-relative MAE")
    ax_b.set_xticks([0, 1], ["Single-channel\nrouter", "Subset\nrouter"])
    ax_b.set_xlim(-0.05, 1.05)
    ax_b.grid(axis="y", color=LIGHT_GREY, lw=0.5)
    ax_b.legend(loc="upper right", handletextpad=0.3)
    panel_label(ax_b, "b")

    # c, balanced response contrasts
    order = ["Single-channel router", "Median replacement", "Single-mask dropout", "Multi-mask dropout"]
    effect_colors = [BLUE, ORANGE, GREEN, PURPLE]
    effects = effects.set_index("LeftMethod").loc[order].reset_index()
    y = np.arange(len(effects))[::-1]
    mean = effects["MeanDifference"].to_numpy(float)
    lower = effects["Bootstrap95CI_Lower"].to_numpy(float)
    upper = effects["Bootstrap95CI_Upper"].to_numpy(float)
    for yi, mu, lo, hi, color in zip(y, mean, lower, upper, effect_colors):
        ax_c.errorbar(mu, yi, xerr=[[mu - lo], [hi - mu]], fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=2.4, markersize=4.4)
    ax_c.axvline(0, color=GREY, lw=0.8, ls="--")
    ax_c.set_yticks(y, order)
    ax_c.set_title("Balanced abrupt-response contrasts", loc="left", fontweight="bold")
    ax_c.set_xlabel("Comparator MAE − subset-routing MAE")
    ax_c.set_xlim(-0.16, 0.57)
    ax_c.grid(axis="x", color=LIGHT_GREY, lw=0.5)
    ax_c.text(0.02, -0.28, "favours comparator", transform=ax_c.transAxes, ha="left", va="top", fontsize=6.2, color=GREY)
    ax_c.text(0.98, -0.28, "favours subset", transform=ax_c.transAxes, ha="right", va="top", fontsize=6.2, color=GREY)
    panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.26)
    return fig


def main() -> None:
    fig = build()
    stem = OUT / "figure9_v36_balanced"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
