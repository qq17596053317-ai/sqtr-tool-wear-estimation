from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "additional_validation_results" / "reviewer_improvements"
OUT_DIR = ROOT / "figures" / "deployable_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIT_FILE = RESULT_DIR / "random_fault_experiment_means.csv"
STATS_FILE = RESULT_DIR / "deployable_baseline_statistics.csv"
FULL_STATS_FILE = RESULT_DIR / "random_fault_statistics.csv"

METHODS = [
    ("Fixed all-channel", "FullRelativeMAE"),
    ("Median replacement", "MedianReplacementRelativeMAE"),
    ("Mask-aware dropout", "MaskDropoutRelativeMAE"),
    ("SQTR", "AutoRelativeMAE"),
]


def save_outputs(fig: plt.Figure, stem: Path) -> None:
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    units = pd.read_csv(UNIT_FILE)
    stats = pd.read_csv(STATS_FILE)
    full_stats = pd.read_csv(FULL_STATS_FILE)

    expected = {column for _, column in METHODS} | {"Dataset", "Experiment"}
    missing = expected.difference(units.columns)
    if missing:
        raise ValueError(f"Missing unit-level columns: {sorted(missing)}")
    if len(units) != 12:
        raise ValueError(f"Expected 12 independent units, found {len(units)}")

    long_rows = []
    for label, column in METHODS:
        for row in units.itertuples(index=False):
            long_rows.append(
                {
                    "Dataset": row.Dataset,
                    "Unit": row.Experiment,
                    "Method": label,
                    "RelativeMAE": getattr(row, column),
                }
            )
    long = pd.DataFrame(long_rows)
    long.to_csv(OUT_DIR / "Figure5_deployable_baselines_source_data.csv", index=False)

    dataset_summary = (
        long.groupby(["Dataset", "Method"], sort=False)["RelativeMAE"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    combined_summary = (
        long.groupby("Method", sort=False)["RelativeMAE"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    combined_summary.insert(0, "Dataset", "Combined")
    pd.concat([dataset_summary, combined_summary], ignore_index=True).to_csv(
        OUT_DIR / "deployable_baseline_method_summary.csv", index=False
    )

    clean_columns = {
        "Fixed all-channel": "CleanFullMAE",
        "Median replacement": "CleanMedianReplacementMAE",
        "Mask-aware dropout": "CleanMaskDropoutMAE",
        "SQTR": "CleanAutoMAE",
    }
    clean_rows = []
    for label, column in clean_columns.items():
        for dataset in ["NUAA", "PHM2010", "Combined"]:
            subset = units if dataset == "Combined" else units[units["Dataset"] == dataset]
            values = subset[column].to_numpy(float)
            clean_rows.append(
                {
                    "Dataset": dataset,
                    "Method": label,
                    "IndependentUnitN": len(values),
                    "CaseMacroCleanMAE": values.mean(),
                    "CaseMacroCleanMAESD": values.std(ddof=1),
                }
            )
    pd.DataFrame(clean_rows).to_csv(
        OUT_DIR / "deployable_baseline_clean_input_summary.csv", index=False
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 78 / 25.4),
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.36},
    )
    ax_a, ax_b = axes
    x = np.arange(len(METHODS))
    dataset_style = {
        "NUAA": ("#4C9ED9", "o"),
        "PHM2010": ("#E58B3C", "^"),
    }
    for _, unit in units.iterrows():
        color, marker = dataset_style[unit["Dataset"]]
        y = [unit[column] for _, column in METHODS]
        ax_a.plot(x, y, color=color, alpha=0.40, linewidth=0.75, zorder=1)
        ax_a.scatter(x, y, s=15, facecolor="white", edgecolor=color, marker=marker,
                     linewidth=0.8, zorder=2)
    means = [units[column].mean() for _, column in METHODS]
    ax_a.plot(x, means, color="#222222", linewidth=1.6, marker="D", markersize=4.0,
              markerfacecolor="#222222", zorder=3)
    ax_a.set_xticks(x, [label.replace(" ", "\n", 1) for label, _ in METHODS])
    ax_a.set_ylabel("Unit-relative MAE\n(corrupted / clean)")
    ax_a.grid(axis="y", color="#E3E3E3", linewidth=0.55)
    ax_a.set_title("Deployable methods across independent units", loc="left", weight="bold")
    ax_a.scatter([], [], s=15, facecolor="white", edgecolor=dataset_style["NUAA"][0],
                 marker="o", label="NUAA experiment (n = 9)")
    ax_a.scatter([], [], s=17, facecolor="white", edgecolor=dataset_style["PHM2010"][0],
                 marker="^", label="PHM tool (n = 3)")
    ax_a.plot([], [], color="#222222", marker="D", markersize=4, label="Unit mean")
    ax_a.legend(loc="upper right", fontsize=6.2)

    combined_full = full_stats.loc[full_stats["Dataset"] == "Combined"].iloc[0]
    labels = ["Fixed all-channel", "Median replacement", "Mask-aware dropout"]
    differences = [
        -float(combined_full["MeanPairedDifference"]),
        float(stats.iloc[0]["MeanDifferenceComparatorMinusSQTR"]),
        float(stats.iloc[1]["MeanDifferenceComparatorMinusSQTR"]),
    ]
    ci_low = [
        -float(combined_full["BootstrapCIUpper"]),
        float(stats.iloc[0]["BootstrapCILower"]),
        float(stats.iloc[1]["BootstrapCILower"]),
    ]
    ci_high = [
        -float(combined_full["BootstrapCILower"]),
        float(stats.iloc[0]["BootstrapCIUpper"]),
        float(stats.iloc[1]["BootstrapCIUpper"]),
    ]
    ypos = np.arange(len(labels))[::-1]
    colors = ["#777777", "#C96A25", "#8A66B3"]
    for y, value, lo, hi, color in zip(ypos, differences, ci_low, ci_high, colors):
        ax_b.errorbar(
            value,
            y,
            xerr=np.array([[value - lo], [hi - value]]),
            fmt="o",
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=4.5,
            capsize=2.3,
            linewidth=1.0,
        )
    ax_b.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax_b.set_yticks(ypos, labels)
    ax_b.set_xlabel("Comparator − SQTR relative MAE")
    ax_b.grid(axis="x", color="#E3E3E3", linewidth=0.55)
    ax_b.set_title("Independent-unit mean differences", loc="left", weight="bold")
    ax_b.text(
        0.98,
        0.03,
        "Positive values favour SQTR",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=6.2,
    )

    ax_a.text(-0.12, 1.07, "a", transform=ax_a.transAxes, weight="bold", fontsize=8)
    ax_b.text(-0.20, 1.07, "b", transform=ax_b.transAxes, weight="bold", fontsize=8)
    save_outputs(fig, OUT_DIR / "Figure5_deployable_baseline_comparison")
    plt.close(fig)


if __name__ == "__main__":
    main()
