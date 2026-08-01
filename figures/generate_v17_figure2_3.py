"""Regenerate manuscript Figures 2 and 3 from verified archived outputs.

Figure contract
---------------
Figure 2 conclusion: input-matched automatic routing has a lower pooled MAE
than fixed all-sensor-plus-time ridge, but case effects are heterogeneous and
the case-cluster interval crosses zero.

Figure 3 conclusion: removing cumulative machining time increases the
case-macro MAE under the prespecified no-time-minus-SQTR direction, and
within-case permutation degrades the pooled error without identifying a
causal time effect.

Archetype: quantitative two-panel grids. Export: 178 mm wide; editable SVG/PDF
plus 600-dpi TIFF/PNG. No observations are excluded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "v17"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
WIDTH_MM = 178
RASTER_DPI = 600
WIDTH = WIDTH_MM * MM
HEIGHT = 76 * MM
BLUE = "#0072B2"
ORANGE = "#D55E00"
CHARCOAL = "#2B2B2B"
MID_GREY = "#888888"
LIGHT_GREY = "#D9D9D9"
PALE_GREY = "#F5F5F5"


def panel_label(ax: plt.Axes, label: str, x: float = -0.14) -> None:
    ax.text(x, 1.035, label, transform=ax.transAxes, weight="bold", fontsize=9,
            ha="left", va="bottom")


def save_bundle(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": RASTER_DPI},
        ".tiff": {"dpi": RASTER_DPI, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(OUT / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def load_figure2_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = ROOT / "targeted_reviewer_experiments" / "nasa_qi_role_ablation"
    pred = pd.read_csv(folder / "qi_role_predictions.csv")
    cases = pd.read_csv(folder / "qi_role_cases.csv")
    hyper = pd.read_csv(folder / "qi_role_hyperparameters.csv")
    assert len(pred) == 146 and pred["CaseID"].nunique() == 16
    assert len(cases) == 16 and len(hyper) == 16
    threshold = hyper.set_index("caseList")["RouteOnlyGate"]
    pred["GateThreshold"] = pred["CaseID"].map(threshold)
    pred["RoutingBranch"] = np.where(
        pred["QualityRatio"] >= pred["GateThreshold"],
        "smcDC-exclusion branch",
        "Full-sensor branch",
    )
    assert (pred["RoutingBranch"] == "smcDC-exclusion branch").sum() == 38
    return pred, cases


def figure2() -> None:
    pred, cases = load_figure2_data()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT),
                             gridspec_kw={"width_ratios": [0.95, 1.05]})
    fig.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.19, wspace=0.38)

    ax = axes[0]
    panel_label(ax, "a")
    ax.set_title("Outer-held-out predictions", loc="left", weight="bold")
    actual = pred["ActualVB"].to_numpy(float)
    estimate = pred["PredRouteOnly"].to_numpy(float)
    lo = min(0.0, float(actual.min()), float(estimate.min()))
    hi = max(float(actual.max()), float(estimate.max())) * 1.03
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color=MID_GREY, zorder=1)
    full = pred["RoutingBranch"].eq("Full-sensor branch")
    excluded = ~full
    ax.scatter(actual[full], estimate[full], s=18, c=BLUE, edgecolor="white",
               linewidth=0.35, alpha=0.78,
               label=f"Full-sensor branch (n={full.sum()})", zorder=3)
    ax.scatter(actual[excluded], estimate[excluded], s=19, facecolor="white",
               edgecolor=ORANGE, linewidth=0.8,
               label=f"smcDC-exclusion branch (n={excluded.sum()})", zorder=4)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Measured flank wear, VB (mm)",
           ylabel="Predicted flank wear, VB (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", handletextpad=0.35, borderaxespad=0.2)
    residual = estimate - actual
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    r2 = 1 - np.sum(residual**2) / np.sum((actual - actual.mean())**2)
    assert np.isclose(mae, 0.108773104678773, atol=1e-12)
    ax.text(0.98, 0.05, f"MAE = {mae:.3f} mm\nRMSE = {rmse:.3f} mm\n$R^2$ = {r2:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": PALE_GREY,
                  "edgecolor": LIGHT_GREY, "lw": 0.6})

    ax = axes[1]
    panel_label(ax, "b", x=-0.10)
    ax.set_title("Input-matched case effects", loc="left", weight="bold")
    cases = cases.sort_values("caseList")
    base = cases["AllTimeMAE"].to_numpy(float)
    sqtr = cases["RouteOnlyMAE"].to_numpy(float)
    for b, s in zip(base, sqtr):
        ax.plot([0, 1], [b, s], color=BLUE if s < b else MID_GREY,
                alpha=0.58, lw=0.8, zorder=1)
    ax.scatter(np.zeros(16), base, s=25, facecolor="white", edgecolor=CHARCOAL,
               linewidth=0.8, zorder=3)
    ax.scatter(np.ones(16), sqtr, s=25, facecolor=BLUE, edgecolor="white",
               linewidth=0.4, zorder=4)
    weights = cases["caseSampleCount"].to_numpy(float)
    pooled_base = np.average(base, weights=weights)
    pooled_sqtr = np.average(sqtr, weights=weights)
    reduction = 100 * (pooled_base - pooled_sqtr) / pooled_base
    assert np.isclose(pooled_base, 0.116210316365672, atol=1e-12)
    assert np.isclose(pooled_sqtr, 0.108773104678773, atol=1e-12)
    ax.set_xlim(-0.32, 1.32)
    ax.set_xticks([0, 1], ["Fixed all-sensor-plus-time\nridge", "SQTR"])
    ax.set_ylabel("Case MAE (mm)")
    ax.tick_params(axis="x", length=0)
    ax.text(0.04, 0.97,
            f"Pooled MAE: {pooled_base:.3f} → {pooled_sqtr:.3f} mm\n"
            f"{reduction:.1f}% lower\n"
            "Case-macro Δ = −0.0052 mm\n"
            "95% CI [−0.0183, 0.0040]\nHolm-adjusted p = 0.645",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": PALE_GREY,
                  "edgecolor": LIGHT_GREY, "lw": 0.6})

    pred[["OriginalIndex", "CaseID", "ActualVB", "PredRouteOnly", "QualityRatio",
          "GateThreshold", "RoutingBranch"]].to_csv(
              OUT / "Figure2_source_data_panel_a.csv", index=False)
    source_b = cases[["caseList", "caseSampleCount", "AllTimeMAE", "RouteOnlyMAE"]].copy()
    source_b["SQTRMinusComparator"] = source_b["RouteOnlyMAE"] - source_b["AllTimeMAE"]
    source_b.to_csv(OUT / "Figure2_source_data_panel_b.csv", index=False)
    save_bundle(fig, "Figure2_input_matched_performance")


def figure3() -> None:
    time_cases = pd.read_csv(ROOT / "mill" / "results" / "time_model_cases.csv")
    q_cases = pd.read_csv(ROOT / "targeted_reviewer_experiments" /
                          "nasa_qi_role_ablation" / "qi_role_cases.csv")
    merged = time_cases.merge(q_cases, left_on="CaseID", right_on="caseList",
                              validate="one_to_one")
    definitions = [
        ("$q_i$ also in regression", "RoutePlusQMAE", "RouteOnlyMAE",
         -0.0005010754, 0.0017399719),
        ("No cumulative time", "RidgeGatedMAE", "TimeGatedMAE",
         0.0011803308, 0.0092566446),
        ("Fixed smcDC-exclusion branch", "QualityAwareTimeMAE", "TimeGatedMAE",
         -0.0009501472, 0.0214408561),
        ("Fixed full-sensor branch", "AllSensorsTimeMAE", "TimeGatedMAE",
         -0.0038324266, 0.0185912431),
        ("Process + time only", "TimeOnlyMAE", "TimeGatedMAE",
         -0.0015376170, 0.0639667840),
    ]
    rows = []
    for label, col, ref, lower, upper in definitions:
        delta = merged[col].to_numpy(float) - merged[ref].to_numpy(float)
        rows.append((label, delta.mean(), lower, upper))
    ablation = pd.DataFrame(rows, columns=["Ablation", "MeanDifference",
                                                   "CILower", "CIUpper"])

    permutation = pd.read_csv(ROOT / "figures" / "source_data" /
                              "Figure3_time_permutation.csv")
    assert len(permutation) == 100
    observed = 0.108773104678773
    empirical_p = (1 + np.sum(permutation["MAE"].to_numpy(float) <= observed)) / 101
    assert np.isclose(empirical_p, 0.009900990099009901)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT),
                             gridspec_kw={"width_ratios": [1.03, 0.97]})
    fig.subplots_adjust(left=0.21, right=0.985, top=0.91, bottom=0.20, wspace=0.42)

    ax = axes[0]
    panel_label(ax, "a", x=-0.27)
    ax.set_title("Structural ablations", loc="left", weight="bold")
    y = np.arange(len(ablation))[::-1]
    for pos, row in zip(y, ablation.itertuples(index=False)):
        ax.errorbar(row.MeanDifference, pos,
                    xerr=np.array([[row.MeanDifference-row.CILower],
                                   [row.CIUpper-row.MeanDifference]]),
                    fmt="o", ms=4.5, color=ORANGE, markeredgecolor="white",
                    markeredgewidth=0.45, ecolor=ORANGE, elinewidth=1.0,
                    capsize=2.5, zorder=3)
    ax.axvline(0, color=MID_GREY, lw=0.8, ls="--")
    ax.set_yticks(y, ablation["Ablation"])
    ax.set_xlabel("Case-macro MAE difference: ablation − SQTR (mm)")
    ax.text(0.98, 0.94, "Positive = higher error", transform=ax.transAxes,
            ha="right", va="top", color=CHARCOAL, fontsize=6.4)
    ax.text(ablation.iloc[1]["MeanDifference"], y[1] + 0.28,
            "11/16 cases; exact p = 0.0335", ha="left", va="bottom",
            fontsize=6.1, color=ORANGE)

    ax = axes[1]
    panel_label(ax, "b", x=-0.12)
    ax.set_title("Within-case time permutation", loc="left", weight="bold")
    values = permutation["MAE"].to_numpy(float)
    ax.hist(values, bins=12, color=LIGHT_GREY, edgecolor="white", linewidth=0.5)
    rug_height = ax.get_ylim()[1] * 0.035
    ax.vlines(values, 0, rug_height, color=CHARCOAL, lw=0.4, alpha=0.65)
    ax.axvline(observed, color=BLUE, lw=1.5)
    ax.text(observed + 0.00005, ax.get_ylim()[1] * 0.95,
            f"Observed order\n{observed:.3f} mm\nempirical p = {empirical_p:.4f}",
            color=BLUE, ha="left", va="top", fontsize=6.5, weight="bold")
    ax.set_xlabel("Pooled MAE after within-case permutation (mm)")
    ax.set_ylabel("Permutation count")

    ablation.to_csv(OUT / "Figure3_source_data_panel_a.csv", index=False)
    permutation.to_csv(OUT / "Figure3_source_data_panel_b.csv", index=False)
    save_bundle(fig, "Figure3_time_and_structure")


if __name__ == "__main__":
    figure2()
    figure3()
    print(f"Saved v17 Figures 2 and 3 to {OUT}")
