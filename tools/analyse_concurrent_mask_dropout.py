from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "targeted_reviewer_experiments" / "concurrent_pair_mask_dropout"
OUT = RESULT_ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = {
    "Median replacement": "MedianRelativeMAE",
    "Single-mask-trained dropout": "SingleMaskTrainRelativeMAE",
    "Multi-mask-trained dropout": "MultiMaskTrainRelativeMAE",
    "Single-channel SQTR": "SingleSQTR_RelativeMAE",
    "Subset routing": "SubsetRoutingRelativeMAE",
    "Oracle pair exclusion": "OraclePairExclusionRelativeMAE",
}


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(sampled, [0.025, 0.975]))


def exact_signed_rank_p(left: np.ndarray, right: np.ndarray) -> float:
    differences = np.asarray(left, float) - np.asarray(right, float)
    differences = differences[differences != 0]
    absolute = np.abs(differences)
    order = np.argsort(absolute, kind="mergesort")
    ranks = np.empty(len(absolute), float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and absolute[order[end]] == absolute[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    observed_positive = ranks[differences > 0].sum()
    total = ranks.sum()
    observed = min(observed_positive, total - observed_positive)
    all_codes = np.arange(2 ** len(ranks), dtype=np.uint32)[:, None]
    positive = ((all_codes >> np.arange(len(ranks))) & 1).astype(bool)
    all_positive = positive @ ranks
    all_statistics = np.minimum(all_positive, total - all_positive)
    return float(np.mean(all_statistics <= observed + 1e-12))


summary_rows: list[dict] = []
pair_rows: list[dict] = []
dataset_rows: list[dict] = []
fault_rows: list[dict] = []

for suite_index, suite in enumerate(("abrupt", "structured")):
    result_dir = RESULT_ROOT / f"{suite}_results"
    units = pd.read_csv(result_dir / "concurrent_pair_unit_means.csv")
    scenarios = pd.read_csv(result_dir / "concurrent_pair_scenarios.csv")

    for method, column in {"Fixed fusion": "FullRelativeMAE", **METHODS}.items():
        summary_rows.append(
            {
                "Suite": suite,
                "Method": method,
                "IndependentUnitN": len(units),
                "MeanRelativeMAE": units[column].mean(),
                "SD_across_units": units[column].std(ddof=1),
                "Units_below_fixed": int((units[column] < units["FullRelativeMAE"]).sum())
                if method != "Fixed fusion"
                else np.nan,
            }
        )

    comparisons = [
        ("Single-mask-trained dropout", "Subset routing"),
        ("Multi-mask-trained dropout", "Subset routing"),
        ("Single-mask-trained dropout", "Median replacement"),
        ("Multi-mask-trained dropout", "Single-mask-trained dropout"),
    ]
    for comparison_index, (left, right) in enumerate(comparisons):
        left_values = units[METHODS[left]].to_numpy(float)
        right_values = units[METHODS[right]].to_numpy(float)
        difference = right_values - left_values  # positive favours left
        ci_low, ci_high = bootstrap_ci(
            difference, seed=2026081800 + 100 * suite_index + comparison_index
        )
        exact_p = exact_signed_rank_p(left_values, right_values)
        pair_rows.append(
            {
                "Suite": suite,
                "Left_method": left,
                "Right_method": right,
                "Difference_definition": "right minus left; positive favours left",
                "IndependentUnitN": len(units),
                "Left_mean_relative_MAE": left_values.mean(),
                "Right_mean_relative_MAE": right_values.mean(),
                "Mean_difference": difference.mean(),
                "Bootstrap95CI_lower": ci_low,
                "Bootstrap95CI_upper": ci_high,
                "Units_favour_left": int((difference > 0).sum()),
                "Units_favour_right": int((difference < 0).sum()),
                "Exact_Wilcoxon_p_unadjusted": exact_p,
            }
        )

    for dataset, group in units.groupby("Dataset", sort=False):
        for method, column in {"Fixed fusion": "FullRelativeMAE", **METHODS}.items():
            dataset_rows.append(
                {
                    "Suite": suite,
                    "Dataset": dataset,
                    "Method": method,
                    "IndependentUnitN": len(group),
                    "MeanRelativeMAE": group[column].mean(),
                }
            )

    scenario_columns = {
        "Median replacement": "MedianReplacementMAE",
        "Single-mask-trained dropout": "SingleMaskTrainMAE",
        "Multi-mask-trained dropout": "MultiMaskTrainMAE",
        "Single-channel SQTR": "SingleSQTR_MAE",
        "Subset routing": "SubsetRoutingMAE",
    }
    for (dataset, fault_type), group in scenarios.groupby(["Dataset", "FaultType"], sort=False):
        for method, column in scenario_columns.items():
            reduction = 100 * (group["FullMAE"] - group[column]) / group["FullMAE"]
            fault_rows.append(
                {
                    "Suite": suite,
                    "Dataset": dataset,
                    "FaultType": fault_type,
                    "Method": method,
                    "ScenarioRepeatRows": len(group),
                    "MeanPercentReductionVsFixed": reduction.mean(),
                }
            )

summary = pd.DataFrame(summary_rows)
pairs = pd.DataFrame(pair_rows)
datasets = pd.DataFrame(dataset_rows)
faults = pd.DataFrame(fault_rows)

summary.to_csv(OUT / "method_summary.csv", index=False)
pairs.to_csv(OUT / "paired_comparisons.csv", index=False)
datasets.to_csv(OUT / "dataset_breakdown.csv", index=False)
faults.to_csv(OUT / "fault_type_breakdown.csv", index=False)

lines = [
    "# Concurrent two-channel mask-aware dropout experiment",
    "",
    "This is an exploratory experiment report and is not manuscript text.",
    "Both suites use the same grouped outer units, detector, distortion draws,",
    "30 technical randomisations and clean-unit normalisation as the existing",
    "concurrent two-channel comparison.",
    "",
    "The single-mask-trained model is the existing deployable baseline, evaluated",
    "with all detector-triggered mask bits at inference. Double- and triple-mask",
    "states were absent from its augmentation. The multi-mask-trained model uses",
    "the same augmentation-rate and ridge-lambda grids, but samples uniformly from",
    "all seven non-empty masks during training; inner grouped validation averages",
    "clean and all seven masked states.",
    "",
]

for suite in ("abrupt", "structured"):
    lines += [f"## {suite.title()} distortions", "", "| Method | Mean relative MAE |", "|---|---:|"]
    block = summary[summary["Suite"] == suite]
    for row in block.itertuples(index=False):
        lines.append(f"| {row.Method} | {row.MeanRelativeMAE:.4f} |")
    lines += ["", "Paired exploratory comparisons (12 independent units):", ""]
    for row in pairs[pairs["Suite"] == suite].itertuples(index=False):
        lines.append(
            f"- {row.Left_method} vs {row.Right_method}: "
            f"right-minus-left = {row.Mean_difference:+.4f}, "
            f"95% unit-bootstrap CI [{row.Bootstrap95CI_lower:+.4f}, "
            f"{row.Bootstrap95CI_upper:+.4f}], "
            f"exact unadjusted Wilcoxon p = {row.Exact_Wilcoxon_p_unadjusted:.5f}."
        )
    lines.append("")

(OUT / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

print(summary.to_string(index=False))
print("\nPaired comparisons\n", pairs.to_string(index=False))
print(f"\nOutputs: {OUT}")
