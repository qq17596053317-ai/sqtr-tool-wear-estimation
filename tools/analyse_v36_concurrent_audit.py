"""Build the v36 concurrent-channel detector and response audit.

This analysis uses the same 30 within-unit randomisations as the concurrent
stress test. Detection rates are reported both as descriptive record-scenario
proportions and as independent-unit macro means; paired response comparisons
use the 12 independent-unit means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "targeted_reviewer_experiments" / "concurrent_pair_detector_audit"
OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

SUITES = {
    "Abrupt": BASE / "abrupt_results",
    "Structured gradual/weak": BASE / "structured_results",
}

RATE_COLUMNS = [
    "ExactSetDetectionRate",
    "BothTargetDetectedRate",
    "OnlyOneTargetDetectedRate",
    "NeitherTargetDetectedRate",
    "ExtraChannelTriggerRate",
]


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(float), weights=weights.to_numpy(float)))


def record_weighted_detector_audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for suite, folder in SUITES.items():
        scenarios = pd.read_csv(folder / "concurrent_pair_scenarios.csv")
        groups = [("Overall", scenarios)]
        groups.extend((name, frame) for name, frame in scenarios.groupby("FaultType", sort=False))
        for mode, frame in groups:
            weights = frame["RunLevelN"]
            row: dict[str, object] = {
                "DistortionFamily": suite,
                "DistortionMode": mode,
                "IndependentUnitN": 12,
                "TechnicalRepeatN": 30,
                "TargetPairN": 3,
                "RecordScenarioObservations": int(weights.sum()),
            }
            for column in RATE_COLUMNS:
                row[column] = weighted_mean(frame[column], weights)
            total = (
                row["BothTargetDetectedRate"]
                + row["OnlyOneTargetDetectedRate"]
                + row["NeitherTargetDetectedRate"]
            )
            if not np.isclose(total, 1.0, atol=1e-12):
                raise RuntimeError(f"Target-flag states do not sum to one: {suite}, {mode}, {total}")
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "concurrent_detector_flag_audit_record_weighted.csv", index=False)
    return result


def unit_macro_detector_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    for suite, folder in SUITES.items():
        states = pd.read_csv(folder / "concurrent_pair_unit_detector_states.csv")
        states["UnitKey"] = states["Dataset"].astype(str) + ":" + states["Experiment"].astype(str)
        groups = [("Overall", states)]
        groups.extend((name, frame) for name, frame in states.groupby("FaultType", sort=False))
        for mode, frame in groups:
            unit_means = frame.groupby("UnitKey", sort=False)[RATE_COLUMNS].mean()
            if len(unit_means) != 12:
                raise RuntimeError(f"Expected 12 independent units for {suite}, {mode}; found {len(unit_means)}")
            row: dict[str, object] = {
                "DistortionFamily": suite,
                "DistortionMode": mode,
                "IndependentUnitN": len(unit_means),
                "TechnicalRepeatN": 30,
                "TargetPairN": 3,
            }
            for column in RATE_COLUMNS:
                row[column] = float(unit_means[column].mean())
            total = row["BothTargetDetectedRate"] + row["OnlyOneTargetDetectedRate"] + row["NeitherTargetDetectedRate"]
            if not np.isclose(total, 1.0, atol=1e-12):
                raise RuntimeError(f"Unit-macro target states do not sum to one: {suite}, {mode}, {total}")
            overall_rows.append(row)

            for dataset, dataset_frame in frame.groupby("Dataset", sort=False):
                dataset_units = dataset_frame.groupby("UnitKey", sort=False)[RATE_COLUMNS].mean()
                ds_row: dict[str, object] = {
                    "DistortionFamily": suite,
                    "Dataset": dataset,
                    "DistortionMode": mode,
                    "IndependentUnitN": len(dataset_units),
                    "TechnicalRepeatN": 30,
                    "TargetPairN": 3,
                }
                for column in RATE_COLUMNS:
                    ds_row[column] = float(dataset_units[column].mean())
                dataset_rows.append(ds_row)
    overall = pd.DataFrame(overall_rows)
    by_dataset = pd.DataFrame(dataset_rows)
    overall.to_csv(OUT / "concurrent_detector_flag_audit_unit_macro.csv", index=False)
    by_dataset.to_csv(OUT / "concurrent_detector_flag_audit_unit_macro_by_dataset.csv", index=False)
    # Backwards-compatible name now points to the inferentially aligned summary.
    overall.to_csv(OUT / "concurrent_detector_flag_audit.csv", index=False)
    return overall, by_dataset


def paired_summary(
    suite: str,
    units: pd.DataFrame,
    left_label: str,
    left_column: str,
    right_label: str,
    right_column: str,
    seed: int,
) -> dict[str, object]:
    left = units[left_column].to_numpy(float)
    right = units[right_column].to_numpy(float)
    difference = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(10_000, len(difference)))
    boot = difference[indices].mean(axis=1)
    ci_lower, ci_upper = np.quantile(boot, [0.025, 0.975])
    nonzero = difference[difference != 0]
    ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy(float)
    observed_positive = float(ranks[nonzero > 0].sum())
    assignments = np.arange(2 ** len(ranks), dtype=np.uint32)[:, None]
    bit_positions = np.arange(len(ranks), dtype=np.uint32)[None, :]
    positive = ((assignments >> bit_positions) & 1).astype(bool)
    possible_positive = (positive * ranks).sum(axis=1)
    lower = np.mean(possible_positive <= observed_positive + 1e-12)
    upper = np.mean(possible_positive >= observed_positive - 1e-12)
    exact_p = float(min(1.0, 2.0 * min(lower, upper)))
    return {
        "DistortionFamily": suite,
        "LeftMethod": left_label,
        "RightMethod": right_label,
        "DifferenceDefinition": "left minus right; negative values favour the left method",
        "IndependentUnitN": len(difference),
        "TechnicalRepeatN": 30,
        "LeftMeanRelativeMAE": left.mean(),
        "RightMeanRelativeMAE": right.mean(),
        "MeanDifference": difference.mean(),
        "Bootstrap95CI_Lower": ci_lower,
        "Bootstrap95CI_Upper": ci_upper,
        "UnitsFavourLeft": int(np.sum(difference < 0)),
        "UnitsFavourRight": int(np.sum(difference > 0)),
        "ExactWilcoxonP_Unadjusted": exact_p,
    }


def paired_response_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    direct_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []
    direct_pairs = [
        ("Single-mask dropout", "SingleMaskTrainRelativeMAE", "Median replacement", "MedianRelativeMAE"),
        ("Multi-mask dropout", "MultiMaskTrainRelativeMAE", "Single-mask dropout", "SingleMaskTrainRelativeMAE"),
    ]
    figure_pairs = [
        ("Single-channel router", "SingleSQTR_RelativeMAE"),
        ("Median replacement", "MedianRelativeMAE"),
        ("Single-mask dropout", "SingleMaskTrainRelativeMAE"),
        ("Multi-mask dropout", "MultiMaskTrainRelativeMAE"),
    ]
    seed = 2026082800
    for suite_index, (suite, folder) in enumerate(SUITES.items()):
        units = pd.read_csv(folder / "concurrent_pair_unit_means.csv")
        if len(units) != 12:
            raise RuntimeError(f"Expected 12 independent units for {suite}, found {len(units)}")
        for pair_index, (left_label, left_column, right_label, right_column) in enumerate(direct_pairs):
            direct_rows.append(
                paired_summary(
                    suite,
                    units,
                    left_label,
                    left_column,
                    right_label,
                    right_column,
                    seed + 100 * suite_index + pair_index,
                )
            )
        if suite == "Abrupt":
            for pair_index, (label, column) in enumerate(figure_pairs):
                figure_rows.append(
                    paired_summary(
                        suite,
                        units,
                        label,
                        column,
                        "Subset routing",
                        "SubsetRoutingRelativeMAE",
                        seed + 500 + pair_index,
                    )
                )
    direct = pd.DataFrame(direct_rows)
    effects = pd.DataFrame(figure_rows)
    direct.to_csv(OUT / "concurrent_direct_response_contrasts.csv", index=False)
    effects.to_csv(OUT / "figure9c_balanced_effects.csv", index=False)
    return direct, effects


def verification() -> None:
    old_base = ROOT / "targeted_reviewer_experiments" / "concurrent_pair_mask_dropout"
    checks = {
        "Abrupt": old_base / "abrupt_results" / "concurrent_pair_unit_means.csv",
        "Structured gradual/weak": old_base / "structured_results" / "concurrent_pair_unit_means.csv",
    }
    columns = [
        "FullRelativeMAE",
        "MedianRelativeMAE",
        "SingleMaskTrainRelativeMAE",
        "MultiMaskTrainRelativeMAE",
        "SingleSQTR_RelativeMAE",
        "SubsetRoutingRelativeMAE",
        "OraclePairExclusionRelativeMAE",
    ]
    lines = ["# v36 concurrent-channel audit", ""]
    for suite, old_path in checks.items():
        new_path = SUITES[suite] / "concurrent_pair_unit_means.csv"
        old = pd.read_csv(old_path)[columns].to_numpy(float)
        new = pd.read_csv(new_path)[columns].to_numpy(float)
        max_difference = float(np.max(np.abs(old - new)))
        if max_difference > 1e-12:
            raise RuntimeError(f"Recomputed predictions changed for {suite}: {max_difference}")
        lines.append(f"- {suite}: maximum absolute difference from archived v35 unit means = {max_difference:.3g}.")
    lines.extend(
        [
            "",
            "The rerun changed only detector-state reporting. Predictive results and random seeds were unchanged.",
            "Detection percentages are available as both record-scenario-weighted and independent-unit-macro summaries. Thirty randomisations remain technical repeats.",
            "Paired response contrasts use 12 independent-unit means and unadjusted exact Wilcoxon tests.",
        ]
    )
    (OUT / "ANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    record_weighted = record_weighted_detector_audit()
    detector, detector_by_dataset = unit_macro_detector_audit()
    direct, effects = paired_response_audit()
    verification()
    print("Record-scenario-weighted detector audit\n", record_weighted.to_string(index=False))
    print("\nIndependent-unit-macro detector audit\n", detector.to_string(index=False))
    print("\nDataset-stratified independent-unit-macro detector audit\n", detector_by_dataset.to_string(index=False))
    print("\nDirect response contrasts\n", direct.to_string(index=False))
    print("\nFigure 9c effects\n", effects.to_string(index=False))


if __name__ == "__main__":
    main()
