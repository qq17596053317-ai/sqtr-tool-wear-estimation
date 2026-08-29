"""Independent-unit comparisons for the concurrent-channel stress tests.

The technical randomisations have already been averaged within each experiment/tool
in the input CSVs.  This script therefore treats the 12 machining units as the
inferential observations, matching the manuscript's statistical hierarchy.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "targeted_reviewer_experiments" / "concurrent_pair_subset_routing"
OUTPUT = EXPERIMENT / "results" / "concurrent_pair_median_vs_subset.csv"
SEED = 20260815
N_BOOT = 10_000


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def exact_two_sided_wilcoxon(difference: np.ndarray) -> float:
    nonzero = difference[difference != 0]
    ranks = average_ranks(np.abs(nonzero))
    observed = ranks[nonzero > 0].sum()
    centre = ranks.sum() / 2
    observed_distance = abs(observed - centre)
    sums = np.zeros(1 << len(ranks), dtype=float)
    for mask in range(1 << len(ranks)):
        sums[mask] = sum(rank for index, rank in enumerate(ranks) if mask & (1 << index))
    return float(np.mean(np.abs(sums - centre) >= observed_distance - 1e-12))


def analyse(label: str, relative_csv: Path) -> dict[str, object]:
    frame = pd.read_csv(relative_csv)
    difference = frame["MedianRelativeMAE"].to_numpy() - frame["SubsetRoutingRelativeMAE"].to_numpy()
    rng = np.random.default_rng(SEED + (0 if label == "Abrupt" else 1))
    indices = rng.integers(0, len(difference), size=(N_BOOT, len(difference)))
    bootstrap = difference[indices].mean(axis=1)
    exact_p = exact_two_sided_wilcoxon(difference)
    return {
        "FaultFamily": label,
        "IndependentUnitN": len(difference),
        "MedianRelativeMAE": frame["MedianRelativeMAE"].mean(),
        "SubsetRoutingRelativeMAE": frame["SubsetRoutingRelativeMAE"].mean(),
        "MedianMinusSubset": difference.mean(),
        "BootstrapCI_Lower": np.quantile(bootstrap, 0.025),
        "BootstrapCI_Upper": np.quantile(bootstrap, 0.975),
        "ExactP": exact_p,
        "UnitsFavouringSubset": int((difference > 0).sum()),
        "UnitsFavouringMedian": int((difference < 0).sum()),
        "Ties": int((difference == 0).sum()),
        "BootstrapSeed": SEED + (0 if label == "Abrupt" else 1),
        "BootstrapResamples": N_BOOT,
    }


rows = [
    analyse("Abrupt", EXPERIMENT / "results" / "concurrent_pair_unit_means.csv"),
    analyse("Gradual/weak", EXPERIMENT / "structured_results" / "concurrent_pair_unit_means.csv"),
]
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print(OUTPUT)
print(pd.DataFrame(rows).to_string(index=False))
