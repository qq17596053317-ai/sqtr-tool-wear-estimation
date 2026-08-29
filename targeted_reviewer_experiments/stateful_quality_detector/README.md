# Stateful quality-detector extension

## Question

This experiment tested whether adding detector memory improves response to gradual and intermittent single-channel distortions. It is an outer-grouped stress test, not a validation of physical sensor faults.

## Design

- Independent units: nine NUAA experiments and three PHM 2010 tools (`n = 12`).
- Technical randomisations: three per training unit for detector-state selection and 30 per outer-held-out unit for evaluation.
- Synthetic sequence-level distortions: linear bias drift, gain drift, coloured-noise ramp and intermittent record dropout.
- Candidate state models: exponentially weighted moving average (EWMA) and cumulative sum (CUSUM), with prespecified parameter grids and one- to three-record persistence rules.
- Selection: performed using outer-training units only. Configurations satisfying a training clean-trigger rate of at most 5% were preferred; event detection and delay broke ties.
- Comparator: the original record-wise point detector using the same outer-fitted full and exclusion branches.
- Inference: paired unit-level comparison. The 10,000-resample independent-unit bootstrap confidence interval is descriptive and the exact Wilcoxon signed-rank test is unadjusted because this was one prespecified primary comparison.

## Result

The stateful detector did not improve the stress-test outcome. Mean unit-relative MAE increased from 1.0825 for point detection to 1.1370 for stateful detection. The paired difference, defined as point minus stateful, was −0.0545 (95% independent-unit bootstrap CI: −0.1385 to −0.0034; exact Wilcoxon `p = 0.1099`); three of 12 units favoured the stateful detector.

Mean event-detection rate decreased from 58.15% to 52.04%. The stateful clean-trigger rate was 13.78% on outer-held-out clean sequences despite the training-side 5% preference rule. Two NUAA units showed especially poor transfer of the selected state rule, illustrating that detector memory can propagate isolated score excursions instead of filtering them.

## Interpretation

This is a bounded negative result. A short EWMA or CUSUM history is not sufficient to handle the tested gradual and intermittent distortions. The failure reflects both weak instantaneous evidence and unstable transfer of state parameters across independent machining units. It does not establish that all sequential detectors are ineffective; change-point models trained on independent physical-fault trajectories remain untested.

## Reproducible outputs

- `results/stateful_detector_primary_comparison.csv`
- `results/stateful_detector_dataset_summary.csv`
- `results/stateful_detector_selected_configs.csv`
- `results/stateful_detector_unit_results.csv`
- `results/stateful_detector_scenarios.csv`
- `results/stateful_detector_results.mat`

