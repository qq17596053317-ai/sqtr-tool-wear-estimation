# Repository code map for manuscript v17

| Analysis component | Authoritative implementation | Principal archived output |
|---|---|---|
| NASA feature preparation and main inputs | `step15timeawareridge.m` | `mill/results/quality_aware_with_time.mat` |
| Nested grouped LOCO and hard routing | `groupedGatedRidgePrediction.m` | `mill/results/time_model_predictions.csv`; `mill/results/time_model_cases.csv` |
| Full-sensor and smcDC-exclusion ridge branches | `step15timeawareridge.m`; `groupedGatedRidgePrediction.m` | `mill/results/time_model_nested_hyperparameters.csv` |
| q_i role ablation and input-matched Figure 2b comparison | `targeted_reviewer_experiments/nasa_qi_role_ablation/step33_qi_role_ablation.m` | `targeted_reviewer_experiments/nasa_qi_role_ablation/qi_role_cases.csv`; `qi_role_route_comparisons.csv` |
| Cumulative-time ablation | `step15timeawareridge.m`; `audit_v17_results.m` | `audit_outputs/time_ablation_case_results.csv`; `time_ablation_case_macro_audit.csv` |
| Within-case time permutation | `step23_time_shortcut_validation.m` | `additional_validation_results/time_permutation/` and figure source data |
| Record-wise versus grouped protocol comparison | `step19_split_protocol_validation.m` | `protocol_validation_results/` |
| Randomised artificial-degradation stress test | `step28_randomized_fault_repeats.m` | `additional_validation_results/reviewer_improvements/random_fault_*.csv` |
| Fixed-severity-grid and Oracle sensitivity benchmark | `external_multisensor_multifault_robustness.m` | `multifault_robustness_results/multifault_*.csv` |
| Training-fold alert calibration | `targeted_reviewer_experiments/alarm_calibration/step34_high_wear_alert_calibration.m` | `targeted_reviewer_experiments/alarm_calibration/alert_*.csv` |
| High-wear residual audit | `step32_high_wear_underestimation_audit.m` | `targeted_reviewer_experiments/high_wear_underestimation/` |
| NUAA and PHM grouped validation | `external_validation.m`; `external_multisensor_multifault_robustness.m` | `external_validation_results/`; `multifault_robustness_results/` |
| Archived-result integrity checks | `verify_archived_results.m`; `audit_v17_results.m` | console pass status; `audit_outputs/key_result_ledger.csv` |

The raw third-party datasets are not redistributed. All inferential summaries are based on outer-held-out machining units; technical degradation repeats are averaged within independent units before paired inference.
