# Part C code-verification report (manuscript v17)

Repository: <https://github.com/qq17596053317-ai/sqtr-tool-wear-estimation>  
Audit branch: `codex/manuscript-v17-verification`  
Starting commit: `2fe1abb286361bdea64604b8a1587b36828856d1`  
Software used for execution checks: MATLAB R2024b

## Executive result

`verify_archived_results.m` completed with `All archived-result checks passed.` The additional `audit_v17_results.m` execution completed with `V17 audit passed.` No core method item remains `NOT VERIFIABLE`. Two v16 manuscript inconsistencies were identified and are corrected in v17: the Figure 2b baseline was not input matched, and the time-ablation paragraph mislabelled an oppositely signed record-weighted pooled interval as a case-level paired interval.

| Issue | Code location | Verification level | Finding | Manuscript action |
|---|---|---|---|---|
| Figure 2b comparator | `step15timeawareridge.m`; `targeted_reviewer_experiments/nasa_qi_role_ablation/step33_qi_role_ablation.m`; `qi_role_route_comparisons.csv` | VERIFIED BY EXECUTION | The input-matched comparator is fixed all-sensor-plus-time ridge (0.116210 mm) versus SQTR (0.108773 mm), a 6.3998% pooled reduction. Case-macro difference = -0.005184 mm, 95% percentile CI [-0.018274, 0.004043], Holm-adjusted p = 0.644531. | Regenerate Figure 2b and describe 6.4% as numerical/inconclusive; retain 11.74% only as the broader conventional-baseline comparison. |
| Time-ablation sign | `step15timeawareridge.m`; `audit_v17_results.m` | VERIFIED BY EXECUTION | With `no-time - SQTR`, mean case-macro difference = +0.005203 mm, 95% percentile CI [0.001180, 0.009257], exact p = 0.03353882; 11/16 cases worsened. | Regenerate Figure 3a and replace the v16 negative sign/incorrect CI. |
| Detector quantities | `step28_randomized_fault_repeats.m:447-473`; `external_multisensor_multifault_robustness.m:389-429` | VERIFIED BY STATIC CODE INSPECTION | Seven signal statistics plus four raw descriptors enter robust standardisation and the max-z channel score: 11 quantities in total. | State “each of the 11 quantities”; specify fallback scale, threshold and one-channel exclusion rule. |
| Random degradation start | `step28_randomized_fault_repeats.m:375-398` | VERIFIED BY STATIC CODE INSPECTION | Each degradation occupies a contiguous segment. Start is uniform over all feasible integer starts; no overrun occurs. | Add the exact start and duration rules to Supplementary Methods S2. |
| Local SD definition | `step28_randomized_fault_repeats.m:375-379`; `external_multisensor_multifault_robustness.m:184-206` | VERIFIED BY STATIC CODE INSPECTION | Sample SD of the complete clean record; fallback `max(1e-6, 0.01*max(abs(x)))`. | Define consistently in S2 and S4. |
| Bias-sign level | `step28_randomized_fault_repeats.m:394-395`; `external_multisensor_multifault_robustness.m:203-204` | VERIFIED BY STATIC CODE INSPECTION | Randomised test: sign sampled independently per record. Fixed grid: positive bias only, applied to the full record. | Remove the incorrect random-sign statement from S4 and distinguish the two experiments. |
| Fixed-grid position | `external_multisensor_multifault_robustness.m:194-206` | VERIFIED BY STATIC CODE INSPECTION | Plateau/dropout use evenly distributed discrete indices from `linspace`; no contiguous or centred segment and no location dimension. | Rewrite S4 as a separate fixed-severity-grid upper-bound sensitivity benchmark. |
| Alarm fallback | `targeted_reviewer_experiments/alarm_calibration/step34_high_wear_alert_calibration.m:290-316`; archived threshold selections | VERIFIED BY STATIC CODE INSPECTION and VERIFIED FROM ARCHIVED OUTPUTS | Candidate thresholds are unique scores not above c plus c, evaluated descending. No-positive folds cause an assertion; no-negative folds yield NaN specificity; a below-minimum threshold is the no-feasible-candidate fallback. No fallback or failed fold occurred in 192 archived selections. | Add Supplementary Methods S6 without inventing unimplemented rules. |
| Bootstrap CI type | analysis scripts using `prctile`; `audit_outputs/bootstrap_ci_map.csv` | VERIFIED BY STATIC CODE INSPECTION | Two-sided percentile 95% intervals; independent-unit resampling; CIs are not multiplicity adjusted. Holm applies to p values within declared families. | State interval type, resampling unit and lack of CI multiplicity adjustment in Methods/S15. |

## Additional verified findings

- Randomised degradation uses seeds 2026071801-2026071830 and averages 30 technical repeats within 9 NUAA experiments and 3 PHM cutters before inference. Combined degradation-to-clean relative MAE decreased by 22.2365% (10/12 units improved; Holm-adjusted p = 0.0146484).
- The separate fixed grid contains 48 conditions per dataset and evaluates fixed, automatic and Oracle policies on the same degraded variants. Combined relative MAE values were 1.374499, 1.086247 and 1.052557, respectively.
- Clean-input false-trigger rates were 3.95% for NUAA and 2.96% for PHM; automatic-minus-fixed clean MAE differences were +0.000101 and +0.000070 mm.
- Alarm calibration uses only inner-LOCO out-of-fold predictions from the outer-training cases. All 192 archived fold/threshold/target selections contained positives and met the requested inner sensitivity.

## Reproduction commands

```matlab
cd('<repository root>')
verify_archived_results
audit_v17_results
```

The public package does not redistribute third-party raw datasets. The archived derived predictions, case/unit summaries and figure source data are sufficient to verify all numerical claims listed in `audit_outputs/key_result_ledger.csv`.
