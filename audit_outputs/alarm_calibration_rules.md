# Training-fold alarm-calibration rules

**Status: VERIFIED BY STATIC CODE INSPECTION and VERIFIED FROM ARCHIVED OUTPUTS.**

For each outer-held-out NASA case and each wear threshold `c`, the calibration labels are `VB >= c` and the scores are inner-LOCO out-of-fold SQTR point predictions from the corresponding outer-training cases. Candidate alert thresholds are the unique values in `{c} union {score: score <= c}`, sorted from highest to lowest. The algorithm selects the highest candidate meeting the target inner sensitivity; this ordering also resolves ties. Candidate thresholds are therefore never above `c`.

The implementation asserts that at least one positive training record is available and asserts finite inner predictions; it does not silently impute missing or infinite scores. If no negative record exists, specificity is returned as `NaN`. A lower fallback threshold (`min(score)` minus a numerical epsilon) is initialised for the theoretical case in which no listed candidate reaches the target; no such fallback was used in the archived 16 × 4 × 3 fold/threshold/target combinations. All 192 combinations contained positive training records and achieved or exceeded the requested inner sensitivity. No outer fold failed calibration.

The selected threshold is frozen and applied once to the outer-held-out case. The primary operating point is `c = 0.40 mm` with 90% target inner sensitivity; 85% and 95% targets and `c = 0.30, 0.60, 0.80 mm` are operating-characteristic analyses.
