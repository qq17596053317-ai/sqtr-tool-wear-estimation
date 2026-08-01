# Cumulative-time ablation verification

**Status: VERIFIED BY EXECUTION.**

The prespecified difference is `MAE(no-time) - MAE(SQTR)`, so a positive value means that removing `(t, t^2)` increased error. The no-time model removes only these two cumulative-time variables; the sensor inputs, process inputs, two ridge branches, quality ratio, gate grid, grouped nested-LOCO workflow and preprocessing remain unchanged.

Across 16 independent NASA cases, the mean case-macro difference was +0.005203 mm, the median difference was +0.006545 mm, and 11 cases worsened while 5 improved. A 10,000-resample equal-case bootstrap with seed 20260730 gave a two-sided percentile 95% CI of [0.001180, 0.009257] mm. The exact two-sided Wilcoxon signed-rank p value was 0.03353882. This is a single prespecified ablation comparison and is not Holm adjusted.

The previously reported interval [-0.01199, -0.00110] mm described a record-weighted pooled contrast in the opposite direction and must not be labelled as the case-level paired mean.
