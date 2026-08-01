# Bootstrap confidence-interval verification

All manuscript confidence intervals audited here are two-sided percentile intervals obtained from the 2.5th and 97.5th percentiles of the resampled statistic. The code uses `prctile`; it does not use BCa, basic intervals or `bootci`. Resampling is performed at the independent machining-unit level (case, experiment or cutter) unless the analysis is explicitly a within-case permutation. Confidence limits are not multiplicity adjusted. Holm adjustment is applied to the corresponding Wilcoxon p values within the prespecified comparison families described in Supplementary Table S15.

Verification level: **VERIFIED BY STATIC CODE INSPECTION** for the interval definitions and **VERIFIED BY EXECUTION** for the v17 time-ablation audit.
