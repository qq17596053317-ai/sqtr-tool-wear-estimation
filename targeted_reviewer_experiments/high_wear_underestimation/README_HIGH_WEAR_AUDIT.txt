HIGH-WEAR UNDERESTIMATION AND ALERT-MISS AUDIT
Generated with MATLAB 24.2.0.2712019 (R2024b)

Analysis units
- Total machining cases: n=16.
- Cases containing VB >= 0.40: n=15.
- Run-level rows are descriptive observations, not independent inferential replicates.
- Wilcoxon tests and bootstrap confidence intervals use paired case-level summaries.
- Holm adjustment covers the three prespecified case-level endpoints.

Definitions
- Residual = prediction - actual; negative values denote underestimation.
- Alert miss = actual wear at or above a threshold but prediction below that same threshold.
- Large underestimation is a descriptive margin of 0.20 VB; it is not asserted to be a universal industrial safety limit.
- Dangerous miss count is reported descriptively as actual VB >= 0.40 with predicted VB < 0.20.

Record-level descriptive results
PrimaryTimeGated, actual VB >= 0.30: n=70, MAE=0.141558, mean residual=-0.070980, underprediction rate=0.5571, alert miss rate=0.1000, large-underprediction n=10.
PrimaryTimeGated, actual VB >= 0.40: n=52, MAE=0.162860, mean residual=-0.101726, underprediction rate=0.6154, alert miss rate=0.2308, large-underprediction n=10.
PrimaryTimeGated, actual VB >= 0.60: n=20, MAE=0.280370, mean residual=-0.261834, underprediction rate=0.9000, alert miss rate=0.7000, large-underprediction n=9.
PrimaryTimeGated, actual VB >= 0.80: n=8, MAE=0.525327, mean residual=-0.525327, underprediction rate=1.0000, alert miss rate=1.0000, large-underprediction n=7.
SafetyWeightedTimeGated, actual VB >= 0.30: n=70, MAE=0.140992, mean residual=-0.071231, underprediction rate=0.5714, alert miss rate=0.1000, large-underprediction n=10.
SafetyWeightedTimeGated, actual VB >= 0.40: n=52, MAE=0.164717, mean residual=-0.099444, underprediction rate=0.6346, alert miss rate=0.2308, large-underprediction n=10.
SafetyWeightedTimeGated, actual VB >= 0.60: n=20, MAE=0.283717, mean residual=-0.258487, underprediction rate=0.9000, alert miss rate=0.7000, large-underprediction n=9.
SafetyWeightedTimeGated, actual VB >= 0.80: n=8, MAE=0.525327, mean residual=-0.525327, underprediction rate=1.0000, alert miss rate=1.0000, large-underprediction n=7.

Paired case-level inference for actual VB >= 0.40
High-wear MAE: n=15, primary=0.135237, safety=0.137382, difference=0.002146, bootstrap 95% CI=[-0.001743, 0.008179], raw p=1, Holm p=1.
Underprediction rate: n=15, primary=0.605556, safety=0.627778, difference=0.022222, bootstrap 95% CI=[0.000000, 0.066667], raw p=1, Holm p=1.
Alert miss rate: n=15, primary=0.251111, safety=0.251111, difference=0.000000, bootstrap 95% CI=[0.000000, 0.000000], raw p=1, Holm p=1.

Interpretation boundary
- The safety-weighted model is a secondary analysis and does not replace the prespecified primary model.
- Any improvement in miss rate must be interpreted together with overall MAE and false-alert trade-offs.
