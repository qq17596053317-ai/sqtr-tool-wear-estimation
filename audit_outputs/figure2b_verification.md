# Figure 2b verification

**Status: VERIFIED FROM ARCHIVED OUTPUTS and VERIFIED BY EXECUTION.**

The defensible input-matched comparison is the fixed all-sensor-plus-time ridge versus route-only SQTR. Both models include the process variables and cumulative machining-time terms `(t, t^2)`; only SQTR performs record-wise quality routing.

- Fixed all-sensor-plus-time pooled MAE: 0.116210 mm.
- SQTR pooled MAE: 0.108773 mm.
- Record-weighted pooled reduction: 6.3998%.
- Case-macro difference, SQTR minus comparator: -0.005184 mm.
- 95% case-cluster percentile bootstrap CI: [-0.018274, 0.004043] mm.
- Raw Wilcoxon p = 0.322266; Holm-adjusted p = 0.644531.
- Cases improved by SQTR: 6/16; worsened: 10/16; ties: 0/16.

The 11.74% comparison (0.12324 to 0.10877 mm) uses the conventional process-plus-all-sensor ridge without matching cumulative-time inputs and is retained only as a broader conventional-baseline comparison. It is not labelled as the isolated routing effect.
