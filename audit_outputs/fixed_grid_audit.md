# Positive-bias fixed-severity-grid and Oracle audit

**Status: VERIFIED BY STATIC CODE INSPECTION and VERIFIED FROM ARCHIVED OUTPUTS.**

This is a separate positive-bias fixed-severity-grid sensitivity benchmark, not the 30-repeat randomised stress test.

- Each dataset contains 3 target channels × 4 degradation modes × 4 severity levels = 48 conditions (96 across both datasets).
- Distributed extreme-value replacement and distributed zero replacement affect indices selected by `unique(round(linspace(1,n,k)))`; they are evenly distributed samples, not contiguous or centred segments, and are not equivalent to the contiguous clipping/dropout modes in the randomised experiment.
- Replacement fractions are 0.10, 0.30, 0.50 and 0.70. Bias levels are +0.50, +1.00, +2.00 and +3.00 local SD and are always positive. Negative bias is not covered. Noise levels are 0.25, 0.50, 1.00 and 2.00 local SD.
- Bias and noise affect the complete record. The local-SD definition and fallback match the code statement above.
- Seed 20260717 fixes the sequence of Gaussian-noise realisations. Each record/condition receives its next independent draw from that stream; deterministic replacement and bias operators consume no random draws.
- Fixed, automatic and Oracle policies are evaluated on the same generated degraded variants. Oracle uses only the known identity of the degraded channel to select the matching exclusion branch; it is a non-deployable upper-bound sensitivity comparator.

Combined relative MAE was 1.374499 for fixed fusion, 1.086247 for automatic routing and 1.052557 for Oracle routing.
