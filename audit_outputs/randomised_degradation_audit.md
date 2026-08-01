# Randomised artificial-degradation audit

**Status: VERIFIED BY STATIC CODE INSPECTION and VERIFIED FROM ARCHIVED OUTPUTS.**

- Plateau clipping, dropout, bias and additive Gaussian noise are each applied to one contiguous segment of every outer-held-out record.
- Duration fractions are independently sampled per record: Uniform(0.10, 0.70) for plateau clipping and dropout, and Uniform(0.20, 1.00) for bias and noise.
- Segment length is `round(duration_fraction * record_length)`, bounded to [1, record_length]. The start index is sampled uniformly from all feasible integer starts `1:(n - block_length + 1)`, so the segment never overruns the record.
- Local SD is the sample standard deviation of the complete clean record. If it is non-finite or below 1e-10, the fallback is `max(1e-6, 0.01 * max(abs(x)))`.
- Bias magnitude is Uniform(0.50, 3.00) local SD; its sign is sampled independently for each record. Noise SD is Uniform(0.25, 2.00) local SD.
- Repeat seeds are 2026071801 through 2026071830. Within a repeat, the seeded stream is consumed sequentially across NUAA and PHM conditions.
- Models, preprocessing, exclusion branches and quality detectors are fitted on clean outer-training units and frozen before the corresponding held-out unit is degraded and predicted.
- The 30 randomisations quantify within-unit Monte Carlo variability. They are averaged within each of the 9 NUAA experiments and 3 PHM cutters before paired inference.

Archived results confirm a combined reduction in degradation-to-clean relative MAE from 1.392267 to 1.082676 (22.2365%; 10/12 units improved; Holm-adjusted p = 0.0146484).
