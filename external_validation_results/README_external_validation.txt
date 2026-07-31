EXTERNAL VALIDATION: NUAA AND PHM2010
Generated with MATLAB 24.2.0.2712019 (R2024b)

Design
- NUAA independent units: W1-W9 experimental sequences (n=9).
- PHM2010 independent units: c1, c4 and c6 cutter sequences (n=3).
- Raw high-frequency rows were treated as technical subsamples and aggregated to cutting-run level.
- PHM2010 cutting-run boundaries were detected from timestamp-cadence discontinuities; wear labels were not used for segmentation.
- Outer validation: leave one entire experiment/cutter sequence out.
- Hyperparameter selection: inner leave-one-training-experiment-out validation.
- Lambda grid: [0.0001 0.000316227766016838 0.001 0.00316227766016838 0.01 0.0316227766016838 0.1 0.316227766016838 1 3.16227766016838 10 31.6227766016838 100 316.227766016838 1000 3162.27766016838 10000].
- Sensor features: mean, RMS, standard deviation, peak-to-peak, skewness, kurtosis, median, MAD and crest factor.
- Operational-history features: elapsed machining time and run index; neither uses the wear label.
- No held-out experiment labels were used for preprocessing, tuning or model fitting.

Quality-gating stress test
- Common channel: force_z.
- Injected upper-rail ratios: [0 0.1 0.3 0.5 0.7 1].
- Gate threshold: 1%. At or above the threshold, prediction switches to a branch trained without force_z.

Statistical unit
- Inferential n is the number of independent experiment/cutter sequences, not the number of raw samples or run-level windows.
- Paired Wilcoxon signed-rank tests compare experiment-level MAE. Two dataset-specific tests use Holm adjustment.
- Bootstrap confidence intervals resample independent experiment-level paired differences.

Important limitations
- PHM2010 contains only three independent cutter sequences, so dataset-specific statistical power is low.
- Wear is strongly monotonic with elapsed time in both curated bundles. Results support predictive validity under these benchmark conditions, not a causal wear mechanism.
- This is dataset-specific retraining under identical methodology, not zero-shot transfer of NASA model coefficients across incompatible sensor units.

Statistics summary
NUAA: n=9, sensor macro-MAE=0.043565, sensor+time macro-MAE=0.040194, delta=-0.003371, p=0.203125, bootstrap 95% CI=[-0.007186, 0.000750].
PHM2010: n=3, sensor macro-MAE=0.012052, sensor+time macro-MAE=0.012539, delta=0.000487, p=0.25, bootstrap 95% CI=[0.000074, 0.001236].
Combined: n=12, sensor macro-MAE=0.035687, sensor+time macro-MAE=0.033280, delta=-0.002406, p=0.233398, bootstrap 95% CI=[-0.005566, 0.000673].
