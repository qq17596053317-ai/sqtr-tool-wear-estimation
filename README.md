# SQTR tool-wear estimation

Reproducibility package for **Sensor-Quality-Aware Temporal Routing for Milling
Tool-Wear Estimation under Unseen Machining Cases and Detectable Sensor
Degradation**.

## Scope

This repository contains the MATLAB R2024b analysis code, derived fold-level
predictions, unit-level summaries, statistical outputs, and figure source
files. Third-party raw datasets are not redistributed.

The inferential unit is always the independent machining case, experiment, or
tool-life sequence. Outer held-out units are not used for preprocessing,
hyperparameter selection, quality-threshold estimation, or alarm calibration.

## Software

- MATLAB R2024b
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox
- Tested on Windows 11, AMD Ryzen 7 5800H, 16 GB RAM

## Raw-data placement

Download the datasets from their original sources and place them as follows:

```text
mill/mill.mat
nuaa_orthogonal_bundle_high_resolution.csv
phm2010_bundle_high_resolution.csv
```

- NASA Milling: https://data.nasa.gov/dataset/milling-wear
- NUAA: https://doi.org/10.21227/3AA1-5E83
- UniWear processed bundles: https://github.com/katulu-io/uniwear-dataset
  (commit `66c894f2449f4393126e906e086485a8f9d95171`)
- PHM 2010 description: https://phmsociety.org/2010-phm-data-challenge-comes-to-a-close/

## Main execution order

1. `compare1.m`, `extract1run2features.m`, `grouped1baseline2models.m`
2. `step09scansensorsaturation.m` through `step20buildpapersummary.m`
3. `external_validation_nuaa_phm2010.m`
4. `external_multisensor_multifault_robustness.m`
5. `step22_tree_ensemble_baselines.m` through
   `step32_high_wear_underestimation_audit.m`

Several scripts consume intermediate files produced by earlier steps. Derived
outputs used in the manuscript are included so that tables and reported
statistics can be inspected without redistributing the raw signals.

## Key reproducibility settings

- Ridge: 17 log-spaced values from `10^-4` to `10^4`
- Gate threshold: `{1%, 10%, 50%}`, ties favour the larger threshold
- SVR: Gaussian kernel, `C={1,10,100}`, epsilon `0.05`,
  kernel scale `{2,5,10,20}`
- Bagged trees: 100 learners, minimum leaf size `{2,5,10}`
- LSBoost: 100 learners, learning rate `{0.03,0.10}`, minimum leaf size `{2,5}`
- PLS: `{2,5,10,15,20}` latent variables, clipped by training dimensions
- GPR: Matern 3/2 kernel, constant basis, exact fit/prediction,
  standardisation enabled; kernel/noise parameters are estimated within each
  training fold by `fitrgp`
- Randomised degradation seeds: `2026071801`–`2026071830`
- Fixed-grid seed: `20260717`
- Case-cluster bootstrap: 10,000 resamples

## External-dataset feature mapping

NUAA channel order: Fz, bending moment X, bending moment Y, torsion,
vibration 1, vibration 2, spindle power, spindle current. Process variables:
feed per tooth, spindle speed, axial depth of cut. Target feature blocks:
Fz columns 1–9, vibration X columns 37–45, vibration Y columns 46–54.

PHM channel order: Fx, Fy, Fz, vibration X, vibration Y, vibration Z, AE RMS.
Target blocks: Fz columns 19–27, vibration X columns 28–36, vibration Y
columns 37–45.

## Results

- `mill/results`: NASA features, predictions, ablations and statistics
- `external_validation_results`: clean external-dataset validation
- `multifault_robustness_results`: fixed-severity-grid and oracle analyses
- `additional_validation_results/reviewer_improvements`: 30-repeat
  randomised degradation and reviewer-requested checks
- `targeted_reviewer_experiments/high_wear_underestimation`: high-wear audit

## Notes

The term **plateau clipping** denotes a synthetic operator or an empirical
upper-end plateau. It is not a documented hardware saturation rail.

No software licence has been selected yet. Add an appropriate licence before
public release.
