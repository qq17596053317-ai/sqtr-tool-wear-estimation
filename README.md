# SQTR tool-wear estimation

Reproducibility package for **Sensor-Quality-Aware Temporal Routing for Milling
Tool-Wear Estimation under Unseen Machining Cases and Detectable Sensor
Degradation**.

## Scope

This repository contains MATLAB R2024b analysis code, derived fold-level
predictions, unit-level summaries, statistical outputs and figure source data.
Third-party raw datasets are not redistributed.

The inferential unit is the independent machining case, experiment or
tool-life sequence. Outer-held-out units are not used for preprocessing,
hyperparameter selection, quality-threshold estimation or alarm calibration.

## Software

- MATLAB R2024b
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox
- Tested on Windows 11, AMD Ryzen 7 5800H, 16 GB RAM
- Python 3.12 with matplotlib, pandas and NumPy is used only to regenerate the
  publication figures from archived CSV source data.

## Raw-data placement

Download the datasets from their original sources and place them as follows:

```text
mill/mill.mat
nuaa_orthogonal_bundle_high_resolution.csv
phm2010_bundle_high_resolution.csv
```

- NASA Milling: <https://data.nasa.gov/dataset/milling-wear>
- NUAA: <https://doi.org/10.21227/3AA1-5E83>
- UniWear processed bundles: <https://github.com/katulu-io/uniwear-dataset>
  (commit `66c894f2449f4393126e906e086485a8f9d95171`)
- PHM 2010 description: <https://phmsociety.org/2010-phm-data-challenge-comes-to-a-close/>

## Main execution order

1. `compare1.m`, `extract1run2features.m`, `grouped1baseline2models.m`
2. `step09scansensorsaturation.m` through `step20buildpapersummary.m`
3. `external_validation_nuaa_phm2010.m`
4. `external_multisensor_multifault_robustness.m`
5. `step22_tree_ensemble_baselines.m` through
   `step32_high_wear_underestimation_audit.m`
6. `targeted_reviewer_experiments/nasa_qi_role_ablation/step33_qi_role_ablation.m`
7. `targeted_reviewer_experiments/alarm_calibration/step34_high_wear_alert_calibration.m`
8. `verify_archived_results.m` and `audit_v17_results.m`
9. `figures/generate_v17_figure2_3.py`

Several scripts consume intermediate files produced by earlier steps. Derived
outputs used in the manuscript are included so that numerical claims can be
verified without redistributing raw signals.

## Key reproducibility settings

- Ridge: 17 log-spaced values from `10^-4` to `10^4`
- Gate threshold: `{1%, 10%, 50%}`, with ties favouring the larger threshold
- SVR: Gaussian kernel, `C={1,10,100}`, epsilon `0.05`, kernel scale
  `{2,5,10,20}`, and MATLAB duplicate-row removal after fold-specific
  standardisation
- Bagged trees: 100 learners; minimum leaf size `{2,5,10}`
- LSBoost: 100 learners; learning rate `{0.03,0.10}`; minimum leaf size `{2,5}`
- PLS: `{2,5,10,15,20}` latent variables, capped by training dimensions
- GPR: Matern 3/2 kernel, constant basis, exact fit/prediction and
  standardisation; `fitrgp` estimates kernel/noise parameters within each fold
- Randomised artificial-degradation seeds: `2026071801-2026071830`
- Fixed-severity-grid seed: `20260717`
- V17 time-ablation bootstrap seed: `20260730`
- Case/unit-cluster bootstrap: 10,000 resamples unless otherwise stated

## External-dataset feature mapping

NUAA channel order: Fz, bending moment X, bending moment Y, torsion,
vibration 1, vibration 2, spindle power and spindle current. Process variables:
feed per tooth, spindle speed and axial depth of cut. Target feature blocks:
Fz columns 1-9, vibration X columns 37-45 and vibration Y columns 46-54.

PHM channel order: Fx, Fy, Fz, vibration X, vibration Y, vibration Z and AE RMS.
Target blocks: Fz columns 19-27, vibration X columns 28-36 and vibration Y
columns 37-45.

## Result and audit locations

- `mill/results`: NASA features, predictions, ablations and statistics
- `external_validation_results`: clean external-dataset validation
- `multifault_robustness_results`: fixed-severity-grid and Oracle analyses
- `additional_validation_results/reviewer_improvements`: 30-repeat randomised
  artificial degradation and reviewer-requested checks
- `targeted_reviewer_experiments/nasa_qi_role_ablation`: q_i role ablation and
  input-matched routing comparison
- `targeted_reviewer_experiments/alarm_calibration`: fold-specific alarm rules
- `targeted_reviewer_experiments/high_wear_underestimation`: high-wear audit
- `audit_outputs`: v17 code map, recomputed case effects, rule audits and
  machine-readable verification ledgers
- `figures/v17`: regenerated Figures 2 and 3 in SVG, PDF, TIFF and PNG formats

## Verification

From the repository root in MATLAB R2024b:

```matlab
verify_archived_results
audit_v17_results
```

Both commands must complete without an assertion error. The v17 audit writes
its results to `audit_outputs/`.

## Notes

The term **plateau clipping** denotes an artificial operator or an empirical
upper-end plateau; it is not a documented hardware saturation rail. The
30-repeat randomised stress test and separate fixed-severity-grid sensitivity
benchmark use different injection designs, as documented in the manuscript
supplement and audit reports.

## Licence

Code in this repository is released under the MIT License. Third-party datasets
retain their original licences and terms and are not redistributed here.
