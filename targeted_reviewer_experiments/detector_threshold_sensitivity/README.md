# Detector-threshold sensitivity

This folder contains a prespecified 3 × 3 sensitivity grid for the general-purpose detector:

- robust-scale floor: 3, 4 or 5;
- training-score quantile: 97.5, 99 or 99.5;
- 30 identical artificial-fault randomisations per setting;
- nine NUAA experiments and three PHM 2010 tools as the independent units.

The grid is an operating-point sensitivity analysis, not a test-set hyperparameter search.

Run from PowerShell:

```powershell
$env:SQTR_GRID_DATA_ROOT = 'PATH_TO_RAW_DATA_FOLDER'
& .\run_threshold_grid.ps1
```

Then compile the grid in MATLAB:

```matlab
run('step40_compile_detector_threshold_sensitivity.m')
```

The main compiled outputs are in `compiled_results/`.

