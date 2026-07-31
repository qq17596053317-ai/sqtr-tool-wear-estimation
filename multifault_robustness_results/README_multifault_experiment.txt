MULTI-SENSOR MULTI-FAULT ROBUSTNESS EXPERIMENT
MATLAB 24.2.0.2712019 (R2024b)

Independent units: NUAA W1-W9 (n=9) and PHM2010 c1/c4/c6 (n=3).
Technical run-level observations were not treated as independent inferential n.
Sensors: force_z, vibration_x, vibration_y.
Faults and severity values:
- Saturation: [0.1 0.3 0.5 0.7] fraction
- Dropout: [0.1 0.3 0.5 0.7] fraction
- Bias: [0.5 1 2 3] signal_sd
- Noise: [0.25 0.5 1 2] signal_sd

Automatic detector training:
Robust median/MAD ranges and the 99th percentile threshold were estimated using clean outer-training experiments only.
The detector used sensor statistics plus rail, zero, flatline and difference-energy descriptors.
OracleGate uses the injected fault identity only as an upper-bound comparison; it is not the proposed deployable result.

Clean-data results:
NUAA FullClean: MAE=0.039539, R2=0.5480, clean false-alarm rate=0.0395.
NUAA AutoGatedClean: MAE=0.039640, R2=0.5457, clean false-alarm rate=0.0395.
PHM2010 FullClean: MAE=0.012539, R2=0.8436, clean false-alarm rate=0.0296.
PHM2010 AutoGatedClean: MAE=0.012608, R2=0.8423, clean false-alarm rate=0.0296.

Experiment-level robustness statistics:
NUAA: n=9, full relative MAE=1.4213, automatic-gate relative MAE=1.0933, improvement=23.08%, raw p=0.0117188, Holm p=0.0234375, CI=[-0.5536, -0.1317].
PHM2010: n=3, full relative MAE=1.2341, automatic-gate relative MAE=1.0652, improvement=13.69%, raw p=0.25, Holm p=0.25, CI=[-0.3660, -0.0137].
Combined: n=12, full relative MAE=1.3745, automatic-gate relative MAE=1.0862, improvement=20.97%, raw p=0.00244141, Holm p=0.00732422, CI=[-0.4711, -0.1326].
