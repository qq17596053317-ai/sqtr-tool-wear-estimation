# High-resolution signal-to-feature reconstruction audit

The external artificial-degradation pipeline reads the high-resolution NUAA and PHM 2010 signal bundles, modifies the selected signal sequence and then recomputes the nine run-level statistics used by the prediction model. This audit independently reconstructed every clean feature from those signal bundles and compared it with the stored modelling matrix.

All 152 NUAA records (eight channels; 234,902 signal samples) and all 945 PHM 2010 records (seven channels; 104,675 signal samples) were reproduced exactly. Across every record, channel and statistic, the maximum absolute difference was zero at the stored numerical precision.

This verifies implementation consistency and rules out direct fault injection into precomputed feature columns. It does **not** establish equivalence to raw data-acquisition voltages: the public high-resolution bundles may already include source-side processing and smoothing.

Outputs:

- `results/waveform_reconstruction_dataset_summary.csv`
- `results/waveform_reconstruction_sensor_summary.csv`
- `results/waveform_reconstruction_detail.csv`
- `results/waveform_reconstruction_audit.mat`

