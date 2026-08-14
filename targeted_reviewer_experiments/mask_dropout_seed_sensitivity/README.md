# Mask-aware sensor-dropout training-seed sensitivity

This sensitivity analysis repeats the locked randomised artificial-degradation
workflow while changing only the base seed used to create masked training
copies for the mask-aware sensor-dropout ridge baseline.

The five base seeds are `20260718`, `20260801`, `20260802`, `20260803` and
`20260804`. Outer partitions, inner grouped hyperparameter selection, detector
outputs and the 30 degradation seeds (`2026071801`--`2026071830`) remain fixed.
The five runs are sensitivity replicates and are not independent inferential
units.

To reproduce one run, set `SQTR_DROPOUT_SEED_BASE` and optionally
`SQTR_RANDOM_FAULT_RESULT_DIR`, then run `step28_randomized_fault_repeats.m`.
The original paper result is recovered when the seed variable is absent or is
set to `20260718`.

The archived summary is in `mask_dropout_seed_sensitivity_summary.csv`.
