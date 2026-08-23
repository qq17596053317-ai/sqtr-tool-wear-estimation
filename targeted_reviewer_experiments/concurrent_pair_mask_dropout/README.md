# Concurrent two-channel mask-aware dropout audit

This directory archives the exploratory reviewer-requested comparison of:

- fixed all-channel fusion;
- detector-guided median replacement;
- the original single-mask-trained mask-aware dropout ridge;
- a multi-mask-trained mask-aware dropout ridge;
- the original single-channel SQTR router;
- concurrent subset routing; and
- non-deployable oracle pair exclusion.

The analyses use the same outer-held-out units and 30 within-unit technical
randomisations as the concurrent-pair routing experiment. The independent
units are nine NUAA experiments and three PHM 2010 tools.

## Reproduction

The experiment is implemented in `step28_randomized_fault_repeats.m`. It
requires the existing `external_validation_results` artefacts and the public
signal bundles described in the repository README.

Run the abrupt suite by setting:

```text
SQTR_FAULT_CARDINALITY=2
SQTR_FAULT_SET=abrupt
SQTR_RANDOM_FAULT_REPEAT_COUNT=30
SQTR_RANDOM_FAULT_RESULT_DIR=targeted_reviewer_experiments/concurrent_pair_mask_dropout/abrupt_results
```

Run the structured suite with the same settings except:

```text
SQTR_FAULT_SET=structured
SQTR_RANDOM_FAULT_RESULT_DIR=targeted_reviewer_experiments/concurrent_pair_mask_dropout/structured_results
```

After both MATLAB runs, execute:

```text
python tools/analyse_concurrent_mask_dropout.py
```

The `analysis` directory contains the cross-suite summaries used in the v34
manuscript revision. Confidence intervals and exact signed-rank tests treat the
12 machining units as independent; the 30 randomisations are technical repeats
and are averaged before inference.
