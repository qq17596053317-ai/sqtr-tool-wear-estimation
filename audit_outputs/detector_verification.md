# Detector verification

**Status: CONFIRMED: 11 quantities.**

Static inspection of `step28_randomized_fault_repeats.m` and `external_multisensor_multifault_robustness.m` confirms that `selectedStats = [1 2 3 4 7 8 9]` selects mean, RMS, standard deviation, peak-to-peak amplitude, median, MAD and crest factor. These seven quantities are concatenated with four raw quality descriptors.

All 11 quantities are centred by the outer-training median and scaled by 1.4826 times the outer-training MAD. A non-finite or smaller-than-1e-9 robust scale falls back first to the outer-training standard deviation and then to 1e-9. The channel score is the maximum absolute robust z-score across the 11 quantities. Its threshold is the larger of the outer-training 99th percentile and 4 robust-scale units. If several channels cross their thresholds, only the channel with the largest score-to-threshold ratio is excluded; if the largest ratio is at most 1, no channel is excluded.

Verification level: **VERIFIED BY STATIC CODE INSPECTION**.
