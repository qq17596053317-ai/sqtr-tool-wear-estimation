# Concurrent two-channel mask-aware dropout experiment

This is an exploratory experiment report and is not manuscript text.
Both suites use the same grouped outer units, detector, distortion draws,
30 technical randomisations and clean-unit normalisation as the existing
concurrent two-channel comparison.

The single-mask-trained model is the existing deployable baseline, evaluated
with all detector-triggered mask bits at inference. Double- and triple-mask
states were absent from its augmentation. The multi-mask-trained model uses
the same augmentation-rate and ridge-lambda grids, but samples uniformly from
all seven non-empty masks during training; inner grouped validation averages
clean and all seven masked states.

## Abrupt distortions

| Method | Mean relative MAE |
|---|---:|
| Fixed fusion | 1.7702 |
| Median replacement | 1.1142 |
| Single-mask-trained dropout | 1.1021 |
| Multi-mask-trained dropout | 1.1661 |
| Single-channel SQTR | 1.4339 |
| Subset routing | 1.1501 |
| Oracle pair exclusion | 1.1170 |

Paired exploratory comparisons (12 independent units):

- Single-mask-trained dropout vs Subset routing: right-minus-left = +0.0480, 95% unit-bootstrap CI [-0.0198, +0.1226], exact unadjusted Wilcoxon p = 0.38037.
- Multi-mask-trained dropout vs Subset routing: right-minus-left = -0.0161, 95% unit-bootstrap CI [-0.0778, +0.0522], exact unadjusted Wilcoxon p = 0.79102.
- Single-mask-trained dropout vs Median replacement: right-minus-left = +0.0121, 95% unit-bootstrap CI [-0.0345, +0.0603], exact unadjusted Wilcoxon p = 0.51855.
- Multi-mask-trained dropout vs Single-mask-trained dropout: right-minus-left = -0.0640, 95% unit-bootstrap CI [-0.1609, +0.0069], exact unadjusted Wilcoxon p = 0.17627.

## Structured distortions

| Method | Mean relative MAE |
|---|---:|
| Fixed fusion | 1.2449 |
| Median replacement | 1.1020 |
| Single-mask-trained dropout | 1.0960 |
| Multi-mask-trained dropout | 1.1465 |
| Single-channel SQTR | 1.1321 |
| Subset routing | 1.1255 |
| Oracle pair exclusion | 1.1170 |

Paired exploratory comparisons (12 independent units):

- Single-mask-trained dropout vs Subset routing: right-minus-left = +0.0295, 95% unit-bootstrap CI [-0.0302, +0.0979], exact unadjusted Wilcoxon p = 0.67725.
- Multi-mask-trained dropout vs Subset routing: right-minus-left = -0.0210, 95% unit-bootstrap CI [-0.0925, +0.0500], exact unadjusted Wilcoxon p = 0.67725.
- Single-mask-trained dropout vs Median replacement: right-minus-left = +0.0060, 95% unit-bootstrap CI [-0.0587, +0.0773], exact unadjusted Wilcoxon p = 0.73340.
- Multi-mask-trained dropout vs Single-mask-trained dropout: right-minus-left = -0.0505, 95% unit-bootstrap CI [-0.1382, +0.0122], exact unadjusted Wilcoxon p = 0.20361.
