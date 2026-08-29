# Detector-threshold sensitivity and concurrent subset routing

## Scope and inferential unit

These experiments extend the original artificial-degradation analysis without changing its inferential hierarchy. The independent units remain nine NUAA experiments and three PHM 2010 tools (`n = 12`). Thirty fault randomisations are averaged within each unit and are technical repeats, not additional independent observations. Confidence intervals are descriptive 10,000-resample independent-unit bootstrap intervals. Wilcoxon tests are based on unit-level paired differences.

## 1. Detector-threshold grid sensitivity

The detector floor was varied over `{3, 4, 5}` robust-scale units and the training-score quantile over `{97.5, 99, 99.5}`. All nine settings used the same 30 fault seeds. This was a sensitivity analysis; it was not used to select the best-performing threshold after observing the test results.

Across the nine settings, combined SQTR relative MAE ranged from 1.0669 to 1.0828, compared with 1.3923 for fixed all-channel fusion. The relative MAE reduction ranged from 22.23% to 23.37%; 10 of 12 independent units favoured SQTR at every grid point. The default setting (floor 4, 99th percentile) produced a relative MAE of 1.0827 and a clean-input trigger rate of 3.78%.

The 97.5th-percentile settings gave slightly lower degraded-input MAE (1.0669–1.0676) but increased the clean-input trigger rate to 14.40%–16.12%. Raising the quantile to 99.5 reduced the clean trigger rate to 3.13% without materially changing degraded-input MAE relative to the default setting. The robust-scale floor had little influence at the 99th and 99.5th percentiles because the quantile term dominated the maximum rule.

**Interpretation.** The performance gain over fixed fusion did not depend on one isolated threshold setting. The grid also exposes a genuine operating trade-off: the more permissive 97.5th-percentile detector slightly improves the stress-test MAE by switching more often, but produces substantially more false switching on clean inputs. The default 99th-percentile rule should therefore be retained as the prespecified compromise rather than replaced by the numerically best grid point.

## 2. Concurrent two-channel subset routing

All three monitored channel pairs were degraded: axial force–vibration X, axial force–vibration Y, and vibration X–vibration Y. The original policy can exclude at most the single channel with the largest normalised score. The extension fits an exclusion branch for every non-empty subset of the three monitored channels and routes to the branch corresponding to the detected subset. Median replacement and non-deployable Oracle pair exclusion were retained as reference responses.

### 2.1 Abrupt stress tests

For the original abrupt degradation operators, fixed fusion had a combined relative MAE of 1.7702. Original single-channel SQTR reduced this to 1.4339, whereas subset routing reduced it to 1.1501. The prespecified unit-level difference (single-channel SQTR minus subset routing) was 0.2839 (95% bootstrap CI: 0.0732 to 0.5287; exact Wilcoxon `p = 0.0342`); 8 of 12 units favoured subset routing.

Relative to fixed fusion, median replacement, original single-channel SQTR and subset routing all reduced error after Holm correction (`p = 0.00977`, `p = 0.00732` and `p = 0.00977`, respectively). Median replacement remained numerically slightly better than subset routing (1.1142 versus 1.1501), so the result does not identify subset routing as the universally preferred response.

### 2.2 Structured gradual and intermittent stress tests

Under soft clipping, intermittent dropout, gradual drift and heteroscedastic noise, fixed fusion had a combined relative MAE of 1.2449. Original single-channel SQTR and subset routing achieved 1.1321 and 1.1255, respectively. Their paired difference was only 0.0066 (95% bootstrap CI: −0.0221 to 0.0311; exact Wilcoxon `p = 0.470`); 7 of 12 units favoured subset routing.

None of median replacement, single-channel SQTR or subset routing differed from fixed fusion after Holm correction (`p = 0.231` for all three comparisons). Detection of both target channels was high for intermittent dropout but poor for soft clipping, gradual drift and heteroscedastic noise. The absence of a stable subset-routing gain is therefore detector-limited rather than evidence that subset exclusion is intrinsically ineffective.

## 3. Contribution supported by the new experiments

The defensible extension is conditional, not universal:

1. detector-threshold sensitivity shows that the original response benefit is stable across a prespecified local threshold grid, while revealing the cost of aggressive switching on clean inputs;
2. subset routing solves a real structural limitation of the original one-channel router under concurrently detectable abrupt faults;
3. the advantage disappears when gradual or weakly detectable faults are used, locating the bottleneck in multi-channel detection rather than in the existence of subset branches;
4. median replacement remains competitive, so the paper should present subset routing as an auditable response option, not as a generally optimal fault-recovery method.

The manuscript should not claim robustness to real hardware faults or general multi-channel degradation. A precise claim is: *within the tested synthetic stress tests, subset routing reduced the additional error caused by concurrently detectable abrupt two-channel degradation, but it provided no stable advantage under gradual or weakly detected distortions.*

## 4. Reproducible outputs

- Threshold-grid summary: `detector_threshold_sensitivity/compiled_results/detector_threshold_grid_summary.csv`
- Threshold stability summary: `detector_threshold_sensitivity/compiled_results/detector_threshold_grid_stability.csv`
- Abrupt pairwise statistics: `concurrent_pair_subset_routing/results/concurrent_pair_strategy_statistics.csv`
- Abrupt subset-versus-single comparison: `concurrent_pair_subset_routing/results/concurrent_pair_subset_vs_single.csv`
- Structured pairwise statistics: `concurrent_pair_subset_routing/structured_results/concurrent_pair_strategy_statistics.csv`
- Structured subset-versus-single comparison: `concurrent_pair_subset_routing/structured_results/concurrent_pair_subset_vs_single.csv`

## 5. Stateful detection under gradual and intermittent distortions

EWMA and CUSUM score histories were selected using outer-training units only and evaluated on 30 randomised sequence-level distortions in each held-out unit. The stateful extension did not improve the record-wise point detector. Mean unit-relative MAE increased from 1.0825 to 1.1370; the paired point-minus-stateful difference was −0.0545 (95% independent-unit bootstrap CI: −0.1385 to −0.0034; exact Wilcoxon `p = 0.1099`). Event detection decreased from 58.15% to 52.04%, and the outer-held-out clean trigger rate was 13.78% despite a training-side preference for configurations at or below 5%.

This result rules out a superficial innovation claim that detector memory alone resolves gradual or intermittent degradation. It instead motivates change-point models trained on independent physical-fault trajectories.

## 6. Cross-fitted recovery-action selection

A separate experiment treated full fusion, median replacement, mask-aware dropout and channel exclusion as candidate actions. Four shallow regression trees predicted action regret from label-free quality scores and candidate-prediction disagreement. All meta-model training used outer-training units; the held-out unit did not contribute wear labels, base-model fitting or action-policy fitting.

The unconstrained selector was worse than the training-fold-selected fixed response (1.0909 versus 1.0658 mean unit-relative MAE). A conservative selective policy therefore used the fixed response as a fallback and overrode it only when cross-fitted training predictions exceeded a selected gain threshold. In the formal 30-repeat analysis, it overrode the fallback in only one of 12 independent units and changed mean relative MAE from 1.0658 to 1.0651 (exact Wilcoxon `p = 1.00`). The result does not support a claim that the available quality evidence identifies the preferred recovery action record by record.

Transfer audits reached the same conclusion. When each degradation family was excluded from action-policy training, the transferred selector had relative MAE 1.0852 versus 1.0697 for the training-selected fixed response (fixed minus transferred: −0.0155; 95% CI: −0.0428 to 0.0036; Holm-adjusted `p = 0.303`). Training the action selector on the other dataset only gave 1.0880 versus 1.0567 (difference: −0.0313; 95% CI: −0.0701 to −0.0011; Holm-adjusted `p = 0.303`). These audits distinguish anomaly detection from action identification: detecting that a channel is unusual does not determine which recovery action will minimise wear-estimation error in a new fault family or dataset.

## 7. Signal-to-feature reconstruction audit

All external model features were independently reconstructed from the same high-resolution signal sequences used by the artificial-degradation pipeline. The audit covered 152 NUAA records across eight channels and 945 PHM 2010 records across seven channels. Every reconstructed feature matched the stored modelling matrix exactly at the stored numerical precision (maximum absolute difference zero). Thus, the degradation operators act on signal sequences before feature recomputation rather than directly editing precomputed feature columns. These public bundles may already be processed or smoothed, so the audit does not establish equivalence to raw data-acquisition voltages.

## 8. Publication figure

The source-controlled multi-panel summary is generated by `figures/generate_innovation_extension_figure.py` and exported as SVG, PDF, 600-dpi PNG and 600-dpi TIFF under `figures/innovation_extension/`. Panel-level points represent independent machining units where applicable; technical randomisations are not plotted as independent observations.
