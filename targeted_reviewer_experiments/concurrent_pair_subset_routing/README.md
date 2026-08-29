# Concurrent two-channel subset routing

This experiment degrades every pair among the three monitored channels and compares:

1. fixed all-channel fusion;
2. detector-guided median replacement;
3. the original one-channel SQTR policy;
4. detector-guided subset routing;
5. non-deployable Oracle pair exclusion.

Subset routing fits an exclusion branch for every non-empty subset of the monitored channels and selects the branch indexed by the detected set. Thirty randomisations are averaged within each of nine NUAA experiments and three PHM 2010 tools before unit-level inference.

The `results/` folder contains the abrupt stress tests. The `structured_results/` folder contains soft clipping, intermittent dropout, gradual drift and heteroscedastic noise. The two fault families are reported separately because their physical interpretation and detectability differ.

