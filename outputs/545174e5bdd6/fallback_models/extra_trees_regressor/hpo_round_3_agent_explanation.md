# HPO Round 3 Agent Grid Proposal

**Strategy:** Constrain tree growth and increase randomness to reduce variance: test shallow-to-moderate depths, larger leaf and split thresholds, and feature subsampling levels that are more regularizing than the previous best settings. Keep n_estimators in a moderate-high range to stabilize predictions without exploding the search space. Total combinations are capped at 120.

The previous rounds indicate persistent severe overfitting with very high variance across folds on a tiny dataset (20 samples, 4 features). The next search should prioritize stronger regularization and reduced tree complexity while still exploring enough ensemble size and feature subsampling to improve stability. I therefore narrow the grid toward larger min_samples_leaf/min_samples_split, shallower max_depth, and both bootstrap settings, while keeping a modest range of n_estimators.

**Expected overfitting effect:** Should reduce overfitting substantially by limiting tree depth, increasing minimum samples per split/leaf, and enabling bootstrap-based randomness in part of the grid. This should lower train-CV gap and CV variance relative to the prior rounds.

**Expected underfitting effect:** Some configurations, especially very shallow trees with larger leaf sizes, may underfit on this small dataset. However, the grid still includes moderate depths and smaller regularization settings to avoid collapsing performance too much.

**Cost estimate:** 120 combinations total; moderate cost for ExtraTrees on a very small dataset, but still manageable. The search is intentionally compact to fit the candidate limit while exploring the main regularization axes.
