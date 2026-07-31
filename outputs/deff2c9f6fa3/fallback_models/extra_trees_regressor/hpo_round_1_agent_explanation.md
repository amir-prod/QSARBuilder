# HPO Round 1 Agent Grid Proposal

**Strategy:** Use a compact, regularization-focused grid centered on reducing tree complexity and variance. Prioritize shallow max_depth values, larger min_samples_leaf/min_samples_split, and max_features below 1.0. Include bootstrap as an additional variance-reduction option. Keep n_estimators moderate-to-high to stabilize ensemble estimates without exploding the search space.

The baseline shows severe overfitting and high fold-to-fold variability on a very small dataset (21 samples, 4 features). The grid therefore emphasizes stronger regularization via shallower trees, larger leaf sizes, larger split thresholds, and feature subsampling, while still retaining a small set of higher-capacity settings for comparison.

**Expected overfitting effect:** Should reduce overfitting substantially by constraining tree depth, increasing leaf and split minimums, and allowing feature subsampling/bootstrap to lower variance.

**Expected underfitting effect:** May slightly increase underfitting risk for the most constrained settings, but the grid retains some less restrictive options (null depth, smaller split/leaf values) to preserve capacity if needed.

**Cost estimate:** Moderate. The full Cartesian grid is 3 x 5 x 4 x 4 x 4 x 2 = 1920 combinations, which exceeds the candidate limit; in practice this should be sampled or reduced before execution. If pruned to a near-120 candidate subset, cost would be low-to-moderate for ExtraTrees on this small dataset.
