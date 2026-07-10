# HPO Round 1 Agent Grid Proposal

**Strategy:** Bias the search toward simpler forests first: shallow max_depth, larger min_samples_leaf/min_samples_split, and reduced max_features. Include bootstrap with max_samples to further decorrelate trees and reduce variance. Keep a small set of higher-capacity values to avoid excessive underfitting.

Baseline shows severe overfitting on a very small dataset (20 samples, 3 features). The grid prioritizes stronger regularization via shallower trees, larger leaf/split sizes, feature subsampling, and optional bootstrap/max_samples constraints while keeping a few higher-capacity settings for comparison. The grid size is kept near the requested limit.

**Expected overfitting effect:** Should reduce overfitting substantially by constraining tree depth, increasing leaf/split minimums, and lowering feature/sample usage per tree. Bootstrap with max_samples is expected to lower variance further.

**Expected underfitting effect:** Some configurations may underfit, especially the shallowest trees with large leaf sizes and low max_features. The inclusion of moderate-depth and null max_depth options should help identify a balanced region.

**Cost estimate:** Moderate to high. The full Cartesian grid is 3*6*4*4*4*2*4*2 = 9216 combinations, which exceeds the stated cap; in practice this grid should be pruned or sampled to stay near 120 candidates.
