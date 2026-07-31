# HPO Round 1 Agent Grid Proposal

**Strategy:** Use a compact, regularization-focused grid centered on depth limits, larger leaf/split constraints, and reduced max_features; include bootstrap as an additional variance-reduction option. Keep combinations near the requested cap while covering the main bias-variance tradeoffs.

Baseline shows strong overfitting and high fold-to-fold variability on a very small dataset, so the grid emphasizes stronger regularization, shallower trees, and feature subsampling while still allowing a few higher-capacity settings for comparison.

**Expected overfitting effect:** Should reduce overfitting by constraining tree complexity, increasing leaf/split minimums, and optionally bootstrapping; lower max_features may further decorrelate trees and improve stability.

**Expected underfitting effect:** May slightly increase underfitting risk for the most constrained settings, but the inclusion of moderate depths and less restrictive configurations should preserve enough capacity to fit the signal.

**Cost estimate:** Moderate; 3 x 5 x 4 x 4 x 4 x 2 = 1920 raw combinations, so this grid is too large for exhaustive search and should be paired with randomized search or reduced by sampling. If exhaustive search is required, prune to a smaller subset of values to stay near 120 candidates.
