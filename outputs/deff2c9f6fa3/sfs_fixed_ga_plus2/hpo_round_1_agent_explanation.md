# HPO Round 1 Agent Grid Proposal

**Strategy:** Use a constrained grid centered on regularization-heavy RandomForestRegressor settings: shallow-to-moderate max_depth, higher min_samples_split/min_samples_leaf, limited max_features choices, and bootstrap enabled with optional subsampling. Include a small number of criterion options to test robustness without expanding the grid excessively.

The baseline shows severe overfitting and high fold-to-fold variability on a very small dataset (21 samples, 4 features). The grid therefore emphasizes stronger regularization, shallower trees, larger leaf sizes, and a mix of bootstrap settings while keeping the search space compact.

**Expected overfitting effect:** Expected to reduce overfitting substantially by limiting tree complexity and increasing leaf/split constraints; bootstrap subsampling may further improve generalization and reduce variance.

**Expected underfitting effect:** Some configurations may underfit, especially the shallowest trees with large leaf sizes, but the grid retains moderate-depth and lower-regularization options to preserve fit capacity.

**Cost estimate:** Grid size: 3 x 5 x 4 x 4 x 4 x 1 x 3 x 2 = 5760 combinations before any pruning; this exceeds the target, so it should be treated as a candidate-value grid for randomized or staged search rather than exhaustive grid search. If exhaustive search is required, reduce dimensions or use a smaller subset to stay near <=120 combinations.
