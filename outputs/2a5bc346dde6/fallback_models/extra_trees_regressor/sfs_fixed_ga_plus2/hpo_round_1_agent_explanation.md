# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid for ExtraTreesRegressor aimed at reducing variance/overfitting on a small dataset; prioritize shallow-to-moderate depth, larger leaf/split thresholds, and reduced max_features, while keeping a small number of estimator counts for stability checks.

Round 1 uses the baseline only because there is no prior HPO feedback. The baseline shows severe overfitting (train R² 0.983 vs CV R² 0.485; large gap 0.498), so the grid is biased toward stronger regularization. Given the small dataset size (n_train=153) and only 4 descriptors, I kept the grid compact and emphasized shallower trees, larger min_samples_split/min_samples_leaf, and subspace feature sampling below 1.0 to reduce variance while still allowing a few moderate-capacity settings.

**Expected overfitting effect:** Should materially reduce overfitting versus baseline by limiting tree depth, requiring larger splits/leaves, and reducing feature usage per split; bootstrap may further lower variance for some settings.

**Expected underfitting effect:** Some combinations, especially max_depth=3 with min_samples_leaf=8 and min_samples_split=24, may underfit; the inclusion of moderate settings (max_depth=8 or 12, smaller leaf sizes) preserves capacity to avoid excessive bias.

**Cost estimate:** Moderate: 2 x 4 x 3 x 3 x 3 x 2 = 432 raw combinations, which exceeds the requested maximum candidate count of 120 if run as a full Cartesian grid.
