# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid for a very small dataset, prioritizing stability and reduced overfitting over model complexity.

Round 1 uses the baseline assessment only, since there is no prior-round feedback. The baseline is unstable and severely overfit (large train-CV R² gap and high CV variability). Because the dataset is very small (n_train=20, only 3 features, ~6.7 samples per feature), this grid emphasizes stronger regularization and a compact search: shallower trees, larger min_samples_split/min_samples_leaf, restricted feature subsampling choices, and bootstrap with subsampling to reduce variance and overfitting risk while keeping total combinations near the requested limit.

**Expected overfitting effect:** Should reduce overfitting by limiting tree depth, requiring larger leaf/split sizes, and using bootstrap subsampling.

**Expected underfitting effect:** Some risk of mild underfitting exists because the grid is intentionally regularized for stability on a very small dataset.

**Cost estimate:** Moderate; although the full Cartesian product is larger than the candidate budget, the grid is intentionally compact and suitable for capped sampling up to 50 candidates.
