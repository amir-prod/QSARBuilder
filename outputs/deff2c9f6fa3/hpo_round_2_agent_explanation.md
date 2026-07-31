# HPO Round 2 Agent Grid Proposal

**Strategy:** Bias the search toward variance reduction: use shallower trees, larger leaf and split constraints, and bootstrap with optional max_samples to stabilize predictions. Keep a small number of settings for max_features and criterion to preserve some flexibility, but avoid an overly broad grid. Total combinations are kept near the requested limit.

The baseline shows severe overfitting and high fold-to-fold instability on a very small dataset (21 samples, 2 features). The next search should prioritize stronger regularization and simpler trees while still allowing a few moderately flexible settings to avoid underfitting. I therefore narrow the grid toward larger min_samples_leaf/min_samples_split, limited depths, and a mix of bootstrap settings with subsampling to reduce variance.

**Expected overfitting effect:** Expected to reduce overfitting substantially by constraining tree complexity and increasing averaging/stability, especially with larger min_samples_leaf/min_samples_split and shallower max_depth. Bootstrap with max_samples below 1.0 should further reduce variance.

**Expected underfitting effect:** Some configurations may underfit, particularly very shallow trees with large leaf sizes. Including a few moderate-depth and lower-regularization settings should help identify a balance without reverting to the highly flexible baseline.

**Cost estimate:** Moderate. The grid is intentionally compact for a small dataset, but the combination count is still around the upper target range; training cost remains manageable due to low n_samples and only 2 features.
