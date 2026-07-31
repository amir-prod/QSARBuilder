# HPO Round 3 Agent Grid Proposal

**Strategy:** Variance-reduction focused grid search: constrain tree complexity, increase sample requirements per split/leaf, and use bootstrap subsampling with max_samples < 1.0. Include a few null-depth and slightly deeper options to test whether mild complexity helps without reintroducing instability. Keep the grid compact and within the candidate budget.

The baseline and prior rounds indicate severe overfitting and high fold sensitivity on a very small dataset (21 samples, 4 features). The next search should prioritize stronger regularization and simpler trees while still retaining a few moderately flexible settings to avoid underfitting. I therefore narrow the grid toward larger min_samples_leaf/min_samples_split, limited depths, and lower max_features, while keeping bootstrap enabled and using max_samples below 1.0 to reduce variance. A small number of criterion choices are retained, but absolute_error is emphasized for robustness.

**Expected overfitting effect:** Should reduce overfitting substantially by limiting tree depth, increasing leaf/split sizes, and using bootstrap subsampling. Expected to lower train-CV gap and improve stability across folds.

**Expected underfitting effect:** May slightly increase underfitting risk relative to the previous best settings, especially for the most regularized combinations, but the inclusion of moderate depths and multiple n_estimators should preserve enough flexibility.

**Cost estimate:** Moderate. The grid has 3*5*4*4*3*1*3*2 = 4320 raw combinations, but the intended search should be sampled or pruned to stay near the 120-candidate budget; per-fit cost is low to moderate due to small dataset size.
