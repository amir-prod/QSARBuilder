# HPO Round 2 Agent Grid Proposal

**Strategy:** Use a compact grid centered on regularization: increase min_samples_leaf and min_samples_split, limit max_depth, test feature subsampling, and compare bootstrap with and without max_samples. Keep n_estimators moderate to high for stability, but avoid an overly large grid. Exclude extreme high-capacity settings.

The baseline and prior round indicate severe overfitting and high fold sensitivity on a very small dataset (21 samples, 4 features). The next search should prioritize stronger regularization, smaller trees, and more conservative sampling while keeping a few moderate-capacity settings to avoid underfitting.

**Expected overfitting effect:** Should reduce overfitting materially by constraining tree depth, increasing leaf and split minimums, and using bootstrap subsampling. Feature subsampling may further reduce variance and fold sensitivity.

**Expected underfitting effect:** Could slightly increase underfitting risk, especially with very shallow trees or larger min_samples_leaf, but the grid retains moderate-depth and lower-regularization options to preserve fit capacity.

**Cost estimate:** 96 candidates total; moderate cost for a small dataset, but absolute_error will be slower than squared_error. Overall feasible within the stated limit.
