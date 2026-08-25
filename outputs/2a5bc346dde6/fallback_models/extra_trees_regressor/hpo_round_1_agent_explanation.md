# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid for ExtraTreesRegressor, emphasizing variance reduction and avoiding overly flexible trees on a small dataset.

Round 1 uses the baseline only: the model is severely overfit (train R² 0.965 vs CV R² 0.427, gap 0.538). With a small training set (n=153) and only 2 descriptors, I biased the grid toward stronger regularization and a compact search: shallower trees, larger min_samples_split/min_samples_leaf, and feature subsampling options that can reduce variance. I excluded very deep/unconstrained settings from dominating the grid because the immediate goal is to reduce overfitting rather than maximize training fit.

**Expected overfitting effect:** Should reduce overfitting by constraining tree depth, requiring larger node sizes, and testing bootstrap/feature subsampling to lower variance.

**Expected underfitting effect:** There is some risk of underfitting from the stronger regularization, but inclusion of max_depth=null and moderate leaf/split settings preserves some flexibility.

**Cost estimate:** Moderate; 180 total combinations before CV, which exceeds the requested near/below-120 target.
