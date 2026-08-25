# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid emphasizing shallow-to-moderate tree depth, larger leaf/split thresholds, and bootstrap subsampling; includes a small contrast set with bootstrap=false and max_features=1.0 to avoid over-constraining the search.

Round 1 uses the baseline only because there is no prior HPO feedback. The baseline shows severe overfitting (train R² 0.902 vs CV R² 0.430; gap 0.472). With a small training set (n_train=153) and only 2 descriptors, I biased the grid toward stronger regularization and a compact search space: shallower trees, larger min_samples_split/min_samples_leaf, restricted feature subsampling choices relevant to 2 features, and bootstrap with subsampling to reduce variance. The grid is kept compact to stay near the candidate limit while focusing on reducing overfit first.

**Expected overfitting effect:** Should reduce overfitting by limiting tree complexity and increasing minimum split/leaf sizes; bootstrap with max_samples < 1.0 should further lower variance and shrink the train-CV gap.

**Expected underfitting effect:** There is some risk of underfitting from the stronger regularization, but inclusion of max_depth=null/8, min_samples_leaf=2, and max_features=1.0 preserves some model flexibility.

**Cost estimate:** Moderate. Raw Cartesian size is 576 combinations, but max_samples is only applicable when bootstrap=true; valid combinations are 384. This exceeds the requested maximum and should be pruned during execution, prioritizing bootstrap=true settings with subsampling and dropping lower-priority contrasts first.
