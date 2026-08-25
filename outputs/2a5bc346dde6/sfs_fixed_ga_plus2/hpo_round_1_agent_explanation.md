# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid for baseline correction, emphasizing variance reduction and simpler trees while keeping total combinations under the candidate budget.

Round 1 uses the baseline only (no prior HPO feedback yet). The baseline shows severe overfitting (train-CV R² gap 0.319) on a small dataset (n_train=153) with very few features (4), so the grid is intentionally compact and biased toward stronger regularization: shallower trees, larger min_samples_split/min_samples_leaf, restricted feature subsampling, and bootstrap with subsampling to reduce variance. The grid avoids overly large/deep forests that would likely worsen memorization on this small training set.

**Expected overfitting effect:** Should reduce overfitting by limiting tree complexity and adding sample/feature subsampling; especially max_depth, larger min_samples_split/min_samples_leaf, and bootstrap with max_samples < 1.0 should shrink the train-CV gap.

**Expected underfitting effect:** There is some risk of underfitting from stronger regularization, but inclusion of max_depth=null/8, min_samples_leaf=2, and max_samples=1.0 preserves a moderate-capacity region of the search space.

**Cost estimate:** Moderate. Total grid size = 2 x 4 x 2 x 3 x 3 x 1 x 3 x 1 = 432 raw combinations, but intended as a compact regularization-focused candidate pool to be pruned/filtered because the max candidate budget is 120.
