# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid for a very small dataset, prioritizing variance reduction and overfitting control over broad exploration.

Round 1 uses the baseline assessment only because there is no prior HPO feedback yet. The baseline is unstable and overfit (high train-CV R2 gap and high CV R2 std), so the grid emphasizes stronger regularization and reduced variance. Given the very small dataset size (n_train=20) and only 3 features, I kept the grid compact and biased toward shallower trees, larger split/leaf sizes, and feature subsampling settings that can improve stability while staying within the candidate budget.

**Expected overfitting effect:** Should reduce overfitting by limiting tree complexity, requiring larger leaves/splits, and using moderate feature subsampling; higher n_estimators may also reduce fold-to-fold variance.

**Expected underfitting effect:** There is some risk of mild underfitting from the stronger regularization, but inclusion of max_depth=null and moderate split/leaf settings preserves some flexibility.

**Cost estimate:** Low; 32 total combinations, feasible even with cross-validation.
