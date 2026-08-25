# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid emphasizing stable SVR regimes for a small dataset: lower-to-moderate C, moderate epsilon, limited gamma choices, and inclusion of linear/rbf/poly kernels with restrained complexity.

Round 1 uses the baseline assessment only. The baseline is unstable and overfit (train-CV R2 gap 0.225, CV R2 std 0.151), so the grid is biased toward stronger regularization and smoother SVR settings. Because the dataset is small (n_train=153) with only 4 features, I kept the grid compact and emphasized lower-to-moderate C, nonzero epsilon, and mostly smoother kernels/gamma choices to reduce variance and improve fold stability while staying within the 120-candidate limit.

**Expected overfitting effect:** Should reduce overfitting risk by prioritizing smaller C values, moderate epsilon, and smoother gamma settings, which can lower model variance and shrink the train-CV gap.

**Expected underfitting effect:** There is some increased underfitting risk at the lowest C and highest epsilon settings, but inclusion of C up to 10 and multiple kernels preserves enough flexibility to recover predictive performance.

**Cost estimate:** Moderate: 180 raw combinations would exceed the target if fully crossed, so this grid should be interpreted as a compact candidate pool to prune or sample from, with priority on linear/rbf and lower C settings first.
