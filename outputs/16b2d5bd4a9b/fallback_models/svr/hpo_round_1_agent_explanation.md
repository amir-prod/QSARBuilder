# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid for small-sample stabilization; prioritize simpler kernels and conservative C/epsilon/gamma ranges to reduce variance and overfitting while staying under the candidate budget.

Round 1 uses the baseline assessment only, since there is no prior-round feedback. The baseline indicates severe overfitting and instability (large train-CV R² gap and high CV variance). Given the very small dataset size (n_train=20) and only 1 feature, I biased the grid toward stronger regularization and a compact search space: lower-to-moderate C values, larger epsilon values than a purely performance-seeking grid, and a limited kernel set emphasizing simpler/smoother behavior. Gamma is included mainly for rbf/poly with conservative values to reduce sensitivity on small data.

**Expected overfitting effect:** Should reduce overfitting risk by favoring smaller C, moderate epsilon, and conservative gamma values, with linear and low-degree polynomial options included for simpler decision functions.

**Expected underfitting effect:** There is some increased underfitting risk from the stronger regularization bias, but inclusion of C up to 3 and rbf/poly kernels preserves moderate flexibility.

**Cost estimate:** Low to moderate: 36 total combinations, suitable for small-data cross-validation.
