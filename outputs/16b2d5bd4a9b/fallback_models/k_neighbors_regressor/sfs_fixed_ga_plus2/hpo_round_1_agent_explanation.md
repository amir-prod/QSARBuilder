# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid centered on larger n_neighbors values for stability on small-sample data, while still testing both weighting schemes and valid distance formulations without exceeding the candidate budget.

Round 1 uses the baseline assessment only, since there is no prior HPO feedback. The baseline is unstable with severe overfitting and high CV variance, so the grid is biased toward stronger regularization for a very small dataset (n_train=20, only 3 features, ~6.7 samples per feature). To reduce sensitivity to fold composition, the search emphasizes larger neighborhood sizes and distance smoothing while keeping the grid compact.

**Expected overfitting effect:** Should reduce overfitting by avoiding very small neighborhoods and favoring smoother local averaging on a small training set.

**Expected underfitting effect:** There is some risk of mild underfitting at the largest n_neighbors values, but the grid retains moderate values to balance bias and variance.

**Cost estimate:** Low; 24 total combinations, well below the maximum of 50.
