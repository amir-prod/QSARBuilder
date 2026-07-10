# HPO Round 2 Agent Grid Proposal

**Strategy:** Bias the search toward smoother, less variance-prone KNN settings by emphasizing larger n_neighbors values and both weighting schemes. Use a compact grid over n_neighbors, weights, p, and metric to probe whether increased neighborhood size and distance-based weighting reduce the train-CV gap and CV instability.

With only 20 training samples and 1 feature, KNN is highly sensitive to neighborhood size and distance weighting. The previous best used a very small neighborhood (k=4) and still showed instability and overfitting, so this grid expands toward larger n_neighbors values while keeping both weights options and both common Minkowski norms. Euclidean and Manhattan are included explicitly for completeness, but the grid is kept compact to stay within the candidate limit.

**Expected overfitting effect:** Increasing n_neighbors should reduce variance and shrink the train-CV gap; distance weighting may still preserve some flexibility, while uniform weighting should be more regularizing.

**Expected underfitting effect:** Larger neighborhoods may increase bias and could worsen underfitting if the true signal is highly local, but the grid retains smaller k values to avoid over-smoothing.

**Cost estimate:** 120 candidates total (10 n_neighbors x 2 weights x 2 p x 3 metrics), which is at the requested upper bound and remains lightweight for KNeighborsRegressor.
