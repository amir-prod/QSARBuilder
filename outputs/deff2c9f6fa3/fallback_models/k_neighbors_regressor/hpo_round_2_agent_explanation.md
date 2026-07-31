# HPO Round 2 Agent Grid Proposal

**Strategy:** Prioritize larger n_neighbors values to reduce variance, while retaining a limited set of small and medium k values for coverage. Use a compact grid over weights, p, and metric that stays within the allowed parameter space and keeps total combinations near the target budget.

The previous best setting used very small neighborhoods and distance weighting, which likely contributed to overfitting and instability on a tiny dataset. This grid broadens neighborhood sizes toward more regularized KNN settings while still including a few small-k options for comparison. It also tests both weighting schemes and both Minkowski norms, plus the equivalent Euclidean/Manhattan metric choices, to identify a more stable configuration without exceeding the candidate budget.

**Expected overfitting effect:** Likely to reduce overfitting by allowing larger neighborhoods and uniform weighting, which smooth predictions and lower sensitivity to individual training points.

**Expected underfitting effect:** May slightly increase underfitting for very large n_neighbors, but the inclusion of smaller k values should preserve the ability to recover a more flexible model if needed.

**Cost estimate:** 120 candidates total (20 n_neighbors x 2 weights x 2 p x 3 metrics), which is at the requested upper bound and feasible for a small dataset.
