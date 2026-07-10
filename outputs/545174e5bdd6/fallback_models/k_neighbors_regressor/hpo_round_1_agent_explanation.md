# HPO Round 1 Agent Grid Proposal

**Strategy:** Prioritize variance reduction by sweeping n_neighbors across small-to-moderately-large values, combined with weights and metric choices that can change local sensitivity. Keep p aligned with Minkowski-compatible metrics and include Euclidean/Manhattan explicitly to cover the main distance behaviors without exceeding the candidate budget.

Use a compact KNeighborsRegressor grid that explores smoothing via larger neighborhoods and both distance/uniform weighting, while keeping metric choices limited to the allowed set. With only 20 training samples and severe overfitting/instability, the search should emphasize higher n_neighbors values to reduce variance and test whether simpler distance definitions improve robustness.

**Expected overfitting effect:** Increasing n_neighbors and testing uniform weighting should reduce model variance and likely narrow the train-CV gap, improving stability across folds.

**Expected underfitting effect:** Very large n_neighbors or uniform weighting may oversmooth the signal and increase underfitting risk, so the grid retains smaller neighborhoods and distance weighting to preserve flexibility.

**Cost estimate:** 90 combinations total (15 n_neighbors x 2 weights x 3 metrics); low cost for n_train=20 and 1 feature.
