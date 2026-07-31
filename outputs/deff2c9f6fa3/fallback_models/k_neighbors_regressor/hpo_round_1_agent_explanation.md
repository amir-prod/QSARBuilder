# HPO Round 1 Agent Grid Proposal

**Strategy:** Use a small-to-moderate n_neighbors sweep to reduce variance, compare uniform vs distance weighting, and test Minkowski with p=1/2 alongside explicit Euclidean and Manhattan metrics to capture both L1 and L2 behavior. Total combinations are 30.

Baseline KNeighborsRegressor shows overfitting and high CV instability on a very small dataset, so the grid focuses on regularizing neighborhood behavior by varying neighborhood size, distance weighting, and distance metric choices while keeping the search compact.

**Expected overfitting effect:** Increasing n_neighbors and using distance weighting may reduce sensitivity to individual training points and lower overfitting; metric choice may further stabilize local neighborhoods.

**Expected underfitting effect:** Very large n_neighbors could increase bias and worsen underfitting, but the grid is capped at 10 to limit that risk on the small dataset.

**Cost estimate:** Low to moderate: 30 total candidates, each inexpensive for n_train=21 and n_features=5.
