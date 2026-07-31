# HPO Round 3 Agent Grid Proposal

**Strategy:** Use a compact grid centered on the previously best neighborhood size, expanding to nearby n_neighbors values and testing both Minkowski-compatible distance settings plus the direct Euclidean/Manhattan metrics. Keep the grid within the candidate budget while emphasizing configurations that may smooth predictions and improve robustness.

The previous best setting has been stable across rounds but the overall CV R² variability remains high, so this grid focuses on small-to-moderate neighborhood sizes and both distance weighting options to reduce sensitivity to fold composition while still allowing enough flexibility to avoid underfitting.

**Expected overfitting effect:** Slightly reduce overfitting risk by exploring larger neighborhoods and distance weighting, which can smooth local noise and reduce dependence on individual training points.

**Expected underfitting effect:** Should not materially increase underfitting risk; the inclusion of smaller k values and both p norms preserves enough local flexibility to fit the data.

**Cost estimate:** 30 total combinations; low computational cost for n_train=21 and 7 features.
