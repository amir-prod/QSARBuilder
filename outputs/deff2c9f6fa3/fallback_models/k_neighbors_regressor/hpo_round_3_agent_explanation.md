# HPO Round 3 Agent Grid Proposal

**Strategy:** Use a compact, stability-oriented grid centered on larger n_neighbors values, while keeping a small set of low-k candidates for comparison. Explore both uniform and distance weights, and test metric/p combinations that correspond to Minkowski, Euclidean, and Manhattan distances. Total combinations are kept at 108 to stay within the budget.

Previous rounds repeatedly selected a very small neighborhood (n_neighbors=2) with distance weighting and Minkowski p=1, which likely contributes to overfitting and instability on this small dataset. The new grid broadens neighborhood size toward smoother models while still retaining a few low-k options, and includes both weighting schemes and all allowed distance formulations to test whether a less sensitive configuration improves CV stability.

**Expected overfitting effect:** Likely to reduce overfitting by allowing larger neighborhoods and uniform weighting, which should smooth predictions and reduce sensitivity to individual training points.

**Expected underfitting effect:** May slightly increase underfitting for larger n_neighbors and uniform weights, but the inclusion of smaller k values should preserve the ability to fit local structure if needed.

**Cost estimate:** 108 candidate configurations total (9 x 2 x 3 x 2), which is moderate and within the stated limit for a small dataset.
