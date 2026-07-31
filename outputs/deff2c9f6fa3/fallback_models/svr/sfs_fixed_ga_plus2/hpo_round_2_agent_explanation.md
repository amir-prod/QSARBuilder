# HPO Round 2 Agent Grid Proposal

**Strategy:** Bias the search toward simpler SVR settings to reduce overfitting: emphasize lower C, moderate epsilon, and include linear kernel candidates. Use a compact grid with a few gamma values for rbf/poly and a limited polynomial option to test whether reduced flexibility improves generalization. Total combinations are kept near or below the requested limit.

The previous best SVR configuration remained overfit with a large train-CV gap. To reduce model flexibility while still exploring useful nonlinear behavior, the grid shifts toward smaller C values, slightly larger epsilon values, and includes linear and polynomial kernels alongside rbf. Gamma is kept mostly in the lower-to-mid range for rbf/poly to avoid overly complex fits, while still retaining scale/auto as baselines.

**Expected overfitting effect:** Should reduce overfitting by lowering effective model capacity through smaller C, larger epsilon, and simpler kernels; linear kernel is especially likely to narrow the train-CV gap.

**Expected underfitting effect:** May slightly increase underfitting risk for the smallest C and largest epsilon settings, but the inclusion of moderate C values and rbf/poly options should preserve enough flexibility.

**Cost estimate:** 60 candidates total (5 x 4 x 3), which is within the 120-candidate limit and moderate for a small dataset.
