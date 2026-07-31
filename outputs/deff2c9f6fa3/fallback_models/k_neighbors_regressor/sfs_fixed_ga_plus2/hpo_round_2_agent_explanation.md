# HPO Round 2 Agent Grid Proposal

**Strategy:** Use a focused grid around the prior best configuration (k=5, minkowski, p=2, uniform) and expand to nearby neighbor counts plus alternative metrics/weights to reduce fold sensitivity. Keep total combinations at 72.

The baseline is unstable with high CV variance, so the grid emphasizes robustness-oriented KNN settings by exploring smaller and moderate neighborhood sizes, both weighting schemes, and all allowed distance formulations while keeping the search compact.

**Expected overfitting effect:** Distance weighting and slightly larger neighborhoods may reduce variance and soften overfitting, though very small k values may remain high-variance.

**Expected underfitting effect:** Including smaller k values and both p norms helps avoid excessive smoothing; if underfitting appears, lower k and distance weighting should improve fit.

**Cost estimate:** 72 candidates total (6 x 2 x 2 x 3), which is lightweight for this dataset.
