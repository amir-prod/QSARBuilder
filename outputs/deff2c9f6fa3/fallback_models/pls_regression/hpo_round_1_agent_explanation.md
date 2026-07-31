# HPO Round 1 Agent Grid Proposal

**Strategy:** Grid search over n_components from 1 to 5 (bounded by min(n_features, n_train-1)=6 and kept below the full maximum to limit complexity), both scale settings, and all allowed max_iter values. This yields 5 x 2 x 4 = 40 combinations, well within the 120-candidate limit.

Use a compact PLSRegression grid that prioritizes lower latent dimensionality to reduce overfitting and instability, while still testing whether moderate component counts improve fit. Include scaling and a small set of iteration limits to check solver convergence sensitivity without exceeding the candidate budget.

**Expected overfitting effect:** Likely to reduce overfitting relative to higher-component settings by favoring simpler latent structures; scaling may further stabilize coefficient estimation.

**Expected underfitting effect:** May slightly increase underfitting risk at the lowest component counts, but the inclusion of up to 5 components should allow recovery if the baseline is too constrained.

**Cost estimate:** Low to moderate: 40 total fits per CV split configuration, with inexpensive PLSRegression training on a small dataset.
