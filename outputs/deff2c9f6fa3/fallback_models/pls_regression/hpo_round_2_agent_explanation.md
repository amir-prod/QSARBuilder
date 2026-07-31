# HPO Round 2 Agent Grid Proposal

**Strategy:** Constrain the grid to low-to-moderate latent dimensionality and compare scaled vs unscaled preprocessing. Prioritize n_components from 1 to 5, since 6 components would be the maximum but may be too flexible for this small dataset. Keep max_iter across the allowed range to check whether convergence settings affect stability, while staying well below the 120-candidate limit.

With only 21 training samples and 6 features, the main instability risk is overfitting from too many latent components. The search should therefore emphasize smaller n_components values while still testing whether scaling improves robustness. max_iter is included at all allowed values, but it is unlikely to be the main driver of performance; it is kept broad to avoid missing convergence-related effects.

**Expected overfitting effect:** Likely to reduce overfitting relative to the current best setting by allowing selection of simpler models with fewer components; scaling may also improve numerical stability and reduce sensitivity to fold composition.

**Expected underfitting effect:** Low risk of increasing underfitting because the grid still includes moderately expressive models up to 5 components; if the current model is under-regularized, smaller component counts may slightly lower training fit but improve generalization.

**Cost estimate:** 40 total combinations (5 x 2 x 4), which is comfortably within the 120-candidate budget and inexpensive for PLSRegression on this dataset size.
