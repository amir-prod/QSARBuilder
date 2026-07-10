# HPO Round 1 Agent Grid Proposal

**Strategy:** Prioritize low-dimensional PLS models first (1-2 components), while still evaluating the full allowable range up to 4 components to detect whether additional latent variables help. Include both scale settings because scaling can materially affect PLS stability. Use a small set of max_iter values to cover convergence robustness without expanding the grid excessively.

Baseline shows moderate overfitting and high CV instability, so the grid focuses on reducing model complexity via n_components while also testing scaling and a small set of iteration limits. With n_train=20 and n_features=4, n_components is constrained to 1-4; keeping the grid compact limits search cost and variance from excessive flexibility.

**Expected overfitting effect:** Lower n_components should reduce overfitting and may improve CV stability; higher n_components may preserve fit but could worsen overfitting. Testing scale may also reduce sensitivity to feature magnitude and improve generalization.

**Expected underfitting effect:** If the baseline is not underfit, the lowest component counts may still be sufficient; higher n_components are included to avoid excessive simplification and to check whether the model benefits from additional latent structure.

**Cost estimate:** 24 total combinations (4 x 2 x 3), which is well below the 120-candidate limit and inexpensive for a small dataset.
