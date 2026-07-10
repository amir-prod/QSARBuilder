# HPO Round 2 Agent Grid Proposal

**Strategy:** Use a compact grid centered on lower-to-mid latent dimensionality to test whether fewer components reduce variance and improve stability, while retaining the previously best setting as a reference. Include both scaling options and all allowed max_iter values to check sensitivity to preprocessing and convergence without exceeding the candidate budget.

Previous best used the maximum feasible n_components (4) with scale=true and low max_iter, but the model remains unstable with a notable train-CV gap. For a small dataset (n_train=20, n_features=4), the main controllable regularization lever in PLSRegression is reducing n_components; keeping scale as a toggle and expanding max_iter only as a convergence safeguard is appropriate.

**Expected overfitting effect:** Lower n_components should reduce overfitting and may narrow the train-CV gap; scale=false may slightly regularize in some cases but could also hurt performance depending on feature scales.

**Expected underfitting effect:** If the current model is already near-optimal in complexity, reducing n_components to 1 or 2 could increase underfitting and lower CV R².

**Cost estimate:** 32 total combinations; low computational cost for this dataset size.
