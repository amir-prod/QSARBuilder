# HPO Round 3 Agent Grid Proposal

**Strategy:** Focused local search around the prior best n_components=5, with a compact sweep over lower and near-max component counts plus both scaling options and a limited max_iter set. This targets reduced variance and potential overfitting while keeping total combinations at 48.

Previous rounds consistently favored n_components=5 with scale=false, but the model remains unstable with a notable train-CV gap. With only 21 training samples and 6 features, the main useful search dimension is n_components near the previously best region, while also testing lower complexity to reduce overfitting and scaling to improve stability. max_iter is included at a small set of values to guard against convergence sensitivity without expanding the grid excessively.

**Expected overfitting effect:** Lower n_components and scale=true may reduce overfitting and improve fold-to-fold stability; higher n_components may preserve current performance but could retain the existing gap.

**Expected underfitting effect:** Very low n_components (1-2) may increase underfitting risk, especially if the signal requires multiple latent factors; n_components 4-6 should help avoid underfitting.

**Cost estimate:** Low to moderate: 48 total candidates, each inexpensive for n_train=21 and n_features=6.
