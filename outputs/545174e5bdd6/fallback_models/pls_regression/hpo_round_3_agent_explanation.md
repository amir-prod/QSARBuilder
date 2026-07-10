# HPO Round 3 Agent Grid Proposal

**Strategy:** Prioritize simpler PLS structures by sweeping n_components from 1 to 4, test both scale settings, and include a small max_iter set focused on convergence robustness. Keep the grid small enough to stay well below the candidate limit while targeting reduced overfitting and improved fold stability.

Previous rounds repeatedly selected n_components=4 with scaling enabled, but the model remains unstable with a notable train-CV gap. With only 20 training samples and 4 features, the most useful search is a compact grid that explores lower latent dimensionality to reduce variance, while also checking whether disabling scaling changes stability. max_iter is included at a small set of values to guard against convergence sensitivity without expanding the grid excessively.

**Expected overfitting effect:** Lower n_components should reduce model flexibility and may shrink the train-CV gap; disabling scaling may also improve stability if feature scaling is amplifying fold sensitivity.

**Expected underfitting effect:** If the current model is already near the bias-variance balance, n_components=1 or 2 could underfit; the inclusion of 3 and 4 preserves the ability to recover higher-capacity settings if needed.

**Cost estimate:** 24 combinations total (4 x 2 x 3), which is low and comfortably within the 120-candidate budget.
