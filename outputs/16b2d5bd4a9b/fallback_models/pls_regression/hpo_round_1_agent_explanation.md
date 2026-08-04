# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact baseline stabilization search focused on lower-complexity latent spaces, with full valid n_components coverage and limited iteration settings to control search size.

Round 1 uses the baseline assessment only: the main issue is instability (high CV R² std) rather than clear overfitting or underfitting. With a small training set (n_train=20) and only 4 features, I kept the grid compact and biased toward stronger regularization by emphasizing lower n_components values while still covering the full valid range 1 to min(n_features, n_train-1)=4. I included scale as a key stability-related preprocessing choice and a small set of allowed max_iter values, while keeping total combinations well below the 50-candidate limit.

**Expected overfitting effect:** Lower n_components values should reduce variance and may improve fold-to-fold stability; scale may also improve robustness depending on descriptor magnitudes.

**Expected underfitting effect:** Including the full valid n_components range up to 4 preserves the ability to recover fit if very small component counts underfit.

**Cost estimate:** Low: 16 total combinations.
