# HPO Round 3 Agent Grid Proposal

**Strategy:** Variance-reduction focused grid: emphasize shallower trees, larger min_samples_split/min_samples_leaf, and bootstrap with subsampling. Keep max_features limited to sqrt/log2 and moderate fractions to reduce sensitivity. Include a small number of criterion choices, but avoid an overly large Cartesian product. Total combinations are kept near or below 120.

The model is severely overfitting and highly unstable on a very small dataset (21 samples, 2 features). The next search should prioritize stronger regularization and simpler trees while keeping bootstrap enabled to reduce variance. I will avoid very deep trees, very small leaf sizes, and overly large feature fractions. The grid focuses on combinations that can reduce train-CV gap and CV variability, while still allowing a few moderately flexible settings in case the current model is under-regularized.

**Expected overfitting effect:** Should reduce overfitting substantially by constraining tree complexity and increasing ensemble randomness/regularization, especially through larger leaves, larger split thresholds, and bootstrap subsampling.

**Expected underfitting effect:** May slightly increase underfitting risk for the most regularized settings (very shallow trees, large leaves, low max_samples), but the grid still includes moderately flexible options to recover performance if the current model is too constrained.

**Cost estimate:** Moderate. The grid is approximately 3 x 6 x 4 x 4 x 4 x 1 x 3 x 2 = 6912 raw combinations, which is above the target if fully expanded; however, as a search proposal it should be pruned or sampled. If exhaustive search is required, reduce by sampling a subset of max_depth/min_samples settings. Recommended practical evaluation budget: 60-120 candidates.
