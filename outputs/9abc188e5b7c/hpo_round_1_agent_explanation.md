# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid centered on variance reduction and stability improvement for a small dataset. Prioritize shallow-to-moderate depth, larger min_samples_split/min_samples_leaf, restricted max_features choices relevant to 2 descriptors, and bootstrap with optional subsampling. Include a small non-bootstrap branch for comparison while avoiding an overly large search.

Round 1 uses the baseline assessment only, since no prior HPO feedback exists yet. The baseline shows severe overfitting and instability (large train-CV R2 gap and high CV variability). With a small training set (n_train=151) and only 2 features, this grid is intentionally compact and regularized: shallower trees, larger split/leaf sizes, and bootstrap-enabled subsampling are emphasized to reduce variance and improve stability.

**Expected overfitting effect:** Should reduce overfitting by limiting tree complexity and increasing minimum node sizes, while bootstrap subsampling (especially max_samples=0.7) should lower variance and improve fold-to-fold stability.

**Expected underfitting effect:** There is some risk of underfitting from stronger regularization, but inclusion of max_depth=null/8 and min_samples_leaf=2 preserves enough flexibility to avoid making the search overly restrictive.

**Cost estimate:** Moderate; 288 raw combinations before any conditional filtering. In execution, max_samples should only be applied with bootstrap=true, and the grid should be pruned or sampled to stay at or below 120 evaluated candidates.
