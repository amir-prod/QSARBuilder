# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid for small-sample baseline correction. Prioritize variance reduction over model flexibility, with bootstrap/subsampling emphasized and only a small set of tree sizes and split constraints explored. Keep total combinations below the candidate budget.

Round 1 uses the baseline only, since there is no prior HPO feedback. The dataset is small (n_train=153) with only 2 descriptors, so the grid is intentionally compact and biased toward stronger regularization to address the severe baseline overfitting (train R² 0.900 vs CV R² 0.409, gap 0.491). I reduced tree complexity via shallow-to-moderate max_depth, larger min_samples_split and min_samples_leaf, limited max_features to values appropriate for 2 features, and emphasized bootstrap with subsampling through max_samples to lower variance. The grid still retains a few less-constrained settings to avoid pushing too far into underfitting.

**Expected overfitting effect:** Should reduce overfitting by constraining tree growth, requiring larger leaves/splits, and using bootstrap subsampling. This is expected to lower train R² and improve the train-CV gap relative to baseline.

**Expected underfitting effect:** There is some risk of underfitting from the stronger regularization, but inclusion of max_depth=null, max_depth=8, min_samples_leaf=2, min_samples_split=8, and max_samples=1.0 preserves moderately flexible configurations.

**Cost estimate:** Low to moderate. Total combinations: 2 x 4 x 2 x 2 x 2 x 1 x 3 x 2 = 192 if fully crossed; therefore this grid should be interpreted as a constrained search space to be sampled rather than exhaustively enumerated, or reduced by conditional selection. Given only 2 features and 153 samples, per-fit cost remains low.
