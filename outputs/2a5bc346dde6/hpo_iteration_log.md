# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.900) is much higher than CV R² (0.409); gap=0.491 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid for small-sample baseline correction. Prioritize variance reduction over model flexibility, with bootstrap/subsampling emphasized and only a small set of tree sizes and split constraints explored. Keep total combinations below the candidate budget..
Candidates searched: 64.
Best CV R²: 0.39.
Train-CV R² gap: 0.36.
Assessment: overfit.

HPO round 2/3: Local, regularization-focused refinement around the round-1 best configuration for a small-sample, low-feature QSAR setting. Emphasize variance reduction by tightening tree depth and node-size constraints and by testing moderate-to-strong bootstrap subsampling, while retaining the best-performing criteria and feature-selection settings from top candidates. Keep the grid compact and near the candidate budget..
Candidates searched: 96.
Best CV R²: 0.39.
Train-CV R² gap: 0.31.
Assessment: overfit.

HPO round 3/3: Local refinement around the round-2 best configuration with a stronger regularization bias for a very small, 2-feature QSAR dataset. Tighten max_depth around 3-5, increase min_samples_split and min_samples_leaf modestly above the current best, and test moderate bootstrap subsampling near the previously promising 0.7-1.0 range. Keep the grid compact and focused on the best-performing criterion/feature settings while preserving one nearby robust alternative..
Candidates searched: 108.
Best CV R²: 0.39.
Train-CV R² gap: 0.29.
Assessment: overfit.

Final selected model: hpo_round_2.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.