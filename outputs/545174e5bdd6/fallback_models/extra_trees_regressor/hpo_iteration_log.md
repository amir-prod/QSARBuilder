# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.709 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Use a compact, regularization-focused grid that explores depth limitation, minimum leaf/split sizes, feature subsampling, and bootstrap. Keep n_estimators moderate-to-high for stability, but avoid an overly large search space given the small sample size and the 120-candidate cap..
Candidates searched: 96.
Best CV R²: 0.30.
Train-CV R² gap: 0.70.
Assessment: unstable.

HPO round 2/3: Regularization-focused grid search emphasizing reduced tree complexity and increased averaging. Explore shallow-to-moderate depths, larger leaf/split constraints, and both bootstrap settings with feature subsampling options to improve stability and reduce train-CV gap. Keep the grid compact and within the candidate budget..
Candidates searched: 96.
Best CV R²: 0.30.
Train-CV R² gap: 0.70.
Assessment: unstable.

HPO round 3/3: Constrain tree growth and increase randomness to reduce variance: test shallow-to-moderate depths, larger leaf and split thresholds, and feature subsampling levels that are more regularizing than the previous best settings. Keep n_estimators in a moderate-high range to stabilize predictions without exploding the search space. Total combinations are capped at 120..
Candidates searched: 96.
Best CV R²: 0.30.
Train-CV R² gap: 0.70.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.