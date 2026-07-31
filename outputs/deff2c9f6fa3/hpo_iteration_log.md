# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.339 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Use a variance-reduction focused grid: shallow to moderate tree depths, larger leaf sizes, conservative split thresholds, and limited feature subsampling choices. Include both bootstrap and non-bootstrap settings, but restrict max_samples to bootstrap=true configurations only. Keep criterion choices broad enough to test robustness without exploding the grid size..
Candidates searched: 64.
Best CV R²: 0.51.
Train-CV R² gap: 0.46.
Assessment: unstable.

HPO round 2/3: Bias the search toward variance reduction: use shallower trees, larger leaf and split constraints, and bootstrap with optional max_samples to stabilize predictions. Keep a small number of settings for max_features and criterion to preserve some flexibility, but avoid an overly broad grid. Total combinations are kept near the requested limit..
Candidates searched: 64.
Best CV R²: 0.50.
Train-CV R² gap: 0.50.
Assessment: unstable.

HPO round 3/3: Variance-reduction focused grid: emphasize shallower trees, larger min_samples_split/min_samples_leaf, and bootstrap with subsampling. Keep max_features limited to sqrt/log2 and moderate fractions to reduce sensitivity. Include a small number of criterion choices, but avoid an overly large Cartesian product. Total combinations are kept near or below 120..
Candidates searched: 64.
Best CV R²: 0.49.
Train-CV R² gap: 0.47.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.