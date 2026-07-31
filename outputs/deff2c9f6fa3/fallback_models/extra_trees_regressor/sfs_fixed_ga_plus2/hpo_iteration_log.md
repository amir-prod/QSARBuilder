# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.155 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Use a compact, regularization-focused grid centered on depth limits, larger leaf/split constraints, and reduced max_features; include bootstrap as an additional variance-reduction option. Keep combinations near the requested cap while covering the main bias-variance tradeoffs..
Candidates searched: 96.
Best CV R²: 0.79.
Train-CV R² gap: 0.21.
Assessment: unstable.

HPO round 2/3: Regularization-focused grid search around the prior best configuration, emphasizing shallower trees and larger leaf sizes to reduce variance. Keep n_estimators in a moderate-high range for stability, but avoid an overly large grid. Include a null max_depth option plus several constrained depths to probe the bias-variance tradeoff..
Candidates searched: 96.
Best CV R²: 0.79.
Train-CV R² gap: 0.21.
Assessment: unstable.

HPO round 3/3: Constrain tree complexity and increase ensemble stability. Search a small set of regularized configurations: shallower depths, larger leaf and split thresholds, bootstrap on/off, and feature subsampling options. Keep n_estimators moderately high but not excessive to reduce variance without expanding the grid too much..
Candidates searched: 96.
Best CV R²: 0.79.
Train-CV R² gap: 0.21.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.