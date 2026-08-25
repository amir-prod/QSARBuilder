# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.555) is much higher than CV R² (0.337); gap=0.219 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid centered on moderate-to-larger neighborhood sizes, with both weighting schemes and valid distance metrics/norms to address baseline overfitting without creating an unnecessarily large search space..
Candidates searched: 108.
Best CV R²: 0.38.
Train-CV R² gap: 0.62.
Assessment: unstable.

HPO round 2/3: Compact, local regularization-focused follow-up grid centered on n_neighbors near the prior best (11), expanded modestly upward to stabilize folds and reduce variance; retain both weighting schemes with emphasis on uniform as a regularizing alternative; keep only the most relevant metric/p combinations near the prior optimum and remove redundant broad regions..
Candidates searched: 64.
Best CV R²: 0.38.
Train-CV R² gap: 0.62.
Assessment: unstable.

HPO round 3/3: Tight, local, regularization-focused follow-up centered on n_neighbors near 11 but expanded upward to 31%-16% of training size (11 to 25) to reduce fold sensitivity; retain both weights with explicit inclusion of uniform for stability; restrict metrics to the most relevant local forms and remove redundant euclidean to avoid wasting budget on equivalent 1D behavior..
Candidates searched: 56.
Best CV R²: 0.38.
Train-CV R² gap: 0.62.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.