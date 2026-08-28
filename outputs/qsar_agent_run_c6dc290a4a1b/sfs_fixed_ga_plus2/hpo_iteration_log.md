# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.858) is much higher than CV R² (0.491); gap=0.367 exceeds 0.15.

HPO round 1/3: Grid search with a focus on regularization parameters to combat overfitting..
Candidates searched: 64.
Best CV R²: 0.54.
Train-CV R² gap: 0.40.
Assessment: overfit.

HPO round 2/3: Grid search with a focus on regularization parameters to combat overfitting..
Candidates searched: 36.
Best CV R²: 0.32.
Train-CV R² gap: 0.08.
Assessment: underfit.

HPO round 3/3: Grid search with a focus on increasing model capacity to combat underfitting..
Candidates searched: 54.
Best CV R²: 0.31.
Train-CV R² gap: 0.08.
Assessment: underfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.