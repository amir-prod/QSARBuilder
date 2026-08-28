# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.736) is much higher than CV R² (0.535); gap=0.201 exceeds 0.15.

HPO round 1/3: Grid search with a focus on regularization and compactness due to small dataset size..
Candidates searched: 72.
Best CV R²: 0.54.
Train-CV R² gap: 0.19.
Assessment: overfit.

HPO round 2/3: Grid search with a focus on regularization adjustments to mitigate overfitting..
Candidates searched: 90.
Best CV R²: 0.54.
Train-CV R² gap: 0.19.
Assessment: overfit.

HPO round 3/3: Grid search with a focus on regularization adjustments to mitigate overfitting..
Candidates searched: 36.
Best CV R²: 0.54.
Train-CV R² gap: 0.19.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.