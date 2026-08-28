# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.943) is much higher than CV R² (0.560); gap=0.382 exceeds 0.15.

HPO round 1/3: Grid search with a focus on regularization parameters to mitigate overfitting..
Candidates searched: 96.
Best CV R²: 0.55.
Train-CV R² gap: 0.45.
Assessment: overfit.

HPO round 2/3: Grid search with a focus on regularization parameters to mitigate overfitting..
Candidates searched: 48.
Best CV R²: 0.50.
Train-CV R² gap: 0.28.
Assessment: overfit.

HPO round 3/3: Grid search with a focus on regularization parameters to further mitigate overfitting..
Candidates searched: 8.
Best CV R²: 0.43.
Train-CV R² gap: 0.19.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.