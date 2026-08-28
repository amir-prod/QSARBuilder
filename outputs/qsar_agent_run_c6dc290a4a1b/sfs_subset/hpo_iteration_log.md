# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.925) is much higher than CV R² (0.538); gap=0.386 exceeds 0.15.

HPO round 1/3: Grid search with a focus on regularization parameters to combat overfitting..
Candidates searched: 64.
Best CV R²: 0.56.
Train-CV R² gap: 0.38.
Assessment: overfit.

HPO round 2/3: Grid search with a focus on regularization parameters to combat overfitting..
Candidates searched: 96.
Best CV R²: 0.56.
Train-CV R² gap: 0.38.
Assessment: overfit.

HPO round 3/3: Grid search with a focus on regularization parameters to combat overfitting..
Candidates searched: 96.
Best CV R²: 0.56.
Train-CV R² gap: 0.38.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.