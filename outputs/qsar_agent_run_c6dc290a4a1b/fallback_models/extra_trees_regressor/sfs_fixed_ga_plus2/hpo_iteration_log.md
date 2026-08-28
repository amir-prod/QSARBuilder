# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.961) is much higher than CV R² (0.535); gap=0.426 exceeds 0.15.

HPO round 1/3: Grid search with a focus on regularization parameters to mitigate overfitting..
Candidates searched: 96.
Best CV R²: 0.55.
Train-CV R² gap: 0.39.
Assessment: overfit.

HPO round 2/3: Grid search with a focus on regularization parameters to mitigate overfitting, while maintaining sufficient model capacity..
Candidates searched: 108.
Best CV R²: 0.54.
Train-CV R² gap: 0.39.
Assessment: overfit.

HPO round 3/3: Grid search with a focus on regularization parameters to mitigate overfitting, while maintaining sufficient model capacity..
Candidates searched: 36.
Best CV R²: 0.51.
Train-CV R² gap: 0.28.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.