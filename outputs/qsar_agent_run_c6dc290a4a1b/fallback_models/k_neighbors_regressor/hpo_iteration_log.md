# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.555) is much higher than CV R² (0.337); gap=0.219 exceeds 0.15.

HPO round 1/3: Grid search with compact parameter values to mitigate overfitting..
Candidates searched: 40.
Best CV R²: 0.38.
Train-CV R² gap: 0.62.
Assessment: unstable.

HPO round 2/3: Grid search with a refined parameter space to address instability and overfitting..
Candidates searched: 40.
Best CV R²: 0.38.
Train-CV R² gap: 0.62.
Assessment: unstable.

HPO round 3/3: Grid search with a refined parameter space to address instability and overfitting..
Candidates searched: 24.
Best CV R²: 0.38.
Train-CV R² gap: 0.62.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.