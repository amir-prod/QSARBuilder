# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.218 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Grid search with a focus on regularization and stability..
Candidates searched: 20.
Best CV R²: 0.40.
Train-CV R² gap: 0.60.
Assessment: unstable.

HPO round 2/3: Grid search with a focus on reducing overfitting and improving stability..
Candidates searched: 20.
Best CV R²: 0.41.
Train-CV R² gap: 0.59.
Assessment: unstable.

HPO round 3/3: Grid search with a focus on reducing overfitting and improving stability..
Candidates searched: 20.
Best CV R²: 0.41.
Train-CV R² gap: 0.59.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.