# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² (0.406) is below the minimum acceptable threshold (0.50).

HPO round 1/3: Grid search with a focus on regularization and kernel selection to improve model performance..
Candidates searched: 72.
Best CV R²: 0.42.
Train-CV R² gap: 0.12.
Assessment: poor_performance.

HPO round 2/3: Grid search with a focus on increasing model capacity and exploring epsilon values to improve performance..
Candidates searched: 60.
Best CV R²: 0.42.
Train-CV R² gap: 0.12.
Assessment: poor_performance.

HPO round 3/3: Grid search with a focus on increasing model capacity and exploring a wider range of epsilon values to improve performance..
Candidates searched: 48.
Best CV R²: 0.42.
Train-CV R² gap: 0.12.
Assessment: poor_performance.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.