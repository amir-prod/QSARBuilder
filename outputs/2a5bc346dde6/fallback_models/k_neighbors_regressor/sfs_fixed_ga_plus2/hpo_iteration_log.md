# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.563) is much higher than CV R² (0.315); gap=0.247 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid centered on larger n_neighbors values, with standard distance metrics and both weighting schemes..
Candidates searched: 120.
Best CV R²: 0.38.
Train-CV R² gap: 0.62.
Assessment: unstable.

HPO round 2/3: Local regularization-focused refinement around the prior best, emphasizing slightly larger n_neighbors and both weighting schemes while retaining the promising metric/p settings seen in top candidates..
Candidates searched: 96.
Best CV R²: 0.38.
Train-CV R² gap: 0.62.
Assessment: unstable.

HPO round 3/3: Local instability-focused refinement centered on k=5, biased toward larger neighborhood sizes and both weighting schemes, while preserving the previously tied metric/p variants from top candidates..
Candidates searched: 120.
Best CV R²: 0.38.
Train-CV R² gap: 0.62.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.