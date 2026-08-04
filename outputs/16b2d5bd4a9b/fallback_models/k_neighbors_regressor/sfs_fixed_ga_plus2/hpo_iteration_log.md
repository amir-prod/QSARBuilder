# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.268 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Compact regularization-focused grid centered on larger n_neighbors values for stability on small-sample data, while still testing both weighting schemes and valid distance formulations without exceeding the candidate budget..
Candidates searched: 24.
Best CV R²: 0.47.
Train-CV R² gap: 0.40.
Assessment: unstable.

HPO round 2/3: Local regularization-focused refinement centered on the previous best neighborhood, prioritizing stability on a 20-sample dataset by testing slightly larger neighbor counts and both weighting schemes, with emphasis on uniform weighting as a likely variance-reducing alternative to the overfit distance-weighted best model..
Candidates searched: 24.
Best CV R²: 0.50.
Train-CV R² gap: 0.37.
Assessment: unstable.

HPO round 3/3: Local regularization-focused refinement around the round-2 best model, preserving the strongest nearby settings while biasing the grid toward variance reduction via uniform weights and modestly larger n_neighbors..
Candidates searched: 16.
Best CV R²: 0.50.
Train-CV R² gap: 0.37.
Assessment: unstable.

Final selected model: hpo_round_2.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.