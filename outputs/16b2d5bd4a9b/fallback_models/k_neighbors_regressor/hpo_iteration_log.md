# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.506 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Compact regularization-focused grid for small-sample KNN. Emphasize higher n_neighbors values to smooth predictions and improve stability, use uniform weights only, and split metric/p into valid paired subspaces to avoid redundant or invalid combinations while keeping total candidates below the limit..
Candidates searched: 48.
Best CV R²: 0.40.
Train-CV R² gap: 0.10.
Assessment: unstable.

HPO round 2/3: Compact local refinement around n_neighbors=7 with a stability-focused bias toward modestly larger neighborhoods; keep uniform weights only, preserve the best and tied nearby metric/p options, and avoid broad exploration because n_train=20 favors conservative regularization..
Candidates searched: 36.
Best CV R²: 0.45.
Train-CV R² gap: 0.06.
Assessment: unstable.

HPO round 3/3: Compact stability-focused local refinement centered at n_neighbors=8, with a modest bias toward slightly larger n_neighbors values for stronger smoothing on a 20-sample dataset; retain uniform weights only and preserve the previously tied metric/p neighborhood while avoiding redundant broad search..
Candidates searched: 36.
Best CV R²: 0.45.
Train-CV R² gap: 0.06.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.