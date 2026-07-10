# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.954 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Prioritize variance reduction by sweeping n_neighbors across small-to-moderately-large values, combined with weights and metric choices that can change local sensitivity. Keep p aligned with Minkowski-compatible metrics and include Euclidean/Manhattan explicitly to cover the main distance behaviors without exceeding the candidate budget..
Candidates searched: 120.
Best CV R²: 0.35.
Train-CV R² gap: 0.38.
Assessment: unstable.

HPO round 2/3: Bias the search toward smoother, less variance-prone KNN settings by emphasizing larger n_neighbors values and both weighting schemes. Use a compact grid over n_neighbors, weights, p, and metric to probe whether increased neighborhood size and distance-based weighting reduce the train-CV gap and CV instability..
Candidates searched: 120.
Best CV R²: 0.28.
Train-CV R² gap: 0.11.
Assessment: unstable.

HPO round 3/3: Use a focused but broader grid emphasizing larger n_neighbors to reduce overfitting and fold sensitivity, while retaining a few smaller values to avoid missing locally optimal settings. Test both uniform and distance weighting, and compare p=1 vs p=2 across minkowski/euclidean/manhattan. Total combinations: 10 x 2 x 2 x 3 = 120..
Candidates searched: 120.
Best CV R²: 0.28.
Train-CV R² gap: 0.11.
Assessment: unstable.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.