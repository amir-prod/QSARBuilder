# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.259 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Use a small, structured grid over neighborhood size, weighting, distance metric, and Minkowski power. Prioritize moderate-to-larger n_neighbors values to improve stability, while retaining a few smaller values to avoid missing a better bias-variance tradeoff..
Candidates searched: 120.
Best CV R²: 0.73.
Train-CV R² gap: 0.14.
Assessment: unstable.

HPO round 2/3: Use a focused grid around the prior best configuration (k=5, minkowski, p=2, uniform) and expand to nearby neighbor counts plus alternative metrics/weights to reduce fold sensitivity. Keep total combinations at 72..
Candidates searched: 72.
Best CV R²: 0.73.
Train-CV R² gap: 0.14.
Assessment: unstable.

HPO round 3/3: Use a compact grid centered on the previously best neighborhood size, expanding to nearby n_neighbors values and testing both Minkowski-compatible distance settings plus the direct Euclidean/Manhattan metrics. Keep the grid within the candidate budget while emphasizing configurations that may smooth predictions and improve robustness..
Candidates searched: 60.
Best CV R²: 0.73.
Train-CV R² gap: 0.14.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.