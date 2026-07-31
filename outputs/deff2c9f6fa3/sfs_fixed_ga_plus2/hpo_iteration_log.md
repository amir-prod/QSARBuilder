# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.164 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Use a constrained grid centered on regularization-heavy RandomForestRegressor settings: shallow-to-moderate max_depth, higher min_samples_split/min_samples_leaf, limited max_features choices, and bootstrap enabled with optional subsampling. Include a small number of criterion options to test robustness without expanding the grid excessively..
Candidates searched: 64.
Best CV R²: 0.65.
Train-CV R² gap: 0.31.
Assessment: unstable.

HPO round 2/3: Use a compact grid centered on regularization: increase min_samples_leaf and min_samples_split, limit max_depth, test feature subsampling, and compare bootstrap with and without max_samples. Keep n_estimators moderate to high for stability, but avoid an overly large grid. Exclude extreme high-capacity settings..
Candidates searched: 64.
Best CV R²: 0.65.
Train-CV R² gap: 0.31.
Assessment: unstable.

HPO round 3/3: Variance-reduction focused grid search: constrain tree complexity, increase sample requirements per split/leaf, and use bootstrap subsampling with max_samples < 1.0. Include a few null-depth and slightly deeper options to test whether mild complexity helps without reintroducing instability. Keep the grid compact and within the candidate budget..
Candidates searched: 64.
Best CV R²: 0.63.
Train-CV R² gap: 0.28.
Assessment: overfit.

Final selected model: hpo_round_3.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.