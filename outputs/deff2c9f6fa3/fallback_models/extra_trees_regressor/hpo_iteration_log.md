# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.185 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Use a compact, regularization-focused grid centered on reducing tree complexity and variance. Prioritize shallow max_depth values, larger min_samples_leaf/min_samples_split, and max_features below 1.0. Include bootstrap as an additional variance-reduction option. Keep n_estimators moderate-to-high to stabilize ensemble estimates without exploding the search space..
Candidates searched: 96.
Best CV R²: 0.71.
Train-CV R² gap: 0.29.
Assessment: unstable.

HPO round 2/3: Regularization-focused grid search emphasizing reduced tree complexity and variance control: test shallow-to-moderate depths, larger split/leaf constraints, both bootstrap settings, and a compact set of feature-subsampling choices. Keep the grid small enough for exhaustive evaluation while covering the most likely overfitting-mitigating configurations..
Candidates searched: 96.
Best CV R²: 0.71.
Train-CV R² gap: 0.29.
Assessment: unstable.

HPO round 3/3: Focus on variance reduction and stability: explore shallow-to-moderate max_depth, larger min_samples_split and min_samples_leaf, and bootstrap enabled/disabled. Keep n_estimators moderately high but not excessive, since tree count mainly affects stability rather than bias. Use a compact grid near the 120-combination limit to cover regularization strength without over-expanding search..
Candidates searched: 96.
Best CV R²: 0.72.
Train-CV R² gap: 0.25.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.