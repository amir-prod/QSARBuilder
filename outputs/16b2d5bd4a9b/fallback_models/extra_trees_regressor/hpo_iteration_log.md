# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.395 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Compact regularization-focused grid for a very small dataset, emphasizing reduced tree complexity and variance control while staying well below the candidate limit..
Candidates searched: 48.
Best CV R²: 0.37.
Train-CV R² gap: 0.22.
Assessment: unstable.

HPO round 2/3: Local refinement around the round-1 best candidate with a variance-reduction/regularization emphasis for a tiny 1-feature dataset; preserve the strongest prior settings, probe only nearby simpler alternatives, and avoid broad exploration..
Candidates searched: 48.
Best CV R²: 0.38.
Train-CV R² gap: 0.21.
Assessment: unstable.

HPO round 3/3: Tight local refinement around the round-2 best candidate with a variance-reduction emphasis: preserve the best-performing no-bootstrap/full-feature setting, probe only adjacent tree-count and mild regularization values, and avoid broader exploration that is unlikely to be reliable for a 20-sample, 1-feature dataset..
Candidates searched: 24.
Best CV R²: 0.38.
Train-CV R² gap: 0.21.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.