# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.203 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Compact regularization-focused grid for a very small dataset, prioritizing variance reduction and overfitting control over broad exploration..
Candidates searched: 48.
Best CV R²: 0.43.
Train-CV R² gap: 0.15.
Assessment: unstable.

HPO round 2/3: Local refinement around the round-1 best configuration with a compact variance-reduction focus: preserve the best-performing neighborhood, test only small regularization increases, and avoid broader or higher-capacity regions that were not supported by the latest results on this very small dataset..
Candidates searched: 36.
Best CV R²: 0.42.
Train-CV R² gap: 0.15.
Assessment: unstable.

HPO round 3/3: Tight local refinement around the round-2 best configuration with a variance-reduction emphasis: preserve the strongest neighborhood, test only small regularization increases, and avoid broad capacity expansion on this very small dataset..
Candidates searched: 48.
Best CV R²: 0.42.
Train-CV R² gap: 0.15.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.