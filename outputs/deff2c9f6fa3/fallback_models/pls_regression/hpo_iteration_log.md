# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.154 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Grid search over n_components from 1 to 5 (bounded by min(n_features, n_train-1)=6 and kept below the full maximum to limit complexity), both scale settings, and all allowed max_iter values. This yields 5 x 2 x 4 = 40 combinations, well within the 120-candidate limit..
Candidates searched: 40.
Best CV R²: 0.77.
Train-CV R² gap: 0.17.
Assessment: unstable.

HPO round 2/3: Constrain the grid to low-to-moderate latent dimensionality and compare scaled vs unscaled preprocessing. Prioritize n_components from 1 to 5, since 6 components would be the maximum but may be too flexible for this small dataset. Keep max_iter across the allowed range to check whether convergence settings affect stability, while staying well below the 120-candidate limit..
Candidates searched: 40.
Best CV R²: 0.77.
Train-CV R² gap: 0.17.
Assessment: unstable.

HPO round 3/3: Focused local search around the prior best n_components=5, with a compact sweep over lower and near-max component counts plus both scaling options and a limited max_iter set. This targets reduced variance and potential overfitting while keeping total combinations at 48..
Candidates searched: 48.
Best CV R²: 0.77.
Train-CV R² gap: 0.17.
Assessment: unstable.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.