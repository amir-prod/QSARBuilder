# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.197 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Prioritize low-dimensional PLS models first (1-2 components), while still evaluating the full allowable range up to 4 components to detect whether additional latent variables help. Include both scale settings because scaling can materially affect PLS stability. Use a small set of max_iter values to cover convergence robustness without expanding the grid excessively..
Candidates searched: 24.
Best CV R²: 0.67.
Train-CV R² gap: 0.21.
Assessment: unstable.

HPO round 2/3: Use a compact grid centered on lower-to-mid latent dimensionality to test whether fewer components reduce variance and improve stability, while retaining the previously best setting as a reference. Include both scaling options and all allowed max_iter values to check sensitivity to preprocessing and convergence without exceeding the candidate budget..
Candidates searched: 32.
Best CV R²: 0.67.
Train-CV R² gap: 0.21.
Assessment: unstable.

HPO round 3/3: Prioritize simpler PLS structures by sweeping n_components from 1 to 4, test both scale settings, and include a small max_iter set focused on convergence robustness. Keep the grid small enough to stay well below the candidate limit while targeting reduced overfitting and improved fold stability..
Candidates searched: 24.
Best CV R²: 0.67.
Train-CV R² gap: 0.21.
Assessment: unstable.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.