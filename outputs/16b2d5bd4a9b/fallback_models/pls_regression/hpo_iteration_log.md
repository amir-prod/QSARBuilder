# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.270 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Compact baseline stabilization search focused on lower-complexity latent spaces, with full valid n_components coverage and limited iteration settings to control search size..
Candidates searched: 16.
Best CV R²: 0.62.
Train-CV R² gap: 0.10.
Assessment: unstable.

HPO round 2/3: Compact stabilization-focused local search centered on the prior best configuration, biasing toward slightly reduced latent dimensionality and preserving only nearby promising settings from top candidates..
Candidates searched: 9.
Best CV R²: 0.62.
Train-CV R² gap: 0.10.
Assessment: unstable.

HPO round 3/3: Tight stabilization-focused local refinement around the round-2 best configuration, emphasizing nearby lower-capacity and equal-capacity settings, retaining only promising scaled variants, and using a minimal max_iter sweep to confirm convergence stability..
Candidates searched: 9.
Best CV R²: 0.62.
Train-CV R² gap: 0.10.
Assessment: unstable.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.