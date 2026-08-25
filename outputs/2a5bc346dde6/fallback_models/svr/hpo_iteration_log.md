# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.151 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Compact regularization-focused grid emphasizing stable SVR regimes for a small dataset: lower-to-moderate C, moderate epsilon, limited gamma choices, and inclusion of linear/rbf/poly kernels with restrained complexity..
Candidates searched: 108.
Best CV R²: 0.38.
Train-CV R² gap: 0.13.
Assessment: poor_performance.

HPO round 2/3: Focused local refinement around the best rbf configuration to address poor performance by modestly increasing model capacity while preserving the stable regime observed in round 1. Restrict to rbf kernel, retain scale/auto, add nearby numeric gamma values, expand C slightly upward from 0.3, and probe epsilon just below 0.2..
Candidates searched: 60.
Best CV R²: 0.41.
Train-CV R² gap: 0.20.
Assessment: overfit.

HPO round 3/3: Focused local refinement around the round-2 best rbf configuration with stronger regularization: lower-to-nearby C around 0.7, epsilon centered at 0.1 with modest upward adjustments, and gamma restricted to scale/auto plus a few small nearby numeric values. Exclude linear/poly and avoid larger C or aggressive gamma regions that were less favorable or likely to worsen overfit..
Candidates searched: 40.
Best CV R²: 0.41.
Train-CV R² gap: 0.20.
Assessment: overfit.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.