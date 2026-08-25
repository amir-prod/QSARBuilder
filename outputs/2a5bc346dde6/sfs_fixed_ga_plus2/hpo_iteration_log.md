# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.710) is much higher than CV R² (0.392); gap=0.319 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid for baseline correction, emphasizing variance reduction and simpler trees while keeping total combinations under the candidate budget..
Candidates searched: 96.
Best CV R²: 0.40.
Train-CV R² gap: 0.34.
Assessment: overfit.

HPO round 2/3: Localized regularization-focused refinement around the prior best configuration, emphasizing reduced tree complexity and slightly smaller bootstrap sample fractions while preserving the strongest-performing neighborhood from top candidates and keeping the grid under the candidate budget..
Candidates searched: 96.
Best CV R²: 0.40.
Train-CV R² gap: 0.34.
Assessment: overfit.

HPO round 3/3: Localized regularization-focused refinement around the current best neighborhood, preserving the best-performing settings and testing slightly stronger constraints on depth, split size, and leaf size while keeping max_features fixed at the strongest value and avoiding previously weaker subsampling regions..
Candidates searched: 54.
Best CV R²: 0.40.
Train-CV R² gap: 0.34.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.