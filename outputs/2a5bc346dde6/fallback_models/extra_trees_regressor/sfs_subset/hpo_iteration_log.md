# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.967) is much higher than CV R² (0.466); gap=0.501 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid for initial overfit correction: prioritize shallow-to-moderate tree depth, larger split/leaf thresholds, and reduced feature subsampling; include both bootstrap settings and a modest range of ensemble sizes..
Candidates searched: 96.
Best CV R²: 0.34.
Train-CV R² gap: 0.16.
Assessment: overfit.

HPO round 2/3: Compact local refinement around the round-1 best configuration with an overfit-correction bias: keep the strongest region from top candidates, test slightly stronger regularization via lower max_depth and higher min_samples_split/min_samples_leaf, retain both equivalent feature-subsetting choices seen near the top, and include bootstrap only as a limited check rather than a broad branch..
Candidates searched: 96.
Best CV R²: 0.34.
Train-CV R² gap: 0.16.
Assessment: overfit.

HPO round 3/3: Local refinement around the round-2 best region with a mild regularization bias: keep the winning n_estimators and feature-subsetting neighborhood, drop clearly weaker broad branches, emphasize bootstrap=false, and probe nearby stronger-regularization settings via max_depth 10-12, min_samples_split 8-12, and min_samples_leaf 2-4 while keeping total combinations below the budget..
Candidates searched: 72.
Best CV R²: 0.34.
Train-CV R² gap: 0.16.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.