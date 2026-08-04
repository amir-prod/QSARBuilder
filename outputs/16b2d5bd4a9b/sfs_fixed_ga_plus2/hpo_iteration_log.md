# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.965) is much higher than CV R² (0.766); gap=0.199 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid for a very small dataset, centered on reducing tree complexity and variance while keeping total combinations under the candidate budget..
Candidates searched: 32.
Best CV R²: 0.66.
Train-CV R² gap: 0.17.
Assessment: unstable.

HPO round 2/3: Compact local refinement centered on the round-1 best configuration, with small regularization increases to reduce fold sensitivity on a very small dataset. Drop clearly worse regions from prior top candidates and keep total combinations well below the 50-candidate budget..
Candidates searched: 18.
Best CV R²: 0.66.
Train-CV R² gap: 0.17.
Assessment: unstable.

HPO round 3/3: Compact local refinement around the round-2 best configuration, targeting instability on a very small dataset by testing only nearby, slightly more regularized settings while preserving the strongest prior region. Keep bootstrap=true, criterion=squared_error, max_features=0.7, and max_samples=1.0 fixed; probe max_depth just below/at the best, min_samples_leaf just above the best, min_samples_split at/just above the best, and a modest increase in n_estimators for averaging stability..
Candidates searched: 16.
Best CV R²: 0.66.
Train-CV R² gap: 0.17.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.