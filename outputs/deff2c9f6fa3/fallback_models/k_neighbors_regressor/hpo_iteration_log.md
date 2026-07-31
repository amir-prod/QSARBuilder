# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.327 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Use a small-to-moderate n_neighbors sweep to reduce variance, compare uniform vs distance weighting, and test Minkowski with p=1/2 alongside explicit Euclidean and Manhattan metrics to capture both L1 and L2 behavior. Total combinations are 30..
Candidates searched: 120.
Best CV R²: 0.80.
Train-CV R² gap: 0.20.
Assessment: overfit.

HPO round 2/3: Prioritize larger n_neighbors values to reduce variance, while retaining a limited set of small and medium k values for coverage. Use a compact grid over weights, p, and metric that stays within the allowed parameter space and keeps total combinations near the target budget..
Candidates searched: 120.
Best CV R²: 0.80.
Train-CV R² gap: 0.20.
Assessment: overfit.

HPO round 3/3: Use a compact, stability-oriented grid centered on larger n_neighbors values, while keeping a small set of low-k candidates for comparison. Explore both uniform and distance weights, and test metric/p combinations that correspond to Minkowski, Euclidean, and Manhattan distances. Total combinations are kept at 108 to stay within the budget..
Candidates searched: 108.
Best CV R²: 0.80.
Train-CV R² gap: 0.20.
Assessment: overfit.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.