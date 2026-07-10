# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.960) is much higher than CV R² (0.613); gap=0.347 exceeds 0.15.

HPO round 1/3: Bias the search toward simpler forests first: shallow max_depth, larger min_samples_leaf/min_samples_split, and reduced max_features. Include bootstrap with max_samples to further decorrelate trees and reduce variance. Keep a small set of higher-capacity values to avoid excessive underfitting..
Candidates searched: 64.
Best CV R²: 0.66.
Train-CV R² gap: 0.30.
Assessment: overfit.

HPO round 2/3: Constrain tree complexity aggressively and explore a compact grid centered on regularization. Use bootstrap=true only, with max_samples below 1.0 to increase randomness and reduce overfitting. Prefer squared_error and absolute_error; include log2/sqrt and low fractional max_features. Keep the grid size well below 120 combinations..
Candidates searched: 64.
Best CV R²: 0.69.
Train-CV R² gap: 0.26.
Assessment: overfit.

HPO round 3/3: Constrain tree complexity aggressively: test shallow to moderate depths, larger leaf sizes, larger split thresholds, and feature subsampling. Keep bootstrap on and include max_samples below 1.0 to further reduce variance. Use a compact grid near the 120-candidate limit to explore regularization strength without reintroducing highly flexible configurations..
Candidates searched: 64.
Best CV R²: 0.64.
Train-CV R² gap: 0.21.
Assessment: unstable.

Final selected model: hpo_round_3.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.