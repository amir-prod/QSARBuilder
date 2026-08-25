# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.983) is much higher than CV R² (0.485); gap=0.498 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid for ExtraTreesRegressor aimed at reducing variance/overfitting on a small dataset; prioritize shallow-to-moderate depth, larger leaf/split thresholds, and reduced max_features, while keeping a small number of estimator counts for stability checks..
Candidates searched: 96.
Best CV R²: 0.42.
Train-CV R² gap: 0.26.
Assessment: overfit.

HPO round 2/3: Local regularization-focused refinement around the round-1 best ExtraTreesRegressor settings: keep n_estimators near 500 for stability, preserve top-performing max_features values, and increase regularization modestly via shallower max_depth plus slightly larger min_samples_split and min_samples_leaf. Include bootstrap as a local check but bias the grid around bootstrap=false..
Candidates searched: 96.
Best CV R²: 0.42.
Train-CV R² gap: 0.26.
Assessment: overfit.

HPO round 3/3: Local overfitting-focused refinement around the round-2 best ExtraTreesRegressor settings. Increase regularization modestly by testing shallower max_depth and slightly larger min_samples_split/min_samples_leaf, while preserving the strongest nearby max_features values from top candidates and keeping n_estimators near 500 for stability. Include bootstrap as a narrow local check without expanding into unrelated regions..
Candidates searched: 96.
Best CV R²: 0.42.
Train-CV R² gap: 0.26.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.