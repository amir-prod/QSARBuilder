# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.902) is much higher than CV R² (0.430); gap=0.472 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid emphasizing shallow-to-moderate tree depth, larger leaf/split thresholds, and bootstrap subsampling; includes a small contrast set with bootstrap=false and max_features=1.0 to avoid over-constraining the search..
Candidates searched: 64.
Best CV R²: 0.40.
Train-CV R² gap: 0.39.
Assessment: overfit.

HPO round 2/3: Compact local refinement around the round-1 best and nearby top candidates, biased toward more regularization: compare bootstrap false vs true with subsampling, keep squared_error as primary criterion with a small absolute_error check, test shallow-to-moderate depth near the prior unbounded optimum, and increase min_samples_leaf/min_samples_split modestly to reduce variance without moving far from the best region..
Candidates searched: 64.
Best CV R²: 0.40.
Train-CV R² gap: 0.39.
Assessment: overfit.

HPO round 3/3: Compact round-3 local refinement around the prior best, explicitly biased toward more regularization to reduce the train-CV gap: keep n_estimators fixed at the prior best, compare bootstrap=false against bootstrap=true with moderate subsampling, retain squared_error as primary with a small absolute_error check, test null vs shallow depth near the best region, and increase min_samples_leaf/min_samples_split slightly above the current optimum..
Candidates searched: 96.
Best CV R²: 0.40.
Train-CV R² gap: 0.39.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.