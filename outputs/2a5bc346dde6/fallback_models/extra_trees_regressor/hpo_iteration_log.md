# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.965) is much higher than CV R² (0.427); gap=0.538 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid for ExtraTreesRegressor, emphasizing variance reduction and avoiding overly flexible trees on a small dataset..
Candidates searched: 96.
Best CV R²: 0.37.
Train-CV R² gap: 0.29.
Assessment: overfit.

HPO round 2/3: Localized round-2 refinement around the prior best with mild additional regularization to reduce overfitting on a small 153-sample, 2-feature dataset; preserve the best-performing neighborhood and probe only nearby depth/split/leaf/bootstrap choices under a compact grid..
Candidates searched: 96.
Best CV R²: 0.37.
Train-CV R² gap: 0.29.
Assessment: overfit.

HPO round 3/3: Localized round-3 refinement around the round-2 best with targeted extra regularization for a 153-sample, 2-feature dataset: preserve the best-performing non-bootstrapped, max_features=1.0 region while probing only nearby depth/split/leaf settings that should reduce variance without collapsing capacity..
Candidates searched: 72.
Best CV R²: 0.37.
Train-CV R² gap: 0.29.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.