# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.977) is much higher than CV R² (0.824); gap=0.154 exceeds 0.15.

HPO round 1/3: Use a small, regularization-focused grid spanning linear, RBF, and poly kernels. Prioritize lower C and moderate epsilon values, with gamma restricted to conservative values for nonlinear kernels. Total combinations are kept at 108 to stay within the budget..
Candidates searched: 108.
Best CV R²: 0.84.
Train-CV R² gap: 0.16.
Assessment: overfit.

HPO round 2/3: Bias the search toward simpler SVR settings to reduce overfitting: emphasize lower C, moderate epsilon, and include linear kernel candidates. Use a compact grid with a few gamma values for rbf/poly and a limited polynomial option to test whether reduced flexibility improves generalization. Total combinations are kept near or below the requested limit..
Candidates searched: 108.
Best CV R²: 0.85.
Train-CV R² gap: 0.14.
Assessment: good.

Final selected model: hpo_round_2.