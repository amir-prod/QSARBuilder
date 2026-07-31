# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.969) is much higher than CV R² (0.773); gap=0.196 exceeds 0.15.

HPO round 1/3: Use a compact grid centered on regularization and kernel simplicity: prioritize linear and RBF kernels, include a limited polynomial option, and span C from strong to moderate regularization with a few epsilon values. Keep gamma restricted to scale/auto for non-linear kernels plus one small numeric value to test smoother RBF behavior..
Candidates searched: 108.
Best CV R²: 0.82.
Train-CV R² gap: 0.17.
Assessment: overfit.

HPO round 2/3: Use a compact grid centered on stronger regularization and larger epsilon values, while comparing linear and low-complexity RBF/poly settings. Keep gamma restricted to scale/auto for kernelized models and avoid very small epsilon/C values that can encourage overfitting. Total combinations are kept at 108..
Candidates searched: 90.
Best CV R²: 0.82.
Train-CV R² gap: 0.17.
Assessment: overfit.

HPO round 3/3: Constrain the search to a compact grid emphasizing regularization and reduced sensitivity: lower C values, broader epsilon values, and a mix of kernels including linear, rbf, and poly. For rbf/poly, include both scale/auto and a few numeric gamma values spanning weak to moderate locality. Keep the grid under the candidate limit while exploring simpler models first..
Candidates searched: 108.
Best CV R²: 0.81.
Train-CV R² gap: 0.18.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.