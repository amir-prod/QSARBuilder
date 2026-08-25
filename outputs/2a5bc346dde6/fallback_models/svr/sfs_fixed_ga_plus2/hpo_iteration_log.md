# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.647) is much higher than CV R² (0.472); gap=0.175 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid centered on reducing SVR flexibility: prioritize lower C, moderate epsilon, conservative gamma choices, and mainly rbf/linear kernels with a limited poly option..
Candidates searched: 120.
Best CV R²: 0.56.
Train-CV R² gap: 0.28.
Assessment: overfit.

HPO round 2/3: Local regularization-focused refinement around the previous rbf optimum: keep kernel fixed to rbf, retain gamma values near the best region (auto/scale plus smaller numeric gamma), move C downward from 10, and nudge epsilon upward around 0.2 to reduce variance and overfitting..
Candidates searched: 25.
Best CV R²: 0.56.
Train-CV R² gap: 0.28.
Assessment: overfit.

HPO round 3/3: Local regularization-focused refinement around the current rbf optimum: keep kernel fixed to rbf, retain the strongest nearby gamma choices ('auto' and 'scale') plus smaller numeric gamma values, move C downward from 10 with emphasis on 2-7.5, and shift epsilon upward around 0.2 to 0.35 to directly target overfitting while preserving the best local region..
Candidates searched: 25.
Best CV R²: 0.56.
Train-CV R² gap: 0.28.
Assessment: overfit.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.