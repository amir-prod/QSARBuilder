# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.928) is much higher than CV R² (0.520); gap=0.409 exceeds 0.15.

HPO round 1/3: Grid search with a focus on regularization parameters to combat overfitting..
Candidates searched: 64.
Best CV R²: 0.53.
Train-CV R² gap: 0.34.
Assessment: overfit.

HPO round 2/3: Grid search with a focus on regularization parameters to combat overfitting, while exploring the impact of bootstrap..
Candidates searched: 96.
Best CV R²: 0.53.
Train-CV R² gap: 0.34.
Assessment: overfit.

HPO round 3/3: Grid search with a focus on regularization parameters to combat overfitting, while exploring the impact of bootstrap..
Candidates searched: 72.
Best CV R²: 0.54.
Train-CV R² gap: 0.44.
Assessment: overfit.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.