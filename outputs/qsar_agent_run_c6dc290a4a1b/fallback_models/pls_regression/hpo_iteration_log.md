# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Both training R² (0.281) and CV R² (0.239) are low. The model lacks capacity or informative descriptors.

HPO round 1/3: Grid search with a focus on compact parameter ranges to address underfitting..
Candidates searched: 4.
Best CV R²: 0.24.
Train-CV R² gap: 0.04.
Assessment: underfit.

HPO round 2/3: Grid search with a focus on compact parameter ranges to address underfitting..
Candidates searched: 2.
Best CV R²: 0.24.
Train-CV R² gap: 0.04.
Assessment: underfit.

HPO round 3/3: Grid search with a focus on increasing model capacity to address underfitting..
Candidates searched: 2.
Best CV R²: 0.24.
Train-CV R² gap: 0.04.
Assessment: underfit.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.