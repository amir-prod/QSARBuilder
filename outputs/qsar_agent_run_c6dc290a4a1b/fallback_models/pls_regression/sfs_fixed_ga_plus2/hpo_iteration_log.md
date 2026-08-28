# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Both training R² (0.377) and CV R² (0.327) are low. The model lacks capacity or informative descriptors.

HPO round 1/3: Grid search with a focus on regularization and compactness due to the small dataset size..
Candidates searched: 8.
Best CV R²: 0.32.
Train-CV R² gap: 0.05.
Assessment: underfit.

HPO round 2/3: Grid search focusing on increasing model capacity and exploring scaling options due to underfitting..
Candidates searched: 8.
Best CV R²: 0.33.
Train-CV R² gap: 0.05.
Assessment: underfit.

HPO round 3/3: Grid search focusing on increasing model capacity by expanding the range of n_components while maintaining the current max_iter and exploring both scaling options..
Candidates searched: 4.
Best CV R²: 0.33.
Train-CV R² gap: 0.05.
Assessment: underfit.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.