# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.700) is much higher than CV R² (0.522); gap=0.178 exceeds 0.15.

HPO round 1/3: Compact grid with stronger regularization to mitigate overfitting..
Candidates searched: 72.
Best CV R²: 0.53.
Train-CV R² gap: 0.17.
Assessment: overfit.

HPO round 2/3: Refined grid with increased regularization and slight adjustments to epsilon and gamma to address overfitting..
Candidates searched: 54.
Best CV R²: 0.53.
Train-CV R² gap: 0.17.
Assessment: overfit.

HPO round 3/3: Refined grid with increased regularization and adjustments to epsilon and gamma to mitigate overfitting..
Candidates searched: 96.
Best CV R²: 0.56.
Train-CV R² gap: 0.27.
Assessment: overfit.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.