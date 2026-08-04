# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.293 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Compact regularization-focused grid for a very small dataset, prioritizing stability and reduced overfitting over model complexity..
Candidates searched: 32.
Best CV R²: 0.47.
Train-CV R² gap: 0.28.
Assessment: unstable.

HPO round 2/3: Local, regularization-focused refinement around the prior best and nearest strong candidate, with narrow changes to improve stability while preserving the only competitive region found previously..
Candidates searched: 32.
Best CV R²: 0.49.
Train-CV R² gap: 0.24.
Assessment: unstable.

HPO round 3/3: Very local, stability-focused refinement around the latest best and nearest top candidates, emphasizing mild additional regularization suitable for a 20-sample dataset while preserving the strongest-performing parameter neighborhood..
Candidates searched: 32.
Best CV R²: 0.49.
Train-CV R² gap: 0.24.
Assessment: unstable.

Final selected model: baseline.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.