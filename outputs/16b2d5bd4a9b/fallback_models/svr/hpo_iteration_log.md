# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: CV R² variability is high (std=0.332 > 0.15). The model may be sensitive to training fold composition.

HPO round 1/3: Compact regularization-focused grid for small-sample stabilization; prioritize simpler kernels and conservative C/epsilon/gamma ranges to reduce variance and overfitting while staying under the candidate budget..
Candidates searched: 36.
Best CV R²: 0.26.
Train-CV R² gap: 0.12.
Assessment: unstable.

HPO round 2/3: Local refinement around the linear best region: keep kernel fixed to linear, probe small upward/downward adjustments in C around 3.0 to address underfitting, and test epsilon values centered at 0.2 with slight expansion toward stronger smoothing for stability. Exclude gamma because it is irrelevant for linear SVR and exclude nonlinear kernels because prior top candidates were all linear on this 1-feature, small-sample dataset..
Candidates searched: 5.
Best CV R²: 0.26.
Train-CV R² gap: 0.12.
Assessment: unstable.

HPO round 3/3: Tight local refinement in the linear region only: keep kernel fixed to linear, probe modestly lower and nearby C values around 1.5 to target instability via stronger regularization, and test a narrow epsilon band centered at 0.2 with slight upward extension for smoothing. Exclude gamma because it is irrelevant for linear SVR, and exclude nonlinear kernels because prior best and top candidates were consistently linear on this 1-feature dataset..
Candidates searched: 5.
Best CV R²: 0.26.
Train-CV R² gap: 0.12.
Assessment: unstable.

Final selected model: hpo_round_2.
Warning: No acceptable model found after HPO; selected highest CV R² candidate. Final model may still be overfit, unstable, or poor-performing.