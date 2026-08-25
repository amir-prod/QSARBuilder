# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Both training R² (0.377) and CV R² (0.327) are low. The model lacks capacity or informative descriptors.

HPO round 1/3: Compact exhaustive grid over all valid latent dimensions, both scale settings, and standard convergence limits to address baseline underfit without exceeding candidate budget..
Candidates searched: 24.
Best CV R²: 0.33.
Train-CV R² gap: 0.05.
Assessment: underfit.

HPO round 2/3: Tight local confirmation grid around the prior best, emphasizing the maximum valid capacity (n_components=3) to address underfit, while retaining immediate neighboring latent dimensions and both scale settings because prior top candidates were tied. max_iter is restricted to the best and nearest standard value since larger values showed no benefit in the previous round..
Candidates searched: 12.
Best CV R²: 0.33.
Train-CV R² gap: 0.05.
Assessment: underfit.

HPO round 3/3: Very tight local refinement around the maximum-capacity valid solution, preserving tied top candidates from the previous round and adding only small convergence-limit checks. Because underfitting persists but n_components cannot exceed 3 with only 3 features, the grid focuses on confirming whether scaling and minor max_iter adjustments can extract any remaining benefit while avoiding a restart of the search..
Candidates searched: 16.
Best CV R²: 0.33.
Train-CV R² gap: 0.05.
Assessment: underfit.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.