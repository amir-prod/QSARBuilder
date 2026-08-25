# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Both training R² (0.281) and CV R² (0.239) are low. The model lacks capacity or informative descriptors.

HPO round 1/3: Compact baseline-focused grid exploring all valid model capacity and preprocessing options under the one-feature constraint..
Candidates searched: 8.
Best CV R²: 0.24.
Train-CV R² gap: 0.04.
Assessment: underfit.

HPO round 2/3: Exhaust the only valid local neighborhood around the prior best under the one-feature constraint; preserve tied nearby candidates and both scale options, while acknowledging that no true capacity expansion is possible for PLSRegression on this dataset..
Candidates searched: 8.
Best CV R²: 0.24.
Train-CV R² gap: 0.04.
Assessment: underfit.

HPO round 3/3: Re-test the only valid local neighborhood centered on the prior best under the one-feature constraint: preserve best_params, retain both scale settings and allowed max_iter values from promising tied candidates, and acknowledge that further underfit reduction is unlikely via PLS hyperparameters alone..
Candidates searched: 8.
Best CV R²: 0.24.
Train-CV R² gap: 0.04.
Assessment: underfit.

Final selected model: hpo_round_1.
Warning: No acceptable model found after HPO; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.