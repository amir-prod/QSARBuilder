# Hyperparameter Optimization Log

HPO round 0/3: Baseline model evaluated.
Decision: Training R² (0.928) is much higher than CV R² (0.770); gap=0.158 exceeds 0.15.

HPO round 1/3: Compact regularization-focused grid for small-sample SVR, prioritizing lower C, moderate epsilon, and mostly smooth kernels to address baseline overfitting..
Candidates searched: 48.
Best CV R²: 0.83.
Train-CV R² gap: 0.10.
Assessment: good.

Final selected model: hpo_round_1.