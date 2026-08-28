# HPO Round 3 Agent Grid Proposal

**Strategy:** Grid search with a refined parameter space to address instability and overfitting.

In this round, I refined the hyperparameter grid around the best_params from the previous round, which were {'metric': 'minkowski', 'n_neighbors': 10, 'p': 1, 'weights': 'distance'}. Given the dataset size of 153 samples, I focused on reducing instability and overfitting by exploring nearby values for n_neighbors and adjusting the weights. The grid includes a mix of uniform and distance weights to assess their impact on model stability. The adjustments target the high train-CV R² gap and the instability indicated in the latest feedback.

**Expected overfitting effect:** The proposed grid aims to reduce overfitting by testing both uniform and distance weights, which may help in stabilizing the model's predictions across different folds.

**Expected underfitting effect:** The range of n_neighbors is kept tight to avoid underfitting while still allowing for exploration of slightly larger neighbor counts to capture more data variability.

**Cost estimate:** The grid contains 24 combinations (3 n_neighbors * 2 weights * 2 p * 2 metrics), which is computationally manageable given the dataset size.
