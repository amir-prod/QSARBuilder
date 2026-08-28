# HPO Round 2 Agent Grid Proposal

**Strategy:** Grid search focusing on increasing model capacity and exploring scaling options due to underfitting.

In this round, I refined the hyperparameter grid around the best_params from the previous round, which were {'max_iter': 100, 'n_components': 2, 'scale': false}. Given the dataset size (n_train_samples = 153, n_features = 3), the model is currently underfitting, as indicated by low training and CV R² values. To address this, I increased the number of components to include 3, which may provide more capacity to capture the underlying patterns in the data. I also included both scaling options to explore their effects on model performance. The max_iter remains at 100 and 200 to ensure convergence. This grid aims to enhance model capacity while maintaining a manageable search space.

**Expected overfitting effect:** The adjustments are not expected to increase overfitting, as the focus is on enhancing model capacity without excessive complexity.

**Expected underfitting effect:** Increasing n_components and exploring scaling options should help mitigate underfitting by allowing the model to better capture the variance in the data.

**Cost estimate:** The computational cost is expected to remain moderate, as the grid size is limited to 8 combinations (2 n_components * 2 scale * 2 max_iter).
