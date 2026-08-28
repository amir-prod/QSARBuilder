# HPO Round 1 Agent Grid Proposal

**Strategy:** Grid search with a focus on regularization and kernel selection to improve model performance.

In this initial round, I focused on a compact grid for the SVR model due to the small dataset size (n_train_samples = 153). Stronger regularization is preferred to avoid overfitting, which is crucial given the baseline performance is poor (CV R² = 0.406). The grid includes a limited range of C values and epsilon to ensure a manageable number of combinations while exploring different kernels and gamma settings.

**Expected overfitting effect:** The proposed grid aims to reduce the risk of overfitting by incorporating lower C values and a variety of kernels, which should help in generalizing better to unseen data.

**Expected underfitting effect:** The inclusion of different kernels and a range of epsilon values is expected to help mitigate underfitting by allowing the model to capture more complex relationships in the data.

**Cost estimate:** The total number of combinations is 24 (3 C values * 3 epsilon values * 4 gamma values * 2 kernels), which is computationally feasible given the dataset size.
