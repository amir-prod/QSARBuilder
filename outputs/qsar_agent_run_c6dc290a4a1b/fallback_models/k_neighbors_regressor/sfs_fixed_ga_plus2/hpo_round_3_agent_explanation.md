# HPO Round 3 Agent Grid Proposal

**Strategy:** Grid search with a focus on reducing overfitting and improving stability.

In this round, I refined the hyperparameter grid around the best_params from the previous round, which was {'metric': 'minkowski', 'n_neighbors': 6, 'p': 2, 'weights': 'distance'}. Given the dataset size of 153 samples, I focused on reducing overfitting and improving stability. I included n_neighbors values around 6 (5, 6, 7, 8, 9) and retained both 'uniform' and 'distance' for weights. I also kept p fixed at 2, as it showed stability. I introduced 'euclidean' as an additional metric to explore potential improvements in stability. This grid aims to address the instability and overfitting issues highlighted in the latest assessment.

**Expected overfitting effect:** The proposed grid aims to reduce overfitting by exploring a range of n_neighbors values and different weightings, which may help stabilize the model's performance across different folds.

**Expected underfitting effect:** The grid is designed to avoid underfitting by maintaining a sufficient number of neighbors (5-9) and allowing for both uniform and distance weighting, which should capture the underlying data patterns effectively.

**Cost estimate:** The computational cost is expected to be moderate, as the grid contains 10 combinations (5 n_neighbors * 2 weights * 1 p * 2 metrics).
