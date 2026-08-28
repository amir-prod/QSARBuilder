# HPO Round 3 Agent Grid Proposal

**Strategy:** Grid search with a focus on increasing model capacity to address underfitting.

In this round, I focused on addressing the underfitting issue identified in the previous assessments. The best_params from round 2 were {'max_iter': 200, 'n_components': 1, 'scale': false}. Given the dataset size (n_train_samples = 153, n_features = 1), I increased the n_components to 2 to enhance model capacity while keeping max_iter at 200. I also included both scaling options to explore their effects on performance. This adjustment aims to provide more flexibility in capturing the underlying data patterns, which may help improve the model's predictive power.

**Expected overfitting effect:** The adjustments are unlikely to increase overfitting due to the limited number of features and the controlled max_iter.

**Expected underfitting effect:** Increasing n_components should help capture more variance in the data, potentially improving model performance and addressing the underfitting issue.

**Cost estimate:** Low, as the grid search involves only a few combinations (4 total).
