# HPO Round 3 Agent Grid Proposal

**Strategy:** Grid search with a focus on regularization parameters to combat overfitting, while exploring the impact of bootstrap.

In this round, I refined the hyperparameter grid around the best_params from the previous round, focusing on reducing overfitting. Given the dataset size (153 samples), I opted for a more constrained max_depth (5-10) and increased min_samples_split (2-10) to promote regularization. I retained the criterion as 'squared_error' and explored both bootstrap options to assess their impact on model stability. This adjustment aims to reduce the severe overfitting observed in prior assessments.

**Expected overfitting effect:** The proposed adjustments are expected to reduce overfitting by increasing the minimum samples required to split nodes and limiting the maximum depth of trees, which should help in generalizing better to unseen data.

**Expected underfitting effect:** There is a slight risk of underfitting due to the constraints on max_depth and min_samples_split, but the selected ranges should still allow for sufficient model complexity given the dataset size.

**Cost estimate:** The computational cost is expected to be moderate, as the grid size is limited to 12 combinations, which is manageable given the dataset size.
