# HPO Round 2 Agent Grid Proposal

**Strategy:** Grid search with a focus on regularization parameters to combat overfitting, while exploring the impact of bootstrap.

In this round, I focused on addressing the overfitting issue identified in the previous assessment. The best_params from round 1 were used as a foundation, specifically maintaining the max_depth of 5 and the criterion of squared_error. To combat overfitting, I reduced the max_depth options and increased the min_samples_leaf to allow for more regularization. I also introduced bootstrap=True to explore its potential for reducing overfitting, while keeping the n_estimators low to manage computational cost given the dataset size of 153 samples.

**Expected overfitting effect:** The adjustments are expected to reduce overfitting by increasing the minimum samples required at leaves and exploring bootstrap sampling, which can help generalize the model better.

**Expected underfitting effect:** There is a slight risk of underfitting due to increased min_samples_leaf and reduced max_depth options, but the focus is primarily on addressing overfitting.

**Cost estimate:** Moderate, as the grid size is kept manageable with a maximum of 12 combinations.
