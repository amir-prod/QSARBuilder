# HPO Round 3 Agent Grid Proposal

**Strategy:** Grid search with a focus on regularization adjustments to mitigate overfitting.

In this round, I refined the hyperparameter grid around the best_params from the previous round, which were C=1.0, epsilon=0.01, gamma='scale', and kernel='rbf'. Given the dataset size of 153 samples, I focused on increasing regularization to address the overfitting issue indicated by the previous assessments. I included a lower C value (0.1) to enhance regularization and a higher epsilon (0.1, 0.2) to allow for more margin in the SVR model. I also retained 'scale' and 'auto' for gamma while dropping numeric values that did not yield promising results in prior rounds. This grid aims to explore a balance between regularization and model capacity to mitigate overfitting while maintaining performance.

**Expected overfitting effect:** The adjustments in C and epsilon are expected to reduce the model's complexity, thereby decreasing the likelihood of overfitting.

**Expected underfitting effect:** There is a risk of underfitting if the regularization is too strong; however, the inclusion of higher epsilon values aims to maintain a balance.

**Cost estimate:** The proposed grid has 24 combinations, which is manageable within computational limits.
