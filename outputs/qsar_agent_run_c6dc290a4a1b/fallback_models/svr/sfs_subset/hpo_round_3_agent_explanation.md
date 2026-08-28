# HPO Round 3 Agent Grid Proposal

**Strategy:** Refined grid with increased regularization and adjustments to epsilon and gamma to mitigate overfitting.

In this round, I refined the hyperparameter grid around the best_params from the previous round, which was identified as overfitting. The dataset size of 153 samples suggests a need for increased regularization. I focused on reducing the risk of overfitting by adjusting 'C' to lower values and exploring a wider range of 'epsilon' values. The 'gamma' values were also adjusted to include lower numeric values to help control model complexity. This grid aims to balance the model's capacity while addressing the overfitting issue observed in the previous assessments.

**Expected overfitting effect:** The adjustments to 'C' and 'gamma' are expected to reduce model complexity, thereby decreasing the likelihood of overfitting.

**Expected underfitting effect:** The inclusion of higher 'C' values and a range of 'epsilon' values allows for flexibility, which should help avoid underfitting while still addressing overfitting.

**Cost estimate:** The proposed grid has 64 combinations, which is manageable given the dataset size and should not significantly increase computational costs.
