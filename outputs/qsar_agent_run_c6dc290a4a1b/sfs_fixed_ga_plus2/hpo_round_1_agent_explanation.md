# HPO Round 1 Agent Grid Proposal

**Strategy:** Grid search with a focus on regularization parameters to combat overfitting.

In this initial round, I focused on addressing the severe overfitting observed in the baseline assessment. Given the small dataset size (153 training samples and 8 features), I opted for stronger regularization by limiting the max_depth and adjusting other parameters to create a compact grid. This approach aims to reduce the model's complexity and improve generalization.

**Expected overfitting effect:** The proposed grid is expected to reduce overfitting by limiting the model complexity through max_depth and adjusting min_samples_split and min_samples_leaf.

**Expected underfitting effect:** There is a risk of underfitting if the parameters are too restrictive; however, the chosen values aim to balance complexity and performance given the dataset size.

**Cost estimate:** Moderate, as the grid size is compact and the number of combinations is kept below 120.
