# HPO Round 1 Agent Grid Proposal

**Strategy:** Grid search with a focus on regularization and compactness due to the small dataset size.

In this initial grid proposal, I considered the dataset size with 153 training samples and 3 features. Given the underfitting status, I opted for a compact grid with a focus on stronger regularization by limiting the number of components. The maximum number of components is set to 2 (min(3, 153-1)). I included both scaling options to assess their impact on model performance. The max_iter values are chosen to allow for sufficient convergence without excessive computational cost.

**Expected overfitting effect:** The proposed grid is expected to reduce the risk of overfitting by limiting the number of components and including scaling options.

**Expected underfitting effect:** By exploring a range of components and scaling options, the grid aims to improve model capacity and potentially address the underfitting issue.

**Cost estimate:** The total combinations are 8 (2 n_components * 2 scale * 2 max_iter), which is computationally feasible.
