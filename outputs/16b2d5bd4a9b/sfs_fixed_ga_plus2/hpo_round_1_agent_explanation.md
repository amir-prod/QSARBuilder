# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid for a very small dataset, centered on reducing tree complexity and variance while keeping total combinations under the candidate budget.

Round 1 uses the baseline assessment only, since there is no prior-round feedback. The baseline indicates overfitting (train-CV R² gap about 0.199), so this grid shifts toward stronger regularization. Because the dataset is very small (n_train=20, n_features=5, samples_per_feature=4.0), the search is intentionally compact and avoids overly flexible forests by favoring shallower trees, larger leaf/split sizes, reduced feature subsampling, and bootstrap with optional row subsampling.

**Expected overfitting effect:** Should reduce overfitting by limiting tree depth, requiring larger leaves/splits, and using bootstrap with optional subsampling to lower variance.

**Expected underfitting effect:** There is some risk of mild underfitting because the grid is deliberately regularized for a tiny dataset, but the included depth and sample-threshold ranges should still allow adequate flexibility.

**Cost estimate:** Low to moderate. Total combinations: 48, which fits the max candidate budget.
