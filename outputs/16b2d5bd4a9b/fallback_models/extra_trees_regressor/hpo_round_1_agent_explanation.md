# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid for a very small dataset, emphasizing reduced tree complexity and variance control while staying well below the candidate limit.

Round 1 uses the baseline assessment only, since there is no prior-round feedback. The baseline shows severe overfitting and instability (very high train-CV R2 gap and high CV variance). Because the dataset is extremely small (n_train=20) with only 1 selected descriptor, I biased the grid toward stronger regularization and a compact search: shallower trees, larger min_samples_split, larger min_samples_leaf, and bootstrap enabled as an option to reduce variance. I also kept max_features fixed to 1.0 because with a single feature, alternative feature-subsampling settings are effectively not meaningful.

**Expected overfitting effect:** Should reduce overfitting by limiting tree depth and requiring larger terminal/support sample sizes; bootstrap may further reduce variance and instability.

**Expected underfitting effect:** There is some risk of underfitting from the stronger regularization, but this is acceptable in round 1 given the baseline severe overfitting and instability.

**Cost estimate:** Low to moderate: 144 total combinations before any pruning, with only 1 feature and 20 training samples making each fit inexpensive.
