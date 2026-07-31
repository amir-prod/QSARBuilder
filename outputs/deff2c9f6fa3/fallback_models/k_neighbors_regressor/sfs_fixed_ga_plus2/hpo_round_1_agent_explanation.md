# HPO Round 1 Agent Grid Proposal

**Strategy:** Use a small, structured grid over neighborhood size, weighting, distance metric, and Minkowski power. Prioritize moderate-to-larger n_neighbors values to improve stability, while retaining a few smaller values to avoid missing a better bias-variance tradeoff.

The baseline is unstable with high CV variance, so the grid emphasizes settings that can smooth predictions and reduce sensitivity to fold composition, while still covering both local and slightly broader neighborhood behavior. The search is kept compact to stay within the candidate limit.

**Expected overfitting effect:** Increasing n_neighbors and testing distance weighting should generally reduce variance and overfitting risk. Smaller neighborhoods remain included to detect if the baseline benefits from more local fits.

**Expected underfitting effect:** Larger n_neighbors may increase bias and can worsen underfitting if the true signal is highly local. Including smaller n_neighbors and both p values helps preserve flexibility.

**Cost estimate:** 60 candidates total (10 n_neighbors x 2 weights x 2 p x 3 metrics), which is within the limit and inexpensive for this dataset size.
