# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact exhaustive grid over all valid latent dimensions, both scale settings, and standard convergence limits to address baseline underfit without exceeding candidate budget.

Round 1 uses the baseline only. The baseline indicates underfitting, with both train and CV R² low and only 3 selected descriptors available. Given n_train=153 and n_features=3 (samples_per_feature=51.0), the dataset is not small relative to feature count, so the main capacity limit in PLS is the very low maximum latent dimensionality. I therefore propose a compact but complete grid over all valid n_components values (1 to min(3, 152)=3), both scaling options, and standard max_iter values. This keeps the search small while directly testing whether increasing latent components from stronger regularization (1 component) to maximum capacity (3 components) reduces underfitting.

**Expected overfitting effect:** Low overfitting risk overall because feature count is only 3 and maximum n_components is capped at 3; testing scale and convergence settings should not materially increase variance.

**Expected underfitting effect:** Primary improvement should come from allowing higher n_components (2 and 3) versus a potentially too-restrictive baseline; scale may also help if descriptor magnitudes differ.

**Cost estimate:** Very low: 24 total combinations.
