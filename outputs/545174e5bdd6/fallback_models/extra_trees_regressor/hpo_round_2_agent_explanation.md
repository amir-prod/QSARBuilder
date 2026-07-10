# HPO Round 2 Agent Grid Proposal

**Strategy:** Regularization-focused grid search emphasizing reduced tree complexity and increased averaging. Explore shallow-to-moderate depths, larger leaf/split constraints, and both bootstrap settings with feature subsampling options to improve stability and reduce train-CV gap. Keep the grid compact and within the candidate budget.

The baseline and prior round indicate severe overfitting and high fold sensitivity on a very small dataset (20 samples, 4 features). The next search should prioritize stronger regularization and simpler tree structures while still allowing a few higher-capacity settings for comparison. I therefore narrow the grid toward larger min_samples_leaf/min_samples_split, shallower max_depth, and feature subsampling, while keeping a small set of bootstrap and non-bootstrap options.

**Expected overfitting effect:** Should reduce overfitting substantially by constraining tree growth and increasing bias, especially via shallower depths and larger leaf/split sizes; bootstrap may further stabilize predictions.

**Expected underfitting effect:** May slightly increase underfitting risk for the most constrained settings, but the inclusion of moderate depths and smaller split/leaf values should preserve enough flexibility to avoid severe underfitting.

**Cost estimate:** Moderate. The full Cartesian grid would exceed the candidate budget, so this proposal is intended to be sampled or pruned to stay near the 120-candidate limit; individual fits remain relatively fast on 20 samples.
