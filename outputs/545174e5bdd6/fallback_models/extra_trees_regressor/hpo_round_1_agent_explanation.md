# HPO Round 1 Agent Grid Proposal

**Strategy:** Use a compact, regularization-focused grid that explores depth limitation, minimum leaf/split sizes, feature subsampling, and bootstrap. Keep n_estimators moderate-to-high for stability, but avoid an overly large search space given the small sample size and the 120-candidate cap.

The baseline shows extreme overfitting and unstable cross-validation on a very small dataset (20 samples, 4 features). The grid therefore emphasizes stronger regularization through shallower trees, larger leaf/split constraints, and feature subsampling, while still including a few less constrained settings to avoid underfitting.

**Expected overfitting effect:** Should reduce overfitting substantially by constraining tree complexity and increasing randomness; bootstrap and feature subsampling may further improve generalization stability.

**Expected underfitting effect:** Some configurations may underfit, especially very shallow trees with large leaf/split constraints, but the grid retains a few less-regularized options to balance this risk.

**Cost estimate:** Moderate. The grid is intentionally small enough for exhaustive or near-exhaustive evaluation under the 120-candidate limit, with ExtraTrees remaining relatively efficient on 20 samples.
