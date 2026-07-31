# HPO Round 2 Agent Grid Proposal

**Strategy:** Regularization-focused grid search around the prior best configuration, emphasizing shallower trees and larger leaf sizes to reduce variance. Keep n_estimators in a moderate-high range for stability, but avoid an overly large grid. Include a null max_depth option plus several constrained depths to probe the bias-variance tradeoff.

The previous best setting still shows strong train-CV gap and high fold variability on a very small dataset (21 samples), so the next search should prioritize stronger regularization and simpler trees while keeping a few higher-capacity references for comparison. I will reduce tree complexity via larger min_samples_leaf/min_samples_split, limit depth, and test both bootstrap modes with moderate feature subsampling.

**Expected overfitting effect:** Should reduce overfitting by increasing tree regularization and lowering model variance, especially through larger min_samples_leaf/min_samples_split and finite max_depth values.

**Expected underfitting effect:** May slightly increase underfitting risk for the most constrained settings, but the grid retains less-regularized references (null depth, smaller leaf sizes) to avoid excessive bias.

**Cost estimate:** 144 candidate combinations; slightly above the requested target, but still manageable for ExtraTrees on 21 samples. If needed, prune one dimension (e.g., n_estimators to [400,800]) to bring it to 96 combinations.
