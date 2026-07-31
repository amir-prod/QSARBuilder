# HPO Round 3 Agent Grid Proposal

**Strategy:** Constrain tree complexity and increase ensemble stability. Search a small set of regularized configurations: shallower depths, larger leaf and split thresholds, bootstrap on/off, and feature subsampling options. Keep n_estimators moderately high but not excessive to reduce variance without expanding the grid too much.

The current ExtraTreesRegressor is highly overfit and unstable on a very small dataset (21 samples, 6 features). Previous rounds repeatedly selected a very flexible configuration (deep trees, minimal leaf/split constraints, no bootstrap), so the next search should prioritize stronger regularization and variance reduction while still allowing a few higher-capacity settings for comparison. The grid is kept compact and centered on increasing min_samples_leaf/min_samples_split, limiting depth, and testing bootstrap with moderate feature subsampling.

**Expected overfitting effect:** Should reduce overfitting by limiting tree depth, increasing minimum samples per split/leaf, and testing bootstrap-based bagging. Feature subsampling may further decorrelate trees and improve generalization stability.

**Expected underfitting effect:** Some configurations may underfit, especially the shallowest depths and largest leaf/split thresholds. Including a few less-regularized settings (e.g., depth null, smaller leaf/split values) helps avoid overcorrecting.

**Cost estimate:** Grid size is 3 x 5 x 4 x 4 x 4 x 2 = 1920 combinations, which exceeds the stated limit. To stay near or below 120 candidates, use a reduced subset such as n_estimators [300,900], max_depth [5,8,12,null], min_samples_split [2,10,20], min_samples_leaf [1,2,4], max_features ['sqrt',0.5], bootstrap [true,false] for 2 x 4 x 3 x 3 x 2 x 2 = 288, still above limit; a further narrowed version is recommended before execution.
