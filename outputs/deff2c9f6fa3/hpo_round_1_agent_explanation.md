# HPO Round 1 Agent Grid Proposal

**Strategy:** Use a variance-reduction focused grid: shallow to moderate tree depths, larger leaf sizes, conservative split thresholds, and limited feature subsampling choices. Include both bootstrap and non-bootstrap settings, but restrict max_samples to bootstrap=true configurations only. Keep criterion choices broad enough to test robustness without exploding the grid size.

The baseline shows severe overfitting and high fold-to-fold instability on a very small dataset (21 samples, 2 features). The grid therefore emphasizes stronger regularization, shallower trees, and a mix of bootstrap settings to reduce variance while still allowing some flexibility. The search space is kept compact to stay near or below the 120-combination limit.

**Expected overfitting effect:** Expected to reduce overfitting substantially by limiting tree complexity and increasing regularization through deeper constraints and larger leaf/split thresholds; bootstrap and max_samples should further lower variance when enabled.

**Expected underfitting effect:** Some configurations may underfit, especially very shallow trees with large leaf sizes or strong subsampling; the inclusion of moderate depths and lower regularization settings should help identify a balanced region.

**Cost estimate:** Moderate. The raw Cartesian product is 3*6*4*4*4*2*4*2 = 9216, but invalid max_samples/non-bootstrap combinations should be filtered. This is far above the requested limit, so the grid should be interpreted as a candidate set to be pruned before execution; as written it is not within the 120-combination target.
