# HPO Round 3 Agent Grid Proposal

**Strategy:** Constrain tree complexity aggressively: test shallow to moderate depths, larger leaf sizes, larger split thresholds, and feature subsampling. Keep bootstrap on and include max_samples below 1.0 to further reduce variance. Use a compact grid near the 120-candidate limit to explore regularization strength without reintroducing highly flexible configurations.

The model remains severely overfit on a very small dataset (20 samples, 3 features). The next search should prioritize stronger regularization and shallower trees while keeping bootstrap enabled and using subsampling where possible. I am excluding very flexible settings such as max_depth=null and min_samples_leaf=1 from the main grid, and focusing on combinations that should reduce variance and the train-CV gap.

**Expected overfitting effect:** Should reduce overfitting substantially by limiting tree depth, increasing minimum leaf/split sizes, and using bootstrap subsampling. This should lower training R² more than CV R², shrinking the train-CV gap.

**Expected underfitting effect:** May introduce mild underfitting if the trees become too constrained, especially at max_depth=2 or with larger min_samples_leaf values. The grid includes moderate settings to balance this risk.

**Cost estimate:** Moderate. Grid size is 3*5*3*3*3*1*3*2 = 810 raw combinations, which exceeds the candidate cap; however, the intended search should be treated as a constrained proposal and pruned by the search controller to stay near or below 120 evaluated candidates. If evaluated exhaustively, cost would be high for this dataset size but still manageable per fit.
