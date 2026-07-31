# HPO Round 3 Agent Grid Proposal

**Strategy:** Focus on variance reduction and stability: explore shallow-to-moderate max_depth, larger min_samples_split and min_samples_leaf, and bootstrap enabled/disabled. Keep n_estimators moderately high but not excessive, since tree count mainly affects stability rather than bias. Use a compact grid near the 120-combination limit to cover regularization strength without over-expanding search.

The model is severely overfitting and unstable on a very small dataset (21 samples, 4 features). Previous rounds showed that changing max_features alone did not improve CV performance, so this grid shifts toward stronger regularization by increasing min_samples_split/min_samples_leaf, limiting tree depth, and testing bootstrap. The grid keeps a few higher-capacity settings as controls while prioritizing simpler trees to reduce variance.

**Expected overfitting effect:** Likely to reduce overfitting substantially by constraining tree growth and increasing leaf/sample requirements; bootstrap may further reduce variance. Some settings may still overfit, but the grid is biased toward more regularized models than prior rounds.

**Expected underfitting effect:** A few configurations with deeper trees, smaller split/leaf thresholds, and no bootstrap preserve enough flexibility to avoid excessive underfitting. However, the overall search intentionally favors regularization, so mild underfitting is possible in the most constrained settings.

**Cost estimate:** 108 total combinations; feasible for a small dataset, with moderate cost due to up to 900 trees per fit. Overall search cost is acceptable and within the 120-candidate limit.
