# HPO Round 3 Agent Grid Proposal

**Strategy:** Use a focused but broader grid emphasizing larger n_neighbors to reduce overfitting and fold sensitivity, while retaining a few smaller values to avoid missing locally optimal settings. Test both uniform and distance weighting, and compare p=1 vs p=2 across minkowski/euclidean/manhattan. Total combinations: 10 x 2 x 2 x 3 = 120.

With only 20 training samples and 1 feature, KNN performance is likely sensitive to neighborhood size and distance weighting. The prior rounds suggest small odd neighbor counts can help, but instability remains high, so this grid broadens n_neighbors toward larger values to reduce variance while still testing local structure. Both p values and all allowed metrics are included to compare Manhattan-like vs Euclidean/Minkowski behavior, while keeping the grid compact.

**Expected overfitting effect:** Increasing n_neighbors should smooth predictions and reduce variance, which may lower the train-CV gap and improve stability. Distance weighting may preserve some local flexibility, but larger neighborhoods should generally counter severe overfitting.

**Expected underfitting effect:** Very large neighborhoods can oversmooth and increase bias, so including smaller and mid-range n_neighbors helps guard against underfitting. Testing both p values and metrics may recover better local geometry if the data benefit from a particular distance definition.

**Cost estimate:** 120 candidate combinations; low-to-moderate cost for a 1-feature KNN search, though cross-validation variance may still be substantial due to the tiny dataset.
