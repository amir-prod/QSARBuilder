# HPO Round 3 Agent Grid Proposal

**Strategy:** Constrain the search to a compact grid emphasizing regularization and reduced sensitivity: lower C values, broader epsilon values, and a mix of kernels including linear, rbf, and poly. For rbf/poly, include both scale/auto and a few numeric gamma values spanning weak to moderate locality. Keep the grid under the candidate limit while exploring simpler models first.

Previous SVR settings repeatedly overfit with an RBF kernel and moderate C. This grid shifts toward lower C and higher epsilon to reduce model flexibility, while also testing linear and polynomial kernels and a small set of gamma values to see whether simpler or less localized decision functions improve CV generalization.

**Expected overfitting effect:** Likely to reduce overfitting by lowering model capacity through smaller C and larger epsilon, and by allowing linear kernel options that are less prone to fitting noise than RBF with higher effective complexity.

**Expected underfitting effect:** May slightly increase underfitting risk for the smallest C and largest epsilon settings, especially with linear kernel; the inclusion of moderate C and RBF/poly options should help recover fit if the data needs nonlinearity.

**Cost estimate:** 60 candidates total (5 x 4 x 5 x 3), which is within the 120-candidate limit and modest for n_train=21.
