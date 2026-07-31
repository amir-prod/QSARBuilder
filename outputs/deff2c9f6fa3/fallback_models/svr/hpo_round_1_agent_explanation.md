# HPO Round 1 Agent Grid Proposal

**Strategy:** Use a compact grid centered on regularization and kernel simplicity: prioritize linear and RBF kernels, include a limited polynomial option, and span C from strong to moderate regularization with a few epsilon values. Keep gamma restricted to scale/auto for non-linear kernels plus one small numeric value to test smoother RBF behavior.

Baseline SVR shows clear overfitting, so the grid emphasizes stronger regularization via lower C, moderate-to-larger epsilon, and simpler kernels while still allowing a small amount of flexibility to recover CV performance.

**Expected overfitting effect:** Should reduce overfitting by lowering model flexibility through smaller C, larger epsilon, and inclusion of linear/simpler settings; RBF/poly options remain but are constrained.

**Expected underfitting effect:** May slightly increase underfitting risk for the most regularized settings, but the grid retains moderate C and RBF/poly combinations to recover fit if needed.

**Cost estimate:** 60 candidates total (5 x 4 x 3), low-to-moderate cost for n_train=21 and 5 features.
