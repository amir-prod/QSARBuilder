# HPO Round 2 Agent Grid Proposal

**Strategy:** Use a compact grid centered on stronger regularization and larger epsilon values, while comparing linear and low-complexity RBF/poly settings. Keep gamma restricted to scale/auto for kernelized models and avoid very small epsilon/C values that can encourage overfitting. Total combinations are kept at 108.

The previous SVR configuration improved CV R² but still showed a substantial train-CV gap, indicating continued overfitting. With a very small dataset (21 samples, 5 features), the next search should bias toward simpler, more regularized models while still allowing a limited amount of nonlinearity to test whether performance can be retained without the same degree of fit complexity.

**Expected overfitting effect:** Should reduce overfitting by favoring simpler decision functions, stronger regularization, and less sensitive residual fitting; linear and higher-epsilon settings are especially likely to narrow the train-CV gap.

**Expected underfitting effect:** May slightly increase underfitting risk for the most regularized linear settings, but the inclusion of moderate C values and kernelized options should preserve a path to adequate fit if the data require nonlinearity.

**Cost estimate:** 108 candidate fits per CV split configuration; low-to-moderate cost for SVR on a 21-sample dataset.
