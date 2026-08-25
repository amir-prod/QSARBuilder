# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid centered on reducing SVR flexibility: prioritize lower C, moderate epsilon, conservative gamma choices, and mainly rbf/linear kernels with a limited poly option.

Round 1 uses the baseline assessment only, since there is no prior-round feedback. The baseline indicates moderate overfitting (train-CV R² gap 0.175, CV R² below threshold). With n_train=153 and only 6 features (25.5 samples/feature), the dataset is small enough to favor stronger regularization and a compact search rather than a wide, high-variance grid. Therefore, this grid emphasizes lower-to-moderate C, larger epsilon values than a purely accuracy-seeking grid, and mostly smoother kernels/gamma settings to reduce model flexibility while still allowing some nonlinearity.

**Expected overfitting effect:** Should reduce overfitting by limiting margin complexity (lower C), increasing insensitivity to small residuals (epsilon), and avoiding overly aggressive gamma values. Including linear provides a strong low-variance baseline.

**Expected underfitting effect:** There is some risk of mild underfitting at the lowest C/highest epsilon settings, but the inclusion of C up to 10 and rbf kernel should preserve enough flexibility to recover nonlinear signal if present.

**Cost estimate:** Low to moderate: 5 x 3 x 4 x 2 = 120 candidate combinations, within the requested maximum.
