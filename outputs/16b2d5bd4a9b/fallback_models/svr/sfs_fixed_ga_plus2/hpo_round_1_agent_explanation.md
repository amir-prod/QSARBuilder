# HPO Round 1 Agent Grid Proposal

**Strategy:** Compact regularization-focused grid for small-sample SVR, prioritizing lower C, moderate epsilon, and mostly smooth kernels to address baseline overfitting.

Round 1 uses the baseline only. The baseline indicates overfitting, so this initial SVR grid is biased toward stronger regularization and a compact search. Because n_train=20 is small and there are only 3 features, I limited the grid size and emphasized lower-to-moderate C values, larger epsilon values, and conservative kernel choices to reduce variance while staying within the 50-candidate limit.

**Expected overfitting effect:** Should reduce overfitting by shrinking model flexibility through smaller C, wider epsilon-insensitive regions, and a compact kernel/gamma set appropriate for very small training size.

**Expected underfitting effect:** There is some risk of mild underfitting at the lowest C and highest epsilon settings, but the inclusion of C up to 5 and both linear/rbf kernels preserves enough flexibility for recovery.

**Cost estimate:** Low; 48 total combinations, which is within the requested maximum and suitable for small-data cross-validation.
