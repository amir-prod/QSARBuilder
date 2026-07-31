# HPO Round 1 Agent Grid Proposal

**Strategy:** Use a small, regularization-focused grid spanning linear, RBF, and poly kernels. Prioritize lower C and moderate epsilon values, with gamma restricted to conservative values for nonlinear kernels. Total combinations are kept at 108 to stay within the budget.

Baseline SVR shows clear overfitting, so the grid emphasizes stronger regularization (lower C), modest-to-larger epsilon, and includes simpler kernels alongside RBF. Gamma is kept mostly in a compact numeric range plus scale/auto to avoid overly flexible settings while still allowing a controlled search.

**Expected overfitting effect:** Likely to reduce overfitting by favoring lower-capacity models, especially through smaller C and inclusion of linear kernel; larger epsilon may further smooth the fit.

**Expected underfitting effect:** Some candidates, especially linear with very low C or high epsilon, may underfit; the grid retains moderate C and nonlinear kernels to preserve flexibility if needed.

**Cost estimate:** 108 candidates total; low-to-moderate cost for SVR on 21 samples, but poly and RBF variants will be somewhat more expensive than linear.
