# Model Comparison (RF + Fallbacks)

**Winner:** SVR (sfs_fixed_ga_plus2) (baseline)

Selected SVR (sfs_fixed_ga_plus2) (baseline) with combined R²=0.5109 (mean CV R²=0.5349, holdout val R²=0.4870), train-CV gap=0.2009, status=overfit. Compared 13 model branch(es); no acceptable models; chose best combined R² with warning.

**Winner source:** SFS-fixed GA expansion (`sfs_fixed_ga_plus2`)

**Warning:** No acceptable model found across estimators; selected highest combined R² candidate. Final model may still be overfit, unstable, or poor-performing.


## All candidates

- RandomForestRegressor (hpo_round_1): combined R²=0.5529, CV R²=0.5343, gap=0.3430, status=overfit, acceptable=False, n_features=6
- RandomForestRegressor (sfs_subset) (baseline): combined R²=0.5248, CV R²=0.5384, gap=0.3864, status=overfit, acceptable=False, n_features=6
- RandomForestRegressor (sfs_fixed_ga_plus2) (baseline): combined R²=0.4610, CV R²=0.4914, gap=0.3670, status=overfit, acceptable=False, n_features=8
- PLSRegression (hpo_round_1): combined R²=0.1950, CV R²=0.2391, gap=0.0420, status=underfit, acceptable=False, n_features=1
- PLSRegression (sfs_fixed_ga_plus2) (hpo_round_1): combined R²=0.1836, CV R²=0.3163, gap=0.0526, status=underfit, acceptable=False, n_features=3
- ExtraTreesRegressor (baseline): combined R²=0.5891, CV R²=0.5269, gap=0.4211, status=overfit, acceptable=False, n_features=8
- ExtraTreesRegressor (sfs_subset) (baseline): combined R²=0.6174, CV R²=0.5605, gap=0.3825, status=overfit, acceptable=False, n_features=8
- ExtraTreesRegressor (sfs_fixed_ga_plus2) (baseline): combined R²=0.5998, CV R²=0.5355, gap=0.4260, status=overfit, acceptable=False, n_features=10
- SVR (baseline): combined R²=0.3003, CV R²=0.4064, gap=0.1223, status=poor_performance, acceptable=False, n_features=6
- SVR (sfs_subset) (baseline): combined R²=0.4717, CV R²=0.5221, gap=0.1782, status=overfit, acceptable=False, n_features=6
- SVR (sfs_fixed_ga_plus2) (baseline): combined R²=0.5109, CV R²=0.5349, gap=0.2009, status=overfit, acceptable=False, n_features=8
- KNeighborsRegressor (baseline): combined R²=0.3255, CV R²=0.3367, gap=0.2187, status=overfit, acceptable=False, n_features=1
- KNeighborsRegressor (sfs_fixed_ga_plus2) (baseline): combined R²=0.2443, CV R²=0.3279, gap=0.2941, status=unstable, acceptable=False, n_features=3