# Model Comparison (RF + Fallbacks)

**Winner:** SVR (sfs_subset) (baseline)

Selected SVR (sfs_subset) (baseline) with combined R²=0.3989 (mean CV R²=0.5015, holdout val R²=0.2962), train-CV gap=0.1161, status=good. Compared 13 model branch(es); 1 acceptable; applied one-SE rule on combined R² (threshold >= 0.2866) with simplicity tie-break.

**Winner source:** SFS subset (`sfs_subset`)


## All candidates

- RandomForestRegressor (hpo_round_2): combined R²=0.3458, CV R²=0.3879, gap=0.3080, status=overfit, acceptable=False, n_features=2
- RandomForestRegressor (sfs_subset) (baseline): combined R²=0.4034, CV R²=0.4301, gap=0.4722, status=overfit, acceptable=False, n_features=2
- RandomForestRegressor (sfs_fixed_ga_plus2) (baseline): combined R²=0.3117, CV R²=0.3916, gap=0.3188, status=overfit, acceptable=False, n_features=4
- PLSRegression (hpo_round_1): combined R²=0.1950, CV R²=0.2391, gap=0.0420, status=underfit, acceptable=False, n_features=1
- PLSRegression (sfs_fixed_ga_plus2) (hpo_round_1): combined R²=0.2169, CV R²=0.3268, gap=0.0506, status=underfit, acceptable=False, n_features=3
- ExtraTreesRegressor (baseline): combined R²=0.2767, CV R²=0.4266, gap=0.5384, status=overfit, acceptable=False, n_features=2
- ExtraTreesRegressor (sfs_subset) (baseline): combined R²=0.4717, CV R²=0.4657, gap=0.5008, status=overfit, acceptable=False, n_features=2
- ExtraTreesRegressor (sfs_fixed_ga_plus2) (baseline): combined R²=0.5439, CV R²=0.4850, gap=0.4976, status=overfit, acceptable=False, n_features=4
- SVR (hpo_round_1): combined R²=0.3210, CV R²=0.3849, gap=0.1327, status=poor_performance, acceptable=False, n_features=4
- SVR (sfs_subset) (baseline): combined R²=0.3989, CV R²=0.5015, gap=0.1161, status=good, acceptable=True, n_features=4
- SVR (sfs_fixed_ga_plus2) (hpo_round_1): combined R²=0.4940, CV R²=0.5612, gap=0.2824, status=overfit, acceptable=False, n_features=6
- KNeighborsRegressor (baseline): combined R²=0.3255, CV R²=0.3367, gap=0.2187, status=overfit, acceptable=False, n_features=1
- KNeighborsRegressor (sfs_fixed_ga_plus2) (baseline): combined R²=0.3331, CV R²=0.3152, gap=0.2475, status=overfit, acceptable=False, n_features=3