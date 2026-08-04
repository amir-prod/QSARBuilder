# Model Comparison (RF + Fallbacks)

**Winner:** SVR (sfs_fixed_ga_plus2) (hpo_round_1)

Selected SVR (sfs_fixed_ga_plus2) (hpo_round_1) with mean CV R²=0.8306, train-CV gap=0.1039, status=good. Compared 10 model branch(es); 2 acceptable; applied one-SE rule (threshold CV R² >= 0.7425) with simplicity tie-break.

**Winner source:** SFS-fixed GA expansion (`sfs_fixed_ga_plus2`)


## All candidates

- RandomForestRegressor (baseline): CV R²=0.5342, gap=0.3969, status=unstable, acceptable=False, n_features=3
- RandomForestRegressor (sfs_fixed_ga_plus2) (baseline): CV R²=0.7658, gap=0.1988, status=overfit, acceptable=False, n_features=5
- PLSRegression (hpo_round_1): CV R²=0.6201, gap=0.0955, status=unstable, acceptable=False, n_features=4
- PLSRegression (sfs_fixed_ga_plus2) (baseline): CV R²=0.7852, gap=0.1489, status=good, acceptable=True, n_features=6
- ExtraTreesRegressor (baseline): CV R²=0.1911, gap=0.8080, status=unstable, acceptable=False, n_features=1
- ExtraTreesRegressor (sfs_fixed_ga_plus2) (baseline): CV R²=0.7384, gap=0.2274, status=unstable, acceptable=False, n_features=3
- SVR (hpo_round_2): CV R²=0.2618, gap=0.1181, status=unstable, acceptable=False, n_features=1
- SVR (sfs_fixed_ga_plus2) (hpo_round_1): CV R²=0.8306, gap=0.1039, status=good, acceptable=True, n_features=3
- KNeighborsRegressor (baseline): CV R²=0.3197, gap=0.2684, status=unstable, acceptable=False, n_features=1
- KNeighborsRegressor (sfs_fixed_ga_plus2) (hpo_round_2): CV R²=0.4970, gap=0.3746, status=unstable, acceptable=False, n_features=3