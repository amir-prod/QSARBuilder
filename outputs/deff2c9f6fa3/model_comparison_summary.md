# Model Comparison (RF + Fallbacks)

**Winner:** SVR (sfs_fixed_ga_plus2) (hpo_round_2)

Selected SVR (sfs_fixed_ga_plus2) (hpo_round_2) with mean CV R²=0.8494, train-CV gap=0.1374, status=good. Compared 10 model branch(es); 2 acceptable; applied one-SE rule (threshold CV R² >= 0.7977) with simplicity tie-break.

**Winner source:** SFS-fixed GA expansion (`sfs_fixed_ga_plus2`)


## All candidates

- RandomForestRegressor (baseline): CV R²=0.5052, gap=0.4579, status=unstable, acceptable=False, n_features=2
- RandomForestRegressor (sfs_fixed_ga_plus2) (hpo_round_3): CV R²=0.6284, gap=0.2825, status=overfit, acceptable=False, n_features=4
- PLSRegression (hpo_round_1): CV R²=0.7715, gap=0.1668, status=unstable, acceptable=False, n_features=6
- PLSRegression (sfs_fixed_ga_plus2) (baseline): CV R²=0.9052, gap=0.0546, status=good, acceptable=True, n_features=8
- ExtraTreesRegressor (baseline): CV R²=0.7168, gap=0.2828, status=unstable, acceptable=False, n_features=4
- ExtraTreesRegressor (sfs_fixed_ga_plus2) (baseline): CV R²=0.7879, gap=0.2120, status=unstable, acceptable=False, n_features=6
- SVR (baseline): CV R²=0.7729, gap=0.1956, status=overfit, acceptable=False, n_features=5
- SVR (sfs_fixed_ga_plus2) (hpo_round_2): CV R²=0.8494, gap=0.1374, status=good, acceptable=True, n_features=7
- KNeighborsRegressor (hpo_round_1): CV R²=0.7958, gap=0.2042, status=overfit, acceptable=False, n_features=5
- KNeighborsRegressor (sfs_fixed_ga_plus2) (baseline): CV R²=0.7335, gap=0.1383, status=unstable, acceptable=False, n_features=7