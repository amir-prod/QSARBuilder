# Model Comparison (RF + Fallbacks)

**Winner:** SVR (baseline)

Selected SVR (baseline) with mean CV R²=0.5766, train-CV gap=0.1453, status=good. Compared 5 model branch(es); 1 acceptable; applied one-SE rule (threshold CV R² >= 0.4267) with simplicity tie-break.


## All candidates

- RandomForestRegressor (hpo_round_3): CV R²=0.6412, gap=0.2092, status=unstable, acceptable=False, n_features=3
- PLSRegression (hpo_round_1): CV R²=0.6732, gap=0.2080, status=unstable, acceptable=False, n_features=4
- ExtraTreesRegressor (baseline): CV R²=0.1988, gap=0.8012, status=unstable, acceptable=False, n_features=4
- SVR (baseline): CV R²=0.5766, gap=0.1453, status=good, acceptable=True, n_features=2
- KNeighborsRegressor (hpo_round_1): CV R²=0.3458, gap=0.3840, status=unstable, acceptable=False, n_features=1