# Feature Count Selection

The highest mean cross-validation R² was 0.8095 at 10 descriptor(s). Applying the one-standard-error rule (threshold = 0.7100), the smallest feature count within one SE of the best is 2 descriptor(s) with CV R² = 0.7517. Training R² exceeds validation R², suggesting some overfitting may be present.

The one-standard-error rule chose **2 descriptors**.

Why:
- The **best mean CV R²** is **0.8095** at **10 features**.
- The reported one-standard-error threshold is **0.7100**.
- Under this rule, we pick the **smallest feature count** whose mean CV R² is still within that threshold.
- **2 features** has **mean CV R² = 0.7517**, which is above the threshold, so it is selected.

Selected 2-feature set:
- **Mordred_GATS1v**
- **Mordred_GeomRadius**

Additional note:
- The **training R²** is higher than the **validation R²** at the selected size, which suggests some overfitting may be present.
