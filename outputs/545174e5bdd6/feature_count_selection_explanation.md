# Feature Count Selection

The highest mean cross-validation R² was 0.8466 at 5 descriptor(s). Applying the one-standard-error rule (threshold = 0.7520), the smallest feature count within one SE of the best is 3 descriptor(s) with CV R² = 0.8152. Training R² exceeds validation R², suggesting some overfitting may be present.

Using the one-standard-error rule, the selected model is the **3-feature** model, not the 5-feature model with the highest mean CV R².

- **Best mean CV R²:** 0.8466 at **5 features**
- **One-SE threshold:** 0.7520
- The **smallest** feature count with mean CV R² within that threshold is **3 features**
- **Selected mean CV R²:** 0.8152

Selected descriptors:
- **AATS3i**
- **ATSC3d**
- **GATS1Z**

For reference, the training R² is higher than the CV R² at the selected size, which suggests some overfitting may still be present.
