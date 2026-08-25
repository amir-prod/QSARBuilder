# Feature Count Selection

The highest combined R² (0.5·CV + 0.5·val) was 0.4503 at 4 descriptor(s) (CV R² = 0.5100, val R² = 0.3906). Applying the one-standard-error rule with CV std as SE (threshold = 0.3540), the smallest feature count within one SE of the best is 2 descriptor(s) with combined R² = 0.4034 (CV R² = 0.4301, val R² = 0.3766). Training R² exceeds CV R², suggesting some overfitting may be present.

Using the one-standard-error rule, the selected model keeps **2 features**.

Why:
- The **best combined R²** occurs at **4 features**:
  - CV R² = **0.5100**
  - val R² = **0.3906**
  - combined R² = **0.4503**
- The one-standard-error threshold is based on the CV variability at that best point:
  - std CV R² at 4 features = **0.0963**
  - threshold = **0.4503 - 0.0963 = 0.3540**

Then we choose the **smallest feature count** whose combined R² is at least **0.3540**:
- 1 feature: combined R² = **0.1839** → below threshold
- 2 features: combined R² = **0.4034** → within one SE
- 3 features: combined R² = **0.4328** → within one SE
- 4 features: combined R² = **0.4503** → best
- 5 features: combined R² = **0.4482** → within one SE

So, by the rule, the selected count is **2**, because it is the **simplest model** still within one standard error of the best-performing model.

Selected 2-feature model:
- Features: **RDKit_MaxAbsPartialCharge**, **RDKit_SMR_VSA6**
- mean train R² = **0.9012**
- mean CV R² = **0.4301**
- std CV R² = **0.1352**
- val R² = **0.3766**
- combined R² = **0.4034**

Context:
- Although **5 features** has the highest CV R² (**0.5278**), its combined R² (**0.4482**) is slightly below the 4-feature model’s **0.4503**, so the deterministic best point is still **4 features**.
- The gap between train R² and CV R² remains noticeable across models; for the selected 2-feature model, train R² (**0.9012**) is well above CV R² (**0.4301**), which is consistent with some overfitting.
