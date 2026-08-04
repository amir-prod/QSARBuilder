# Feature Count Selection

The highest mean cross-validation R² was 0.7883 at 4 descriptor(s). Applying the one-standard-error rule (threshold = 0.7229), the smallest feature count within one SE of the best is 3 descriptor(s) with CV R² = 0.7561. Training R² exceeds validation R², suggesting some overfitting may be present.

Using the one-standard-error rule, the selected model uses **3 descriptors**.

Why:
- The **best mean CV R²** is **0.7883** at **4 features**.
- The one-SE threshold given in the deterministic selection is **0.7229**.
- We then choose the **smallest feature count** whose mean CV R² is at least this threshold.

Checking the SFS results:
- **1 feature:** CV R² = **0.5859** → below threshold
- **2 features:** CV R² = **0.7125** → below threshold
- **3 features:** CV R² = **0.7561** → above threshold
- **4 features:** CV R² = **0.7883** → best overall

So although **4 features** gives the highest CV R², **3 features** is the **simplest model within one standard error of the best**, which is why it is selected.

Selected 3-feature set:
- `RDKit_Chi4n`
- `RDKit_MaxPartialCharge`
- `RDKit_PEOE_VSA12`

Additional interpretation from the reported values:
- The training R² is higher than CV R² at each feature count, including the selected 3-feature model (**0.9746 train vs 0.7561 CV**), which is consistent with some degree of overfitting.
- The one-SE rule favors parsimony here: it accepts a small drop from the maximum CV R² (**0.7883 → 0.7561**) in exchange for using fewer descriptors.
