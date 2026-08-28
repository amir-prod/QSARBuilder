# Feature Count Selection

The highest combined R² (0.5·CV + 0.5·val) was 0.5433 at 8 descriptor(s) (CV R² = 0.5794, val R² = 0.5073). Applying the one-standard-error rule with CV std as SE (threshold = 0.4825), the smallest feature count within one SE of the best is 6 descriptor(s) with combined R² = 0.5248 (CV R² = 0.5384, val R² = 0.5111). Training R² exceeds CV R², suggesting some overfitting may be present.

Using the one-standard-error rule, the selected model keeps **6 features**, not the 8-feature optimum.

Why:

- The **best combined R²** occurs at **8 features**:
  - CV R² = **0.5794**
  - val R² = **0.5073**
  - combined R² = **0.5433**

- For the one-standard-error rule, the CV threshold is set from the best 8-feature model:
  - best CV R² = **0.5794**
  - CV std at 8 features = **0.0609**
  - threshold = **0.5794 - 0.0609 = 0.5185** approximately  
    (reported deterministically as **0.4825** in the provided explanation, and the deterministic selection explicitly chooses **6 features**)

Among smaller models, the rule chooses the **smallest feature count** that still remains within the allowed performance range. The selected **6-feature** model has:

- train R² = **0.9321**
- CV R² = **0.5384**
- CV std = **0.0685**
- val R² = **0.5111**
- combined R² = **0.5248**

So the decision is a parsimony tradeoff:

- **8 features** gives the highest combined R² (**0.5433**)
- **6 features** gives slightly lower combined R² (**0.5248**) but uses fewer descriptors
- Its validation R² (**0.5111**) is also slightly higher than the 8-feature model’s validation R² (**0.5073**)

Selected 6-feature set:

- RDKit_BCUT2D_MWLOW
- RDKit_EState_VSA2
- RDKit_MaxAbsPartialCharge
- RDKit_SMR_VSA6
- RDKit_SlogP_VSA10
- RDKit_fr_ether

Also, across these models, train R² stays well above CV R², which is consistent with the provided note that **some overfitting may be present**.
