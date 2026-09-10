# Feature Count Selection

The highest combined R² (0.5·CV + 0.5·val) was 0.3756 at 3 descriptor(s) (CV R² = 0.5504, val R² = 0.2009). Applying the one-standard-error rule with CV std as SE (threshold = 0.2701), the smallest feature count within one SE of the best is 2 descriptor(s) with combined R² = 0.3221 (CV R² = 0.5089, val R² = 0.1354). Training R² exceeds CV R², suggesting some overfitting may be present.

Using the one-standard-error rule, the selected model keeps **2 features**.

Why:

- The **best combined R²** occurs at **3 features**:
  - combined R² = **0.37562651269284814**
  - CV R² = **0.5503861119809892**
  - val R² = **0.20086691340470708**

- For the one-standard-error rule, the CV standard deviation at that best point is used:
  - std CV R² at 3 features = **0.10552041491425698**

- This gives an acceptance threshold of:
  - **0.37562651269284814 - 0.10552041491425698 = 0.27010609777859116**

Now compare combined R² values to that threshold:

- **1 feature**: combined R² = **-0.07203682590354404** → below threshold
- **2 features**: combined R² = **0.32212408410549526** → above threshold
- **3 features**: combined R² = **0.37562651269284814** → above threshold
- **4 features**: combined R² = **0.36178423956033606** → above threshold
- **5 features**: combined R² = **0.3297370913286512** → above threshold

The one-standard-error rule then chooses the **smallest feature count** that is still within one standard error of the best result. That is **2 features**.

Selected 2-feature model:

- Features: **RDKit_Chi2v, RDKit_qed**
- mean train R² = **0.9035384512743059**
- mean CV R² = **0.5088951322400777**
- std CV R² = **0.11600368831089315**
- val R² = **0.13535303597091286**
- combined R² = **0.32212408410549526**

Interpretation:

- Although **3 features** gives the highest combined R², the **2-feature** model is close enough under the one-standard-error criterion.
- This favors a **simpler model** with only a modest drop from the best combined R²:
  - from **0.37562651269284814** at 3 features
  - to **0.32212408410549526** at 2 features

Also, for the selected 2-feature model, train R² (**0.9035384512743059**) is notably higher than CV R² (**0.5088951322400777**), which is consistent with some overfitting risk.
