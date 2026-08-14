# QSAR Agent

**QSAR Agent** is a Streamlit application for building regression QSAR models from SMILES and experimental activity data. It orchestrates a reproducible, leakage-aware workflow using [DescJocky](https://github.com/StephenSzwiec/descjocky) molecular descriptors (with optional external descriptor merge), UMAP-based cluster splitting, sequential and genetic feature selection, and Random Forest modeling—with optional OpenAI agent coordination for feature-count decisions.

> **Important:** External-test performance is evaluated only after feature selection and final model training. The test set is never used for preprocessing, tuning, or feature selection. A held-out validation set (default 10%) is used together with training cross-validation for feature-count selection, GA fitness, and HPO/model selection.

## Features

- CSV upload with column mapping (SMILES, activity, optional compound ID)
- RDKit validation and canonicalization of SMILES
- DescJocky descriptor calculation (selectable backends; optional xtb geometry optimization)
- Optional external descriptor CSV merge on `compound_id`
- UMAP + KMeans cluster-aware train/validation/external-test split
- Train-only descriptor preprocessing (imputation, variance/correlation filtering, scaling)
- Sequential forward feature selection (mlxtend) with CV R² and validation R² curves
- One-standard-error rule for optimal descriptor count (OpenAI explains the choice)
- DEAP genetic algorithm for final descriptor subset optimization
- Agent-guided hyperparameter optimization (up to 3 rounds; search on training CV, selection uses CV + validation)
- Random Forest final model with train, validation, and external-test metrics
- Williams applicability-domain plot
- Per-run artifact export and ZIP download

## Architecture

```
streamlit_app.py          # Streamlit UI
qsar_agent/
  config.py               # Workflow defaults and OpenAI settings
  app_state.py            # Streamlit session state helpers
  schemas/                # Pydantic models for tools, state, and reports
  tools/                  # Deterministic scientific pipeline stages
  agents/                 # OpenAI-assisted feature-count explanation
  services/               # Workflow runner, plotting, artifact management
examples/                 # Original reference scripts (UMAP split, GA, SFS)
example/                  # Sample input CSV for testing
outputs/<run_id>/         # Isolated artifacts per workflow run
tests/                    # Unit and integration tests
```

The OpenAI agent coordinates the workflow and explains decisions but **never** calculates descriptors, trains models, or fabricates metrics. All scientific work is done by deterministic Python tools.

## Installation

### Requirements

- Python 3.9+ (3.10+ recommended)
- RDKit-compatible environment (conda/micromamba recommended for cheminformatics deps)

### Setup

```bash
cd QSARBuilder
python -m venv .venv          # or: micromamba create -n qsar-agent python=3.11
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Python 3.9, `eval_type_backport` (included in `requirements.txt`) is required for Pydantic type annotations.

Descriptor calculation uses [DescJocky](https://github.com/StephenSzwiec/descjocky) via a patched vendor copy at `vendor/descjocky` (upstream `requires-python = "=3.11"` is invalid for pip; the vendor tree uses `>=3.11`). By default geometry optimization is **off**: the app writes RDKit SDFs and runs DescJocky with `skip_phase1=True`. Enable **Run geometry optimization (xtb)** in the Streamlit sidebar only if [`xtb`](https://github.com/grimme-lab/xtb) is installed and on `PATH`.

Optional external descriptors: upload a CSV with a `compound_id` column (matching the cleaned dataset IDs) plus numeric descriptor columns. Colliding names are renamed with an `ext__` prefix.

### OpenAI configuration (optional)

The workflow runs without OpenAI. For agent explanations of feature-count selection, set:

**Environment variables** (copy from `.env.example`):

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_MODEL="gpt-4o-mini"
```

**Or Streamlit secrets** (copy from `.streamlit/secrets.toml.example`):

```toml
OPENAI_API_KEY = "your_key_here"
OPENAI_MODEL = "gpt-4o-mini"
```

Never commit `.env` or `.streamlit/secrets.toml` with real keys.

## Running the application

From the project root:

```bash
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### Quick start

1. Upload a CSV dataset.
2. Select **SMILES**, **activity**, and optional **compound ID** columns.
3. Optionally click **Validate Dataset**.
4. Adjust settings in the sidebar (test fraction, preprocessing thresholds, GA/SFS parameters).
5. Click **Run QSAR Workflow**.
6. Review results in the dashboard tabs and download artifacts.

### Example dataset

```bash
example/synthetic_qsar_dataset.csv
```

Columns: `compound_id`, `smiles`, `pIC50` (30 compounds).

## Expected CSV format

| Column | Required | Description |
|--------|----------|-------------|
| SMILES | Yes | Structure strings (parsed with RDKit) |
| Activity | Yes | Continuous numeric endpoint (e.g. pIC50, log IC50) |
| Compound ID | No | Unique identifier; auto-generated if omitted |

Invalid SMILES, missing activities, and duplicates are reported separately—not silently dropped without a record.

## Workflow stages

| Stage | Description |
|-------|-------------|
| 1. Dataset validation | Column checks, SMILES parsing, activity cleaning, duplicate handling |
| 2. Descriptor calculation | DescJocky backends (default RDKit+Mordred); optional xtb Phase 1; optional external CSV joined on `compound_id` |
| 3. UMAP split | Provisional unsupervised preprocessing → UMAP embedding → KMeans clustering → per-cluster train/test split |
| 4. Descriptor preprocessing | Train-only filtering, median imputation, StandardScaler; applied unchanged to test |
| 5. Sequential feature selection | Forward SFS for 1…N descriptors with mean training and CV R² |
| 6. Feature count selection | One-standard-error rule on CV R²; OpenAI explains the choice |
| 7. Genetic algorithm | DEAP GA optimizes CV R² for exactly the selected descriptor count |
| 8. Hyperparameter optimization | Baseline CV diagnostics, overfitting assessment, up to 3 agent-guided HPO rounds (training only) |
| 9. Final model selection | Choose baseline or best HPO configuration using training CV only |
| 10. Final model | Random Forest trained on selected features; external test evaluated once |
| 11. Applicability domain | Williams plot (leverage vs standardized residuals) |

## Hyperparameter optimization (HPO)

After genetic feature selection, the workflow can tune Random Forest hyperparameters using **only the preprocessed training set**.

### What overfitting means here

Overfitting is detected from **training cross-validation** diagnostics, not external-test performance:

- **Train–CV R² gap** = mean training-fold R² minus mean validation-fold R²
- Large gap with high training R² → likely overfitting
- Low training and CV R² → underfitting
- High CV R² standard deviation → unstable model
- CV R² below the minimum threshold → poor performance

Default thresholds: gap > 0.15 (overfit), gap > 0.25 (severe warning), minimum CV R² = 0.50, CV std > 0.15 (unstable).

### Why external test is excluded from HPO

The external test set is **never** used to propose grids, score candidates, assess overfitting, or select the final model. It is evaluated **once** after the final configuration is chosen and the model is retrained on all training compounds.

### Agent-guided grids

When `OPENAI_API_KEY` is set, the agent proposes structured JSON hyperparameter grids (Random Forest only). Invalid responses trigger one repair attempt, then a deterministic fallback grid (regularization-focused for overfit, capacity-focused for underfit, stability-focused for unstable CV). The agent does **not** train models or invent metrics.

### Maximum 3 HPO rounds

If the baseline model is acceptable, HPO is skipped. Otherwise the workflow runs up to **3 rounds** of grid search. Each round logs `HPO round X/3` in Streamlit and artifact logs. Search stops early when an acceptable model is found.

### Disabling HPO

In the Streamlit sidebar, uncheck **Enable HPO** or set `hpo.enabled: false` in configuration. The workflow uses the default Random Forest settings from the Model settings panel.

### HPO artifacts

| File | Description |
|------|-------------|
| `baseline_cv_metrics.csv` / `baseline_cv_summary.json` | Baseline K-fold CV on training set |
| `baseline_overfitting_assessment.json` | Baseline overfitting classification |
| `hpo_round_<i>_agent_grid.json` / `_agent_explanation.md` | Agent-proposed grid and rationale |
| `hpo_round_<i>_grid_sanitization.json` | Sanitized grid and shrink log |
| `hpo_round_<i>_search_results.csv` | All candidates scored in the round |
| `hpo_round_<i>_best_params.json` / `_cv_summary.json` | Best candidate per round |
| `hpo_round_<i>_overfitting_assessment.json` | Round best-model assessment |
| `hpo_round_<i>_performance.png` / `.svg` | Train vs CV R² per candidate |
| `hpo_iteration_log.json` / `.md` | Full HPO decision log |
| `hpo_final_selection.json` / `_explanation.md` | Final model source and rationale |
| `hpo_all_rounds_summary.csv` / `hpo_summary.png` | Cross-round summary |
| `hpo_agent_fallback_log.json` | Agent/fallback events (if any) |
| `final_overfitting_assessment.json` | Assessment of selected configuration |

### Limitations

Automated overfitting detection from CV metrics is a heuristic. High descriptor-to-sample ratios, activity noise, and small training sets can produce misleading gaps or unstable CV scores. HPO improves regularization but does not guarantee external-test performance.

## Methodological notes

### Near-constant filtering before StandardScaler

Near-constant descriptors are removed using raw training-set standard deviation (`std < 0.01`) **before** `StandardScaler` is fit. After scaling, every non-constant feature has ~unit variance, so post-scaling variance checks would fail to detect near-constant raw descriptors.

### Preprocessing fitted only on training data

Missing-value thresholds, imputation values, correlation decisions, and scaling parameters are learned from the training set only and applied unchanged to the validation and external test sets.

### Validation set (development) vs external test

Default split is **80% train / 10% validation / 10% test**. Feature selection, GA fitness, and HPO/model selection use **training K-fold CV plus held-out validation** (combined score = 0.5·CV R² + 0.5·val R²). Validation metrics are slightly optimistic because they influenced selection.

### External test set isolation

The external test set is not used during:

- Descriptor preprocessing decisions
- Sequential feature selection
- Feature-count selection (one-standard-error rule on combined CV+val score)
- Genetic algorithm fitness
- Hyperparameter search and model selection

### UMAP is not clustering

UMAP produces a 2D embedding; **KMeans** clusters that embedding (following the `examples/` reference code). Splitting is performed within each cluster to preserve chemical diversity.

### Williams plot limitations

The Williams plot is a classical descriptor-space diagnostic based on leverage and standardized residuals. Interpret with caution for nonlinear models such as Random Forest.

### Corrections from reference `examples/` code

- **GA fitness:** Combined training CV R² and held-out validation R² (the example GA used the test set for fitness—data leakage).
- **Preprocessing order:** Near-constant removal before scaling (examples scaled first).
- **SFS efficiency:** A single mlxtend SFS fit up to `max_features` reads all subset sizes from `sfs.subsets_` (the example `build_each_model` pattern), rather than re-fitting SFS separately for each feature count.

## Output files

Each run writes to `outputs/<run_id>/`:

| File | Description |
|------|-------------|
| `input_dataset.csv` | Uploaded dataset copy |
| `cleaned_dataset.csv` | Valid, deduplicated compounds |
| `invalid_rows.csv` | Invalid SMILES or activities (if any) |
| `duplicate_compounds.csv` | Duplicate SMILES removed (if any) |
| `dataset_validation.json` | Validation summary |
| `descriptors_raw.csv` | Combined generated (+ optional external) descriptor matrix |
| `generated_descriptors.csv` | Final generated-only descriptor matrix (meta + DescJocky features) |
| `generated_descriptors_raw.csv` | Same as generated (compatibility copy) |
| `descjocky_descriptors.csv` | Native DescJocky output CSV copy |
| `generated_descriptor_columns.json` | List of generated descriptor column names |
| `external_descriptors.csv` | Copied user-provided external descriptors (if used) |
| `descriptor_calculation_report.json` / `.md` | Backends used, 3D status, column list, warnings |
| `descjocky/` | DescJocky working files (SMILES, SDFs, backend CSV) |
| `train_set_raw_descriptors.csv` / `val_set_raw_descriptors.csv` / `test_set_raw_descriptors.csv` | Post-split descriptor sets |
| `split_assignments.csv` / `umap_coordinates.csv` | Split and embedding coordinates |
| `umap_split.png` / `.svg` | UMAP split figure |
| `preprocessed_train_descriptors.csv` / `preprocessed_val_descriptors.csv` / `preprocessed_test_descriptors.csv` | Scaled descriptor matrices |
| `descriptor_preprocessor.joblib` | Fitted preprocessing pipeline |
| `descriptor_preprocessing_report.json` | Preprocessing summary |
| `removed_descriptors.csv` | Descriptors removed and reasons |
| `sfs_results.csv` | SFS R² vs descriptor count |
| `sfs_r2_vs_feature_count.png` / `.svg` | SFS performance plot |
| `selected_feature_count.json` | Chosen descriptor count |
| `feature_count_selection_explanation.md` | Agent/rule explanation |
| `ga_selected_features.json` / `ga_history.csv` | GA results |
| `ga_convergence.png` / `.svg` | GA convergence plot |
| `baseline_cv_metrics.csv` / `baseline_overfitting_assessment.json` | HPO baseline diagnostics |
| `hpo_iteration_log.md` / `hpo_final_selection.json` | HPO decisions (when enabled) |
| `predictions.csv` / `model_metrics.json` | Per-compound predictions and metrics |
| `final_model.joblib` | Trained Random Forest |
| `prediction_scatter.png` / `.svg` | Predicted vs experimental plot |
| `applicability_domain.csv` | Per-compound AD classification |
| `williams_plot.png` / `.svg` | Williams plot |
| `run_manifest.json` | Reproducibility metadata |
| `qsar_agent_run_<run_id>.zip` | Complete run archive |

## Configuration

Key defaults (adjustable in the Streamlit sidebar):

| Setting | Default |
|---------|---------|
| Validation fraction | 0.10 |
| External test fraction | 0.10 |
| Random seed | 42 |
| Missing-value threshold | 20% |
| Near-constant std threshold | 0.01 |
| Correlation threshold | 0.95 |
| Max SFS descriptors | 20 |
| CV folds | 5 |
| SFS / GA / model `n_jobs` | -1 (all cores) |
| Random Forest | 100 trees, max_depth=10 |
| GA population / generations | 50 / 30 |
| HPO enabled | true |
| Max HPO rounds | 3 |
| Max grid candidates / round | 120 |
| Minimum acceptable CV R² | 0.50 |
| Overfit gap threshold | 0.15 |

## Running tests

```bash
pytest tests/ -v
```

Skip the slow end-to-end smoke test:

```bash
pytest tests/ -v -m "not slow"
```

## Reproducibility

`run_manifest.json` records Python and package versions, random seeds, model parameters, selected descriptors, and a SHA-256 hash of the input dataset. Re-running with the same data, configuration, and package versions should yield the same results as closely as the libraries allow.

## Troubleshooting

### `streamlit streamlit_app.py` fails

Use `streamlit run streamlit_app.py` (the `run` subcommand is required).

### Workflow stuck at sequential feature selection

SFS evaluates many candidate feature subsets with cross-validation. With hundreds of retained descriptors this stage can take **15–30+ minutes** even with `n_jobs=-1`. Progress appears in the terminal running Streamlit (mlxtend logs like `Features: 3/20`). Reduce `Max SFS descriptors` or tighten preprocessing (higher correlation threshold, lower missing-value threshold) to retain fewer descriptors.

### OpenAI errors

The workflow continues without OpenAI; feature-count selection uses the deterministic one-standard-error rule. Check `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env` or `.streamlit/secrets.toml`.

### `PosixPath ... are the same file`

Fixed: when the dataset is already saved as `input_dataset.csv` in the run directory, the workflow skips redundant copying.

## Reference examples

The `examples/` directory contains the original scripts this project extends:

- `reg_cluster_split.py` — UMAP + KMeans cluster split
- `ga_feature_selection_regression.py` — DEAP genetic algorithm (test-set leakage corrected in QSAR Agent)
- `utils.py` — SFS, preprocessing helpers, UMAP clustering

## License

See [LICENSE](LICENSE).
