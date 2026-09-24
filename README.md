# QSAR Agent

**QSAR Agent** builds QSAR models from SMILES and experimental activity data. It offers two workflows over a shared set of deterministic scientific tools:

| | [Agentic CLI workflow](#agentic-cli-workflow) | [Streamlit workflow](#streamlit-workflow) |
|---|---|---|
| Entry point | `python -m qsar_agent.cli run ...` | `streamlit run streamlit_app.py` |
| Endpoints | Regression **and** classification | Regression only |
| Descriptors | RDKit descriptors, Morgan / MACCS / path / atom-pair fingerprints | DescJocky (RDKit + Mordred), optional xtb geometry |
| Feature selection | Correlation and univariate filters chosen per iteration by an agent | Sequential forward selection + genetic algorithm |
| Validation | Train/test split, K-fold CV, y-scrambling, k-NN applicability domain | Train/validation/test split, CV, Williams plot |
| Acceptance | Criteria you specify; the agents iterate until they are met | Reported for you to judge |
| LLM | Required | Optional |
| Typical runtime | Seconds per iteration | 15–30+ minutes |

> **Important:** in both workflows the test set is evaluated only after feature selection and model training. It is never used for preprocessing, tuning, or feature selection.

## Features

### Agentic CLI workflow

- Autonomous multi-agent loop: ingest, featurise, train, validate, diagnose, revise
- Regression and classification, with the endpoint type detected from the activity column
- RDKit descriptors plus Morgan, MACCS, path and atom-pair fingerprints
- Validation gates: hold-out split, K-fold CV, y-scrambling, k-NN applicability domain
- Acceptance criteria supplied by you, shown with literature-typical suggestions
- Iterates on failure against a structured diagnosis, and reports honestly when criteria are not met
- Tool-using LLM agents with web search; swappable provider including local OpenAI-compatible servers
- Random, stratified, and Bemis-Murcko scaffold splits

### Streamlit workflow

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
streamlit_app.py          # Streamlit UI (regression workflow)
qsar_agent/
  cli.py                  # CLI entry point for the agentic workflow
  config.py               # Workflow defaults and OpenAI settings
  app_state.py            # Streamlit session state helpers
  schemas/                # Pydantic models for tools, state, and reports
  llm/                    # Swappable LLM client, agent tool specs, test double
  tools/                  # Deterministic scientific pipeline stages
  agents/                 # Specialist agents and the LLM strategist
  services/               # Workflow runners, plotting, artifact management
examples/                 # Original reference scripts (UMAP split, GA, SFS)
example/                  # Sample input CSVs
scripts/                  # Sample-data generation and offline utilities
outputs/<run_id>/         # Isolated artifacts per workflow run
tests/                    # Unit and integration tests
```

LLM agents decide and explain; deterministic Python tools compute. No agent ever calculates a descriptor, trains a model, or produces a metric.

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

### LLM configuration

The **agentic CLI workflow requires** a language model and fails immediately with instructions if no key is present. The **Streamlit workflow is optional**: without a key it falls back to the deterministic one-standard-error rule and skips agent explanations.

**Environment variables** (copy from `.env.example`):

```bash
export OPENAI_API_KEY="sk-..."           # required for the agentic workflow
export QSAR_LLM_MODEL="gpt-4o-mini"      # optional, this is the default
```

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI credential | — |
| `QSAR_LLM_API_KEY` | credential for an OpenAI-compatible endpoint | falls back to `OPENAI_API_KEY` |
| `QSAR_LLM_MODEL` | model name (`OPENAI_MODEL` also honoured) | `gpt-4o-mini` |
| `QSAR_LLM_BASE_URL` | OpenAI-compatible endpoint (vLLM, Ollama, OpenRouter, Azure) | unset |
| `QSAR_LLM_PROVIDER` | registered provider name | `openai` |
| `QSAR_LLM_TEMPERATURE` | sampling temperature | `0` |
| `TAVILY_API_KEY` | preferred web-search backend | unset (falls back to DuckDuckGo) |

To use a local model, point `QSAR_LLM_BASE_URL` at any OpenAI-compatible server — no code change is needed:

```bash
export QSAR_LLM_BASE_URL="http://localhost:11434/v1"
export QSAR_LLM_API_KEY="ollama"
export QSAR_LLM_MODEL="llama3.1"
```

**Or Streamlit secrets** for the Streamlit workflow (copy from `.streamlit/secrets.toml.example`):

```toml
OPENAI_API_KEY = "your_key_here"
OPENAI_MODEL = "gpt-4o-mini"
```

Never commit `.env` or `.streamlit/secrets.toml` with real keys.

## Agentic CLI workflow

A team of agents takes a CSV of compounds and autonomously produces a validated model, iterating on its own design when validation falls short of the criteria **you** specify.

### Quick start

```bash
export OPENAI_API_KEY="sk-..."

python -m qsar_agent.cli run \
  --csv example/agentic_regression_sample.csv \
  --smiles-col smiles \
  --activity-col pIC50 \
  --id-col compound_id
```

The CLI detects whether the endpoint is continuous or categorical, then asks what you require of a validated model before any training starts. Press Enter to accept a suggestion, type your own number, or type `off` to drop a criterion:

```
Hold-out test R2 at least >= ?
  Coefficient of determination on the untouched test set.
  Suggested: 0.6 — Tropsha external-validation guidance (R2_test > 0.6)
  Your requirement [0.6]:
```

The exit code is `0` when the criteria were met and `1` when they were not.

### The agents

| Agent | Responsibility |
|---|---|
| **DataAgent** | Ingest the CSV, canonicalise SMILES with RDKit, resolve duplicates, detect the task, split the data |
| **DescriptorAgent** | Build the feature matrix for the current recipe, fitting every filter and scaler on training rows only |
| **ModelingAgent** | Instantiate the planned estimator and cross-validate it, scoring both halves of each fold |
| **ValidationAgent** | Measure hold-out and CV metrics, run y-scrambling, compute the applicability domain |
| **Strategist** | The only agent that changes the plan. Reads the full history and diagnosis, may search the web, and proposes the next feature recipe and model |

The Strategist is LLM-backed and can call tools while reasoning: `web_search`, `list_model_zoo`, `list_feature_blocks`, `get_iteration_history`, `get_dataset_summary`, and `get_acceptance_criteria`. Web search tries Tavily, then DuckDuckGo, and returns an empty result set with an explanatory note when the network is unavailable — it never aborts a run.

### Acceptance criteria

Thresholds are not library defaults, because what counts as good enough depends on your dataset and your decision. See every suggestion and where it comes from:

```bash
python -m qsar_agent.cli show-criteria
```

| Task | Criterion | Flag | Suggested | Basis |
|---|---|---|---|---|
| regression | hold-out test R² ≥ | `--min-test-r2` | 0.60 | Tropsha external-validation guidance |
| regression | cross-validated Q² ≥ | `--min-cv-q2` | 0.50 | OECD internal-validation floor |
| regression | train − CV R² gap ≤ | `--max-train-cv-gap` | 0.30 | overfitting guard |
| regression | test RMSE ≤ | `--max-test-rmse` | off | depends on your activity scale |
| regression | test MAE ≤ | `--max-test-mae` | off | depends on your activity scale |
| classification | hold-out ROC-AUC ≥ | `--min-roc-auc` | 0.70 | common acceptability floor |
| classification | MCC ≥ | `--min-mcc` | 0.30 | imbalance-robust agreement |
| classification | balanced accuracy ≥ | `--min-balanced-accuracy` | 0.65 | imbalance-robust accuracy |
| classification | cross-validated ROC-AUC ≥ | `--min-cv-roc-auc` | 0.70 | internal validation |
| both | mean y-scrambled score ≤ | `--max-scramble-score` | 0.20 / 0.60 | chance-correlation control |
| both | real − scrambled score ≥ | `--min-scramble-margin` | 0.25 / 0.10 | signal must beat chance |
| both | test AD coverage ≥ (%) | `--min-ad-coverage` | 80 | OECD applicability-domain principle |

Where two suggestions are shown, the first is for regression and the second for classification: a scrambled R² near zero is expected, whereas a scrambled ROC-AUC near 0.5 is.

RMSE, MAE, sensitivity, specificity, precision, recall and F1 are always computed and reported whether or not you gate on them.

For scripted runs, supply the values instead of being asked:

```bash
# Individual flags ('off' disables a criterion)
python -m qsar_agent.cli run --csv data.csv --smiles-col smiles --activity-col pIC50 \
  --min-test-r2 0.70 --min-cv-q2 0.60 --max-train-cv-gap 0.25 \
  --max-scramble-score 0.15 --min-scramble-margin 0.30 --min-ad-coverage 85

# A criteria file (null disables a criterion)
python -m qsar_agent.cli run --csv data.csv --smiles-col smiles --activity-col pIC50 \
  --criteria-file my_criteria.json --non-interactive

# The suggested values, without prompting
python -m qsar_agent.cli run --csv data.csv --smiles-col smiles --activity-col pIC50 \
  --accept-suggested
```

```json
{
  "min_test_r2": 0.70,
  "min_cv_q2": 0.60,
  "max_train_cv_gap": 0.25,
  "max_scramble_score": 0.15,
  "min_scramble_margin": 0.30,
  "min_ad_coverage_pct": 85.0
}
```

With no terminal to ask on and no criteria supplied, the CLI **exits with an error** rather than assuming what you wanted. Once a run starts the criteria are frozen: the Strategist can read them but has no way to relax them.

### How the agents iterate on failure

```
plan -> execute -> validate -> diagnose -> revise
```

When a candidate fails, the ValidationAgent classifies *why* — and that label, not the raw metrics, is what the Strategist acts on:

| Diagnosis | Meaning | Typical response |
|---|---|---|
| `chance_correlation` | scrambled models score nearly as well as the real one | cut the feature count hard, simplify the model |
| `insufficient_data` | too few compounds per feature for any metric to mean much | reduce dimensionality aggressively |
| `overfit` | fits training folds far better than held-out folds | constrain capacity, fewer features, simpler family |
| `poor_generalisation` | CV is healthy but the hold-out is not | regularise, broaden features, inspect the split |
| `unstable_cv` | fold scores vary too much for the mean to be trusted | filter noisy features, average more estimators |
| `narrow_applicability_domain` | performance passes but test compounds are extrapolations | drop domain outliers, reduce dimensionality |
| `class_imbalance` | skewed classes are driving the imbalance-sensitive failures | class weighting |
| `underfit` | nothing pathological, just not good enough yet | richer features, higher-capacity model |

Diagnoses are triaged in that order, because chance correlation invalidates any other reading of the metrics, and an overfitting signature calls for the opposite fix to an underfitting one.

The Strategist receives the frozen criteria, every previous iteration's recipe/metrics/diagnosis, the current diagnosis, and the signatures already tried. Its reply is validated against a Pydantic schema and must name registered feature blocks, registered estimators, and hyperparameters the estimator actually accepts; an invalid reply is returned with the error attached for repair. A proposal that repeats an already-tested configuration is rejected and a different one requested, so the loop cannot spin on one idea.

The loop stops when all criteria pass, `--max-iterations` (default 6) is reached, `--budget-seconds` (default 900) is exhausted, or the Strategist reports that no untried change is worth testing. **On stop-without-success the report says so plainly**, names the best iteration, lists exactly which criteria still fail, and shows everything that was tried. No threshold is relaxed to manufacture a pass, and there is no heuristic fallback: if the model cannot produce a valid plan the run ends with an error.

### Feature blocks and estimators

Feature blocks: `rdkit_descriptors` (~210 physicochemical and topological descriptors), `morgan_fp`, `morgan_counts`, `maccs_keys`, `rdkit_fp`, `atom_pair_fp`.

| Task | Estimators |
|---|---|
| Regression | RandomForest, ExtraTrees, GradientBoosting, Ridge, ElasticNet, SVR, KNN, PLS |
| Classification | RandomForest, ExtraTrees, GradientBoosting, LogisticRegression, SVC, KNN |

A recipe also controls fingerprint size and radius, near-constant and correlation filtering, univariate top-k selection, scaling, and whether to drop applicability-domain outliers from the training set.

### CLI options

```bash
python -m qsar_agent.cli run --help
```

| Group | Options |
|---|---|
| Data | `--csv`, `--smiles-col`, `--activity-col`, `--id-col`, `--task {auto,regression,classification}` |
| Splitting | `--split {random,stratified,scaffold}`, `--test-size`, `--cv-folds`, `--seed`, `--min-compounds` |
| Validation | `--scramble-repeats`, `--ad-k`, `--ad-z` |
| Budget | `--max-iterations`, `--budget-seconds`, `--n-jobs`, `--output-dir`, `--run-id` |
| Model | `--llm-provider`, `--llm-model` |
| Criteria | the flags above, `--criteria-file`, `--accept-suggested`, `--non-interactive` |

A `scaffold` split assigns whole Bemis-Murcko scaffold groups to one side, which is a harder and more realistic test of generalisation than a random split.

### Artifacts

Each run writes to `<output-dir>/<run_id>/`:

| File | Description |
|---|---|
| `agentic_cleaned_dataset.csv` | Canonicalised, deduplicated compounds |
| `agentic_invalid_rows.csv` | Rows excluded for unparseable SMILES or missing activity |
| `agentic_ingest_report.json` | Ingest counts, duplicate resolution, task detection |
| `agentic_split_assignments.csv` | Train/test assignment per compound |
| `iteration_NN/predictions.csv` | Per-compound observed and predicted values |
| `iteration_NN/selected_features.json` | Features surviving that iteration's filters |
| `iteration_NN/iteration_record.json` | Plan, metrics, verdict and diagnosis |
| `iteration_NN/model.joblib` | The fitted model, written for the accepted iteration |
| `agentic_run_report.json` | Complete machine-readable run record |
| `agentic_run_report.md` | Human-readable report including the honest assessment |

### Offline demo

`--llm-provider mock` substitutes a scripted test double for the model, so the loop can be demonstrated with no key and no network. It is for demos and tests only — the replies are canned, not reasoned:

```bash
python scripts/generate_sample_datasets.py   # regenerate the samples if needed

python -m qsar_agent.cli run \
  --csv example/agentic_regression_sample.csv \
  --smiles-col smiles --activity-col pIC50 --id-col compound_id \
  --llm-provider mock --accept-suggested --max-iterations 5
```

### Sample datasets

| File | Endpoint | Compounds |
|---|---|---|
| `example/agentic_regression_sample.csv` | `pIC50`, continuous | 284 |
| `example/agentic_classification_sample.csv` | `active`, binary (170 / 114) | 284 |

Both are built by `scripts/generate_sample_datasets.py` from a combinatorial library of real scaffolds, with activities drawn from a deliberately nonlinear function of RDKit descriptors. A linear model underfits them, so the loop has to iterate to succeed.

## Streamlit workflow

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

### Expected CSV format

| Column | Required | Description |
|--------|----------|-------------|
| SMILES | Yes | Structure strings (parsed with RDKit) |
| Activity | Yes | Continuous numeric endpoint (e.g. pIC50, log IC50) |
| Compound ID | No | Unique identifier; auto-generated if omitted |

Invalid SMILES, missing activities, and duplicates are reported separately—not silently dropped without a record.

### Workflow stages

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

### Hyperparameter optimization (HPO)

After genetic feature selection, the workflow can tune Random Forest hyperparameters using **only the preprocessed training set**.

#### What overfitting means here

Overfitting is detected from **training cross-validation** diagnostics, not external-test performance:

- **Train–CV R² gap** = mean training-fold R² minus mean validation-fold R²
- Large gap with high training R² → likely overfitting
- Low training and CV R² → underfitting
- High CV R² standard deviation → unstable model
- CV R² below the minimum threshold → poor performance

Default thresholds: gap > 0.15 (overfit), gap > 0.25 (severe warning), minimum CV R² = 0.50, CV std > 0.15 (unstable).

#### Why external test is excluded from HPO

The external test set is **never** used to propose grids, score candidates, assess overfitting, or select the final model. It is evaluated **once** after the final configuration is chosen and the model is retrained on all training compounds.

#### Agent-guided grids

When `OPENAI_API_KEY` is set, the agent proposes structured JSON hyperparameter grids (Random Forest only). Invalid responses trigger one repair attempt, then a deterministic fallback grid (regularization-focused for overfit, capacity-focused for underfit, stability-focused for unstable CV). The agent does **not** train models or invent metrics.

#### Maximum 3 HPO rounds

If the baseline model is acceptable, HPO is skipped. Otherwise the workflow runs up to **3 rounds** of grid search. Each round logs `HPO round X/3` in Streamlit and artifact logs. Search stops early when an acceptable model is found.

#### Disabling HPO

In the Streamlit sidebar, uncheck **Enable HPO** or set `hpo.enabled: false` in configuration. The workflow uses the default Random Forest settings from the Model settings panel.

#### HPO artifacts

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

#### Limitations

Automated overfitting detection from CV metrics is a heuristic. High descriptor-to-sample ratios, activity noise, and small training sets can produce misleading gaps or unstable CV scores. HPO improves regularization but does not guarantee external-test performance.

### Methodological notes

#### Near-constant filtering before StandardScaler

Near-constant descriptors are removed using raw training-set standard deviation (`std < 0.01`) **before** `StandardScaler` is fit. After scaling, every non-constant feature has ~unit variance, so post-scaling variance checks would fail to detect near-constant raw descriptors.

#### Preprocessing fitted only on training data

Missing-value thresholds, imputation values, correlation decisions, and scaling parameters are learned from the training set only and applied unchanged to the validation and external test sets.

#### Validation set (development) vs external test

Default split is **80% train / 10% validation / 10% test**. Feature selection, GA fitness, and HPO/model selection use **training K-fold CV plus held-out validation** (combined score = 0.5·CV R² + 0.5·val R²). Validation metrics are slightly optimistic because they influenced selection.

#### External test set isolation

The external test set is not used during:

- Descriptor preprocessing decisions
- Sequential feature selection
- Feature-count selection (one-standard-error rule on combined CV+val score)
- Genetic algorithm fitness
- Hyperparameter search and model selection

#### UMAP is not clustering

UMAP produces a 2D embedding; **KMeans** clusters that embedding (following the `examples/` reference code). Splitting is performed within each cluster to preserve chemical diversity.

#### Williams plot limitations

The Williams plot is a classical descriptor-space diagnostic based on leverage and standardized residuals. Interpret with caution for nonlinear models such as Random Forest.

#### Corrections from reference `examples/` code

- **GA fitness:** Combined training CV R² and held-out validation R² (the example GA used the test set for fitness—data leakage).
- **Preprocessing order:** Near-constant removal before scaling (examples scaled first).
- **SFS efficiency:** A single mlxtend SFS fit up to `max_features` reads all subset sizes from `sfs.subsets_` (the example `build_each_model` pattern), rather than re-fitting SFS separately for each feature count.

### Output files

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

### Configuration

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

Skip the slow end-to-end tests:

```bash
pytest tests/ -v -m "not slow"
```

Tests need neither network access nor an API key: the LLM client is stubbed with `MockLLMClient` or a purpose-built scripted double, and the web-search tool is monkeypatched. Coverage includes the tools (descriptors, metrics, model zoo, y-scrambling, applicability domain, splitting, ingest, search), the LLM layer's fail-fast configuration and tool-calling loop, the Strategist's schema validation and repair, the acceptance and diagnosis logic, the CLI's criteria resolution, and the iteration loop's stop conditions and honest reporting.

## Reproducibility

`run_manifest.json` records Python and package versions, random seeds, model parameters, selected descriptors, and a SHA-256 hash of the input dataset. Re-running with the same data, configuration, and package versions should yield the same results as closely as the libraries allow.

## Troubleshooting

### `streamlit streamlit_app.py` fails

Use `streamlit run streamlit_app.py` (the `run` subcommand is required).

### Workflow stuck at sequential feature selection

SFS evaluates many candidate feature subsets with cross-validation. With hundreds of retained descriptors this stage can take **15–30+ minutes** even with `n_jobs=-1`. Progress appears in the terminal running Streamlit (mlxtend logs like `Features: 3/20`). Reduce `Max SFS descriptors` or tighten preprocessing (higher correlation threshold, lower missing-value threshold) to retain fewer descriptors.

### OpenAI errors

In the Streamlit workflow the run continues without OpenAI; feature-count selection uses the deterministic one-standard-error rule. Check `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env` or `.streamlit/secrets.toml`.

### `No LLM API key found` from the CLI

The agentic workflow requires a language model and refuses to start without one. Set `OPENAI_API_KEY`, or `QSAR_LLM_API_KEY` together with `QSAR_LLM_BASE_URL` for a self-hosted endpoint. To demonstrate the loop with no key, use `--llm-provider mock` (scripted replies; not a substitute for real reasoning).

### The agentic run reports that criteria were not met

That is the honest outcome, not a bug. The report names the criteria that still fail and what was tried. Consider whether the criteria are stricter than your decision requires, whether more or cleaner data is available, whether a larger `--max-iterations` would help, or whether the endpoint is simply not predictable from structure at that accuracy.

### `pip` reports a `mordred` / `numpy` conflict

Expected. Mordred declares `numpy==1.*` but imports and runs against numpy 2.x. The warning is safe to ignore; the agentic workflow does not use Mordred at all.

### `PosixPath ... are the same file`

Fixed: when the dataset is already saved as `input_dataset.csv` in the run directory, the workflow skips redundant copying.

## Reference examples

The `examples/` directory contains the original scripts this project extends:

- `reg_cluster_split.py` — UMAP + KMeans cluster split
- `ga_feature_selection_regression.py` — DEAP genetic algorithm (test-set leakage corrected in QSAR Agent)
- `utils.py` — SFS, preprocessing helpers, UMAP clustering

## License

See [LICENSE](LICENSE).
