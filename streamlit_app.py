"""QSAR Agent Streamlit application."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from qsar_agent.app_state import init_session_state, reset_session, set_workflow_state
from qsar_agent.config import (
    ClusteringConfig,
    DescriptorConfig,
    GAConfig,
    HPOSettings,
    ModelConfig,
    ModelFallbackSettings,
    PreprocessingConfig,
    SFSConfig,
    UMAPConfig,
    WorkflowConfig,
    get_openai_api_key_source,
    get_openai_model,
    load_env_file,
)
from qsar_agent.schemas.agentic import AgenticAcceptanceCriteria, AgenticImprovementConfig
from qsar_agent.agentic.approvals import init_agentic_session_state, resolve_pending_approval
from qsar_agent.schemas.workflow import StageStatus
from qsar_agent.services.artifact_manager import generate_run_id, get_run_dir
from qsar_agent.services.workflow_runner import WorkflowRunner
from qsar_agent.tools.dataset_validation import validate_dataset

MAX_UPLOAD_MB = 50
APP_TITLE = "QSAR Agent"
TOTAL_STAGES = 16

# Ensure .env is loaded before the app reads OpenAI settings.
load_env_file()

st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
init_session_state()
init_agentic_session_state(st)


def file_download_button(label: str, path: str, mime: str = "application/octet-stream") -> None:
    p = Path(path)
    if p.exists():
        st.download_button(label, p.read_bytes(), file_name=p.name, mime=mime)


def render_header() -> None:
    st.title(f"🧪 {APP_TITLE}")
    st.markdown(
        "Build regression QSAR models from SMILES and experimental activity using "
        "DescJocky descriptors (optional external merge), UMAP-based cluster splitting, "
        "sequential and genetic feature selection, Random Forest modeling with automatic "
        "fallback to other regressors when HPO fails."
    )
    st.warning(
        "External-test performance is evaluated only after the winning configuration "
        "is locked using training CV (and optional agentic internal evidence). "
        "The external test is never used for tuning, branch comparison, or agent decisions."
    )
    if st.session_state.get("run_id"):
        st.info(f"Current run ID: `{st.session_state.run_id}`")


def render_openai_api_key_controls() -> None:
    """Show API key status; accept a paste fallback when .env/secrets have no key."""
    with st.sidebar.expander("OpenAI API key", expanded=True):
        key, source = get_openai_api_key_source()
        if source in {"environment", "streamlit_secrets"}:
            label = (
                ".env / environment variable"
                if source == "environment"
                else "Streamlit secrets"
            )
            st.success(f"OpenAI API key loaded successfully from {label}.")
            st.caption("Sidebar paste is only used when no key is found in .env or secrets.")
            return

        pasted = st.text_input(
            "Paste OpenAI API key",
            type="password",
            help="Used for HPO grid proposals when OPENAI_API_KEY is not in .env.",
            key="openai_api_key_input",
        )
        cleaned = pasted.strip().strip('"').strip("'") if pasted else ""
        st.session_state.openai_api_key = cleaned or None

        key, source = get_openai_api_key_source()
        if key and source == "ui":
            st.success("OpenAI API key loaded successfully from sidebar input.")
        else:
            st.warning(
                "No OpenAI API key found. HPO will use deterministic fallback grids "
                "unless you paste a key here or set OPENAI_API_KEY in .env."
            )


def render_sidebar_config() -> WorkflowConfig:
    st.sidebar.header("Configuration")
    render_openai_api_key_controls()
    mapping = st.session_state.get("column_mapping", {})

    test_fraction = st.sidebar.slider("External test fraction", 0.1, 0.4, 0.2, 0.05)
    random_seed = st.sidebar.number_input("Random seed", 0, 99999, 42)
    missing_thresh = st.sidebar.slider("Missing value threshold", 0.0, 0.5, 0.2, 0.05)
    near_const = st.sidebar.number_input("Near-constant std threshold", 0.0, 0.1, 0.01, 0.001)
    corr_thresh = st.sidebar.slider("Correlation threshold", 0.8, 0.99, 0.95, 0.01)
    max_sfs = st.sidebar.slider("Max SFS descriptors", 1, 20, 20)
    cv_folds = st.sidebar.number_input("CV folds", 2, 10, 5)
    output_dir = st.sidebar.text_input("Output directory", "outputs")

    with st.sidebar.expander("Model settings"):
        n_estimators = st.number_input("RF n_estimators", 10, 500, 100)
        max_depth = st.number_input("RF max_depth", 2, 50, 10)

    with st.sidebar.expander("Descriptors", expanded=True):
        desc_backends = st.multiselect(
            "DescJocky backends",
            options=["RDKit", "Mordred", "Native", "Pybel"],
            default=["RDKit", "Mordred"],
            help="Mordred/Native/Pybel require their packages. Geometry uses xtb when enabled.",
        )
        run_geom = st.checkbox(
            "Run geometry optimization (xtb)",
            value=False,
            help="Requires xtb on PATH. When off, RDKit SDFs are written and Phase 1 is skipped.",
        )
        desc_workers = st.number_input("Descriptor workers", 1, 32, 4)
        xtb_timeout = st.number_input("xtb timeout (seconds)", 60, 3600, 600)
        external_upload = st.file_uploader(
            "External descriptors CSV (optional)",
            type=["csv"],
            key="external_descriptors_uploader",
            help=(
                "Must include a compound_id column whose values match the dataset "
                "Compound ID column (e.g. C001). If Compound ID is left as (none), "
                "IDs become compound_0, compound_1, … and the merge will fail."
            ),
        )
        if external_upload is not None:
            st.session_state._external_descriptor_bytes = external_upload.getvalue()
        elif "_external_descriptor_bytes" not in st.session_state:
            st.session_state._external_descriptor_bytes = None

    with st.sidebar.expander("UMAP settings"):
        n_neighbors = st.number_input("UMAP n_neighbors", 2, 50, 15)
        min_dist = st.number_input("UMAP min_dist", 0.0, 1.0, 0.1)

    with st.sidebar.expander("GA settings"):
        pop_size = st.number_input("Population size", 10, 200, 50)
        n_gen = st.number_input("Generations", 5, 100, 30)
        cx_prob = st.slider("Crossover probability", 0.0, 1.0, 0.7)
        mut_prob = st.slider("Mutation probability", 0.0, 1.0, 0.2)

    with st.sidebar.expander("Hyperparameter Optimization", expanded=True):
        hpo_enabled = st.checkbox("Enable HPO", value=True)
        max_hpo_rounds = st.number_input("Max HPO rounds", 1, 3, 3)
        max_candidates = st.number_input("Max grid candidates per round", 20, 300, 120)
        min_cv_r2 = st.slider("Minimum acceptable CV R²", 0.0, 1.0, 0.50, 0.05)
        overfit_gap = st.slider("Overfit gap threshold", 0.05, 0.5, 0.15, 0.01)
        severe_overfit_gap = st.slider("Severe overfit gap threshold", 0.1, 0.6, 0.25, 0.01)
        cv_std_thresh = st.slider("CV std threshold", 0.05, 0.5, 0.15, 0.01)
        min_cv_improvement = st.slider("Min CV improvement to stop", 0.0, 0.2, 0.02, 0.01)
        hpo_n_jobs = st.number_input("HPO parallel jobs (-1 = all cores)", -1, 16, -1)
        hpo_openai_model = st.text_input(
            "OpenAI model for HPO grids",
            value=get_openai_model(),
        )

    model_fallback_enabled = st.sidebar.checkbox(
        "Try other models if RF HPO fails",
        value=True,
        help="Runs PLS, ExtraTrees, SVR, and KNN with per-model feature selection and HPO.",
    )

    with st.sidebar.expander("Agentic Improvement", expanded=False):
        agentic_enabled = st.checkbox(
            "Enable agentic improvement",
            value=False,
            help="After the deterministic pipeline, diagnose failed internal acceptance and run bounded experiments.",
        )
        agentic_max_cycles = st.number_input("Max cycles", 1, 10, 3)
        agentic_max_experiments = st.number_input("Max experiments", 1, 20, 8)
        agentic_min_cv = st.slider("Acceptance min mean CV R²", 0.0, 1.0, 0.60, 0.05)
        agentic_max_gap = st.slider("Acceptance max train–CV gap", 0.05, 0.5, 0.15, 0.01)
        agentic_max_std = st.slider("Acceptance max CV R² std", 0.05, 0.5, 0.15, 0.01)
        agentic_model = st.text_input(
            "Agent model",
            value=get_openai_model(),
            help="Used only when agentic mode is enabled and an API key is available.",
        )
        if st.button("Stop agentic loop (next cycle)"):
            st.session_state.agentic_stop_requested = True
        pending = st.session_state.get("agentic_pending_approval")
        if pending and pending.get("status") == "pending":
            st.warning("Pending approval required")
            st.json(pending)
            c1, c2 = st.columns(2)
            run_id = st.session_state.get("run_id")
            if run_id and c1.button("Approve"):
                resolve_pending_approval(Path("outputs") / run_id, approve=True)
                st.success("Approved")
            if run_id and c2.button("Reject"):
                resolve_pending_approval(Path("outputs") / run_id, approve=False)
                st.info("Rejected")

    with st.sidebar.expander("Resume from previous run", expanded=False):
        from qsar_agent.services.resume_agentic import list_resumable_runs

        resumable = list_resumable_runs(output_dir)
        if not resumable:
            st.caption("No resumable runs found (need preprocessed train + HPO/comparison).")
            st.session_state.resume_source_run_id = None
        else:
            def _resume_label(s) -> str:
                cv = f"{s.mean_cv_r2:.3f}" if s.mean_cv_r2 is not None else "?"
                return (
                    f"{s.run_id} | {s.estimator or '?'} | "
                    f"feats={s.feature_count} | CV R²={cv}"
                )

            labels = {s.run_id: _resume_label(s) for s in resumable}
            choice = st.selectbox(
                "Source run",
                options=[s.run_id for s in resumable],
                format_func=lambda rid: labels.get(rid, rid),
                key="resume_source_run_select",
            )
            st.session_state.resume_source_run_id = choice
            selected = next(s for s in resumable if s.run_id == choice)
            st.caption(
                f"Winner: `{selected.estimator}` · features={selected.feature_count} · "
                f"mean CV R²={selected.mean_cv_r2}"
            )
            if selected.external_previously_evaluated:
                st.warning(
                    "This run already evaluated the external test. Agentic resume will "
                    "**fork a new lineage** and cannot continue inside the original run."
                )
                default_eval_ext = False
            else:
                st.info("Source run has no external-test artifacts; fork will still be created.")
                default_eval_ext = True
            st.session_state.resume_evaluate_external = st.checkbox(
                "Evaluate external test after lock",
                value=default_eval_ext,
                help=(
                    "When the source already scored the external holdout, post-lock "
                    "evaluation on that same holdout is exploratory and not an "
                    "untouched independent test."
                ),
                key="resume_evaluate_external_cb",
            )

    return WorkflowConfig(
        test_fraction=test_fraction,
        random_seed=int(random_seed),
        output_dir=output_dir,
        smiles_column=mapping.get("smiles", ""),
        activity_column=mapping.get("activity", ""),
        id_column=mapping.get("id"),
        umap=UMAPConfig(n_neighbors=int(n_neighbors), min_dist=float(min_dist)),
        clustering=ClusteringConfig(),
        preprocessing=PreprocessingConfig(
            missing_value_threshold=missing_thresh,
            near_constant_std_threshold=float(near_const),
            correlation_threshold=corr_thresh,
        ),
        model=ModelConfig(n_estimators=int(n_estimators), max_depth=int(max_depth)),
        descriptors=DescriptorConfig(
            backends=list(desc_backends) if desc_backends else ["RDKit"],
            run_geometry_optimization=bool(run_geom),
            num_workers=int(desc_workers),
            xtb_timeout=int(xtb_timeout),
        ),
        ga=GAConfig(
            population_size=int(pop_size),
            n_generations=int(n_gen),
            crossover_prob=cx_prob,
            mutation_prob=mut_prob,
            cv_folds=int(cv_folds),
        ),
        sfs=SFSConfig(max_features=int(max_sfs), cv_folds=int(cv_folds)),
        hpo=HPOSettings(
            enabled=hpo_enabled,
            max_hpo_rounds=int(max_hpo_rounds),
            cv_folds=int(cv_folds),
            max_candidates_per_round=int(max_candidates),
            minimum_cv_r2=float(min_cv_r2),
            overfit_gap_threshold=float(overfit_gap),
            severe_overfit_gap_threshold=float(severe_overfit_gap),
            cv_std_threshold=float(cv_std_thresh),
            min_cv_improvement=float(min_cv_improvement),
            n_jobs=int(hpo_n_jobs),
            openai_model=hpo_openai_model,
        ),
        model_fallback=ModelFallbackSettings(enabled=model_fallback_enabled),
        agentic=AgenticImprovementConfig(
            enabled=bool(agentic_enabled),
            max_cycles=int(agentic_max_cycles),
            max_total_experiments=int(agentic_max_experiments),
            model=agentic_model,
            acceptance=AgenticAcceptanceCriteria(
                minimum_mean_cv_r2=float(agentic_min_cv),
                maximum_train_cv_gap=float(agentic_max_gap),
                maximum_cv_r2_std=float(agentic_max_std),
            ),
        ),
    )


def render_upload_section() -> bytes | None:
    st.header("Dataset Upload")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if not uploaded:
        return st.session_state.get("_upload_bytes")

    upload_bytes = uploaded.getvalue()
    st.session_state._upload_bytes = upload_bytes
    size_mb = len(upload_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"File too large ({size_mb:.1f} MB). Maximum: {MAX_UPLOAD_MB} MB.")
        return None

    df = pd.read_csv(io.BytesIO(upload_bytes))
    st.session_state.dataset_preview = df
    st.session_state.uploaded_filename = uploaded.name
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        smiles_col = st.selectbox("SMILES column", cols, key="smiles_col")
    with c2:
        activity_col = st.selectbox("Activity column", cols, key="activity_col")
    with c3:
        id_options = ["(none)"] + cols
        # When external descriptors are uploaded, prefer a real ID column so merges
        # do not silently join against auto-generated compound_0 / compound_1 IDs.
        default_id_index = 0
        if st.session_state.get("_external_descriptor_bytes"):
            from qsar_agent.tools.descriptor_calculation import suggest_dataset_id_column

            suggested = suggest_dataset_id_column(cols)
            if suggested and suggested in cols:
                default_id_index = id_options.index(suggested)
        id_col = st.selectbox(
            "Compound ID column (optional)",
            id_options,
            index=default_id_index,
            key="id_col",
            help=(
                "Required for external descriptor merge. Values must match the "
                "compound_id column in the external CSV."
            ),
        )

    st.session_state.column_mapping = {
        "smiles": smiles_col,
        "activity": activity_col,
        "id": None if id_col == "(none)" else id_col,
    }

    numeric = pd.to_numeric(df[activity_col], errors="coerce")
    st.write(
        f"Activity stats: min={numeric.min():.3f}, max={numeric.max():.3f}, "
        f"mean={numeric.mean():.3f}, missing={numeric.isna().sum()}"
    )

    if st.button("Validate Dataset"):
        with st.spinner("Validating..."):
            run_id = generate_run_id()
            run_dir = get_run_dir("outputs", run_id)
            tmp = run_dir / "upload_temp.csv"
            tmp.write_bytes(upload_bytes)
            try:
                result = validate_dataset(
                    tmp,
                    smiles_col,
                    activity_col,
                    None if id_col == "(none)" else id_col,
                    run_dir,
                )
                st.session_state.validation_result = result
                st.session_state.run_id = run_id
                st.success(
                    f"Validation complete: {result.valid_compound_count} valid compounds."
                )
                st.json(result.model_dump())
            except Exception as exc:
                st.error(str(exc))

    return upload_bytes


def render_workflow_execution(config: WorkflowConfig, upload_bytes: bytes | None) -> None:
    st.header("Workflow Execution")
    if not upload_bytes:
        st.info("Upload a dataset to run the workflow.")
        return
    if not config.smiles_column or not config.activity_column:
        st.warning("Select SMILES and activity columns before running.")
        return

    c1, c2 = st.columns(2)
    with c1:
        run_clicked = st.button("Run QSAR Workflow", type="primary")
    with c2:
        if st.button("Reset Session"):
            reset_session()
            st.rerun()

    if run_clicked:
        run_id = generate_run_id()
        run_dir = get_run_dir(config.output_dir, run_id)
        dataset_path = run_dir / "input_dataset.csv"
        dataset_path.write_bytes(upload_bytes)

        external_bytes = st.session_state.get("_external_descriptor_bytes")
        if external_bytes:
            external_path = run_dir / "external_descriptors.csv"
            external_path.write_bytes(external_bytes)
            # Auto-pick a dataset ID column if the user left it as (none).
            if not config.id_column:
                from qsar_agent.tools.descriptor_calculation import suggest_dataset_id_column

                preview = st.session_state.get("dataset_preview")
                cols = list(preview.columns) if preview is not None else []
                suggested = suggest_dataset_id_column(cols)
                if suggested:
                    config = config.model_copy(update={"id_column": suggested})
                    st.info(
                        f"External descriptors detected: using dataset column "
                        f"'{suggested}' as Compound ID so IDs align with the "
                        f"external CSV (e.g. C001)."
                    )
            config = config.model_copy(
                update={
                    "descriptors": config.descriptors.model_copy(
                        update={"external_descriptors_path": str(external_path)}
                    )
                }
            )

        progress = st.progress(0.0)
        status_placeholder = st.empty()
        stage_table = st.empty()

        def on_progress(state) -> None:
            completed = sum(
                1 for s in state.stages if s.status == StageStatus.COMPLETED
            )
            progress.progress(min(completed / TOTAL_STAGES, 1.0))
            running = [s.stage for s in state.stages if s.status == StageStatus.RUNNING]
            if running:
                status_placeholder.info(f"Running: {running[0]}")
            stage_table.dataframe(
                pd.DataFrame([s.model_dump() for s in state.stages]),
                use_container_width=True,
            )

        try:
            with st.status("Executing QSAR workflow...", expanded=True) as status:
                runner = WorkflowRunner(
                    config, dataset_path, progress_callback=on_progress, run_id=run_id
                )
                final_state = runner.run()
                set_workflow_state(final_state)
                status.update(label="Workflow complete!", state="complete")
            st.success("QSAR workflow finished successfully.")
        except Exception as exc:
            st.error(f"Workflow failed: {exc}")

    state = st.session_state.get("workflow_state")
    if state:
        st.subheader("Stage Status")
        st.dataframe(
            pd.DataFrame([s.model_dump() for s in state.stages]),
            use_container_width=True,
        )
        if state.logs:
            with st.expander("Logs"):
                st.text("\n".join(state.logs[-50:]))
        for warning in state.warnings:
            st.warning(warning)


def _load_branch_external_artifacts(run_dir: Path, artifacts: dict) -> list[dict]:
    path = artifacts.get("branch_external_artifacts")
    if not path:
        path = str(run_dir / "branch_external_artifacts.json")
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _render_branch_external_plots(
    run_dir: Path, artifacts: dict, *, show: str
) -> None:
    """Show per-branch scatter or Williams plots from branch_external_artifacts.json."""
    rows = _load_branch_external_artifacts(run_dir, artifacts)
    # Prefer non-winner branch dirs; still show all for comparison.
    if len(rows) <= 1:
        return
    st.caption(
        "Winner-only external evaluation is the default. Additional branch external "
        "plots appear only if legacy multi-branch evaluation was enabled."
    )
    labels = [r.get("label") or r.get("estimator") or f"branch_{i}" for i, r in enumerate(rows)]
    choice = st.selectbox(
        "Branch",
        options=list(range(len(rows))),
        format_func=lambda i: labels[i],
        key=f"branch_plot_select_{show}",
    )
    row = rows[choice]
    if show == "scatter":
        path = row.get("scatter_png_path") or ""
        caption = (
            f"{row.get('label', '')}: train R²={row.get('train_r2', float('nan')):.3f}, "
            f"test R²={row.get('test_r2', float('nan')):.3f}"
        )
    else:
        path = row.get("williams_png_path") or ""
        caption = f"{row.get('label', '')}: Williams plot"
    if path and Path(path).exists():
        st.image(path, caption=caption)
    else:
        st.caption(f"Plot not found for {row.get('label', 'branch')}.")


def _render_agentic_tab(run_dir: Path, artifacts: dict) -> None:
    st.subheader("Agentic improvement")
    st.caption(
        "Deterministic scientific results are authoritative. Agent text is advisory. "
        "Deterministic fallback decisions are labeled explicitly."
    )
    meta_path = run_dir / "agentic_resume_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        st.info(
            f"Resumed from source run `{meta.get('source_run_id')}`. "
            f"External previously evaluated: {meta.get('external_previously_evaluated')}."
        )
        if meta.get("external_previously_evaluated"):
            st.warning(
                "This forked lineage must not report the reused holdout as an untouched "
                "independent external test if that holdout was scored in the source run."
            )
    agent_ws = Path(artifacts.get("agent_workspace") or (run_dir / "agent_workspace"))
    state_path = agent_ws / "project_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        st.write(
            f"**Status:** {state.get('status')} | "
            f"**Cycle:** {state.get('cycle_index')} | "
            f"**Experiments:** {state.get('experiment_count')} | "
            f"**Best:** `{state.get('best_experiment_id')}`"
        )
        if state.get("last_acceptance"):
            st.write("Acceptance:", state["last_acceptance"].get("explanation"))
        if state.get("last_validation_review"):
            st.write("Validation:", state["last_validation_review"].get("summary"))
        if state.get("lock_record"):
            st.json(state["lock_record"])
    ledger = agent_ws / "experiment_ledger.jsonl"
    if ledger.exists():
        rows = []
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rows.append(
                    {
                        "experiment_id": rec.get("experiment_id"),
                        "parent": rec.get("parent_experiment_id"),
                        "action": rec.get("action"),
                        "kind": rec.get("experiment_kind"),
                        "multi_component": rec.get("multi_component"),
                        "estimator": rec.get("estimator"),
                        "mean_cv_r2": (rec.get("internal_metrics") or {}).get("mean_cv_r2"),
                        "decision_source": rec.get("decision_source"),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows))
    report_md = agent_ws / "final_agent_report.md"
    if report_md.exists():
        st.markdown(report_md.read_text(encoding="utf-8"))
        file_download_button("Download agentic report", str(report_md), "text/markdown")
    events = agent_ws / "agent_events.jsonl"
    if events.exists():
        with st.expander("Agent event log"):
            st.text(events.read_text(encoding="utf-8")[-8000:])
    disclaimer = run_dir / "locked_external" / "external_independence_disclaimer.json"
    if disclaimer.exists():
        st.error(json.loads(disclaimer.read_text(encoding="utf-8")).get("disclaimer", ""))
    if not state_path.exists():
        st.info("No agentic workspace for this run (agentic mode disabled or not triggered).")


def render_results_dashboard() -> None:
    report = st.session_state.get("final_report")
    artifacts = st.session_state.get("artifact_paths", {})
    run_id = st.session_state.get("run_id")
    if not report and not (run_id and artifacts):
        return

    run_dir = Path("outputs") / run_id if run_id else Path("outputs")
    if artifacts.get("prediction_scatter"):
        run_dir = Path(artifacts["prediction_scatter"]).parent
    elif artifacts.get("agent_workspace"):
        run_dir = Path(artifacts["agent_workspace"]).parent

    st.header("Results Dashboard")
    tabs = st.tabs(
        [
            "Dataset",
            "Descriptors",
            "Split",
            "Feature Selection",
            "Hyperparameter Optimization",
            "Model",
            "Applicability Domain",
            "Agentic",
            "Downloads",
            "Logs",
        ]
    )

    if report is None:
        # Agentic-resume path: show Agentic / Downloads / Logs only.
        for i in range(7):
            with tabs[i]:
                st.info("Full pipeline report not available (agentic-only resume). See Agentic tab.")
        with tabs[7]:
            _render_agentic_tab(run_dir, artifacts)
        with tabs[8]:
            for name, path in artifacts.items():
                file_download_button(f"Download {name}", path)
        with tabs[9]:
            ws = st.session_state.get("workflow_state")
            if ws and ws.logs:
                st.text("\n".join(ws.logs))
        return

    with tabs[0]:
        st.json(report.model_dump())
        if artifacts.get("cleaned_dataset"):
            st.dataframe(pd.read_csv(artifacts["cleaned_dataset"]).head())

    with tabs[1]:
        initial = getattr(report, "initial_descriptor_count", None)
        if initial is None:
            initial = getattr(report, "initial_mordred_descriptors", "?")
        st.write(
            f"Descriptors (raw): {initial} | "
            f"Preprocessed: {report.final_preprocessed_descriptors}"
        )
        calc_md = artifacts.get("descriptor_calculation_report_md")
        if calc_md and Path(calc_md).exists():
            st.markdown(Path(calc_md).read_text(encoding="utf-8"))
        calc_json = artifacts.get("descriptor_calculation_report")
        if not calc_json and artifacts.get("descriptors_raw"):
            calc_json = str(
                Path(artifacts["descriptors_raw"]).parent
                / "descriptor_calculation_report.json"
            )
        if calc_json and Path(calc_json).exists():
            with st.expander("Descriptor report JSON"):
                st.json(json.loads(Path(calc_json).read_text(encoding="utf-8")))
        if artifacts.get("generated_descriptors"):
            file_download_button(
                "Download generated descriptors CSV",
                artifacts["generated_descriptors"],
                "text/csv",
            )
        if artifacts.get("external_descriptors"):
            st.caption(f"External descriptors: {artifacts['external_descriptors']}")

    with tabs[2]:
        umap_plot = artifacts.get("umap_plot")
        if umap_plot and Path(umap_plot).exists():
            st.image(umap_plot)

    with tabs[3]:
        if artifacts.get("sfs_results"):
            sfs_png = Path(artifacts["sfs_results"]).parent / "sfs_r2_vs_feature_count.png"
            if sfs_png.exists():
                st.image(str(sfs_png))
        st.markdown(report.agent_explanation)
        if artifacts.get("ga_selected_features"):
            st.json(pd.read_json(artifacts["ga_selected_features"]))

    with tabs[4]:
        run_dir = Path(artifacts.get("run_manifest", "")).parent if artifacts.get("run_manifest") else None
        if run_dir and run_dir.exists():
            baseline_json = run_dir / "baseline_overfitting_assessment.json"
            if baseline_json.exists():
                st.subheader("Baseline overfitting assessment")
                st.json(json.loads(baseline_json.read_text(encoding="utf-8")))
            baseline_csv = run_dir / "baseline_cv_metrics.csv"
            if baseline_csv.exists():
                st.subheader("Baseline CV metrics")
                st.dataframe(pd.read_csv(baseline_csv), use_container_width=True)
            hpo_log = run_dir / "hpo_iteration_log.md"
            if hpo_log.exists():
                st.subheader("HPO iteration log")
                st.markdown(hpo_log.read_text(encoding="utf-8"))
            final_sel = run_dir / "hpo_final_selection.json"
            if final_sel.exists():
                st.subheader("Final selected configuration")
                st.json(json.loads(final_sel.read_text(encoding="utf-8")))
            summary_plot = artifacts.get("hpo_summary_plot") or str(run_dir / "hpo_summary.png")
            if Path(summary_plot).exists():
                st.image(summary_plot)
            for i in (1, 2, 3):
                perf = run_dir / f"hpo_round_{i}_performance.png"
                if perf.exists():
                    st.subheader(f"HPO round {i} performance")
                    st.image(str(perf))
            hpo_downloads = [
                "baseline_cv_metrics.csv",
                "baseline_overfitting_assessment.json",
                "final_overfitting_assessment.json",
                "hpo_iteration_log.json",
                "hpo_iteration_log.md",
                "hpo_final_selection.json",
                "hpo_final_selection_explanation.md",
                "hpo_all_rounds_summary.csv",
                "hpo_summary.png",
                "hpo_summary.csv",
            ]
            for name in hpo_downloads:
                p = run_dir / name
                if p.exists():
                    file_download_button(f"Download {name}", str(p))
            comparison_csv = artifacts.get("model_comparison_csv")
            if comparison_csv and Path(comparison_csv).exists():
                st.subheader("Model comparison (RF + fallbacks)")
                st.dataframe(pd.read_csv(comparison_csv), use_container_width=True)
            comparison_md = run_dir / "model_comparison_summary.md"
            if comparison_md.exists():
                st.markdown(comparison_md.read_text(encoding="utf-8"))
        else:
            st.info("HPO artifacts will appear here after a workflow run with HPO enabled.")

    with tabs[5]:
        st.write(f"**Estimator:** {getattr(report, 'estimator', 'RandomForestRegressor')}")
        if getattr(report, "model_comparison_summary", ""):
            st.info(report.model_comparison_summary)
        st.subheader("Winning model")
        scatter = artifacts.get("prediction_scatter")
        if scatter and Path(scatter).exists():
            st.image(scatter)
        st.write("Training:", report.train_metrics)
        st.write("External test:", report.test_metrics)
        if hasattr(report, "train_metrics"):
            ws = st.session_state.get("workflow_state")
            if ws and ws.config_snapshot.get("hpo", {}).get("enabled"):
                st.caption("Final model selected using training CV only before external-test evaluation.")
        _render_branch_external_plots(run_dir, artifacts, show="scatter")

    with tabs[6]:
        st.subheader("Winning model")
        williams = artifacts.get("williams_plot")
        if williams and Path(williams).exists():
            st.image(williams)
        st.write(report.applicability_domain_summary)
        _render_branch_external_plots(run_dir, artifacts, show="williams")

    with tabs[7]:
        _render_agentic_tab(run_dir, artifacts)

    with tabs[8]:
        for name, path in artifacts.items():
            file_download_button(f"Download {name}", path)
        ws = st.session_state.get("workflow_state")
        if ws and ws.zip_path:
            file_download_button("Download complete run ZIP", ws.zip_path, "application/zip")

    with tabs[9]:
        ws = st.session_state.get("workflow_state")
        if ws and ws.logs:
            st.text("\n".join(ws.logs))


def render_agentic_resume_section(config: WorkflowConfig) -> None:
    """Run agentic improvement only from a previously completed deterministic run."""
    st.header("Resume Agentic Only")
    source_id = st.session_state.get("resume_source_run_id")
    if not source_id:
        st.info(
            "Select a previous run under **Resume from previous run** in the sidebar. "
            "Dataset re-upload is not required for this path."
        )
        return

    source_dir = Path(config.output_dir) / source_id
    st.write(f"Source run: `{source_id}`")
    if not config.agentic.enabled:
        st.warning("Enable **Agentic Improvement** in the sidebar before running agentic-only resume.")

    evaluate_external = bool(st.session_state.get("resume_evaluate_external", False))
    run_resume = st.button(
        "Run agentic only",
        type="primary",
        disabled=not config.agentic.enabled,
        key="run_agentic_only_btn",
    )
    if not run_resume:
        return

    from qsar_agent.services.resume_agentic import (
        InPlaceResumeForbiddenError,
        detect_external_access,
        run_agentic_only,
    )

    ext = detect_external_access(source_dir)
    if ext.external_previously_evaluated:
        st.warning(
            "Forking a new lineage because the source run already has external-test artifacts. "
            f"Reasons: {', '.join(ext.reasons)}"
        )

    progress = st.empty()
    logs: list[str] = []

    def _log(msg: str) -> None:
        logs.append(msg)
        progress.text("\n".join(logs[-20:]))

    try:
        with st.spinner("Running agentic improvement on forked lineage..."):
            result = run_agentic_only(
                source_dir,
                workflow_config=config.model_copy(
                    update={"agentic": config.agentic.model_copy(update={"enabled": True})}
                ),
                output_root=config.output_dir,
                evaluate_external_after_lock=evaluate_external,
                log_callback=_log,
                stop_check=lambda: bool(st.session_state.get("agentic_stop_requested")),
            )
        st.session_state.run_id = result.forked_run_id
        st.session_state.artifact_paths = dict(result.artifact_paths)
        st.session_state.agentic_project_state = result.agentic_state.model_dump()
        st.session_state.agentic_stop_requested = False
        # Minimal workflow_state-like fields for downloads/logs
        from qsar_agent.schemas.workflow import WorkflowState

        ws = WorkflowState.create(
            result.forked_run_id, config_snapshot=config.to_dict()
        )
        ws.artifact_paths = dict(result.artifact_paths)
        ws.logs = logs
        set_workflow_state(ws)
        st.success(
            f"Agentic resume complete. Forked run: `{result.forked_run_id}` "
            f"(source `{result.source_run_id}`)."
        )
        if result.disclaimer:
            st.error(result.disclaimer)
        st.json(
            {
                "forked_run_id": result.forked_run_id,
                "external_previously_evaluated": result.external_previously_evaluated,
                "evaluated_external": result.evaluated_external,
                "status": result.agentic_state.status,
                "best_experiment_id": result.agentic_state.best_experiment_id,
            }
        )
    except InPlaceResumeForbiddenError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Agentic resume failed: {exc}")
        raise


def main() -> None:
    render_header()
    upload_bytes = render_upload_section()
    config = render_sidebar_config()
    render_workflow_execution(config, upload_bytes)
    render_agentic_resume_section(config)
    render_results_dashboard()


if __name__ == "__main__":
    main()
