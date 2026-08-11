"""Read-only post-external-test audit (no retrain, retune, or lineage mutation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from qsar_agent.agentic.ledger import get_experiment, load_project_state
from qsar_agent.schemas.post_test_audit import (
    BootstrapCI,
    DiagnosticFlag,
    MetricBlock,
    PostTestAuditCriteria,
    PostTestAuditResult,
    SubgroupMetrics,
)
from qsar_agent.services.artifact_manager import save_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_frozen_criteria(run_dir: Path) -> PostTestAuditCriteria:
    """Load criteria snapshot; fall back to defaults if missing."""
    for candidate in (
        Path(run_dir) / "agent_workspace" / "post_test_audit_criteria.json",
        Path(run_dir) / "locked_external" / "post_test_audit_criteria.json",
    ):
        data = _load_json(candidate)
        if data:
            return PostTestAuditCriteria(**data)
    return PostTestAuditCriteria()


def _metric_block(
    *,
    mean_r2: float | None = None,
    rmse: float | None = None,
    mae: float | None = None,
    n: int | None = None,
    source: str = "",
) -> MetricBlock:
    return MetricBlock(mean_r2=mean_r2, rmse=rmse, mae=mae, n=n, source=source)


def _metrics_from_y(y_true: np.ndarray, y_pred: np.ndarray, source: str) -> MetricBlock:
    if len(y_true) == 0:
        return _metric_block(n=0, source=source)
    return _metric_block(
        mean_r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        n=int(len(y_true)),
        source=source,
    )


def _bootstrap_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    criteria: PostTestAuditCriteria,
) -> list[BootstrapCI]:
    n = len(y_true)
    out: list[BootstrapCI] = []
    if n < criteria.minimum_bootstrap_n:
        warn = (
            f"External n={n} < minimum_bootstrap_n={criteria.minimum_bootstrap_n}; "
            "bootstrap CIs skipped."
        )
        for name in ("r2", "rmse", "mae"):
            out.append(
                BootstrapCI(
                    metric=name,
                    n_samples=n,
                    n_bootstrap=0,
                    available=False,
                    warning=warn,
                )
            )
        return out

    rng = np.random.default_rng(criteria.bootstrap_seed)
    alpha = 1.0 - criteria.confidence_level
    lower_q = 100 * (alpha / 2.0)
    upper_q = 100 * (1.0 - alpha / 2.0)

    def _point(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
        return {
            "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
        }

    point = _point(y_true, y_pred)
    samples = {k: [] for k in point}
    for _ in range(criteria.bootstrap_samples):
        idx = rng.integers(0, n, size=n)
        try:
            boot = _point(y_true[idx], y_pred[idx])
        except Exception:
            continue
        for k, v in boot.items():
            samples[k].append(v)

    for name in ("r2", "rmse", "mae"):
        arr = np.asarray(samples[name], dtype=float)
        if len(arr) == 0:
            out.append(
                BootstrapCI(
                    metric=name,
                    estimate=point[name],
                    n_samples=n,
                    n_bootstrap=0,
                    available=False,
                    warning="Bootstrap failed to produce samples.",
                )
            )
            continue
        out.append(
            BootstrapCI(
                metric=name,
                estimate=point[name],
                lower=float(np.percentile(arr, lower_q)),
                upper=float(np.percentile(arr, upper_q)),
                n_samples=n,
                n_bootstrap=len(arr),
                available=True,
            )
        )
    return out


def _subgroup_metrics(
    pred_df: pd.DataFrame,
    mask: pd.Series,
    label: str,
    criteria: PostTestAuditCriteria,
) -> SubgroupMetrics:
    sub = pred_df.loc[mask]
    n = len(sub)
    if n == 0:
        return SubgroupMetrics(
            label=label,
            n=0,
            reliable=False,
            warning="Empty subgroup.",
        )
    if n < criteria.minimum_subgroup_n:
        m = _metrics_from_y(
            sub["activity"].to_numpy(dtype=float),
            sub["predicted_activity"].to_numpy(dtype=float),
            source=label,
        )
        return SubgroupMetrics(
            label=label,
            n=n,
            r2=m.mean_r2,
            rmse=m.rmse,
            mae=m.mae,
            reliable=False,
            warning=(
                f"Subgroup n={n} < minimum_subgroup_n={criteria.minimum_subgroup_n}; "
                "metrics shown but not reliable."
            ),
        )
    m = _metrics_from_y(
        sub["activity"].to_numpy(dtype=float),
        sub["predicted_activity"].to_numpy(dtype=float),
        source=label,
    )
    return SubgroupMetrics(
        label=label,
        n=n,
        r2=m.mean_r2,
        rmse=m.rmse,
        mae=m.mae,
        reliable=True,
    )


def _find_grouped_cv_artifacts(run_dir: Path) -> dict[str, Any]:
    """Report random vs grouped CV only if pre-lock artifacts already exist."""
    candidates = [
        Path(run_dir) / "agent_workspace" / "grouped_cv_comparison.json",
        Path(run_dir) / "grouped_cv_comparison.json",
        Path(run_dir) / "cluster_cv_comparison.json",
        Path(run_dir) / "agent_workspace" / "random_vs_grouped_cv.json",
    ]
    for path in candidates:
        data = _load_json(path)
        if data:
            return {
                "available": True,
                "status": "available",
                "path": str(path),
                "comparison": data,
                "recommendation": None,
            }
    return {
        "available": False,
        "status": "unavailable",
        "recommendation": (
            "Grouped/cluster-aware CV was not generated before model lock. "
            "Recommend generating it in a future development lineage before locking."
        ),
    }


def _build_recommendations(
    *,
    primary_failed: bool,
    flags: list[DiagnosticFlag],
    failed_criteria: list[str],
    grouped_cv: dict[str, Any],
) -> list[str]:
    recs: list[str] = []
    if primary_failed:
        recs.append(
            "External validation failed against frozen criteria. "
            "Do not unlock or continue agentic optimization in this lineage. "
            "Start a new improvement lineage (fork) if remediation is needed."
        )
    if "external_distribution_shift" in flags:
        recs.append(
            "Consider collecting more representative external compounds or "
            "revisiting chemical-space coverage in a future lineage."
        )
    if "applicability_domain_failure" in flags:
        recs.append(
            "Low AD coverage on the external set suggests representation or "
            "domain-definition changes in a future lineage."
        )
    if "external_metric_unstable" in flags or "small_external_test" in flags:
        recs.append(
            "External metrics are unstable or based on a small holdout; "
            "gather a larger independent test set before claiming performance."
        )
    if "internal_validation_mismatch" in flags:
        recs.append(
            "Large CV–external gap: prefer stronger internal validation "
            "(e.g. grouped CV) and avoid over-interpreting CV scores."
        )
    if "influential_compounds_detected" in flags:
        recs.append(
            "Review influential / high-leverage external compounds; "
            "consider data quality checks in a future lineage."
        )
    if not grouped_cv.get("available"):
        recs.append(str(grouped_cv.get("recommendation") or ""))
    for item in failed_criteria:
        if item not in " ".join(recs):
            recs.append(f"Failed criterion: {item}")
    return [r for r in recs if r]


def run_post_test_audit(
    run_dir: Path,
    *,
    criteria: PostTestAuditCriteria | None = None,
) -> PostTestAuditResult:
    """
    Read-only audit after locked external evaluation.

    Never trains models, mutates the lock record, or appends ledger experiments.
    """
    run_dir = Path(run_dir)
    locked_ext = run_dir / "locked_external"
    criteria_path = run_dir / "agent_workspace" / "post_test_audit_criteria.json"
    if criteria is None:
        criteria = load_frozen_criteria(run_dir)
    else:
        # Prefer frozen snapshot if present
        frozen = _load_json(criteria_path)
        if frozen:
            criteria = PostTestAuditCriteria(**frozen)

    warnings: list[str] = []
    evidence: dict[str, str] = {}

    metrics_path = locked_ext / "model_metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "model_metrics.json"
    metrics_data = _load_json(metrics_path) or {}
    if metrics_path.exists():
        evidence["model_metrics"] = str(metrics_path)

    pred_path = locked_ext / "predictions.csv"
    if not pred_path.exists():
        pred_path = run_dir / "predictions.csv"
    if pred_path.exists():
        evidence["predictions"] = str(pred_path)

    ad_path = locked_ext / "applicability_domain.csv"
    if not ad_path.exists():
        ad_path = run_dir / "applicability_domain.csv"
    if ad_path.exists():
        evidence["applicability_domain"] = str(ad_path)

    train_block = _metric_block(source="model_metrics.train")
    external_block = _metric_block(source="model_metrics.test")
    if metrics_data.get("train"):
        t = metrics_data["train"]
        train_block = _metric_block(
            mean_r2=t.get("r2"),
            rmse=t.get("rmse"),
            mae=t.get("mae"),
            source="model_metrics.train",
        )
    if metrics_data.get("test"):
        t = metrics_data["test"]
        external_block = _metric_block(
            mean_r2=t.get("r2"),
            rmse=t.get("rmse"),
            mae=t.get("mae"),
            source="model_metrics.test",
        )

    # Internal / agent-val from lock / experiment
    cv_block = _metric_block(source="unavailable")
    agent_val_block = _metric_block(source="unavailable")
    state = load_project_state(run_dir)
    locked_exp = None
    if state and state.locked_experiment_id:
        locked_exp = get_experiment(run_dir, state.locked_experiment_id)
        if locked_exp is not None:
            evidence["locked_experiment"] = (
                f"agent_workspace/experiments/{locked_exp.experiment_id}"
            )
            im = locked_exp.internal_metrics or {}
            if im.get("mean_cv_r2") is not None:
                cv_block = _metric_block(
                    mean_r2=float(im["mean_cv_r2"]),
                    source="locked_experiment.internal_metrics",
                )
            if im.get("mean_train_r2") is not None and train_block.mean_r2 is None:
                train_block = _metric_block(
                    mean_r2=float(im["mean_train_r2"]),
                    source="locked_experiment.internal_metrics",
                )
            if im.get("agent_val_r2") is not None:
                agent_val_block = _metric_block(
                    mean_r2=float(im["agent_val_r2"]),
                    source="locked_experiment.internal_metrics",
                )

    # Also check HPO metadata in model_metrics
    hpo = metrics_data.get("hyperparameter_optimization") or {}
    if cv_block.mean_r2 is None and hpo.get("mean_cv_r2") is not None:
        cv_block = _metric_block(mean_r2=float(hpo["mean_cv_r2"]), source="hpo_metadata")
    if cv_block.mean_r2 is None and hpo.get("winning_mean_cv_r2") is not None:
        cv_block = _metric_block(
            mean_r2=float(hpo["winning_mean_cv_r2"]), source="hpo_metadata"
        )

    # Agent-visible summary if present
    if state and state.locked_experiment_id:
        summary_path = (
            run_dir
            / "agent_workspace"
            / "experiments"
            / state.locked_experiment_id
            / "agent_visible_summary.json"
        )
        summary = _load_json(summary_path)
        if summary:
            evidence["agent_visible_summary"] = str(summary_path)
            m = summary.get("metrics") or summary
            if cv_block.mean_r2 is None and m.get("mean_cv_r2") is not None:
                cv_block = _metric_block(
                    mean_r2=float(m["mean_cv_r2"]),
                    source="agent_visible_summary",
                )
            if agent_val_block.mean_r2 is None and m.get("agent_val_r2") is not None:
                agent_val_block = _metric_block(
                    mean_r2=float(m["agent_val_r2"]),
                    source="agent_visible_summary",
                )

    pred_df = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    test_pred = pred_df[pred_df["split"] == "test"] if not pred_df.empty else pred_df
    if not test_pred.empty:
        y_true = test_pred["activity"].to_numpy(dtype=float)
        y_pred = test_pred["predicted_activity"].to_numpy(dtype=float)
        external_block = _metrics_from_y(y_true, y_pred, source="predictions.csv:test")
        external_n = len(test_pred)
        act_min = float(np.min(y_true))
        act_max = float(np.max(y_true))
        act_var = float(np.var(y_true))
    else:
        y_true = np.array([])
        y_pred = np.array([])
        external_n = int(external_block.n or 0)
        act_min = act_max = act_var = None
        warnings.append("No external predictions found for audit.")

    bootstrap_cis = _bootstrap_cis(y_true, y_pred, criteria) if len(y_true) else []

    residual_outlier_ids: list[str] = []
    influential_ids: list[str] = []
    ad_coverage: float | None = None
    in_domain_m: SubgroupMetrics | None = None
    out_domain_m: SubgroupMetrics | None = None

    if ad_path.exists() and not test_pred.empty:
        ad_df = pd.read_csv(ad_path)
        ad_test = ad_df[ad_df["split"] == "test"].copy()
        if "compound_id" in ad_test.columns:
            merged = test_pred.merge(
                ad_test[
                    [
                        c
                        for c in (
                            "compound_id",
                            "in_domain",
                            "leverage",
                            "standardized_residual",
                            "applicability_domain",
                            "h_star",
                        )
                        if c in ad_test.columns
                    ]
                ],
                on="compound_id",
                how="left",
            )
        else:
            merged = test_pred.copy()
            merged["in_domain"] = True

        if "in_domain" in merged.columns:
            in_mask = merged["in_domain"].fillna(False).astype(bool)
            ad_coverage = float(in_mask.mean()) if len(merged) else None
            in_domain_m = _subgroup_metrics(merged, in_mask, "in_domain", criteria)
            out_domain_m = _subgroup_metrics(merged, ~in_mask, "out_of_domain", criteria)
            if in_domain_m.warning:
                warnings.append(in_domain_m.warning)
            if out_domain_m.warning:
                warnings.append(out_domain_m.warning)

        z = criteria.residual_outlier_z
        if "standardized_residual" in merged.columns:
            residual_outlier_ids = [
                str(cid)
                for cid in merged.loc[
                    merged["standardized_residual"].abs() > z, "compound_id"
                ].tolist()
            ]
        # Influential: high leverage and/or residual outlier on external
        if "leverage" in merged.columns and "h_star" in merged.columns:
            high_lev = merged["leverage"] > merged["h_star"]
            resid_flag = (
                merged["standardized_residual"].abs() > z
                if "standardized_residual" in merged.columns
                else pd.Series(False, index=merged.index)
            )
            influential_ids = [
                str(cid)
                for cid in merged.loc[high_lev | resid_flag, "compound_id"].tolist()
            ]
        elif residual_outlier_ids:
            influential_ids = list(residual_outlier_ids)

    train_cv_gap = None
    if train_block.mean_r2 is not None and cv_block.mean_r2 is not None:
        train_cv_gap = float(train_block.mean_r2 - cv_block.mean_r2)
    cv_test_gap = None
    if cv_block.mean_r2 is not None and external_block.mean_r2 is not None:
        cv_test_gap = float(cv_block.mean_r2 - external_block.mean_r2)

    grouped_cv = _find_grouped_cv_artifacts(run_dir)
    if grouped_cv.get("path"):
        evidence["grouped_cv_comparison"] = str(grouped_cv["path"])

    failed_criteria: list[str] = []
    flags: list[DiagnosticFlag] = []

    ext_r2 = external_block.mean_r2
    if ext_r2 is None or ext_r2 < criteria.minimum_external_r2:
        failed_criteria.append(
            f"minimum_external_r2 ({ext_r2} < {criteria.minimum_external_r2})"
        )
    if cv_test_gap is not None and cv_test_gap > criteria.maximum_cv_test_r2_gap:
        failed_criteria.append(
            f"maximum_cv_test_r2_gap ({cv_test_gap:.4f} > {criteria.maximum_cv_test_r2_gap})"
        )
        flags.append("internal_validation_mismatch")
    if ad_coverage is not None and ad_coverage < criteria.minimum_ad_coverage:
        failed_criteria.append(
            f"minimum_ad_coverage ({ad_coverage:.4f} < {criteria.minimum_ad_coverage})"
        )
        flags.append("applicability_domain_failure")

    primary: Any = (
        "external_validation_failed" if failed_criteria else "external_validation_passed"
    )

    if external_n < criteria.minimum_bootstrap_n:
        flags.append("small_external_test")
    unstable = any(
        (ci.available and ci.metric == "r2" and ci.lower is not None and ci.lower < 0.0)
        for ci in bootstrap_cis
    )
    if unstable or any(not ci.available for ci in bootstrap_cis):
        if external_n > 0:
            flags.append("external_metric_unstable")
    if ad_coverage is not None and ad_coverage < 0.9 and "applicability_domain_failure" not in flags:
        # Soft distribution-shift signal when coverage is reduced but above hard fail
        if ad_coverage < criteria.minimum_ad_coverage + 0.15:
            flags.append("external_distribution_shift")
    if residual_outlier_ids or influential_ids:
        flags.append("influential_compounds_detected")
    if (
        train_cv_gap is not None
        and abs(train_cv_gap) > criteria.maximum_cv_test_r2_gap
        and "internal_validation_mismatch" not in flags
    ):
        # Large train–CV gap as soft mismatch flag
        flags.append("internal_validation_mismatch")

    # Deduplicate flags preserving order
    seen: set[str] = set()
    uniq_flags: list[DiagnosticFlag] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            uniq_flags.append(f)

    recommendations = _build_recommendations(
        primary_failed=primary == "external_validation_failed",
        flags=uniq_flags,
        failed_criteria=failed_criteria,
        grouped_cv=grouped_cv,
    )

    explanation = (
        f"Primary outcome: {primary}. "
        + (
            f"Failed criteria: {', '.join(failed_criteria)}. "
            if failed_criteria
            else "All frozen pass/fail criteria satisfied. "
        )
        + (f"Flags: {', '.join(uniq_flags)}." if uniq_flags else "No diagnostic flags.")
    )

    result = PostTestAuditResult(
        primary_outcome=primary,
        diagnostic_flags=uniq_flags,
        criteria=criteria,
        criteria_snapshot_path=str(criteria_path) if criteria_path.exists() else "",
        train_metrics=train_block,
        cv_metrics=cv_block,
        agent_val_metrics=agent_val_block,
        external_metrics=external_block,
        train_cv_r2_gap=train_cv_gap,
        cv_test_r2_gap=cv_test_gap,
        bootstrap_cis=bootstrap_cis,
        external_n=external_n,
        external_activity_min=act_min,
        external_activity_max=act_max,
        external_activity_variance=act_var,
        residual_outlier_ids=residual_outlier_ids,
        influential_compound_ids=influential_ids,
        ad_coverage=ad_coverage,
        in_domain_metrics=in_domain_m,
        out_of_domain_metrics=out_domain_m,
        random_vs_grouped_cv=grouped_cv,
        evidence_paths=evidence,
        warnings=warnings,
        recommendations=recommendations,
        failed_criteria=failed_criteria,
        remediation_allowed=primary == "external_validation_failed" or bool(uniq_flags),
        explanation=explanation,
    )

    out_json = locked_ext / "post_test_audit.json"
    out_md = locked_ext / "post_test_audit.md"
    locked_ext.mkdir(parents=True, exist_ok=True)
    save_json(out_json, result.model_dump())
    out_md.write_text(_render_audit_markdown(result), encoding="utf-8")
    result.evidence_paths["post_test_audit_json"] = str(out_json)
    result.evidence_paths["post_test_audit_md"] = str(out_md)
    # Refresh JSON with evidence paths including audit outputs
    save_json(out_json, result.model_dump())
    return result


def _render_audit_markdown(result: PostTestAuditResult) -> str:
    lines = [
        "# Post-Test Audit (read-only)",
        "",
        f"**Primary outcome:** `{result.primary_outcome}`",
        "",
        result.explanation,
        "",
        "## Frozen criteria",
        "",
        "```json",
        json.dumps(result.criteria.model_dump(), indent=2),
        "```",
        "",
        "## Metric table",
        "",
        "| Split | R² | RMSE | MAE | n | source |",
        "|---|---|---|---|---|---|",
    ]
    for name, block in (
        ("Train", result.train_metrics),
        ("CV", result.cv_metrics),
        ("Agent-val", result.agent_val_metrics),
        ("External", result.external_metrics),
    ):
        lines.append(
            f"| {name} | {_fmt(block.mean_r2)} | {_fmt(block.rmse)} | "
            f"{_fmt(block.mae)} | {block.n if block.n is not None else '—'} | {block.source} |"
        )
    lines += [
        "",
        f"- Train–CV R² gap: `{_fmt(result.train_cv_r2_gap)}`",
        f"- CV–external R² gap: `{_fmt(result.cv_test_r2_gap)}`",
        f"- External n: `{result.external_n}`",
        f"- AD coverage: `{_fmt(result.ad_coverage)}`",
        "",
        "## Bootstrap CIs (external)",
        "",
    ]
    if result.bootstrap_cis:
        for ci in result.bootstrap_cis:
            if ci.available:
                lines.append(
                    f"- **{ci.metric}**: estimate={_fmt(ci.estimate)}, "
                    f"CI=[{_fmt(ci.lower)}, {_fmt(ci.upper)}] "
                    f"(n={ci.n_samples}, B={ci.n_bootstrap})"
                )
            else:
                lines.append(f"- **{ci.metric}**: unavailable — {ci.warning or ''}")
    else:
        lines.append("_No bootstrap results._")

    lines += ["", "## AD subgroups", ""]
    for sg in (result.in_domain_metrics, result.out_of_domain_metrics):
        if sg is None:
            continue
        lines.append(
            f"- **{sg.label}** (n={sg.n}, reliable={sg.reliable}): "
            f"R²={_fmt(sg.r2)}, RMSE={_fmt(sg.rmse)}, MAE={_fmt(sg.mae)}"
            + (f" — {sg.warning}" if sg.warning else "")
        )

    lines += [
        "",
        "## Random vs grouped CV",
        "",
        f"Status: `{result.random_vs_grouped_cv.get('status', 'unavailable')}`",
        "",
        str(result.random_vs_grouped_cv.get("recommendation") or ""),
        "",
        "## Diagnostic flags",
        "",
    ]
    if result.diagnostic_flags:
        for f in result.diagnostic_flags:
            lines.append(f"- `{f}`")
    else:
        lines.append("_None_")

    lines += ["", "## Recommendations", ""]
    if result.recommendations:
        for r in result.recommendations:
            lines.append(f"- {r}")
    else:
        lines.append("_None_")

    if result.residual_outlier_ids:
        lines += [
            "",
            "## Residual outliers",
            "",
            ", ".join(result.residual_outlier_ids[:50]),
        ]
    if result.influential_compound_ids:
        lines += [
            "",
            "## Influential compounds",
            "",
            ", ".join(result.influential_compound_ids[:50]),
        ]

    lines += ["", "## Evidence paths", ""]
    for k, v in result.evidence_paths.items():
        lines.append(f"- `{k}`: `{v}`")

    lines += [
        "",
        "---",
        "",
        "_This audit is strictly read-only. It does not retrain, retune, change "
        "features, or restart the locked lineage. Remediation requires a new forked lineage._",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"
