"""The iteration controller for the agentic QSAR workflow.

Runs the closed loop: plan, execute, validate, diagnose, revise. Stops when the
user's acceptance criteria are met, the budget runs out, or the Strategist
declares there is nothing useful left to try — and reports honestly either way.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from qsar_agent.agents.roles import (
    DataAgent,
    DatasetBundle,
    DescriptorAgent,
    ModelingAgent,
    ValidationAgent,
    encode_targets,
)
from qsar_agent.agents.strategist import Strategist, StrategistError
from qsar_agent.llm.provider import LLMClient, get_llm_client
from qsar_agent.logging_utils import get_logger
from qsar_agent.schemas.agentic import (
    AgenticRunReport,
    IterationPlan,
    IterationRecord,
    SplitMethod,
)
from qsar_agent.services.artifact_manager import generate_run_id, get_run_dir, save_json
from qsar_agent.tools.acceptance import (
    AcceptanceCriteria,
    DiagnosisThresholds,
    diagnose_failure,
    evaluate_acceptance,
)
from qsar_agent.tools.metrics import primary_metric_name
from qsar_agent.tools.model_zoo import decision_scores

logger = get_logger()


@dataclass
class AgenticConfig:
    """Everything the loop needs that is not the dataset or the criteria."""

    smiles_column: str
    activity_column: str
    id_column: str | None = None
    forced_task: str | None = None
    split_method: SplitMethod = "random"
    test_size: float = 0.2
    cv_folds: int = 5
    scramble_repeats: int = 10
    ad_k: int = 5
    ad_z_factor: float = 3.0
    random_seed: int = 42
    max_iterations: int = 6
    budget_seconds: float = 900.0
    n_jobs: int = 1
    output_dir: str = "outputs"
    min_compounds: int = 20
    diagnosis: DiagnosisThresholds = field(default_factory=DiagnosisThresholds)


ProgressCallback = Callable[[str], None]


class AgenticQSARWorkflow:
    """Orchestrates the specialist agents and the Strategist across iterations."""

    def __init__(
        self,
        dataset_path: str | Path,
        criteria: AcceptanceCriteria,
        config: AgenticConfig,
        llm_client: LLMClient | None = None,
        run_id: str | None = None,
        progress: ProgressCallback | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.criteria = criteria
        self.config = config
        self.run_id = run_id or generate_run_id()
        self.run_dir = get_run_dir(config.output_dir, self.run_id)
        self.progress = progress or (lambda message: logger.info(message))
        # Raises immediately when no provider is configured: the workflow needs a
        # language model, so failing here is better than failing mid-run.
        self.llm = llm_client or get_llm_client()
        self.records: list[IterationRecord] = []
        self.web_searches = 0
        self.warnings: list[str] = []
        self._bundle: DatasetBundle | None = None
        self._stop_reason = ""

    # -- context exposed to the Strategist and its tools -----------------------

    def _history_payload(self) -> list[dict[str, Any]]:
        payload = []
        for record in self.records:
            entry: dict[str, Any] = {
                "iteration": record.index,
                "features": record.plan.features.model_dump(),
                "model": record.plan.model.model_dump(),
                "signature": record.plan.signature(),
            }
            if record.error:
                entry["error"] = record.error
            if record.validation is not None:
                entry["n_features_after_filtering"] = record.validation.n_features
                entry["train_metrics"] = record.validation.train_metrics
                entry["test_metrics"] = record.validation.test_metrics
                entry["cross_validation"] = record.validation.cv.model_dump()
                if record.validation.y_scrambling is not None:
                    entry["y_scrambling"] = {
                        "mean_scrambled_score": record.validation.y_scrambling.mean_scrambled_score,
                        "margin": record.validation.y_scrambling.margin,
                    }
                if record.validation.applicability_domain is not None:
                    entry["ad_test_coverage_pct"] = (
                        record.validation.applicability_domain.test_coverage_pct
                    )
            if record.verdict is not None:
                entry["accepted"] = record.verdict.accepted
                entry["failed_criteria"] = [
                    {
                        "name": check.name,
                        "observed": check.observed,
                        "threshold": check.threshold,
                        "direction": check.direction,
                    }
                    for check in record.verdict.failed_checks
                ]
            if record.diagnosis is not None:
                entry["diagnosis"] = record.diagnosis.model_dump()
            payload.append(entry)
        return payload

    def _summary_payload(self) -> dict[str, Any]:
        if self._bundle is None:
            return {"note": "Dataset not ingested yet."}
        return self._bundle.summary()

    def _criteria_payload(self) -> dict[str, Any]:
        task = self._bundle.task if self._bundle else "regression"
        return {
            "note": "Set by the user before the run; immutable during the run.",
            "criteria": self.criteria.describe(task),
        }

    def _count_web_search(self) -> None:
        self.web_searches += 1

    # -- the loop --------------------------------------------------------------

    def run(self) -> AgenticRunReport:
        started = time.monotonic()

        self.progress("DataAgent: ingesting dataset and detecting the endpoint type")
        data_agent = DataAgent(self.run_dir, random_seed=self.config.random_seed)
        bundle = data_agent.prepare(
            dataset_path=self.dataset_path,
            smiles_column=self.config.smiles_column,
            activity_column=self.config.activity_column,
            id_column=self.config.id_column,
            forced_task=self.config.forced_task,
            split_method=self.config.split_method,
            test_size=self.config.test_size,
            min_compounds=self.config.min_compounds,
        )
        self._bundle = bundle
        self.warnings.extend(bundle.ingest.warnings)
        self.progress(
            f"DataAgent: {bundle.n_train} train / {bundle.n_test} test compounds, "
            f"task={bundle.task}, split={bundle.split_method}"
        )
        self.progress(f"DataAgent: {bundle.task_detection.reason}")

        enabled = self.criteria.enabled_names(bundle.task)
        if not enabled:
            raise ValueError(
                f"No acceptance criteria are enabled for a {bundle.task} task. The loop would "
                "have nothing to satisfy."
            )

        descriptor_agent = DescriptorAgent(bundle)
        modeling_agent = ModelingAgent(
            task=bundle.task,
            random_seed=self.config.random_seed,
            cv_folds=self.config.cv_folds,
            n_jobs=self.config.n_jobs,
        )
        validation_agent = ValidationAgent(
            task=bundle.task,
            random_seed=self.config.random_seed,
            scramble_repeats=self.config.scramble_repeats,
            ad_k=self.config.ad_k,
            ad_z_factor=self.config.ad_z_factor,
            n_jobs=self.config.n_jobs,
        )

        strategist = Strategist(
            client=self.llm,
            task=bundle.task,
            history_provider=self._history_payload,
            summary_provider=self._summary_payload,
            criteria_provider=self._criteria_payload,
            on_web_search=self._count_web_search,
        )

        y_all, encoder = encode_targets(bundle.task, bundle.frame["activity"])

        self.progress("Strategist: proposing the opening plan")
        plan = strategist.initial_plan(
            {
                "dataset": bundle.summary(),
                "task": bundle.task,
                "acceptance_criteria": self.criteria.describe(bundle.task),
                "current_plan": None,
            }
        )
        self.progress(f"Strategist: iteration 1 plan -> {plan.signature()}")

        tried: set[str] = set()
        accepted = False
        stop_reason = ""

        for index in range(1, self.config.max_iterations + 1):
            elapsed = time.monotonic() - started
            if elapsed > self.config.budget_seconds:
                stop_reason = (
                    f"Time budget of {self.config.budget_seconds:.0f}s exhausted after "
                    f"{index - 1} iterations."
                )
                break

            iteration_started = time.monotonic()
            tried.add(plan.signature())
            record = IterationRecord(index=index, plan=plan)
            iteration_dir = self.run_dir / f"iteration_{index:02d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            record.artifact_dir = str(iteration_dir)

            try:
                self.progress(
                    f"Iteration {index}/{self.config.max_iterations}: "
                    f"{'+'.join(plan.features.blocks)} -> {plan.model.estimator}"
                )
                validation, fitted = self._execute(
                    plan=plan,
                    bundle=bundle,
                    descriptor_agent=descriptor_agent,
                    modeling_agent=modeling_agent,
                    validation_agent=validation_agent,
                    y_all=y_all,
                    iteration_dir=iteration_dir,
                    encoder=encoder,
                )
            except Exception as exc:
                record.error = f"{type(exc).__name__}: {exc}"
                record.elapsed_seconds = time.monotonic() - iteration_started
                self.records.append(record)
                logger.warning("Iteration %d failed: %s", index, record.error)
                self.progress(f"Iteration {index}: execution failed — {record.error}")
                plan_or_stop = self._next_plan(strategist, bundle, plan, tried, index)
                if plan_or_stop is None:
                    stop_reason = self._stop_reason or "The strategist stopped the run."
                    break
                plan = plan_or_stop
                continue

            record.validation = validation
            verdict = evaluate_acceptance(self.criteria, validation)
            record.verdict = verdict
            diagnosis = diagnose_failure(
                validation, verdict, bundle.task_detection, self.config.diagnosis
            )
            record.diagnosis = diagnosis
            record.elapsed_seconds = time.monotonic() - iteration_started
            self.records.append(record)

            self._log_iteration(index, validation, verdict, diagnosis)
            save_json(
                iteration_dir / "iteration_record.json",
                json.loads(record.model_dump_json()),
            )

            if verdict.accepted:
                accepted = True
                stop_reason = "All user-specified acceptance criteria were met."
                joblib.dump(fitted, iteration_dir / "model.joblib")
                break

            if index == self.config.max_iterations:
                stop_reason = (
                    f"Reached the maximum of {self.config.max_iterations} iterations without "
                    "meeting the acceptance criteria."
                )
                break

            plan_or_stop = self._next_plan(strategist, bundle, plan, tried, index)
            if plan_or_stop is None:
                stop_reason = self._stop_reason or "The strategist stopped the run."
                break
            plan = plan_or_stop

        if not stop_reason:
            stop_reason = "Loop ended without an explicit stop condition."

        report = self._build_report(
            bundle=bundle,
            accepted=accepted,
            stop_reason=stop_reason,
            elapsed=time.monotonic() - started,
        )
        self._persist(report)
        return report

    # -- one iteration ---------------------------------------------------------

    def _execute(
        self,
        plan: IterationPlan,
        bundle: DatasetBundle,
        descriptor_agent: DescriptorAgent,
        modeling_agent: ModelingAgent,
        validation_agent: ValidationAgent,
        y_all: np.ndarray,
        iteration_dir: Path,
        encoder,
    ):
        y_train_provisional = y_all[bundle.train_idx]
        matrices = descriptor_agent.build(plan.features, y_train_encoded=y_train_provisional)
        y_train = y_all[matrices.train_idx]
        y_test = y_all[matrices.test_idx]
        test_ids = bundle.compound_ids(matrices.test_idx)

        estimator = modeling_agent.build(plan)
        validation, fitted = validation_agent.validate(
            modeling_agent=modeling_agent,
            estimator=estimator,
            matrices=matrices,
            y_train=y_train,
            y_test=y_test,
            test_ids=test_ids,
        )

        self._save_predictions(
            iteration_dir=iteration_dir,
            bundle=bundle,
            matrices=matrices,
            fitted=fitted,
            y_train=y_train,
            y_test=y_test,
            task=bundle.task,
            encoder=encoder,
        )
        (iteration_dir / "selected_features.json").write_text(
            json.dumps(matrices.feature_names, indent=2), encoding="utf-8"
        )
        return validation, fitted

    def _save_predictions(
        self,
        iteration_dir: Path,
        bundle: DatasetBundle,
        matrices,
        fitted,
        y_train: np.ndarray,
        y_test: np.ndarray,
        task: str,
        encoder,
    ) -> None:
        rows = []
        for split, indices, X, y in (
            ("train", matrices.train_idx, matrices.X_train, y_train),
            ("test", matrices.test_idx, matrices.X_test, y_test),
        ):
            predicted = fitted.predict(X)
            scores = decision_scores(fitted, X) if task == "classification" else None
            for position, dataset_index in enumerate(indices):
                row = {
                    "compound_id": bundle.frame.iloc[dataset_index]["compound_id"],
                    "canonical_smiles": bundle.frame.iloc[dataset_index]["canonical_smiles"],
                    "split": split,
                }
                if task == "classification" and encoder is not None:
                    row["observed"] = encoder.inverse_transform([int(y[position])])[0]
                    row["predicted"] = encoder.inverse_transform([int(predicted[position])])[0]
                    if scores is not None and np.ndim(scores) == 1:
                        row["score"] = float(scores[position])
                else:
                    row["observed"] = float(y[position])
                    row["predicted"] = float(predicted[position])
                    row["residual"] = float(y[position]) - float(predicted[position])
                rows.append(row)
        pd.DataFrame(rows).to_csv(iteration_dir / "predictions.csv", index=False)

    # -- revision --------------------------------------------------------------

    def _next_plan(
        self,
        strategist: Strategist,
        bundle: DatasetBundle,
        current: IterationPlan,
        tried: set[str],
        index: int,
    ) -> IterationPlan | None:
        """Ask the Strategist for the next plan, rejecting repeats."""
        self._stop_reason = ""
        last = self.records[-1]
        context = {
            "iteration_just_completed": index,
            "iterations_remaining": self.config.max_iterations - index,
            "task": bundle.task,
            "dataset": bundle.summary(),
            "acceptance_criteria": self.criteria.describe(bundle.task),
            "current_plan": current.model_dump(),
            "tried_signatures": sorted(tried),
            "diagnosis": last.diagnosis.model_dump() if last.diagnosis else {},
            "failed_criteria": (
                [check.model_dump() for check in last.verdict.failed_checks]
                if last.verdict
                else []
            ),
            "last_error": last.error,
            "history": self._history_payload(),
        }

        for attempt in range(2):
            try:
                revision = strategist.revise(context)
            except StrategistError as exc:
                self._stop_reason = str(exc)
                self.warnings.append(str(exc))
                return None

            if revision.stop:
                self._stop_reason = (
                    revision.stop_reason
                    or "The strategist saw no further revision worth testing."
                )
                if self.records:
                    self.records[-1].revision = revision
                return None

            candidate = revision.to_plan(current)
            if candidate.signature() in tried:
                self.progress(
                    f"Strategist proposed an already-tested configuration "
                    f"({candidate.signature()}); asking for a different one."
                )
                context = {
                    **context,
                    "rejected_proposal": {
                        "signature": candidate.signature(),
                        "reason": "This exact configuration was already tested. Propose a "
                        "materially different one.",
                    },
                }
                continue

            if self.records:
                self.records[-1].revision = revision
            self.progress(
                f"Strategist: iteration {index + 1} plan -> {candidate.signature()} "
                f"({revision.rationale[:160]})"
            )
            return candidate

        self._stop_reason = (
            "The strategist kept proposing configurations that had already been tested."
        )
        return None

    # -- reporting -------------------------------------------------------------

    def _log_iteration(self, index: int, validation, verdict, diagnosis) -> None:
        primary = validation.primary_metric
        test_score = validation.test_metrics.get(primary)
        pieces = [
            f"Iteration {index} results:",
            f"  features retained: {validation.n_features}",
            f"  CV {primary}: {validation.cv.mean_cv_score:.4f} "
            f"(+/-{validation.cv.std_cv_score:.4f}, train {validation.cv.mean_train_score:.4f}, "
            f"gap {validation.cv.train_cv_gap:.4f})",
            f"  test {primary}: {test_score:.4f}" if test_score is not None else "  test: n/a",
        ]
        if validation.y_scrambling is not None:
            pieces.append(
                f"  y-scrambled {primary}: {validation.y_scrambling.mean_scrambled_score:.4f} "
                f"(margin {validation.y_scrambling.margin:.4f})"
            )
        if validation.applicability_domain is not None:
            pieces.append(
                f"  AD coverage: {validation.applicability_domain.test_coverage_pct:.1f}%"
            )
        for check in verdict.checks:
            pieces.append(f"    {check.render()}")
        pieces.append(f"  verdict: {verdict.summary()}")
        if not verdict.accepted:
            pieces.append(f"  diagnosis: {diagnosis.label} — {diagnosis.explanation}")
        self.progress("\n".join(pieces))

    def _build_report(
        self, bundle: DatasetBundle, accepted: bool, stop_reason: str, elapsed: float
    ) -> AgenticRunReport:
        scored = [r for r in self.records if r.validation is not None]
        best = None
        if scored:
            accepted_records = [r for r in scored if r.verdict and r.verdict.accepted]
            pool = accepted_records or scored
            best = max(
                pool,
                key=lambda r: (
                    r.primary_score() if r.primary_score() is not None else float("-inf")
                ),
            )

        return AgenticRunReport(
            run_id=self.run_id,
            dataset_path=str(self.dataset_path),
            task=bundle.task,
            task_detection=bundle.task_detection,
            criteria=self.criteria.describe(bundle.task),
            n_compounds=int(len(bundle.frame)),
            n_train=bundle.n_train,
            n_test=bundle.n_test,
            split_method=bundle.split_method,
            iterations_run=len(self.records),
            max_iterations=self.config.max_iterations,
            accepted=accepted,
            stop_reason=stop_reason,
            best_iteration_index=best.index if best else None,
            best_validation=best.validation if best else None,
            best_verdict=best.verdict if best else None,
            best_plan=best.plan if best else None,
            iterations=self.records,
            llm_provider=getattr(self.llm, "provider", "unknown"),
            llm_model=getattr(self.llm, "model", "unknown"),
            web_searches=self.web_searches,
            total_elapsed_seconds=elapsed,
            warnings=self.warnings,
        )

    def _persist(self, report: AgenticRunReport) -> None:
        json_path = self.run_dir / "agentic_run_report.json"
        save_json(json_path, json.loads(report.model_dump_json()))
        markdown_path = self.run_dir / "agentic_run_report.md"
        markdown_path.write_text(render_report(report), encoding="utf-8")
        report.artifact_paths = {
            "run_dir": str(self.run_dir),
            "report_json": str(json_path),
            "report_markdown": str(markdown_path),
        }


def render_report(report: AgenticRunReport) -> str:
    """Human-readable final report; states plainly when criteria were not met."""
    lines: list[str] = []
    status = "ACCEPTED" if report.accepted else "NOT ACCEPTED"
    lines.append(f"# Agentic QSAR run {report.run_id} — {status}")
    lines.append("")
    lines.append(f"- Dataset: `{report.dataset_path}`")
    lines.append(f"- Task: {report.task} ({report.task_detection.reason})")
    lines.append(
        f"- Compounds: {report.n_compounds} "
        f"({report.n_train} train / {report.n_test} test, {report.split_method} split)"
    )
    lines.append(f"- Iterations: {report.iterations_run} of {report.max_iterations} allowed")
    lines.append(f"- LLM: {report.llm_provider} / {report.llm_model}")
    lines.append(f"- Web searches performed: {report.web_searches}")
    lines.append(f"- Wall clock: {report.total_elapsed_seconds:.1f}s")
    lines.append(f"- Stop reason: {report.stop_reason}")
    lines.append("")

    lines.append("## Acceptance criteria requested by the user")
    lines.append("")
    lines.append("| Criterion | Requirement | Enabled |")
    lines.append("|---|---|---|")
    for name, spec in report.criteria.items():
        symbol = ">=" if spec["direction"] == "min" else "<="
        threshold = "-" if spec["threshold"] is None else f"{symbol} {spec['threshold']}"
        lines.append(f"| {spec['label']} | {threshold} | {'yes' if spec['enabled'] else 'no'} |")
    lines.append("")

    if report.best_validation is not None and report.best_verdict is not None:
        best = report.best_validation
        lines.append(f"## Best model (iteration {report.best_iteration_index})")
        lines.append("")
        if report.best_plan is not None:
            lines.append(f"- Features: {', '.join(report.best_plan.features.blocks)}")
            lines.append(f"- Retained descriptors: {best.n_features}")
            lines.append(f"- Estimator: {report.best_plan.model.signature()}")
            if report.best_plan.rationale:
                lines.append(f"- Rationale: {report.best_plan.rationale}")
        lines.append("")
        lines.append("### Measured performance")
        lines.append("")
        lines.append("| Metric | Train | Test |")
        lines.append("|---|---|---|")
        for key in sorted(set(best.train_metrics) | set(best.test_metrics)):
            if key == "n_samples":
                continue
            train_value = best.train_metrics.get(key)
            test_value = best.test_metrics.get(key)
            lines.append(
                f"| {key} | "
                f"{'-' if train_value is None else f'{train_value:.4f}'} | "
                f"{'-' if test_value is None else f'{test_value:.4f}'} |"
            )
        lines.append("")
        cv = best.cv
        lines.append(
            f"Cross-validation ({cv.folds}-fold): {cv.primary_metric} = "
            f"{cv.mean_cv_score:.4f} +/- {cv.std_cv_score:.4f} "
            f"(training folds {cv.mean_train_score:.4f}, gap {cv.train_cv_gap:.4f})"
        )
        if best.y_scrambling is not None:
            scramble = best.y_scrambling
            lines.append("")
            lines.append(
                f"y-scrambling ({scramble.n_repeats} shuffles): mean "
                f"{scramble.mean_scrambled_score:.4f} +/- {scramble.std_scrambled_score:.4f}, "
                f"max {scramble.max_scrambled_score:.4f}, margin over chance "
                f"{scramble.margin:.4f}."
            )
        if best.applicability_domain is not None:
            domain = best.applicability_domain
            lines.append("")
            lines.append(
                f"Applicability domain (k-NN, k={domain.k}, z={domain.z_factor}): "
                f"{domain.test_in_domain_count}/{domain.test_total_count} test compounds inside "
                f"({domain.test_coverage_pct:.1f}%)."
            )
        lines.append("")
        lines.append("### Criterion-by-criterion verdict")
        lines.append("")
        for check in report.best_verdict.checks:
            lines.append(f"- {check.render()}")
        lines.append("")

    lines.append("## Iteration history")
    lines.append("")
    lines.append("| # | Features | Estimator | CV | Test | Verdict | Diagnosis |")
    lines.append("|---|---|---|---|---|---|---|")
    for record in report.iterations:
        if record.error:
            lines.append(
                f"| {record.index} | {'+'.join(record.plan.features.blocks)} | "
                f"{record.plan.model.estimator} | - | - | execution error | {record.error} |"
            )
            continue
        validation = record.validation
        test_score = record.primary_score()
        verdict = "accepted" if record.verdict and record.verdict.accepted else "rejected"
        diagnosis = record.diagnosis.label if record.diagnosis else "-"
        lines.append(
            f"| {record.index} | {'+'.join(record.plan.features.blocks)} | "
            f"{record.plan.model.estimator} | "
            f"{validation.cv.mean_cv_score:.4f} | "
            f"{'-' if test_score is None else f'{test_score:.4f}'} | {verdict} | {diagnosis} |"
        )
    lines.append("")

    if not report.accepted:
        lines.append("## Honest assessment")
        lines.append("")
        lines.append(
            "This run did **not** produce a model meeting the criteria you specified. "
            "No threshold was relaxed to manufacture a pass."
        )
        if report.best_verdict is not None:
            failed = report.best_verdict.failed_checks
            if failed:
                lines.append("")
                lines.append("Still failing on the best iteration:")
                lines.append("")
                for check in failed:
                    symbol = ">=" if check.direction == "min" else "<="
                    observed = "not measured" if check.observed is None else f"{check.observed:.4f}"
                    lines.append(
                        f"- {check.description}: observed {observed}, required "
                        f"{symbol} {check.threshold:.4f}"
                    )
        lines.append("")
        lines.append(
            "Options: supply more or cleaner data, relax the criteria if they are stricter "
            "than the decision requires, raise the iteration budget, or accept that this "
            "endpoint may not be predictable from structure at the requested accuracy."
        )
        lines.append("")

    if report.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def run_agentic_workflow(
    dataset_path: str | Path,
    criteria: AcceptanceCriteria,
    config: AgenticConfig,
    llm_client: LLMClient | None = None,
    run_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> AgenticRunReport:
    """Convenience wrapper around :class:`AgenticQSARWorkflow`."""
    workflow = AgenticQSARWorkflow(
        dataset_path=dataset_path,
        criteria=criteria,
        config=config,
        llm_client=llm_client,
        run_id=run_id,
        progress=progress,
    )
    return workflow.run()


__all__ = [
    "AgenticConfig",
    "AgenticQSARWorkflow",
    "primary_metric_name",
    "render_report",
    "run_agentic_workflow",
]
