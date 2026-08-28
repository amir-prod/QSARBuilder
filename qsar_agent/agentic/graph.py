"""LangGraph StateGraph for post-handoff modeling improvement."""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphRecursionError

from qsar_agent.agentic.decisions import propose_decision, validate_decision
from qsar_agent.agentic.handoff_validator import validate_handoff_dir
from qsar_agent.agentic.ledger import write_decisions_jsonl
from qsar_agent.agentic.ranking import rank_from_workflow
from qsar_agent.agentic.reporting import snapshot_state, write_final_report_md
from qsar_agent.agentic.requirements import evaluate_requirements, requirements_from_config
from qsar_agent.agentic.sealing import (
    SealedTestAccessError,
    assert_development_phase,
    assert_no_test_paths,
    development_view_dict,
    phase_value,
)
from qsar_agent.agentic.stopping import (
    STOP_CAPABILITY,
    STOP_REQUIREMENTS_MET,
    check_stopping,
)
from qsar_agent.agentic.tools import agent_dir, execute_tool, load_workflow_config
from qsar_agent.config import WorkflowConfig
from qsar_agent.models.registry import SUPPORTED_ESTIMATORS
from qsar_agent.schemas.agentic import (
    AgentDecision,
    ModelingAgentState,
    PipelinePhase,
)
from qsar_agent.schemas.handoff import HandoffPackage
from qsar_agent.services.artifact_manager import save_json


@dataclass
class GraphDeps:
    decision_fn: Callable[[ModelingAgentState], AgentDecision] | None = None
    execute_fn: Callable[..., Any] | None = None
    config: WorkflowConfig | None = None


_DEPS: contextvars.ContextVar[GraphDeps] = contextvars.ContextVar("modeling_graph_deps")


def get_deps() -> GraphDeps:
    try:
        return _DEPS.get()
    except LookupError:
        return GraphDeps()


def _cfg(state: ModelingAgentState) -> WorkflowConfig:
    deps = get_deps()
    if deps.config is not None:
        return deps.config
    return load_workflow_config(state.run_dir)


def _winner_metrics(view: dict[str, Any]) -> tuple[dict[str, Any], list[str], str]:
    experiments = view.get("experiments") or []
    winner = next((e for e in experiments if e.get("is_winner")), experiments[0] if experiments else {})
    metrics = dict(winner.get("metrics") or {})
    features = list(winner.get("selected_feature_names") or winner.get("selected_features") or [])
    model = str(winner.get("model") or "")
    return metrics, features, model


def _ensure_dirs(run_dir: str) -> Path:
    path = agent_dir(run_dir)
    for name in ("experiments", "plots", "capability_requests", "frozen_pipeline", "exclusion_proposals"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _dispatch_tool(
    state: ModelingAgentState,
    tool: str,
    args: dict[str, Any],
    *,
    phase: PipelinePhase | str | None = None,
    sealed_test_result: Any = None,
) -> Any:
    deps = get_deps()
    if deps.execute_fn is not None:
        return deps.execute_fn(tool, args, state)
    parent = (state.current_best_candidate or {}).get("experiment_id")
    features = (state.current_best_candidate or {}).get("selected_features")
    return execute_tool(
        tool,
        args,
        run_dir=state.run_dir,
        state_phase=phase if phase is not None else state.phase,
        dataset_hash=state.dataset_hash,
        development_split_hash=state.development_split_hash,
        parent_id=parent,
        selected_features=features,
        completed_experiments=state.completed_experiments,
        current_best=state.current_best_candidate,
        exclusion_approved=bool((state.exclusion_decision or {}).get("approved")),
        sealed_test_result=sealed_test_result if sealed_test_result is not None else state.sealed_test_result,
    )


def _as_tool_dump(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return dict(result)


def load_and_validate_handoff(state: ModelingAgentState) -> dict[str, Any]:
    run_dir = Path(state.run_dir)
    report_dir = run_dir / "final_report"
    result = validate_handoff_dir(report_dir, run_dir)
    out: dict[str, Any] = {
        "project_id": run_dir.name,
        "handoff_validation": result.model_dump(mode="json"),
        "validation_passed": result.passed,
        "validation_errors": result.errors,
        "dataset_hash": result.dataset_hash or state.dataset_hash,
        "development_split_hash": result.development_split_hash or state.development_split_hash,
        "sealed_test_hash": result.sealed_test_hash or state.sealed_test_hash,
        "started_at": state.started_at or datetime.now(timezone.utc).isoformat(),
        "phase": PipelinePhase.DEVELOPMENT,
        "route": "valid_handoff" if result.passed else "invalid_handoff",
    }
    if not result.passed:
        out["stopping_reason"] = "integrity_check_failed"
    _ensure_dirs(state.run_dir)
    snapshot_state(state.model_copy(update=out), _ensure_dirs(state.run_dir))
    return out


def create_development_safe_view(state: ModelingAgentState) -> dict[str, Any]:
    assert_development_phase(state.phase, sealed_test_result=state.sealed_test_result)
    raw = (state.handoff_validation or {}).get("package")
    if not raw:
        return {"route": "invalid_handoff", "stopping_reason": "integrity_check_failed", "validation_passed": False}
    package = HandoffPackage.model_validate(raw)
    view = development_view_dict(package)
    config = _cfg(state)
    req = requirements_from_config(config)
    return {
        "development_view": view,
        "dataset_summary": {
            "train_size": package.dataset_audit.train_size,
            "validation_size": package.dataset_audit.validation_size,
            "test_size": package.dataset_audit.test_size,
        },
        "available_representations": list(package.representation_preprocessing.descriptor_backends),
        "available_model_families": list(SUPPORTED_ESTIMATORS),
        "requirements": req.model_dump(),
        "route": "evaluate_requirements",
    }


def evaluate_requirements_node(state: ModelingAgentState) -> dict[str, Any]:
    if state.sealed_test_result and state.phase == PipelinePhase.DEVELOPMENT:
        raise SealedTestAccessError("evaluate_requirements refused sealed-test results.")
    config = _cfg(state)
    req = requirements_from_config(config)
    view = state.development_view or {}
    metrics, features, model = _winner_metrics(view)
    if state.current_best_candidate:
        metrics = state.current_best_candidate.get("metrics") or metrics
        features = state.current_best_candidate.get("selected_features") or features
    evaluation = evaluate_requirements(metrics, req, feature_count=len(features) if features else None)
    if evaluation.acceptance_status == "passed":
        route = "requirements_satisfied"
    else:
        limits = config.agentic_improvement.limits
        stop_reason = check_stopping(
            acceptance_status="failed",
            agent_iteration=state.agent_iteration,
            adaptive_experiments_used=state.adaptive_experiments_used,
            compute_budget_used_hours=state.compute_budget_used,
            stagnation_count=state.stagnation_count,
            model_families_used=len(state.model_families_used),
            representation_changes_used=state.representation_changes_used,
            pending_capability=bool(state.pending_capability_request),
            integrity_ok=state.validation_passed,
            limits=limits,
            action_error=state.action_error,
        )
        route = "check_stopping" if stop_reason else "requirements_failed"
    best = state.current_best_candidate or {
        "experiment_id": (view.get("experiments") or [{}])[0].get("run_id"),
        "metrics": metrics,
        "selected_features": features,
        "model": model,
    }
    return {
        "requirement_evaluation": evaluation.model_dump(mode="json"),
        "acceptance_status": evaluation.acceptance_status,
        "failed_requirements": [f.model_dump() for f in evaluation.failed_requirements],
        "current_best_candidate": best,
        "route": route,
    }


def diagnose_failure(state: ModelingAgentState) -> dict[str, Any]:
    assert_development_phase(state.phase, sealed_test_result=state.sealed_test_result)
    assert_no_test_paths(state.development_view, label="development_view")
    return {"agent_iteration": state.agent_iteration + 1, "route": "form_hypothesis"}


def form_hypothesis(state: ModelingAgentState) -> dict[str, Any]:
    assert_development_phase(state.phase, sealed_test_result=state.sealed_test_result)
    deps = get_deps()
    decision = propose_decision(
        state,
        use_openai=state.use_openai,
        openai_model=state.openai_model,
        decision_fn=deps.decision_fn,
    )
    write_decisions_jsonl(
        agent_dir(state.run_dir),
        {"iteration": state.agent_iteration, "decision": decision.model_dump(mode="json")},
    )
    return {
        "current_diagnosis": decision.diagnosis,
        "current_hypothesis": decision.hypothesis,
        "proposed_action": decision.action.model_dump(),
        "last_decision": decision.model_dump(mode="json"),
        "route": "validate_action",
    }


def validate_action(state: ModelingAgentState) -> dict[str, Any]:
    assert_development_phase(state.phase, sealed_test_result=state.sealed_test_result)
    if not state.last_decision:
        return {"route": "invalid_action", "action_error": "no_justified_action", "stagnation_count": state.stagnation_count + 1}
    decision = AgentDecision.model_validate(state.last_decision)
    reason = validate_decision(decision, state, _cfg(state))
    tool = decision.action.tool_name
    if reason:
        return {
            "route": "invalid_action",
            "action_error": reason,
            "stagnation_count": state.stagnation_count + 1,
        }
    if tool == "request_new_capability":
        return {"route": "missing_capability", "action_error": ""}
    if tool == "run_exclusion_sensitivity_analysis" or (
        tool == "detect_persistent_outliers" and (decision.action.arguments or {}).get("propose_exclusion")
    ):
        # Exclusion itself is interrupt-gated; detect_persistent_outliers still executes.
        if tool == "run_exclusion_sensitivity_analysis":
            return {
                "route": "outlier_approval",
                "pending_exclusion_proposal": decision.action.arguments,
                "action_error": "",
            }
    return {"route": "execute", "action_error": ""}


def execute_deterministic_tool(state: ModelingAgentState) -> dict[str, Any]:
    action = state.proposed_action or {}
    tool = str(action.get("tool_name") or "")
    args = dict(action.get("arguments") or {})
    if tool != "evaluate_sealed_test":
        assert_development_phase(state.phase, sealed_test_result=state.sealed_test_result)
        assert_no_test_paths(args, label="execute.arguments")
    if state.exclusion_decision and state.exclusion_decision.get("approved"):
        tool = "run_exclusion_sensitivity_analysis"
        args = {**(state.pending_exclusion_proposal or {}), **args, "approved": True}
    result = _dispatch_tool(state, tool, args)
    return {"last_tool_result": _as_tool_dump(result), "route": "ingest"}


def ingest_result(state: ModelingAgentState) -> dict[str, Any]:
    assert_development_phase(state.phase, sealed_test_result=state.sealed_test_result)
    result = dict(state.last_tool_result or {})
    exp_id = str(result.get("experiment_id") or "")
    ids = list(state.completed_experiment_ids)
    experiments = list(state.completed_experiments)
    if exp_id and exp_id not in ids:
        ids.append(exp_id)
        experiments.append(
            {
                "experiment_id": exp_id,
                "tool_name": result.get("tool_name"),
                "metrics": result.get("metrics") or {},
                "selected_features": result.get("selected_features") or [],
                "feature_count": len(result.get("selected_features") or []),
                "model": (result.get("extra") or {}).get("estimator")
                or (state.current_best_candidate or {}).get("model"),
                "runtime_seconds": result.get("runtime_seconds"),
            }
        )
    used = state.adaptive_experiments_used + (0 if exp_id in state.completed_experiment_ids else 1)
    history = list(state.best_cv_r2_history)
    cv = (result.get("metrics") or {}).get("cv_r2")
    if isinstance(cv, (int, float)):
        history.append(float(cv))
    limits = _cfg(state).agentic_improvement.limits
    improved = True
    if len(history) >= 2 and (history[-1] - max(history[:-1])) < limits.minimum_meaningful_cv_improvement:
        improved = False
    families = list(state.model_families_used)
    extra = result.get("extra") or {}
    if extra.get("estimator") and extra["estimator"] not in families:
        families.append(str(extra["estimator"]))
    rep_changes = state.representation_changes_used + (
        1 if result.get("tool_name") == "run_representation_experiment" else 0
    )
    best = state.current_best_candidate
    if cv is not None and (best is None or (best.get("metrics") or {}).get("cv_r2") is None or cv >= (best.get("metrics") or {}).get("cv_r2", float("-inf"))):
        best = experiments[-1] if experiments else best
    return {
        "completed_experiment_ids": ids,
        "completed_experiments": experiments,
        "adaptive_experiments_used": used,
        "best_cv_r2_history": history,
        "stagnation_count": 0 if improved else state.stagnation_count + 1,
        "model_families_used": families,
        "representation_changes_used": rep_changes,
        "current_best_candidate": best,
        "compute_budget_used": state.compute_budget_used + float(result.get("runtime_seconds") or 0) / 3600.0,
        "exclusion_decision": None,
        "route": "compare",
    }


def compare_candidates(state: ModelingAgentState) -> dict[str, Any]:
    assert_development_phase(state.phase, sealed_test_result=state.sealed_test_result)
    config = _cfg(state)
    pool = list(state.completed_experiments)
    view = state.development_view or {}
    if view.get("experiments"):
        winner = next((e for e in view["experiments"] if e.get("is_winner")), view["experiments"][0])
        pool = [
            {
                "experiment_id": winner.get("run_id"),
                "metrics": winner.get("metrics") or {},
                "selected_features": winner.get("selected_feature_names") or [],
                "feature_count": winner.get("feature_count"),
                "model": winner.get("model"),
            }
        ] + pool
    rankings = rank_from_workflow(pool, config)
    best = None
    if rankings:
        top_id = rankings[0].experiment_id
        best = next((e for e in pool if str(e.get("experiment_id") or e.get("run_id")) == top_id), pool[0])
    return {"rankings": [r.model_dump() for r in rankings], "current_best_candidate": best, "route": "evaluate_requirements"}


def check_stopping_conditions(state: ModelingAgentState) -> dict[str, Any]:
    config = _cfg(state)
    limits = config.agentic_improvement.limits
    reason = check_stopping(
        acceptance_status=state.acceptance_status,
        agent_iteration=state.agent_iteration,
        adaptive_experiments_used=state.adaptive_experiments_used,
        compute_budget_used_hours=state.compute_budget_used,
        stagnation_count=state.stagnation_count,
        model_families_used=len(state.model_families_used),
        representation_changes_used=state.representation_changes_used,
        pending_capability=bool(state.pending_capability_request),
        integrity_ok=state.validation_passed,
        limits=limits,
        action_error=state.action_error,
    )
    if not reason and state.acceptance_status == "passed":
        reason = STOP_REQUIREMENTS_MET
    if reason == STOP_REQUIREMENTS_MET:
        return {"stopping_reason": reason, "route": "freeze"}
    if reason:
        return {"stopping_reason": reason, "route": "stop"}
    return {"stopping_reason": "", "route": "continue_diagnosis"}


def request_capability(state: ModelingAgentState) -> dict[str, Any]:
    action = state.proposed_action or {}
    result = _dispatch_tool(state, "request_new_capability", action.get("arguments") or {})
    dumped = _as_tool_dump(result)
    extra = dumped.get("extra") if isinstance(dumped, dict) else getattr(result, "extra", None)
    return {
        "pending_capability_request": extra,
        "stopping_reason": STOP_CAPABILITY,
        "route": "stop",
        "last_tool_result": dumped,
    }


def request_outlier_approval(state: ModelingAgentState) -> dict[str, Any]:
    from langgraph.types import interrupt

    proposal = state.pending_exclusion_proposal or (state.proposed_action or {}).get("arguments") or {}
    cap_dir = agent_dir(state.run_dir) / "exclusion_proposals"
    cap_dir.mkdir(parents=True, exist_ok=True)
    save_json(cap_dir / "pending.json", proposal)
    decision = interrupt({"type": "outlier_exclusion", "proposal": proposal})
    approved = bool(isinstance(decision, dict) and decision.get("approved"))
    if approved:
        return {
            "exclusion_decision": decision if isinstance(decision, dict) else {"approved": True},
            "proposed_action": {
                "tool_name": "run_exclusion_sensitivity_analysis",
                "arguments": {**proposal, "approved": True},
            },
            "route": "execute",
        }
    return {
        "exclusion_decision": {"approved": False},
        "pending_exclusion_proposal": None,
        "route": "continue_diagnosis",
    }


def freeze_and_refit_pipeline(state: ModelingAgentState) -> dict[str, Any]:
    result = _dispatch_tool(
        state,
        "freeze_pipeline",
        {"candidate": state.current_best_candidate},
        phase=PipelinePhase.DEVELOPMENT,
        sealed_test_result=None,
    )
    return {
        "phase": PipelinePhase.FROZEN,
        "last_tool_result": _as_tool_dump(result),
        "route": "frozen",
    }


def evaluate_sealed_test_node(state: ModelingAgentState) -> dict[str, Any]:
    if phase_value(state.phase) != PipelinePhase.FROZEN.value:
        raise SealedTestAccessError("evaluate_sealed_test requires FROZEN phase.")
    if state.sealed_test_result:
        return {"phase": PipelinePhase.EXTERNAL_EVALUATED, "route": "external_done"}
    result = _dispatch_tool(
        state,
        "evaluate_sealed_test",
        {"selected_features": (state.current_best_candidate or {}).get("selected_features")},
        phase=PipelinePhase.FROZEN,
        sealed_test_result=None,
    )
    return {
        "phase": PipelinePhase.EXTERNAL_EVALUATED,
        "sealed_test_result": _as_tool_dump(result),
        "route": "external_done",
    }


def write_final_report(state: ModelingAgentState) -> dict[str, Any]:
    config = _cfg(state)
    path = write_final_report_md(state, config, agent_dir(state.run_dir))
    snapshot_state(state.model_copy(update={"report_path": path}), agent_dir(state.run_dir))
    return {"report_path": path, "route": "end"}


def _route_from_load(state: ModelingAgentState) -> str:
    return "create_development_safe_view" if state.route == "valid_handoff" else "write_final_report"


def _route_from_eval(state: ModelingAgentState) -> str:
    if state.route == "requirements_satisfied":
        return "freeze_and_refit_pipeline"
    if state.route == "check_stopping":
        return "check_stopping_conditions"
    return "diagnose_failure"


def _route_from_validate(state: ModelingAgentState) -> str:
    return {
        "execute": "execute_deterministic_tool",
        "missing_capability": "request_capability",
        "outlier_approval": "request_outlier_approval",
        "invalid_action": "check_stopping_conditions",
    }.get(state.route, "check_stopping_conditions")


def _route_from_stop(state: ModelingAgentState) -> str:
    if state.route == "freeze" or state.stopping_reason == STOP_REQUIREMENTS_MET:
        return "freeze_and_refit_pipeline"
    if state.route == "continue_diagnosis":
        return "diagnose_failure"
    return "write_final_report"


def _route_from_approval(state: ModelingAgentState) -> str:
    if state.route == "execute":
        return "execute_deterministic_tool"
    return "diagnose_failure"


def build_modeling_graph(checkpointer: Any):
    """Compile the modeling-improvement StateGraph with the given checkpointer."""
    graph = StateGraph(ModelingAgentState)
    graph.add_node("load_and_validate_handoff", load_and_validate_handoff)
    graph.add_node("create_development_safe_view", create_development_safe_view)
    graph.add_node("evaluate_requirements", evaluate_requirements_node)
    graph.add_node("diagnose_failure", diagnose_failure)
    graph.add_node("form_hypothesis", form_hypothesis)
    graph.add_node("validate_action", validate_action)
    graph.add_node("execute_deterministic_tool", execute_deterministic_tool)
    graph.add_node("ingest_result", ingest_result)
    graph.add_node("compare_candidates", compare_candidates)
    graph.add_node("check_stopping_conditions", check_stopping_conditions)
    graph.add_node("request_capability", request_capability)
    graph.add_node("request_outlier_approval", request_outlier_approval)
    graph.add_node("freeze_and_refit_pipeline", freeze_and_refit_pipeline)
    graph.add_node("evaluate_sealed_test", evaluate_sealed_test_node)
    graph.add_node("write_final_report", write_final_report)

    graph.add_edge(START, "load_and_validate_handoff")
    graph.add_conditional_edges(
        "load_and_validate_handoff",
        _route_from_load,
        {
            "create_development_safe_view": "create_development_safe_view",
            "write_final_report": "write_final_report",
        },
    )
    graph.add_edge("create_development_safe_view", "evaluate_requirements")
    graph.add_conditional_edges(
        "evaluate_requirements",
        _route_from_eval,
        {
            "freeze_and_refit_pipeline": "freeze_and_refit_pipeline",
            "diagnose_failure": "diagnose_failure",
            "check_stopping_conditions": "check_stopping_conditions",
        },
    )
    graph.add_edge("diagnose_failure", "form_hypothesis")
    graph.add_edge("form_hypothesis", "validate_action")
    graph.add_conditional_edges(
        "validate_action",
        _route_from_validate,
        {
            "execute_deterministic_tool": "execute_deterministic_tool",
            "request_capability": "request_capability",
            "request_outlier_approval": "request_outlier_approval",
            "check_stopping_conditions": "check_stopping_conditions",
        },
    )
    graph.add_conditional_edges(
        "request_outlier_approval",
        _route_from_approval,
        {
            "execute_deterministic_tool": "execute_deterministic_tool",
            "diagnose_failure": "diagnose_failure",
        },
    )
    graph.add_edge("execute_deterministic_tool", "ingest_result")
    graph.add_edge("ingest_result", "compare_candidates")
    graph.add_edge("compare_candidates", "evaluate_requirements")
    graph.add_conditional_edges(
        "check_stopping_conditions",
        _route_from_stop,
        {
            "freeze_and_refit_pipeline": "freeze_and_refit_pipeline",
            "diagnose_failure": "diagnose_failure",
            "write_final_report": "write_final_report",
        },
    )
    graph.add_edge("request_capability", "write_final_report")
    graph.add_edge("freeze_and_refit_pipeline", "evaluate_sealed_test")
    graph.add_edge("evaluate_sealed_test", "write_final_report")
    graph.add_edge("write_final_report", END)
    return graph.compile(checkpointer=checkpointer)


logger = logging.getLogger(__name__)


def _sqlite_checkpointer(run_dir: str | Path):
    db = Path(run_dir) / "agent_results" / "langgraph_checkpoints.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(db), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        return checkpointer
    except Exception as exc:
        from langgraph.checkpoint.memory import InMemorySaver

        logger.warning(
            "SQLite checkpointer unavailable (%s). Using an in-memory checkpointer. "
            "Prepend this environment's lib/ to LD_LIBRARY_PATH before starting Python "
            "to persist checkpoints at %s.",
            exc,
            db,
        )
        return InMemorySaver()


def compile_modeling_graph(run_dir: str | Path, checkpointer: Any | None = None):
    if checkpointer is None:
        checkpointer = _sqlite_checkpointer(run_dir)
    return build_modeling_graph(checkpointer)


def thread_config(run_id: str, recursion_limit: int) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": f"qsar-agentic-{run_id}"},
        "recursion_limit": recursion_limit,
    }


def invoke_modeling_graph(
    compiled: Any,
    state: ModelingAgentState,
    *,
    resume: bool = False,
    resume_value: Any = None,
    recursion_limit: int = 140,
) -> ModelingAgentState:
    config = thread_config(Path(state.run_dir).name, recursion_limit)
    try:
        if resume:
            from langgraph.types import Command

            payload = None if resume_value is None else Command(resume=resume_value)
            raw = compiled.invoke(payload, config)
        else:
            raw = compiled.invoke(state.model_dump(mode="json"), config)
    except GraphRecursionError:
        updates = write_final_report(state.model_copy(update={"stopping_reason": "recursion_limit"}))
        return state.model_copy(update={"stopping_reason": "recursion_limit", **updates})
    if isinstance(raw, ModelingAgentState):
        return raw
    if isinstance(raw, dict):
        payload = {k: v for k, v in raw.items() if not str(k).startswith("__")}
        try:
            return ModelingAgentState.model_validate(payload)
        except Exception:
            allowed = set(ModelingAgentState.model_fields)
            return state.model_copy(update={k: v for k, v in payload.items() if k in allowed})
    return state
