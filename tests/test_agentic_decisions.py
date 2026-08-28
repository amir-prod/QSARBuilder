"""AgentDecision schema validation rejects unsafe or duplicate actions."""

from __future__ import annotations

from qsar_agent.agentic.decisions import fallback_decision, validate_decision
from qsar_agent.agentic.ids import make_experiment_id
from qsar_agent.config import WorkflowConfig
from qsar_agent.schemas.agentic import AgentDecision, ModelingAgentState
from tests.agentic_harness import make_decision


def _state(**kwargs) -> ModelingAgentState:
    data = {
        "run_dir": "/tmp/run",
        "dataset_hash": "ds",
        "development_split_hash": "dev",
        "failed_requirements": [{"name": "minimum_cv_r2", "observed": 0.2, "required": 0.5}],
        "acceptance_status": "failed",
        "requirements": {"minimum_cv_r2": 0.5, "maximum_train_cv_gap": 0.15},
    }
    data.update(kwargs)
    return ModelingAgentState.model_validate(data)


def test_duplicate_experiment_id_is_rejected():
    state = _state()
    decision = make_decision("run_model_search", {"estimator": "RandomForestRegressor"})
    exp_id = make_experiment_id(
        tool_name="run_model_search",
        arguments={"estimator": "RandomForestRegressor"},
        dataset_hash="ds",
        development_split_hash="dev",
        parent_experiment_id=None,
    )
    state.completed_experiment_ids = [exp_id]
    reason = validate_decision(decision, state, WorkflowConfig())
    assert "equivalent completed experiment" in reason.lower() or "repeated" in reason.lower()


def test_unsupported_hyperparameter_is_rejected():
    state = _state()
    decision = make_decision(
        "run_model_search",
        {
            "estimator": "RandomForestRegressor",
            "param_grid": {"learning_rate": [0.1], "not_a_param": [1]},
        },
    )
    reason = validate_decision(decision, state, WorkflowConfig())
    assert reason
    assert "allowlisted" in reason.lower() or "not in the allowlisted" in reason.lower()


def test_unsupported_estimator_is_rejected():
    state = _state()
    decision = make_decision("run_model_search", {"estimator": "XGBoostRegressor"})
    reason = validate_decision(decision, state, WorkflowConfig())
    assert "Unsupported model family" in reason


def test_budget_exhausted_is_rejected():
    from qsar_agent.config import AgentLimits, AgenticImprovementSettings

    state = _state(adaptive_experiments_used=12)
    decision = make_decision("run_model_search", {"estimator": "RandomForestRegressor"})
    config = WorkflowConfig(
        agentic_improvement=AgenticImprovementSettings(limits=AgentLimits(maximum_adaptive_experiments=12))
    )
    reason = validate_decision(decision, state, config)
    assert "budget" in reason.lower()


def test_missing_success_conditions_rejected():
    state = _state()
    decision = AgentDecision.model_validate(
        {
            "diagnosis": "underfitting",
            "evidence": [{"observation": "cv_r2", "value": 0.2, "interpretation": "low"}],
            "hypothesis": "try HPO",
            "action": {"tool_name": "run_model_search", "arguments": {"estimator": "RandomForestRegressor"}},
            "success_conditions": {},
            "reason_existing_results_are_insufficient": "failed",
        }
    )
    reason = validate_decision(decision, state, WorkflowConfig())
    assert "success condition" in reason.lower()


def test_missing_evidence_rejected():
    state = _state()
    decision = AgentDecision.model_validate(
        {
            "diagnosis": "underfitting",
            "evidence": [],
            "hypothesis": "try HPO",
            "action": {"tool_name": "run_model_search", "arguments": {}},
            "success_conditions": {"minimum_cv_r2": 0.5},
            "reason_existing_results_are_insufficient": "failed",
        }
    )
    reason = validate_decision(decision, state, WorkflowConfig())
    assert "evidence" in reason.lower()


def test_test_path_in_arguments_rejected():
    state = _state()
    decision = make_decision(
        "run_model_search",
        {"estimator": "RandomForestRegressor", "test_path": "preprocessed_test_descriptors.csv"},
    )
    reason = validate_decision(decision, state, WorkflowConfig())
    assert reason


def test_pca_rejected_when_latent_components_disallowed():
    state = _state()
    decision = make_decision("run_feature_selection_search", {"method": "pca", "n_features": 3})
    reason = validate_decision(decision, state, WorkflowConfig())
    assert "latent" in reason.lower()


def test_fallback_decision_is_schema_valid():
    decision = fallback_decision(_state())
    assert decision.action.tool_name
    assert decision.evidence
    assert decision.success_conditions.minimum_cv_r2 is not None
