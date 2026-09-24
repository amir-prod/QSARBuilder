"""The Strategist: the only agent that decides what to try next.

It is always backed by a language model. There is no rule-based planner behind
it — if the model cannot produce a valid plan, the run stops and says so rather
than quietly substituting a heuristic.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import ValidationError

from qsar_agent.llm.agent_tools import build_strategist_toolset
from qsar_agent.llm.provider import LLMClient, LLMResponseError
from qsar_agent.logging_utils import get_logger
from qsar_agent.schemas.agentic import IterationPlan, PlanRevision
from qsar_agent.tools.model_zoo import list_estimators, validate_hyperparameters
from qsar_agent.tools.rdkit_features import AVAILABLE_BLOCKS

logger = get_logger()

DEFAULT_MAX_REPAIR_ATTEMPTS = 2


class StrategistError(RuntimeError):
    """Raised when the model cannot produce a usable plan."""


_ACTION_GUIDANCE = """\
Guidance by diagnosis (you may deviate with a stated reason):
- underfit: richer features (add fingerprints / MACCS keys), then a higher-capacity
  estimator (RandomForest, ExtraTrees, GradientBoosting), then looser regularisation.
- overfit: stronger regularisation (shallower trees, larger min_samples_leaf, higher
  alpha), then fewer features (tighter correlation_threshold, univariate_top_k), then a
  simpler estimator family.
- unstable_cv: more feature filtering, then a bagged ensemble with more estimators.
- chance_correlation: cut the feature count hard so the descriptor-to-sample ratio drops,
  and use a simple regularised estimator.
- poor_generalisation: regularise and broaden the feature space; the test chemistry
  differs from the training folds.
- narrow_applicability_domain: set drop_ad_outliers, and reduce dimensionality.
- class_imbalance: set class_weight='balanced' on an estimator that supports it.
- insufficient_data: reduce dimensionality aggressively; drop fingerprint blocks.
"""

_INITIAL_SCHEMA_HINT = """\
{
  "features": {
    "blocks": ["rdkit_descriptors" | "morgan_fp" | "morgan_counts" | "maccs_keys" |
               "rdkit_fp" | "atom_pair_fp"],
    "fingerprint_bits": int,
    "fingerprint_radius": int,
    "variance_threshold": float,
    "correlation_threshold": float or null,
    "univariate_top_k": int or null,
    "scale_features": bool,
    "drop_ad_outliers": bool
  },
  "model": {"estimator": str, "hyperparameters": {}},
  "rationale": str
}
"""

_REVISION_SCHEMA_HINT = """\
{
  "stop": bool,
  "stop_reason": str,
  "features": { ...same shape as the initial plan, or null to keep the current recipe... },
  "model": {"estimator": str, "hyperparameters": {}} or null,
  "rationale": str,
  "addresses_diagnosis": str
}
"""


class Strategist:
    """Wraps an LLM client with schema validation and repair."""

    def __init__(
        self,
        client: LLMClient,
        task: str,
        history_provider: Callable[[], list[dict[str, Any]]],
        summary_provider: Callable[[], dict[str, Any]],
        criteria_provider: Callable[[], dict[str, Any]],
        on_web_search: Callable[[], None] | None = None,
        max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
    ):
        self.client = client
        self.task = task
        self.max_repair_attempts = max_repair_attempts
        self.tools = build_strategist_toolset(
            task=task,
            history_provider=history_provider,
            summary_provider=summary_provider,
            criteria_provider=criteria_provider,
            on_web_search=on_web_search,
        )

    # -- prompts ---------------------------------------------------------------

    def _system_prompt(self) -> str:
        estimators = ", ".join(list_estimators(self.task))
        blocks = ", ".join(sorted(AVAILABLE_BLOCKS))
        return (
            "You are the strategist of an autonomous QSAR modelling team. You decide which "
            "molecular features and which machine-learning model the team should try next. "
            "You never compute or estimate a metric yourself: deterministic tools measure "
            "everything, and you only receive their results.\n\n"
            f"Task type: {self.task}.\n"
            f"Available estimators: {estimators}.\n"
            f"Available feature blocks: {blocks}.\n"
            "Hyperparameters must be real scikit-learn parameters of the estimator you pick; "
            "call list_model_zoo if you are unsure.\n\n"
            "The acceptance criteria were set by the user before the run and are immutable. "
            "Your job is to reach them, not to reinterpret them.\n\n"
            + _ACTION_GUIDANCE
        )

    # -- planning --------------------------------------------------------------

    def initial_plan(self, context: dict[str, Any]) -> IterationPlan:
        """Ask for the opening plan."""
        payload = {
            "request": "initial_plan",
            "instruction": (
                "Propose the first feature recipe and model for this dataset. Prefer a "
                "defensible, modest starting point: it is better to start simple and let the "
                "measured results justify added complexity."
            ),
            **context,
        }
        return self._ask(
            system=self._system_prompt(),
            user=json.dumps(payload, indent=2, default=str),
            schema_hint=_INITIAL_SCHEMA_HINT,
            validator=self._validate_initial,
        )

    def revise(self, context: dict[str, Any]) -> PlanRevision:
        """Ask how to change the plan after a failed validation."""
        payload = {
            "request": "revision",
            "instruction": (
                "The current plan did not meet the acceptance criteria. Propose the single most "
                "promising change, addressing the reported diagnosis. Set stop=true only if no "
                "untried change is worth testing."
            ),
            **context,
        }
        return self._ask(
            system=self._system_prompt(),
            user=json.dumps(payload, indent=2, default=str),
            schema_hint=_REVISION_SCHEMA_HINT,
            validator=self._validate_revision,
        )

    # -- internals -------------------------------------------------------------

    def _ask(
        self,
        system: str,
        user: str,
        schema_hint: str,
        validator: Callable[[dict[str, Any]], Any],
    ) -> Any:
        """Call the model, validating the reply and feeding errors back for repair."""
        message = user
        last_error: Exception | None = None

        for attempt in range(self.max_repair_attempts + 1):
            try:
                result = self.client.complete_json(
                    system=system,
                    user=message,
                    schema_hint=schema_hint,
                    tools=self.tools,
                )
            except LLMResponseError as exc:
                last_error = exc
                message = self._repair_message(user, str(exc))
                continue

            try:
                validated = validator(result.data)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Strategist reply rejected (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_repair_attempts + 1,
                    exc,
                )
                message = self._repair_message(user, str(exc))
                continue

            if result.tool_calls:
                logger.info("Strategist used tools: %s", ", ".join(result.tool_names()))
            return validated

        raise StrategistError(
            f"The strategist could not produce a valid plan after "
            f"{self.max_repair_attempts + 1} attempts. Last error: {last_error}"
        )

    @staticmethod
    def _repair_message(original: str, error: str) -> str:
        """Fold the validation error into the context.

        The user message stays a single JSON document so every client sees the
        same contract on the first attempt and on a repair.
        """
        try:
            context = json.loads(original)
        except json.JSONDecodeError:
            context = {"original_context": original}
        context["rejected_previous_reply"] = {
            "error": error,
            "instruction": (
                "Your previous reply was rejected for the reason above. Return corrected JSON "
                "only, using registered estimators, registered feature blocks, and "
                "hyperparameters the chosen estimator actually accepts."
            ),
        }
        return json.dumps(context, indent=2, default=str)

    def _validate_initial(self, data: dict[str, Any]) -> IterationPlan:
        plan = IterationPlan.model_validate(data)
        self._check_model(plan.model.estimator, plan.model.hyperparameters)
        return plan

    def _validate_revision(self, data: dict[str, Any]) -> PlanRevision:
        revision = PlanRevision.model_validate(data)
        if revision.stop:
            return revision
        if revision.features is None and revision.model is None:
            raise ValueError(
                "A revision must change the feature recipe, the model, or both — or set "
                "stop=true with a stop_reason."
            )
        if revision.model is not None:
            self._check_model(revision.model.estimator, revision.model.hyperparameters)
        return revision

    def _check_model(self, estimator: str, hyperparameters: dict[str, Any]) -> None:
        available = list_estimators(self.task)
        if estimator not in available:
            raise ValueError(
                f"Estimator {estimator!r} is not available for {self.task}. "
                f"Choose from: {', '.join(available)}."
            )
        validate_hyperparameters(self.task, estimator, hyperparameters)
