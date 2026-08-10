"""Centralized system prompts for agentic specialists and supervisor."""

from qsar_agent.agentic.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from qsar_agent.agentic.prompts.data_quality import DATA_QUALITY_SYSTEM_PROMPT
from qsar_agent.agentic.prompts.descriptor_feature import DESCRIPTOR_FEATURE_SYSTEM_PROMPT
from qsar_agent.agentic.prompts.modeling import MODELING_SYSTEM_PROMPT
from qsar_agent.agentic.prompts.validation import VALIDATION_SYSTEM_PROMPT

__all__ = [
    "SUPERVISOR_SYSTEM_PROMPT",
    "DATA_QUALITY_SYSTEM_PROMPT",
    "DESCRIPTOR_FEATURE_SYSTEM_PROMPT",
    "MODELING_SYSTEM_PROMPT",
    "VALIDATION_SYSTEM_PROMPT",
]
