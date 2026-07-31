"""Backward-compatible re-exports for descriptor calculation.

Prefer importing from ``qsar_agent.tools.descriptor_calculation``.
"""

from qsar_agent.tools.descriptor_calculation import (  # noqa: F401
    META_COLUMNS,
    calculate_descriptors,
    calculate_mordred_descriptors,
    merge_external_descriptors,
)
