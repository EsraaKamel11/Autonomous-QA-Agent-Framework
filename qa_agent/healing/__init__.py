"""Self-healing subsystem: failure classification and patch generation."""

from qa_agent.healing.classifier import (
    ClassificationResult,
    classify_failure,
    classify_all_failures,
    is_auto_healable,
)

__all__ = [
    "ClassificationResult",
    "classify_failure",
    "classify_all_failures",
    "is_auto_healable",
]
