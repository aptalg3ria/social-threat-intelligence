"""
src/moderation/__init__.py
--------------------------
Public exports for Agent 3 (Supervisor & Decision Engine).
"""

from src.moderation.policy import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    SupervisorPolicy,
)
from src.moderation.decision_engine import (
    SupervisorAgent,
    VALID_DECISIONS,
    VALID_RISK_LEVELS,
    evaluate_threat,
)
from src.moderation.human_review import (
    ReviewCase,
    ReviewQueue,
    VALID_HUMAN_DECISIONS,
)
from src.moderation.meta_supervisor import MetaSupervisorAgent
from src.moderation.pipeline import (
    DEFAULT_SYNTHETIC_META_MODEL_PATH,
    REQUIRED_ACCOUNT_METADATA_FIELDS,
    SocialThreatPipeline,
    validate_account_metadata,
    validate_post_text,
)

__all__ = [
    "HIGH_CONFIDENCE",
    "MEDIUM_CONFIDENCE",
    "SupervisorPolicy",
    "SupervisorAgent",
    "MetaSupervisorAgent",
    "SocialThreatPipeline",
    "DEFAULT_SYNTHETIC_META_MODEL_PATH",
    "REQUIRED_ACCOUNT_METADATA_FIELDS",
    "validate_account_metadata",
    "validate_post_text",
    "VALID_DECISIONS",
    "VALID_RISK_LEVELS",
    "evaluate_threat",
    "ReviewCase",
    "ReviewQueue",
    "VALID_HUMAN_DECISIONS",
]
