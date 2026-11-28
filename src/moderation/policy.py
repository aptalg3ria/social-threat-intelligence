"""
src/moderation/policy.py
------------------------
Configurable threshold and risk-scoring policy for Agent 3 (Supervisor).
"""

from dataclasses import dataclass


# Default confidence thresholds required by the Supervisor decision policy
HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.60


@dataclass(frozen=True)
class SupervisorPolicy:
    """
    Configuration for Supervisor confidence tiers, risk scoring, and decisions.

    Attributes
    ----------
    high_confidence : float
        Minimum confidence required for a prediction to be treated as high
        confidence (default: 0.85).
    medium_confidence : float
        Minimum confidence for the medium-confidence band; values strictly
        below this threshold are treated as low confidence (default: 0.60).
    fake_news_weight : float
        Relative weight assigned to the Fake News signal in risk scoring
        (default: 0.60).
    bot_detection_weight : float
        Relative weight assigned to the Bot Detection signal in risk scoring
        (default: 0.40).
    medium_risk_threshold : float
        Risk score (0-100) at or above which risk_level becomes MEDIUM
        (default: 35.0).
    high_risk_threshold : float
        Risk score (0-100) at or above which risk_level becomes HIGH
        (default: 70.0).
    review_on_uncertainty : bool
        When True, any prediction with medium or low confidence routes the
        case to REVIEW instead of APPROVE or RESTRICT.
    restrict_requires_both_high : bool
        When True, RESTRICT is only issued when both agents predict a threat
        ("fake" and "bot") with high confidence.
    require_human_review_on_restrict : bool
        When True, RESTRICT decisions still flag human_review_required=True
        because RESTRICT represents a temporary restriction pending review.
    """

    high_confidence: float = HIGH_CONFIDENCE
    medium_confidence: float = MEDIUM_CONFIDENCE
    fake_news_weight: float = 0.60
    bot_detection_weight: float = 0.40
    medium_risk_threshold: float = 35.0
    high_risk_threshold: float = 70.0
    review_on_uncertainty: bool = True
    restrict_requires_both_high: bool = True
    require_human_review_on_restrict: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.medium_confidence < self.high_confidence <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0.0 < medium_confidence < high_confidence <= 1.0. "
                f"Got medium={self.medium_confidence}, high={self.high_confidence}."
            )
        if self.fake_news_weight < 0.0 or self.bot_detection_weight < 0.0:
            raise ValueError("Risk weights must be non-negative.")
        if (self.fake_news_weight + self.bot_detection_weight) <= 0.0:
            raise ValueError("At least one risk weight must be positive.")
        if not (0.0 <= self.medium_risk_threshold < self.high_risk_threshold <= 100.0):
            raise ValueError(
                "Risk thresholds must satisfy 0.0 <= medium_risk_threshold "
                "< high_risk_threshold <= 100.0."
            )

    @property
    def normalized_weights(self) -> tuple[float, float]:
        """Return (fake_weight, bot_weight) normalized to sum to 1.0."""
        total = self.fake_news_weight + self.bot_detection_weight
        return (self.fake_news_weight / total, self.bot_detection_weight / total)
