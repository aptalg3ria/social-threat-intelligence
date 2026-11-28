"""
src/moderation/meta_supervisor.py
---------------------------------
Part 3 Moderation Integration combining:
  - Agent 1 (FakeNewsDetector, frozen)
  - Agent 2 (BotDetector, frozen)
  - Agent 3 Logistic Regression Meta-Classifier (MetaLogisticRegression)
  - Safety Guardrails & Explainability
  - Human Review Queue compatibility

Preserves the Rule-Based Supervisor (src/moderation/decision_engine.py)
as an intact baseline while providing the exact same output schema contract.
"""

from __future__ import annotations

import copy
from typing import Any

from src.models.logistic_regression import (
    DEFAULT_META_MODEL_PATH,
    MetaLogisticRegression,
)
from src.moderation.decision_engine import (
    SupervisorAgent,
    VALID_BOT_LABELS,
    VALID_FAKE_NEWS_LABELS,
)
from src.moderation.policy import SupervisorPolicy


class MetaSupervisorAgent:
    """
    Moderation Supervisor powered by the Logistic Regression Meta-Classifier
    with deterministic safety guardrails and structured explainability.
    """

    def __init__(
        self,
        meta_classifier: MetaLogisticRegression | None = None,
        model_path: str = DEFAULT_META_MODEL_PATH,
        policy: SupervisorPolicy | None = None,
        meta_confidence_threshold: float = 0.60,
        fake_news_detector: Any = None,
        bot_detector: Any = None,
    ):
        if not (0.0 < meta_confidence_threshold <= 1.0):
            raise ValueError("meta_confidence_threshold must be in (0.0, 1.0].")

        self.policy = policy or SupervisorPolicy()
        self.meta_confidence_threshold = meta_confidence_threshold
        self._fake_news_detector = fake_news_detector
        self._bot_detector = bot_detector

        if meta_classifier is not None:
            if not getattr(meta_classifier, "is_fitted_", False):
                raise ValueError("Provided MetaLogisticRegression instance is not fitted.")
            self.meta_classifier = meta_classifier
        else:
            self.meta_classifier = MetaLogisticRegression.load(model_path)

        # Reuse SupervisorAgent helpers for validation, risk scoring, and tiers.
        self._helper = SupervisorAgent(policy=self.policy)

    def evaluate(self, fake_news_output: dict, bot_output: dict) -> dict:
        """
        Evaluate frozen Agent 1 and Agent 2 outputs using Logistic Regression
        plus safety guardrails, returning the canonical Supervisor output dict.
        """
        fake_pred, fake_conf, fake_signal = SupervisorAgent._validate_agent_output(
            fake_news_output,
            agent_name="fake_news",
            valid_labels=VALID_FAKE_NEWS_LABELS,
            threat_key="fake",
        )
        bot_pred, bot_conf, bot_signal = SupervisorAgent._validate_agent_output(
            bot_output,
            agent_name="bot_detection",
            valid_labels=VALID_BOT_LABELS,
            threat_key="bot",
        )

        fake_tier = self._helper._confidence_tier(fake_conf)
        bot_tier = self._helper._confidence_tier(bot_conf)
        risk_score = self._helper._compute_risk_score(fake_signal, bot_signal)
        risk_level = self._helper._classify_risk_level(risk_score)

        lr_result = self.meta_classifier.predict_from_outputs(fake_news_output, bot_output)
        decision = lr_result["prediction"]
        lr_conf = lr_result["confidence"]
        lr_probs = lr_result["probabilities"]

        reasons: list[str] = [
            f"Fake News Agent classified content as '{fake_pred}' "
            f"(confidence={fake_conf:.4f}, tier={fake_tier}).",
            f"Bot Detection Agent classified account as '{bot_pred}' "
            f"(confidence={bot_conf:.4f}, tier={bot_tier}).",
            f"Logistic Regression Meta-Classifier predicted '{decision}' "
            f"(confidence={lr_conf:.4f}, probabilities={lr_probs}).",
        ]

        guardrail_applied = False

        # Guardrail 0: Never auto-approve when either agent detects a threat.
        if decision == "APPROVE" and (fake_pred == "fake" or bot_pred == "bot"):
            decision = "REVIEW"
            guardrail_applied = True
            reasons.append(
                "Safety guardrail override: automated APPROVE is blocked because at least "
                "one agent detected a threat signal ('fake' or 'bot'); routing to review."
            )

        # Guardrail 1: Bot classification alone never causes automatic restriction.
        if decision == "RESTRICT" and fake_pred == "real":
            decision = "REVIEW"
            guardrail_applied = True
            reasons.append(
                "Safety guardrail override: content is classified as real news; "
                "bot classification alone never triggers automatic restriction."
            )

        # Guardrail 2: Fake news alone (by human account) never auto-restricts under policy.
        elif (
            decision == "RESTRICT"
            and self.policy.restrict_requires_both_high
            and bot_pred == "human"
        ):
            decision = "REVIEW"
            guardrail_applied = True
            reasons.append(
                "Safety guardrail override: account is classified as human; "
                "routing potential misinformation to human review instead of restriction."
            )

        # Guardrail 3: Uncertain base-agent predictions cannot be auto-approved/restricted.
        if (
            self.policy.review_on_uncertainty
            and decision in ("APPROVE", "RESTRICT")
            and (fake_tier != "HIGH" or bot_tier != "HIGH")
        ):
            decision = "REVIEW"
            guardrail_applied = True
            reasons.append(
                "Safety guardrail override: at least one base agent has medium or low "
                "confidence; routing uncertain case to human review."
            )

        # Guardrail 4: Low meta-classifier confidence escalates to REVIEW.
        if decision != "REVIEW" and lr_conf < self.meta_confidence_threshold:
            decision = "REVIEW"
            guardrail_applied = True
            reasons.append(
                f"Safety guardrail override: meta-classifier confidence ({lr_conf:.4f}) "
                f"is below threshold ({self.meta_confidence_threshold:.2f}); routing to review."
            )

        if decision == "APPROVE":
            human_review_required = False
        elif decision == "RESTRICT":
            human_review_required = self.policy.require_human_review_on_restrict
        else:
            human_review_required = True

        return {
            "agent": "supervisor",
            "decision": decision,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "reasons": reasons,
            "human_review_required": human_review_required,
            "meta_classifier": {
                "model": "LogisticRegression",
                "raw_prediction": lr_result["prediction"],
                "confidence": lr_conf,
                "probabilities": lr_probs,
                "guardrail_applied": guardrail_applied,
            },
            "input_predictions": {
                "fake_news": copy.deepcopy(fake_news_output),
                "bot_detection": copy.deepcopy(bot_output),
            },
        }

    def analyze(self, text: str, account_data: dict) -> dict:
        """
        Run frozen Agent 1 and Agent 2 on raw inputs and evaluate via MetaSupervisorAgent.
        """
        if self._fake_news_detector is None:
            from src.detection.fake_news_detector import FakeNewsDetector
            self._fake_news_detector = FakeNewsDetector()
        if self._bot_detector is None:
            from src.detection.bot_detector import BotDetector
            self._bot_detector = BotDetector()

        fn_output = self._fake_news_detector.predict(text)
        bot_output = self._bot_detector.predict(account_data)
        return self.evaluate(fn_output, bot_output)
