"""
src/moderation/decision_engine.py
---------------------------------
Agent 3 — Supervisor & Decision Engine.

Combines independent outputs from Agent 1 (FakeNewsDetector) and
Agent 2 (BotDetector) without conflating automated accounts with
misinformation:
  - A human account can publish fake news.
  - A bot account can publish real news.

Moderation decisions:
  - APPROVE  : Both agents predict benign ("real" + "human") at high confidence.
  - REVIEW   : Single-signal threat ("fake"+"human" or "real"+"bot"), or any
               uncertain (medium/low confidence) prediction.
  - RESTRICT : Temporary restriction / reduced distribution pending review when
               both "fake" and "bot" are predicted at high confidence.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from src.moderation.policy import SupervisorPolicy


VALID_FAKE_NEWS_LABELS = ("fake", "real")
VALID_BOT_LABELS = ("bot", "human")
VALID_DECISIONS = ("APPROVE", "REVIEW", "RESTRICT")
VALID_RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


class SupervisorAgent:
    """
    Coordinates Agent 1 (Fake News) and Agent 2 (Bot Detection) outputs to
    produce a transparent risk score, moderation decision, structured reasons,
    and human-review requirement flag.
    """

    def __init__(
        self,
        policy: SupervisorPolicy | None = None,
        fake_news_detector: Any = None,
        bot_detector: Any = None,
    ):
        self.policy = policy or SupervisorPolicy()
        self._fake_news_detector = fake_news_detector
        self._bot_detector = bot_detector

    # ------------------------------------------------------------------
    # Validation & signal extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_agent_output(
        output: Any,
        agent_name: str,
        valid_labels: tuple[str, ...],
        threat_key: str,
    ) -> tuple[str, float, float]:
        """
        Validate a single agent's output dictionary and return:
        (normalized_prediction_label, confidence, independent_threat_signal).
        """
        if output is None:
            raise ValueError(f"Missing {agent_name} agent output.")
        if not isinstance(output, dict):
            raise TypeError(
                f"{agent_name} output must be a dict, got {type(output).__name__}."
            )

        raw_pred = output.get("prediction")
        if not isinstance(raw_pred, str) or raw_pred.strip().lower() not in valid_labels:
            raise ValueError(
                f"Invalid {agent_name} prediction label: {raw_pred!r}. "
                f"Expected one of {valid_labels}."
            )
        prediction = raw_pred.strip().lower()

        if "confidence" not in output or output["confidence"] is None:
            raise ValueError(f"Missing 'confidence' value in {agent_name} output.")

        conf = output["confidence"]
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            raise ValueError(
                f"Invalid 'confidence' type in {agent_name} output: {type(conf).__name__}."
            )
        conf = float(conf)
        if math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(
                f"Invalid 'confidence' in {agent_name} output: {conf}. "
                "Expected a finite float in [0.0, 1.0]."
            )

        # Extract the independent threat probability proxy (P(fake) or P(bot)).
        scores = output.get("scores")
        if (
            isinstance(scores, dict)
            and threat_key in scores
            and scores[threat_key] is not None
            and not isinstance(scores[threat_key], bool)
            and isinstance(scores[threat_key], (int, float))
        ):
            threat_signal = float(scores[threat_key])
            if math.isnan(threat_signal) or math.isinf(threat_signal) or not (0.0 <= threat_signal <= 1.0):
                raise ValueError(
                    f"Invalid scores['{threat_key}'] in {agent_name} output: {threat_signal}."
                )
        else:
            threat_signal = conf if prediction == threat_key else (1.0 - conf)

        return prediction, conf, threat_signal

    def _confidence_tier(self, confidence: float) -> str:
        """Map a [0.0, 1.0] confidence score to HIGH, MEDIUM, or LOW."""
        if confidence >= self.policy.high_confidence:
            return "HIGH"
        if confidence >= self.policy.medium_confidence:
            return "MEDIUM"
        return "LOW"

    def _compute_risk_score(self, fake_signal: float, bot_signal: float) -> float:
        """
        Compute bounded heuristic risk score in [0.0, 100.0].

        Formula:
            risk_score = 100.0 * (w_fake * P_signal(fake) + w_bot * P_signal(bot))
        where w_fake + w_bot = 1.0 (default 0.60 and 0.40).
        This is a transparent moderation heuristic, not a calibrated probability.
        """
        w_fake, w_bot = self.policy.normalized_weights
        score = 100.0 * (w_fake * fake_signal + w_bot * bot_signal)
        return round(min(100.0, max(0.0, score)), 2)

    def _classify_risk_level(self, risk_score: float) -> str:
        """Map risk_score (0-100) to LOW, MEDIUM, or HIGH."""
        if risk_score >= self.policy.high_risk_threshold:
            return "HIGH"
        if risk_score >= self.policy.medium_risk_threshold:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, fake_news_output: dict, bot_output: dict) -> dict:
        """
        Analyze independent outputs from Agent 1 and Agent 2 and return a
        structured moderation decision.

        Parameters
        ----------
        fake_news_output : dict
            Output dictionary returned by FakeNewsDetector.predict().
        bot_output : dict
            Output dictionary returned by BotDetector.predict().

        Returns
        -------
        dict
            Canonical Supervisor output dictionary.
        """
        fake_pred, fake_conf, fake_signal = self._validate_agent_output(
            fake_news_output,
            agent_name="fake_news",
            valid_labels=VALID_FAKE_NEWS_LABELS,
            threat_key="fake",
        )
        bot_pred, bot_conf, bot_signal = self._validate_agent_output(
            bot_output,
            agent_name="bot_detection",
            valid_labels=VALID_BOT_LABELS,
            threat_key="bot",
        )

        fake_tier = self._confidence_tier(fake_conf)
        bot_tier = self._confidence_tier(bot_conf)

        risk_score = self._compute_risk_score(fake_signal, bot_signal)
        risk_level = self._classify_risk_level(risk_score)

        reasons: list[str] = [
            f"Fake News Agent classified content as '{fake_pred}' "
            f"(confidence={fake_conf:.4f}, tier={fake_tier}).",
            f"Bot Detection Agent classified account as '{bot_pred}' "
            f"(confidence={bot_conf:.4f}, tier={bot_tier}).",
        ]

        uncertain_agents = []
        if fake_tier != "HIGH":
            uncertain_agents.append(f"fake_news ({fake_tier.lower()} confidence {fake_conf:.4f})")
        if bot_tier != "HIGH":
            uncertain_agents.append(f"bot_detection ({bot_tier.lower()} confidence {bot_conf:.4f})")

        # Decision policy evaluates both labels AND confidence tiers.
        if uncertain_agents and self.policy.review_on_uncertainty:
            decision = "REVIEW"
            reasons.append(
                "Uncertain prediction(s) detected for "
                + " and ".join(uncertain_agents)
                + "; routing to human review instead of automated approval or restriction."
            )
            if fake_pred == "fake" or bot_pred == "bot":
                reasons.append(
                    "At least one threat signal is present alongside uncertainty; "
                    "manual verification is required."
                )
        elif fake_pred == "real" and bot_pred == "human":
            decision = "APPROVE"
            reasons.append(
                "Content is classified as real news and account as human, both with high confidence."
            )
        elif fake_pred == "fake" and bot_pred == "human":
            decision = "REVIEW"
            reasons.append(
                "Potential misinformation published by an account classified as human; "
                "requires human review."
            )
        elif fake_pred == "real" and bot_pred == "bot":
            decision = "REVIEW"
            reasons.append(
                "Account is classified as a bot, but content is classified as real news; "
                "bot classification alone never triggers automatic restriction."
            )
        elif fake_pred == "fake" and bot_pred == "bot":
            if (not self.policy.restrict_requires_both_high) or (
                fake_tier == "HIGH" and bot_tier == "HIGH"
            ):
                decision = "RESTRICT"
                reasons.append(
                    "Both fake news and bot activity detected with high confidence; "
                    "applying temporary restriction / reduced distribution pending review."
                )
            else:
                decision = "REVIEW"
                reasons.append(
                    "Both threat signals present without dual high confidence; routing to review."
                )
        else:
            decision = "REVIEW"
            reasons.append("Fallback routing to human review.")

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
            "input_predictions": {
                "fake_news": copy.deepcopy(fake_news_output),
                "bot_detection": copy.deepcopy(bot_output),
            },
        }

    def analyze(self, text: str, account_data: dict) -> dict:
        """
        End-to-end convenience method: run Agent 1 on text and Agent 2 on
        account_data, then evaluate both outputs via the Supervisor.
        """
        if self._fake_news_detector is None:
            from src.detection.fake_news_detector import FakeNewsDetector
            self._fake_news_detector = FakeNewsDetector()
        if self._bot_detector is None:
            from src.detection.bot_detector import BotDetector
            self._bot_detector = BotDetector()

        fake_news_output = self._fake_news_detector.predict(text)
        bot_output = self._bot_detector.predict(account_data)
        return self.evaluate(fake_news_output, bot_output)


def evaluate_threat(
    fake_news_output: dict,
    bot_output: dict,
    policy: SupervisorPolicy | None = None,
) -> dict:
    """Functional helper to run SupervisorAgent.evaluate() in one call."""
    return SupervisorAgent(policy=policy).evaluate(fake_news_output, bot_output)
