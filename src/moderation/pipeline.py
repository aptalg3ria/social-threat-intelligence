"""
src/moderation/pipeline.py
--------------------------
End-to-End Integration Pipeline connecting:
  - Agent 1: FakeNewsDetector (frozen Naive Bayes + TF-IDF)
  - Agent 2: BotDetector (frozen PyTorch GRU + StandardScaler)
  - Agent 3: MetaSupervisorAgent (synthetic-trained Logistic Regression + guardrails)
  - Human Review Queue (ReviewQueue)

Input validation strictly rejects empty/invalid post text and incomplete
account metadata rather than inventing missing features or zero-imputing.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.detection.bot_detector import BotDetector
from src.detection.fake_news_detector import FakeNewsDetector
from src.models.logistic_regression import (
    MetaLogisticRegression,
    extract_meta_features_from_outputs,
)
from src.moderation.human_review import ReviewQueue
from src.moderation.meta_supervisor import MetaSupervisorAgent
from src.moderation.policy import SupervisorPolicy


DEFAULT_SYNTHETIC_META_MODEL_PATH = "models/saved/meta_logreg_synthetic.joblib"

# Exact raw fields needed to compute all 15 Agent 2 features without silent
# zero-imputation (created_at supplies hour_of_day, day_of_week, and year).
REQUIRED_ACCOUNT_METADATA_FIELDS: tuple[str, ...] = (
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "average_tweets_per_day",
    "account_age_days",
    "default_profile",
    "default_profile_image",
    "geo_enabled",
    "verified",
    "created_at",
    "description",
    "screen_name",
)

_NUMERIC_ACCOUNT_FIELDS: tuple[str, ...] = (
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "average_tweets_per_day",
    "account_age_days",
)

_BOOLEAN_ACCOUNT_FIELDS: tuple[str, ...] = (
    "default_profile",
    "default_profile_image",
    "geo_enabled",
    "verified",
)


def validate_post_text(text: Any) -> str:
    """Validate that post text is a non-empty string."""
    if not isinstance(text, str):
        raise TypeError(f"Post text must be a str, got {type(text).__name__}.")
    if not text.strip():
        raise ValueError("Post text cannot be empty or whitespace-only.")
    return text


def validate_account_metadata(account_data: Any) -> dict[str, Any]:
    """
    Validate that account_data contains all required raw metadata fields needed
    by Agent 2 without inventing missing values or silently replacing with zeros.
    """
    if not isinstance(account_data, dict):
        raise TypeError(
            f"Account metadata must be a dict, got {type(account_data).__name__}."
        )
    if not account_data:
        raise ValueError("Account metadata dictionary cannot be empty.")

    missing = [
        key for key in REQUIRED_ACCOUNT_METADATA_FIELDS
        if key not in account_data or account_data[key] is None
    ]
    if missing:
        raise ValueError(
            f"Missing required account metadata fields for Agent 2 (refusing to "
            f"invent or zero-impute missing data): {missing}"
        )

    for col in _NUMERIC_ACCOUNT_FIELDS:
        val = account_data[col]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(
                f"Account field '{col}' must be numeric, got {type(val).__name__}."
            )
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval) or fval < 0.0:
            raise ValueError(
                f"Account field '{col}' must be a finite non-negative number, got {val!r}."
            )

    for col in _BOOLEAN_ACCOUNT_FIELDS:
        val = account_data[col]
        if not isinstance(val, bool) and not (isinstance(val, int) and val in (0, 1)):
            raise ValueError(
                f"Account field '{col}' must be a boolean (or 0/1), got {val!r}."
            )

    created_ts = pd.to_datetime(account_data["created_at"], errors="coerce")
    if pd.isna(created_ts):
        raise ValueError(
            f"Account field 'created_at' is not a valid timestamp: {account_data['created_at']!r}."
        )

    if not isinstance(account_data["description"], str):
        raise ValueError(
            f"Account field 'description' must be a str, got {type(account_data['description']).__name__}."
        )
    if not isinstance(account_data["screen_name"], str) or not account_data["screen_name"].strip():
        raise ValueError("Account field 'screen_name' must be a non-empty str.")

    return account_data


class SocialThreatPipeline:
    """
    End-to-End Social Threat Intelligence pipeline connecting Agent 1, Agent 2,
    Agent 3 (MetaSupervisorAgent), and Human Review Queue.
    """

    def __init__(
        self,
        fake_news_detector: FakeNewsDetector | Any | None = None,
        bot_detector: BotDetector | Any | None = None,
        meta_classifier: MetaLogisticRegression | Any | None = None,
        meta_model_path: str = DEFAULT_SYNTHETIC_META_MODEL_PATH,
        policy: SupervisorPolicy | None = None,
        meta_confidence_threshold: float = 0.60,
        review_queue: ReviewQueue | None = None,
    ):
        self.fake_news_detector = fake_news_detector or FakeNewsDetector()
        self.bot_detector = bot_detector or BotDetector()
        self.policy = policy or SupervisorPolicy()
        self.review_queue = review_queue if review_queue is not None else ReviewQueue()
        self.meta_model_path = meta_model_path

        self.meta_supervisor = MetaSupervisorAgent(
            meta_classifier=meta_classifier,
            model_path=meta_model_path,
            policy=self.policy,
            meta_confidence_threshold=meta_confidence_threshold,
            fake_news_detector=self.fake_news_detector,
            bot_detector=self.bot_detector,
        )

    def run(
        self,
        text: str,
        account_data: dict[str, Any],
        case_id: str | None = None,
        enqueue_review: bool = True,
        validate_inputs: bool = True,
    ) -> dict[str, Any]:
        """
        Execute Steps 1-7 of the End-to-End moderation pipeline:
          1. Run Agent 1 (Fake News Detection) on validated text.
          2. Run Agent 2 (Bot Detection) on validated account metadata.
          3. Collect both agent outputs.
          4. Extract the 6 meta-features via extract_meta_features_from_outputs().
          5. Predict via Agent 3 Logistic Regression Meta-Classifier.
          6. Apply confidence and safety guardrails (MetaSupervisorAgent).
          7. Enqueue REVIEW / pending RESTRICT cases into Human Review Queue
             and return structured unified output.
        """
        if validate_inputs:
            validate_post_text(text)
            validate_account_metadata(account_data)

        # Steps 1-3: Run frozen Agent 1 and Agent 2 independently
        fn_out = self.fake_news_detector.predict(text)
        bot_out = self.bot_detector.predict(account_data)

        # Step 4: Extract the 6 canonical meta-features
        meta_features = extract_meta_features_from_outputs(fn_out, bot_out)

        # Steps 5-6: Agent 3 Meta-Classifier + safety guardrails
        supervisor_out = self.meta_supervisor.evaluate(fn_out, bot_out)

        # Step 7: Route flagged cases to Human Review Queue
        review_case = (
            self.review_queue.submit(supervisor_out, case_id=case_id)
            if enqueue_review
            else None
        )

        fn_scores = fn_out.get("scores", {})
        bot_scores = bot_out.get("scores", {})
        meta_info = supervisor_out["meta_classifier"]

        return {
            "fake_news": {
                "prediction": fn_out["prediction"],
                "probability": float(fn_scores.get("fake", meta_features["fake_probability"])),
                "confidence": float(fn_out["confidence"]),
                "scores": dict(fn_scores),
                "raw_class": fn_out.get("raw_class"),
            },
            "bot_detection": {
                "prediction": bot_out["prediction"],
                "probability": float(bot_scores.get("bot", meta_features["bot_probability"])),
                "confidence": float(bot_out["confidence"]),
                "scores": dict(bot_scores),
                "raw_class": bot_out.get("raw_class"),
            },
            "meta_features": meta_features,
            "moderation": {
                "decision": supervisor_out["decision"],
                "confidence": float(meta_info["confidence"]),
                "probabilities": dict(meta_info["probabilities"]),
                "reason": " ".join(supervisor_out["reasons"]),
                "reasons": list(supervisor_out["reasons"]),
                "risk_level": supervisor_out["risk_level"],
                "risk_score": supervisor_out["risk_score"],
                "human_review_required": supervisor_out["human_review_required"],
                "guardrail_applied": meta_info["guardrail_applied"],
                "raw_meta_prediction": meta_info["raw_prediction"],
                "model_origin": "synthetic_bootstrap",
            },
            "review_queue": {
                "queued": review_case is not None,
                "case_id": review_case.case_id if review_case is not None else None,
                "status": review_case.status if review_case is not None else None,
            },
            "supervisor_output": supervisor_out,
        }
