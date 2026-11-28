"""
tests/test_supervisor.py
------------------------
Unit and integration tests for Agent 3 (Supervisor & Decision Engine).

Unit tests use mocked agent outputs and require no training datasets.
Integration tests invoke the real Agent 1 (FakeNewsDetector) and
Agent 2 (BotDetector) models end-to-end.

Run with:
  pytest tests/test_supervisor.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.moderation import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    ReviewQueue,
    SupervisorAgent,
    SupervisorPolicy,
    VALID_DECISIONS,
    VALID_RISK_LEVELS,
    evaluate_threat,
)


# ---------------------------------------------------------------------------
# Helpers for building mocked Agent 1 & Agent 2 outputs
# ---------------------------------------------------------------------------

def make_fake_news_output(prediction: str, confidence: float) -> dict:
    p_fake = confidence if prediction == "fake" else round(1.0 - confidence, 4)
    p_real = confidence if prediction == "real" else round(1.0 - confidence, 4)
    return {
        "agent": "fake_news",
        "prediction": prediction,
        "confidence": confidence,
        "scores": {"fake": p_fake, "real": p_real},
        "raw_class": 0 if prediction == "fake" else 1,
    }


def make_bot_output(prediction: str, confidence: float) -> dict:
    p_bot = confidence if prediction == "bot" else round(1.0 - confidence, 4)
    p_human = confidence if prediction == "human" else round(1.0 - confidence, 4)
    return {
        "agent": "bot_detection",
        "prediction": prediction,
        "confidence": confidence,
        "scores": {"human": p_human, "bot": p_bot},
        "raw_class": 1 if prediction == "bot" else 0,
        "bot_score": p_bot,
        "bot_class": 1 if prediction == "bot" else 0,
        "bot_label": prediction.upper(),
    }


SUPERVISOR_KEYS = {
    "agent",
    "decision",
    "risk_level",
    "risk_score",
    "reasons",
    "human_review_required",
    "input_predictions",
}


def assert_supervisor_schema(result: dict) -> None:
    assert isinstance(result, dict)
    assert SUPERVISOR_KEYS.issubset(result.keys())
    assert result["agent"] == "supervisor"
    assert result["decision"] in VALID_DECISIONS
    assert result["risk_level"] in VALID_RISK_LEVELS
    assert isinstance(result["risk_score"], float)
    assert 0.0 <= result["risk_score"] <= 100.0
    assert isinstance(result["reasons"], list) and len(result["reasons"]) >= 3
    assert all(isinstance(r, str) and r for r in result["reasons"])
    assert isinstance(result["human_review_required"], bool)
    assert set(result["input_predictions"].keys()) == {"fake_news", "bot_detection"}


# ===========================================================================
# 1. Core Decision Policy Unit Tests (Cases 1-4 + Uncertainty)
# ===========================================================================

class TestSupervisorDecisionPolicy:

    def test_1_real_human_high_confidence_approves(self):
        supervisor = SupervisorAgent()
        fn_out = make_fake_news_output("real", 0.94)
        bot_out = make_bot_output("human", 0.91)

        res = supervisor.evaluate(fn_out, bot_out)
        assert_supervisor_schema(res)
        assert res["decision"] == "APPROVE"
        assert res["risk_level"] == "LOW"
        assert res["human_review_required"] is False

    def test_2_fake_human_high_confidence_reviews(self):
        supervisor = SupervisorAgent()
        fn_out = make_fake_news_output("fake", 0.95)
        bot_out = make_bot_output("human", 0.92)

        res = supervisor.evaluate(fn_out, bot_out)
        assert_supervisor_schema(res)
        assert res["decision"] == "REVIEW"
        assert res["human_review_required"] is True

    def test_3_real_bot_high_confidence_reviews(self):
        supervisor = SupervisorAgent()
        fn_out = make_fake_news_output("real", 0.93)
        bot_out = make_bot_output("bot", 0.96)

        res = supervisor.evaluate(fn_out, bot_out)
        assert_supervisor_schema(res)
        assert res["decision"] == "REVIEW"
        assert res["human_review_required"] is True

    def test_4_fake_bot_high_confidence_restricts(self):
        supervisor = SupervisorAgent()
        fn_out = make_fake_news_output("fake", 0.96)
        bot_out = make_bot_output("bot", 0.94)

        res = supervisor.evaluate(fn_out, bot_out)
        assert_supervisor_schema(res)
        assert res["decision"] == "RESTRICT"
        assert res["risk_level"] == "HIGH"
        assert res["human_review_required"] is True

    def test_5_fake_low_confidence_routes_to_review(self):
        supervisor = SupervisorAgent()
        # Even with high-confidence bot, low-confidence fake prediction must REVIEW.
        fn_out = make_fake_news_output("fake", 0.54)
        bot_out = make_bot_output("bot", 0.95)

        res = supervisor.evaluate(fn_out, bot_out)
        assert_supervisor_schema(res)
        assert res["decision"] == "REVIEW"
        assert res["human_review_required"] is True

    def test_6_bot_low_confidence_routes_to_review(self):
        supervisor = SupervisorAgent()
        # Even with high-confidence fake, low-confidence bot prediction must REVIEW.
        fn_out = make_fake_news_output("fake", 0.95)
        bot_out = make_bot_output("bot", 0.52)

        res = supervisor.evaluate(fn_out, bot_out)
        assert_supervisor_schema(res)
        assert res["decision"] == "REVIEW"
        assert res["human_review_required"] is True

    @pytest.mark.parametrize("conf", [0.60, 0.72, 0.8499, 0.50])
    def test_uncertain_benign_predictions_never_auto_approve(self, conf):
        supervisor = SupervisorAgent()
        fn_out = make_fake_news_output("real", conf)
        bot_out = make_bot_output("human", 0.95)

        res = supervisor.evaluate(fn_out, bot_out)
        assert res["decision"] == "REVIEW"
        assert res["human_review_required"] is True

    @pytest.mark.parametrize("bot_conf", [0.85, 0.95, 0.9999, 1.0])
    def test_bot_alone_never_triggers_restrict(self, bot_conf):
        supervisor = SupervisorAgent()
        fn_out = make_fake_news_output("real", 0.95)
        bot_out = make_bot_output("bot", bot_conf)

        res = supervisor.evaluate(fn_out, bot_out)
        assert res["decision"] != "RESTRICT"
        assert res["decision"] == "REVIEW"


# ===========================================================================
# 2. Input Validation & Error Handling Tests (Cases 7-9)
# ===========================================================================

class TestSupervisorValidation:

    def test_7_missing_confidence_raises(self):
        supervisor = SupervisorAgent()
        bad_fn = {"agent": "fake_news", "prediction": "real", "scores": {"fake": 0.1, "real": 0.9}}
        bot_out = make_bot_output("human", 0.90)

        with pytest.raises(ValueError, match="Missing 'confidence'"):
            supervisor.evaluate(bad_fn, bot_out)

        fn_out = make_fake_news_output("real", 0.90)
        bad_bot = {"agent": "bot_detection", "prediction": "human", "confidence": None}
        with pytest.raises(ValueError, match="Missing 'confidence'"):
            supervisor.evaluate(fn_out, bad_bot)

    def test_8_invalid_prediction_labels_raise(self):
        supervisor = SupervisorAgent()
        bad_fn = make_fake_news_output("real", 0.90)
        bad_fn["prediction"] = "satire"
        bot_out = make_bot_output("human", 0.90)

        with pytest.raises(ValueError, match="Invalid fake_news prediction label"):
            supervisor.evaluate(bad_fn, bot_out)

        fn_out = make_fake_news_output("real", 0.90)
        bad_bot = make_bot_output("human", 0.90)
        bad_bot["prediction"] = "cyborg"
        with pytest.raises(ValueError, match="Invalid bot_detection prediction label"):
            supervisor.evaluate(fn_out, bad_bot)

    def test_9_missing_agent_outputs_raise(self):
        supervisor = SupervisorAgent()
        fn_out = make_fake_news_output("real", 0.90)
        bot_out = make_bot_output("human", 0.90)

        with pytest.raises(ValueError, match="Missing fake_news agent output"):
            supervisor.evaluate(None, bot_out)

        with pytest.raises(ValueError, match="Missing bot_detection agent output"):
            supervisor.evaluate(fn_out, None)

        with pytest.raises(TypeError, match="must be a dict"):
            supervisor.evaluate("real", bot_out)

    @pytest.mark.parametrize("bad_conf", [-0.1, 1.05, float("nan"), float("inf"), True, "0.9"])
    def test_out_of_range_or_invalid_confidence_raises(self, bad_conf):
        supervisor = SupervisorAgent()
        fn_out = {"prediction": "real", "confidence": bad_conf}
        bot_out = make_bot_output("human", 0.90)
        with pytest.raises(ValueError, match="confidence"):
            supervisor.evaluate(fn_out, bot_out)


# ===========================================================================
# 3. Invariants, Risk Bounds, Preservation & Determinism
# ===========================================================================

class TestSupervisorInvariants:

    def test_inputs_preserved_without_mutation(self):
        supervisor = SupervisorAgent()
        fn_out = make_fake_news_output("fake", 0.92)
        bot_out = make_bot_output("human", 0.88)

        res = supervisor.evaluate(fn_out, bot_out)
        assert res["input_predictions"]["fake_news"] == fn_out
        assert res["input_predictions"]["bot_detection"] == bot_out

        # Mutating returned input_predictions must not affect caller dicts.
        res["input_predictions"]["fake_news"]["prediction"] = "tampered"
        assert fn_out["prediction"] == "fake"

    @pytest.mark.parametrize(
        "fn_pred,fn_conf,bot_pred,bot_conf",
        [
            ("real", 1.0, "human", 1.0),
            ("fake", 1.0, "bot", 1.0),
            ("real", 0.0, "human", 0.0),
            ("fake", 0.5, "human", 0.5),
            ("real", 0.85, "bot", 0.85),
        ],
    )
    def test_risk_score_always_between_0_and_100(self, fn_pred, fn_conf, bot_pred, bot_conf):
        res = evaluate_threat(
            make_fake_news_output(fn_pred, fn_conf),
            make_bot_output(bot_pred, bot_conf),
        )
        assert 0.0 <= res["risk_score"] <= 100.0

    def test_deterministic_outputs_for_identical_inputs(self):
        supervisor = SupervisorAgent()
        fn_out = make_fake_news_output("fake", 0.87)
        bot_out = make_bot_output("bot", 0.91)

        first = supervisor.evaluate(fn_out, bot_out)
        second = supervisor.evaluate(fn_out, bot_out)
        assert first == second

    def test_configurable_thresholds_and_policy(self):
        assert HIGH_CONFIDENCE == 0.85
        assert MEDIUM_CONFIDENCE == 0.60

        strict_policy = SupervisorPolicy(high_confidence=0.95, medium_confidence=0.70)
        supervisor = SupervisorAgent(policy=strict_policy)
        # 0.90 is HIGH under default policy, but MEDIUM under strict_policy -> REVIEW.
        res = supervisor.evaluate(
            make_fake_news_output("real", 0.90),
            make_bot_output("human", 0.90),
        )
        assert res["decision"] == "REVIEW"
        assert res["human_review_required"] is True

    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError):
            SupervisorPolicy(high_confidence=0.50, medium_confidence=0.60)
        with pytest.raises(ValueError):
            SupervisorPolicy(fake_news_weight=-1.0)


# ===========================================================================
# 4. Human Review Workflow Tests (Case 10)
# ===========================================================================

class TestHumanReviewWorkflow:

    def test_10_human_review_lifecycle_preserves_ai_decision(self):
        supervisor = SupervisorAgent()
        queue = ReviewQueue()

        # Approve case is NOT enqueued.
        approve_res = supervisor.evaluate(
            make_fake_news_output("real", 0.95),
            make_bot_output("human", 0.95),
        )
        assert queue.submit(approve_res) is None
        assert len(queue.get_pending()) == 0

        # Review case IS enqueued.
        review_res = supervisor.evaluate(
            make_fake_news_output("fake", 0.91),
            make_bot_output("human", 0.89),
        )
        case = queue.submit(review_res, case_id="case-001")
        assert case is not None
        assert case.status == "PENDING"
        assert case.ai_decision == "REVIEW"
        assert case.human_decision is None
        assert len(case.review_reasons) >= 3
        assert len(queue.get_pending()) == 1

        # Authorized reviewer overrides to REJECT while preserving AI decision.
        resolved = queue.resolve(
            "case-001",
            human_decision="REJECT",
            reviewer_id="mod_alice",
            notes="Verified fabricated claim.",
        )
        assert resolved.status == "RESOLVED"
        assert resolved.ai_decision == "REVIEW"
        assert resolved.human_decision == "REJECT"
        assert resolved.reviewer_id == "mod_alice"
        assert resolved.reviewed_at is not None
        assert len(queue.get_pending()) == 0
        assert len(queue.get_resolved()) == 1

    @pytest.mark.parametrize("human_decision", ["APPROVE", "REJECT", "RESTRICT"])
    def test_valid_human_dispositions(self, human_decision):
        supervisor = SupervisorAgent()
        queue = ReviewQueue()
        res = supervisor.evaluate(
            make_fake_news_output("fake", 0.90),
            make_bot_output("bot", 0.90),
        )
        case = queue.submit(res)
        resolved = queue.resolve(case.case_id, human_decision=human_decision, reviewer_id="mod_bob")
        assert resolved.ai_decision == "RESTRICT"
        assert resolved.human_decision == human_decision

    def test_review_validation_errors(self):
        supervisor = SupervisorAgent()
        queue = ReviewQueue()
        res = supervisor.evaluate(
            make_fake_news_output("fake", 0.90),
            make_bot_output("human", 0.90),
        )
        case = queue.submit(res, case_id="case-err")

        with pytest.raises(ValueError, match="Invalid human_decision"):
            queue.resolve("case-err", human_decision="BAN_FOREVER", reviewer_id="mod_1")

        with pytest.raises(ValueError, match="reviewer_id"):
            queue.resolve("case-err", human_decision="APPROVE", reviewer_id="   ")

        with pytest.raises(KeyError):
            queue.resolve("missing-case", human_decision="APPROVE", reviewer_id="mod_1")


# ===========================================================================
# 5. Live Integration Tests with Real Agent 1 & Agent 2
# ===========================================================================

class TestSupervisorLiveIntegration:

    @pytest.fixture(scope="class")
    @classmethod
    def supervisor(cls):
        return SupervisorAgent()

    def test_live_end_to_end_real_and_human_sample(self, supervisor):
        text = (
            "The central bank kept interest rates unchanged on Thursday while "
            "citing steady employment growth and moderating inflation."
        )
        human_account = {
            "followers_count": 640,
            "friends_count": 410,
            "statuses_count": 1850,
            "favourites_count": 1200,
            "average_tweets_per_day": 1.8,
            "account_age_days": 2400,
            "default_profile": False,
            "default_profile_image": False,
            "geo_enabled": True,
            "verified": False,
            "created_at": "2018-05-14 14:22:10",
            "description": "Data journalist covering climate and urban transit.",
            "screen_name": "claire_transit",
        }
        res = supervisor.analyze(text, human_account)
        assert_supervisor_schema(res)
        assert res["input_predictions"]["fake_news"]["prediction"] == "real"
        assert res["input_predictions"]["bot_detection"]["prediction"] == "human"

    def test_live_end_to_end_fake_and_bot_sample(self, supervisor):
        text = (
            "BREAKING: Secret government microchip inside tap water exposed!! "
            "Share before they delete this! Mainstream media refuses to report!"
        )
        bot_account = {
            "followers_count": 15,
            "friends_count": 4800,
            "statuses_count": 45000,
            "favourites_count": 1,
            "average_tweets_per_day": 180.0,
            "account_age_days": 250,
            "default_profile": True,
            "default_profile_image": True,
            "geo_enabled": False,
            "verified": False,
            "created_at": "2023-11-02 03:15:00",
            "description": "",
            "screen_name": "news_bot_9948271",
        }
        res = supervisor.analyze(text, bot_account)
        assert_supervisor_schema(res)
        assert res["input_predictions"]["fake_news"]["prediction"] == "fake"
        assert res["input_predictions"]["bot_detection"]["prediction"] == "bot"
        assert res["decision"] == "RESTRICT"
        assert res["human_review_required"] is True

        queue = ReviewQueue()
        case = queue.submit(res)
        assert case is not None
        assert case.ai_decision == "RESTRICT"
