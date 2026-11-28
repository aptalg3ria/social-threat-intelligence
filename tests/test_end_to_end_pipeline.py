"""
tests/test_end_to_end_pipeline.py
---------------------------------
End-to-End Integration Tests connecting:
  - Agent 1: FakeNewsDetector (frozen Naive Bayes + TF-IDF)
  - Agent 2: BotDetector (frozen PyTorch GRU + StandardScaler + 15 features)
  - Agent 3: MetaLogisticRegression + MetaSupervisorAgent
             (synthetic-trained artifact: models/saved/meta_logreg_synthetic.joblib)
  - Human Review Queue (ReviewQueue)

Covers all 10 required integration scenarios (live frozen-model smoke runs,
deterministic decision-policy integration tests, input validation for missing
metadata / empty text, and Human Review Queue lifecycle checks).

NOTE: Agent 3 uses the SYNTHETIC-trained proof-of-concept artifact. Passing
integration tests verify software orchestration and synthetic policy behavior,
not real-world human moderation accuracy.

Run with:
  pytest tests/test_end_to_end_pipeline.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.data.preprocessing import ACCOUNT_FEATURES
from src.detection.bot_detector import BotDetector
from src.detection.fake_news_detector import FakeNewsDetector
from src.models.logistic_regression import META_FEATURE_COLUMNS, MetaLogisticRegression
from src.moderation import (
    DEFAULT_SYNTHETIC_META_MODEL_PATH,
    MetaSupervisorAgent,
    ReviewQueue,
    SocialThreatPipeline,
    SupervisorAgent,
)
from src.utils.helpers import resolve_project_path


EXPECTED_AGENT2_FEATURES = [
    "log_followers",
    "log_friends",
    "log_statuses",
    "log_favourites",
    "log_tweets_per_day",
    "account_age_days",
    "default_profile",
    "default_profile_image",
    "geo_enabled",
    "verified",
    "hour_of_day",
    "day_of_week",
    "year",
    "desc_len",
    "digits_in_name",
]

REAL_NEWS_STRONG = (
    "WASHINGTON (Reuters) - The U.S. Federal Reserve held benchmark interest rates "
    "steady on Wednesday following its two-day policy meeting, citing steady labor "
    "market conditions and moderating inflation data in the quarterly economic report. "
    "Treasury officials confirmed that bipartisan lawmakers will review the annual "
    "budget proposal before the fiscal deadline next month."
)

FAKE_NEWS_STRONG = (
    "BREAKING: Secret government microchip inside tap water exposed!! "
    "Share before they delete this! Mainstream media refuses to report "
    "this Big Pharma conspiracy cure they are hiding from the public!"
)

HUMAN_ACCOUNT_STRONG = {
    "followers_count": 1250,
    "friends_count": 480,
    "statuses_count": 3400,
    "favourites_count": 2100,
    "average_tweets_per_day": 1.4,
    "account_age_days": 2400,
    "default_profile": False,
    "default_profile_image": False,
    "geo_enabled": True,
    "verified": True,
    "created_at": "2018-04-10 14:20:00",
    "description": "Senior economics correspondent and policy analyst covering central banks.",
    "screen_name": "econ_correspondent",
}

BOT_ACCOUNT_STRONG = {
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


class _StubFakeNewsAgent:
    def __init__(self, output: dict):
        self._output = output

    def predict(self, text: str) -> dict:
        return dict(self._output)


class _StubBotAgent:
    def __init__(self, output: dict):
        self._output = output

    def predict(self, account_data: dict) -> dict:
        return dict(self._output)


def _fn_output(p_fake: float) -> dict:
    pred = "fake" if p_fake >= 0.5 else "real"
    conf = p_fake if pred == "fake" else (1.0 - p_fake)
    return {
        "agent": "fake_news",
        "prediction": pred,
        "confidence": round(conf, 4),
        "scores": {"fake": round(p_fake, 4), "real": round(1.0 - p_fake, 4)},
        "raw_class": 0 if pred == "fake" else 1,
    }


def _bot_output(p_bot: float) -> dict:
    pred = "bot" if p_bot >= 0.5 else "human"
    conf = p_bot if pred == "bot" else (1.0 - p_bot)
    return {
        "agent": "bot_detection",
        "prediction": pred,
        "confidence": round(conf, 4),
        "scores": {"human": round(1.0 - p_bot, 4), "bot": round(p_bot, 4)},
        "raw_class": 1 if pred == "bot" else 0,
        "bot_score": round(p_bot, 4),
        "bot_class": 1 if pred == "bot" else 0,
        "bot_label": "BOT" if pred == "bot" else "HUMAN",
    }


@pytest.fixture(scope="module")
def live_pipeline() -> SocialThreatPipeline:
    return SocialThreatPipeline(meta_model_path=DEFAULT_SYNTHETIC_META_MODEL_PATH)


# ===========================================================================
# 1. Live Model Smoke Tests (Real Saved Artifacts for Agents 1, 2, and 3)
# ===========================================================================

class TestLiveAgentsAndPipelineSmoke:

    def test_agent1_loads_and_predicts_real_and_fake(self, live_pipeline):
        fn = live_pipeline.fake_news_detector
        real_res = fn.predict(REAL_NEWS_STRONG)
        fake_res = fn.predict(FAKE_NEWS_STRONG)

        assert real_res["agent"] == "fake_news"
        assert real_res["prediction"] == "real"
        assert 0.5 < real_res["confidence"] <= 1.0
        assert abs(real_res["scores"]["fake"] + real_res["scores"]["real"] - 1.0) < 1e-4

        assert fake_res["prediction"] == "fake"
        assert 0.5 < fake_res["confidence"] <= 1.0
        assert fake_res["scores"]["fake"] > 0.85

        # Empty text handled without crash by Agent 1 directly; invalid non-str raises TypeError
        empty_res = fn.predict("")
        assert empty_res["prediction"] in ("fake", "real")
        with pytest.raises(TypeError, match="must be a str"):
            fn.predict(None)

    def test_agent2_loads_15_features_in_exact_order_and_predicts(self, live_pipeline):
        bd = live_pipeline.bot_detector
        assert ACCOUNT_FEATURES == EXPECTED_AGENT2_FEATURES
        assert bd.features == EXPECTED_AGENT2_FEATURES
        assert bd.scaler.n_features_in_ == 15

        human_res = bd.predict(HUMAN_ACCOUNT_STRONG)
        bot_res = bd.predict(BOT_ACCOUNT_STRONG)

        assert human_res["agent"] == "bot_detection"
        assert human_res["prediction"] == "human"
        assert 0.5 < human_res["confidence"] <= 1.0
        assert abs(human_res["scores"]["human"] + human_res["scores"]["bot"] - 1.0) < 1e-4

        assert bot_res["prediction"] == "bot"
        assert 0.5 < bot_res["confidence"] <= 1.0
        assert bot_res["scores"]["bot"] > 0.85

    def test_agent3_loads_synthetic_artifact(self, live_pipeline):
        assert os.path.exists(resolve_project_path(DEFAULT_SYNTHETIC_META_MODEL_PATH))
        meta_model = live_pipeline.meta_supervisor.meta_classifier
        assert isinstance(meta_model, MetaLogisticRegression)
        assert meta_model.is_fitted_ is True
        assert meta_model.feature_names == META_FEATURE_COLUMNS

    def test_live_end_to_end_pipeline_four_core_combinations(self, live_pipeline):
        # Reset queue for clean isolated assertions
        live_pipeline.review_queue = ReviewQueue()

        # 1. Live Human + Real News
        res_hr = live_pipeline.run(REAL_NEWS_STRONG, HUMAN_ACCOUNT_STRONG, case_id="live-hr")
        assert set(res_hr.keys()) >= {"fake_news", "bot_detection", "meta_features", "moderation", "review_queue"}
        assert res_hr["fake_news"]["prediction"] == "real"
        assert res_hr["bot_detection"]["prediction"] == "human"
        assert tuple(res_hr["meta_features"].keys()) == META_FEATURE_COLUMNS
        assert res_hr["moderation"]["decision"] in ("APPROVE", "REVIEW")
        assert res_hr["moderation"]["model_origin"] == "synthetic_bootstrap"

        # 2. Live Bot + Real News -> must route to REVIEW (never RESTRICT)
        res_br = live_pipeline.run(REAL_NEWS_STRONG, BOT_ACCOUNT_STRONG, case_id="live-br")
        assert res_br["fake_news"]["prediction"] == "real"
        assert res_br["bot_detection"]["prediction"] == "bot"
        assert res_br["moderation"]["decision"] == "REVIEW"
        assert res_br["moderation"]["human_review_required"] is True
        assert res_br["review_queue"]["queued"] is True

        # 3. Live Human + Fake News -> must route to REVIEW (never auto-RESTRICT)
        res_hf = live_pipeline.run(FAKE_NEWS_STRONG, HUMAN_ACCOUNT_STRONG, case_id="live-hf")
        assert res_hf["fake_news"]["prediction"] == "fake"
        assert res_hf["bot_detection"]["prediction"] == "human"
        assert res_hf["moderation"]["decision"] == "REVIEW"
        assert res_hf["moderation"]["human_review_required"] is True
        assert res_hf["review_queue"]["queued"] is True

        # 4. Live Bot + Fake News -> RESTRICT (when both high conf) or REVIEW if guarded
        res_bf = live_pipeline.run(FAKE_NEWS_STRONG, BOT_ACCOUNT_STRONG, case_id="live-bf")
        assert res_bf["fake_news"]["prediction"] == "fake"
        assert res_bf["bot_detection"]["prediction"] == "bot"
        assert res_bf["moderation"]["decision"] == "RESTRICT"
        assert res_bf["moderation"]["human_review_required"] is True
        assert res_bf["review_queue"]["queued"] is True

        # Rule-Based Supervisor baseline remains intact and produces same safe decision
        baseline = SupervisorAgent().analyze(FAKE_NEWS_STRONG, BOT_ACCOUNT_STRONG)
        assert baseline["decision"] == "RESTRICT"


# ===========================================================================
# 2. Deterministic Integration Scenarios (1-10) + Human Review Queue
# ===========================================================================

class TestIntegrationScenariosAndReviewQueue:

    @staticmethod
    def _build_stubbed_pipeline(p_fake: float, p_bot: float) -> SocialThreatPipeline:
        return SocialThreatPipeline(
            fake_news_detector=_StubFakeNewsAgent(_fn_output(p_fake)),
            bot_detector=_StubBotAgent(_bot_output(p_bot)),
            meta_model_path=DEFAULT_SYNTHETIC_META_MODEL_PATH,
        )

    def test_scenario_1_human_real_news_approves(self):
        pipe = self._build_stubbed_pipeline(p_fake=0.05, p_bot=0.06)
        res = pipe.run(REAL_NEWS_STRONG, HUMAN_ACCOUNT_STRONG, case_id="sc-1")
        assert res["moderation"]["decision"] == "APPROVE"
        assert res["moderation"]["human_review_required"] is False
        assert res["review_queue"]["queued"] is False
        assert pipe.review_queue.get_pending() == []

    def test_scenario_2_bot_real_news_routes_to_review(self):
        pipe = self._build_stubbed_pipeline(p_fake=0.07, p_bot=0.93)
        res = pipe.run(REAL_NEWS_STRONG, BOT_ACCOUNT_STRONG, case_id="sc-2")
        assert res["moderation"]["decision"] == "REVIEW"
        assert res["moderation"]["human_review_required"] is True
        assert res["review_queue"]["queued"] is True

    def test_scenario_3_human_fake_news_routes_to_review(self):
        pipe = self._build_stubbed_pipeline(p_fake=0.94, p_bot=0.08)
        res = pipe.run(FAKE_NEWS_STRONG, HUMAN_ACCOUNT_STRONG, case_id="sc-3")
        assert res["moderation"]["decision"] == "REVIEW"
        assert res["moderation"]["human_review_required"] is True
        assert res["review_queue"]["queued"] is True

    def test_scenario_4_bot_fake_news_restricts_and_queues(self):
        pipe = self._build_stubbed_pipeline(p_fake=0.95, p_bot=0.94)
        res = pipe.run(FAKE_NEWS_STRONG, BOT_ACCOUNT_STRONG, case_id="sc-4")
        assert res["moderation"]["decision"] == "RESTRICT"
        assert res["moderation"]["human_review_required"] is True
        assert res["review_queue"]["queued"] is True

    def test_scenario_5_low_confidence_agent1_escalates_to_review(self):
        pipe = self._build_stubbed_pipeline(p_fake=0.42, p_bot=0.06)
        res = pipe.run(REAL_NEWS_STRONG, HUMAN_ACCOUNT_STRONG, case_id="sc-5")
        assert res["fake_news"]["confidence"] == pytest.approx(0.58, abs=1e-4)
        assert res["meta_features"]["min_agent_confidence"] == pytest.approx(0.58, abs=1e-4)
        assert res["moderation"]["decision"] == "REVIEW"
        assert res["moderation"]["human_review_required"] is True
        assert res["review_queue"]["queued"] is True

    def test_scenario_6_low_confidence_agent2_escalates_to_review(self):
        pipe = self._build_stubbed_pipeline(p_fake=0.06, p_bot=0.45)
        res = pipe.run(REAL_NEWS_STRONG, HUMAN_ACCOUNT_STRONG, case_id="sc-6")
        assert res["bot_detection"]["confidence"] == pytest.approx(0.55, abs=1e-4)
        assert res["meta_features"]["min_agent_confidence"] == pytest.approx(0.55, abs=1e-4)
        assert res["moderation"]["decision"] == "REVIEW"
        assert res["moderation"]["human_review_required"] is True
        assert res["review_queue"]["queued"] is True

    def test_scenario_7_conflicting_signals_never_approve_or_restrict(self):
        # Moderate conflicting signals (0.22 fake vs 0.78 bot) and (0.82 fake vs 0.18 bot)
        for idx, (pf, pb) in enumerate([(0.22, 0.78), (0.82, 0.18)], start=1):
            pipe = self._build_stubbed_pipeline(p_fake=pf, p_bot=pb)
            res = pipe.run(REAL_NEWS_STRONG, HUMAN_ACCOUNT_STRONG, case_id=f"sc-7-{idx}")
            assert res["moderation"]["decision"] == "REVIEW"
            assert res["moderation"]["human_review_required"] is True

    def test_scenario_8_missing_or_invalid_account_metadata_rejected(self, live_pipeline):
        # Empty dict
        with pytest.raises(ValueError, match="cannot be empty"):
            live_pipeline.run(REAL_NEWS_STRONG, {})

        # Missing required BotDetector fields (refuse silent zero-imputation)
        incomplete = {"followers_count": 100, "friends_count": 50, "statuses_count": 200}
        with pytest.raises(ValueError, match="Missing required account metadata fields"):
            live_pipeline.run(REAL_NEWS_STRONG, incomplete)

        # Invalid timestamp
        bad_time = dict(HUMAN_ACCOUNT_STRONG, created_at="invalid-timestamp")
        with pytest.raises(ValueError, match="not a valid timestamp"):
            live_pipeline.run(REAL_NEWS_STRONG, bad_time)

        # Non-dict
        with pytest.raises(TypeError, match="must be a dict"):
            live_pipeline.run(REAL_NEWS_STRONG, ["not", "a", "dict"])

    def test_scenario_9_empty_or_invalid_post_text_rejected(self, live_pipeline):
        with pytest.raises(ValueError, match="cannot be empty"):
            live_pipeline.run("", HUMAN_ACCOUNT_STRONG)

        with pytest.raises(ValueError, match="cannot be empty"):
            live_pipeline.run("   \n\t  ", HUMAN_ACCOUNT_STRONG)

        with pytest.raises(TypeError, match="must be a str"):
            live_pipeline.run(None, HUMAN_ACCOUNT_STRONG)

        with pytest.raises(TypeError, match="must be a str"):
            live_pipeline.run(12345, HUMAN_ACCOUNT_STRONG)

    def test_scenario_10_review_queue_context_and_human_resolution(self):
        pipe = self._build_stubbed_pipeline(p_fake=0.91, p_bot=0.10)
        res = pipe.run(FAKE_NEWS_STRONG, HUMAN_ACCOUNT_STRONG, case_id="hrq-case-101")

        assert res["moderation"]["decision"] == "REVIEW"
        assert res["review_queue"]["queued"] is True
        assert res["review_queue"]["status"] == "PENDING"

        pending = pipe.review_queue.get_pending()
        assert len(pending) == 1
        case = pending[0]
        assert case.case_id == "hrq-case-101"
        assert case.ai_decision == "REVIEW"
        assert case.status == "PENDING"
        assert len(case.review_reasons) > 0
        # Preserves both Agent 1 & Agent 2 outputs intact
        assert case.input_predictions["fake_news"]["prediction"] == "fake"
        assert case.input_predictions["bot_detection"]["prediction"] == "human"

        # Verify ReviewQueue does NOT auto-convert REVIEW; reviewer resolves explicitly
        resolved = pipe.review_queue.resolve(
            case_id="hrq-case-101",
            human_decision="REJECT",
            reviewer_id="mod_alice",
            notes="Confirmed misleading claim during manual verification.",
        )
        assert resolved.status == "RESOLVED"
        assert resolved.human_decision == "REJECT"
        assert resolved.ai_decision == "REVIEW"
        assert pipe.review_queue.get_pending() == []
