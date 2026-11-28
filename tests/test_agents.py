"""
tests/test_agents.py
--------------------
Phase 1 validation tests for Agent 1 (Fake News) and Agent 2 (Bot Detection).

Covers:
  - Model and artefact loading
  - Preprocessing pipeline
  - Inference on known inputs
  - Canonical output schema validation
  - Batch prediction
  - Error handling

Run with:
  pytest tests/test_agents.py -v
"""

import sys
import os
import warnings

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
import joblib

from src.utils.helpers import resolve_project_path

# ---------------------------------------------------------------------------
# Shared schema validator
# ---------------------------------------------------------------------------

CANONICAL_KEYS = {"agent", "prediction", "confidence", "scores", "raw_class"}


def assert_canonical_schema(result: dict, expected_agent: str,
                             expected_predictions: tuple,
                             expected_score_keys: tuple):
    """Raise AssertionError if result dict does not conform to the shared schema."""
    assert isinstance(result, dict), "Result must be a dict"
    missing = CANONICAL_KEYS - result.keys()
    assert not missing, f"Missing canonical keys: {missing}"

    assert result["agent"] == expected_agent, (
        f"Expected agent='{expected_agent}', got '{result['agent']}'"
    )
    assert result["prediction"] in expected_predictions, (
        f"prediction must be one of {expected_predictions}, got '{result['prediction']}'"
    )
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"confidence must be in [0,1], got {result['confidence']}"
    )
    assert isinstance(result["scores"], dict), "scores must be a dict"
    for key in expected_score_keys:
        assert key in result["scores"], f"scores missing key '{key}'"
        assert 0.0 <= result["scores"][key] <= 1.0, (
            f"scores['{key}'] must be in [0,1], got {result['scores'][key]}"
        )
    assert result["raw_class"] in (0, 1), (
        f"raw_class must be 0 or 1, got {result['raw_class']}"
    )
    # Confidence must equal the score of the predicted class
    if result["prediction"] in ("fake",):
        assert abs(result["confidence"] - result["scores"]["fake"]) < 1e-4
    elif result["prediction"] in ("real",):
        assert abs(result["confidence"] - result["scores"]["real"]) < 1e-4
    elif result["prediction"] == "bot":
        assert abs(result["confidence"] - result["scores"]["bot"]) < 1e-4
    elif result["prediction"] == "human":
        assert abs(result["confidence"] - result["scores"]["human"]) < 1e-4


# ===========================================================================
# AGENT 1 — Fake News Detection
# ===========================================================================

class TestFakeNewsArtifactLoading:
    """Verify that saved model artefacts exist and load correctly."""

    def test_vectorizer_file_exists(self):
        assert os.path.exists(resolve_project_path("models/vectorizers/tfidf_vectorizer.joblib")), (
            "TF-IDF vectorizer file is missing."
        )

    def test_model_file_exists(self):
        assert os.path.exists(resolve_project_path("models/saved/fake_news_model.joblib")), (
            "Fake news model file is missing."
        )

    def test_vectorizer_loads(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vect = joblib.load(resolve_project_path("models/vectorizers/tfidf_vectorizer.joblib"))
        assert hasattr(vect, "transform"), "Vectorizer must have transform()"
        assert hasattr(vect, "vocabulary_"), "Vectorizer must have vocabulary_"
        assert len(vect.vocabulary_) > 0, "Vocabulary must not be empty"

    def test_vectorizer_parameters(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vect = joblib.load(resolve_project_path("models/vectorizers/tfidf_vectorizer.joblib"))
        assert vect.max_features == 5000
        assert vect.ngram_range == (1, 2)

    def test_model_loads(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(resolve_project_path("models/saved/fake_news_model.joblib"))
        assert hasattr(model, "predict"), "Model must have predict()"
        assert hasattr(model, "predict_proba"), "Model must have predict_proba()"
        assert hasattr(model, "classes_"), "Model must have classes_"

    def test_model_classes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(resolve_project_path("models/saved/fake_news_model.joblib"))
        assert list(model.classes_) == [0, 1], (
            "Classes must be [0, 1] (0=fake, 1=real)"
        )


class TestFakeNewsDetectorInit:
    """Test FakeNewsDetector instantiation and error handling."""

    def test_default_instantiation(self):
        from src.detection.fake_news_detector import FakeNewsDetector
        det = FakeNewsDetector()
        assert det.vectorizer is not None
        assert det.model is not None

    def test_missing_model_raises(self, tmp_path):
        from src.detection.fake_news_detector import FakeNewsDetector
        with pytest.raises(FileNotFoundError, match="Fake news model not found"):
            FakeNewsDetector(
                model_path=str(tmp_path / "nonexistent_model.joblib"),
                vectorizer_path="models/vectorizers/tfidf_vectorizer.joblib",
            )

    def test_missing_vectorizer_raises(self, tmp_path):
        from src.detection.fake_news_detector import FakeNewsDetector
        with pytest.raises(FileNotFoundError, match="TF-IDF vectorizer not found"):
            FakeNewsDetector(
                model_path="models/saved/fake_news_model.joblib",
                vectorizer_path=str(tmp_path / "nonexistent_vect.joblib"),
            )

    def test_non_string_input_raises(self):
        from src.detection.fake_news_detector import FakeNewsDetector
        det = FakeNewsDetector()
        with pytest.raises(TypeError, match="must be a str"):
            det.predict(12345)


class TestFakeNewsPredictions:
    """Verify inference correctness and output schema on known texts."""

    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls):
        from src.detection.fake_news_detector import FakeNewsDetector
        return FakeNewsDetector()

    # --- Real news ---
    REAL_NEWS = (
        "The government announced a new economic policy "
        "to improve employment and support small businesses. "
        "The minister confirmed the plan during a press conference."
    )

    # --- Fake / sensational news ---
    FAKE_NEWS = (
        "BREAKING: Scientists discover drinking bleach cures all diseases! "
        "Share before it gets deleted! The mainstream media won't tell you "
        "this secret cure that Big Pharma is hiding from the public!"
    )

    def test_real_news_prediction(self, detector):
        result = detector.predict(self.REAL_NEWS)
        assert result["prediction"] == "real", (
            f"Expected 'real', got '{result['prediction']}'. Full result: {result}"
        )

    def test_fake_news_prediction(self, detector):
        result = detector.predict(self.FAKE_NEWS)
        assert result["prediction"] == "fake", (
            f"Expected 'fake', got '{result['prediction']}'. Full result: {result}"
        )

    def test_real_news_schema(self, detector):
        result = detector.predict(self.REAL_NEWS)
        assert_canonical_schema(
            result,
            expected_agent="fake_news",
            expected_predictions=("fake", "real"),
            expected_score_keys=("fake", "real"),
        )

    def test_fake_news_schema(self, detector):
        result = detector.predict(self.FAKE_NEWS)
        assert_canonical_schema(
            result,
            expected_agent="fake_news",
            expected_predictions=("fake", "real"),
            expected_score_keys=("fake", "real"),
        )

    def test_scores_sum_to_one(self, detector):
        result = detector.predict(self.REAL_NEWS)
        total = result["scores"]["fake"] + result["scores"]["real"]
        assert abs(total - 1.0) < 1e-4, f"Scores must sum to 1.0, got {total}"

    def test_real_news_high_real_confidence(self, detector):
        result = detector.predict(self.REAL_NEWS)
        assert result["scores"]["real"] > 0.5, (
            f"Real news should score >0.5 for real, got {result['scores']['real']}"
        )

    def test_fake_news_high_fake_confidence(self, detector):
        result = detector.predict(self.FAKE_NEWS)
        assert result["scores"]["fake"] > 0.5, (
            f"Fake news should score >0.5 for fake, got {result['scores']['fake']}"
        )

    def test_url_containing_text(self, detector):
        """URLs should not crash the detector or skew results."""
        text = "Visit https://example.com/news for the latest update from @reporter."
        result = detector.predict(text)
        assert_canonical_schema(
            result,
            expected_agent="fake_news",
            expected_predictions=("fake", "real"),
            expected_score_keys=("fake", "real"),
        )

    def test_empty_string(self, detector):
        """Empty string should return a valid result dict (NB handles sparse input)."""
        result = detector.predict("")
        assert_canonical_schema(
            result,
            expected_agent="fake_news",
            expected_predictions=("fake", "real"),
            expected_score_keys=("fake", "real"),
        )


class TestFakeNewsBatchPrediction:
    """Verify batch inference returns consistent results."""

    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls):
        from src.detection.fake_news_detector import FakeNewsDetector
        return FakeNewsDetector()

    def test_batch_returns_list(self, detector):
        texts = [
            "New trade deal signed between two nations.",
            "SHOCKING truth they don't want you to know!",
        ]
        results = detector.predict_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_schema_each_item(self, detector):
        texts = [
            "New trade deal signed between two nations.",
            "SHOCKING truth they don't want you to know!",
        ]
        for result in detector.predict_batch(texts):
            assert_canonical_schema(
                result,
                expected_agent="fake_news",
                expected_predictions=("fake", "real"),
                expected_score_keys=("fake", "real"),
            )

    def test_empty_batch(self, detector):
        assert detector.predict_batch([]) == []

    def test_batch_rejects_single_string(self, detector):
        with pytest.raises(TypeError, match="expects a sequence of strings"):
            detector.predict_batch("single string instead of list")

    def test_batch_matches_single(self, detector):
        """Each batch result should match the corresponding single call."""
        texts = [
            "The senate passed the new infrastructure bill today.",
            "ALIENS are real and government is hiding it! Wake up!",
        ]
        batch_results = detector.predict_batch(texts)
        for text, batch_res in zip(texts, batch_results):
            single_res = detector.predict(text)
            assert batch_res["prediction"] == single_res["prediction"]
            assert abs(batch_res["confidence"] - single_res["confidence"]) < 1e-4


class TestFakeNewsPreprocessing:
    """Verify clean_text() integration is consistent and safe."""

    def test_clean_text_removes_urls(self):
        from src.data.preprocessing import clean_text
        assert "http" not in clean_text("Visit https://t.co/xyz now!")

    def test_clean_text_removes_mentions(self):
        from src.data.preprocessing import clean_text
        assert "@" not in clean_text("Hey @elonmusk check this!")

    def test_clean_text_lowercases(self):
        from src.data.preprocessing import clean_text
        result = clean_text("BREAKING NEWS")
        assert result == result.lower()

    def test_preprocess_does_not_alter_model_output(self):
        """
        Applying clean_text() to plain text (no URLs/mentions) must not
        change the prediction result — confirming backwards compatibility.
        """
        from src.detection.fake_news_detector import FakeNewsDetector
        from src.data.preprocessing import clean_text
        import joblib, warnings

        det = FakeNewsDetector()
        plain_text = "the government announced a new economic policy"

        result_via_detector = det.predict(plain_text)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vect  = joblib.load(resolve_project_path("models/vectorizers/tfidf_vectorizer.joblib"))
            model = joblib.load(resolve_project_path("models/saved/fake_news_model.joblib"))

        # Direct call without clean_text (what the old code did)
        X_raw     = vect.transform([plain_text])
        pred_raw  = int(model.predict(X_raw)[0])

        # Call with clean_text (what new code does)
        X_clean   = vect.transform([clean_text(plain_text)])
        pred_clean = int(model.predict(X_clean)[0])

        # For plain text, both must agree
        assert pred_raw == pred_clean, (
            "clean_text() changes prediction on plain text — "
            "training/inference mismatch detected!"
        )
        assert result_via_detector["raw_class"] == pred_clean


# ===========================================================================
# AGENT 2 — Bot Detection
# ===========================================================================

# Shared sample accounts for testing
HUMAN_SAMPLE = {
    "followers_count": 500,
    "friends_count": 400,
    "statuses_count": 1200,
    "favourites_count": 800,
    "verified": False,
    "default_profile": False,
    "default_profile_image": False,
    "geo_enabled": True,
    "average_tweets_per_day": 2.5,
    "account_age_days": 1200,
    "description": "Software engineer and open source contributor.",
    "screen_name": "developer_john",
    "created_at": "2020-01-15 09:30:00",
}

BOT_SAMPLE = {
    "followers_count": 50000,
    "friends_count": 50100,
    "statuses_count": 100000,
    "favourites_count": 0,
    "verified": False,
    "default_profile": True,
    "default_profile_image": True,
    "geo_enabled": False,
    "average_tweets_per_day": 500,
    "account_age_days": 10,
    "description": "",
    "screen_name": "user12345678",
    "created_at": "2023-12-01 03:00:00",
}


def torch_available():
    """Return True if torch can be imported."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


skip_if_no_torch = pytest.mark.skipif(
    not torch_available(),
    reason="torch not installed — install with: pip install torch",
)


class TestBotDetectorArtifacts:
    """Verify that all 3 Bot Detection artefacts exist and load."""

    def test_model_file_exists(self):
        assert os.path.exists(resolve_project_path("models/saved/bot_rnn_model.pth")), (
            "Bot RNN model weights missing."
        )

    def test_scaler_file_exists(self):
        assert os.path.exists(resolve_project_path("models/vectorizers/bot_scaler.joblib")), (
            "Bot scaler missing."
        )

    def test_meta_file_exists(self):
        assert os.path.exists(resolve_project_path("models/vectorizers/bot_input_meta.joblib")), (
            "Bot input meta missing."
        )

    def test_scaler_loads(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = joblib.load(resolve_project_path("models/vectorizers/bot_scaler.joblib"))
        assert hasattr(scaler, "transform")
        assert scaler.n_features_in_ == 15, (
            f"Scaler expects 15 features, found {scaler.n_features_in_}"
        )

    def test_meta_loads_and_matches_source(self):
        from src.data.preprocessing import ACCOUNT_FEATURES
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta = joblib.load(resolve_project_path("models/vectorizers/bot_input_meta.joblib"))
        assert meta["input_size"] == 3
        assert meta["seq_len"] == 5
        assert meta["features"] == ACCOUNT_FEATURES, (
            "Saved meta features do not match ACCOUNT_FEATURES in source."
        )


class TestBotDetectorInit:
    """Test BotDetector instantiation and error handling."""

    def test_missing_torch_raises_clear_error(self, monkeypatch):
        import src.detection.bot_detector as bd
        monkeypatch.setattr(bd, "torch", None)
        with pytest.raises(ImportError, match="PyTorch"):
            bd.BotDetector()

    @skip_if_no_torch
    def test_default_instantiation(self):
        from src.detection.bot_detector import BotDetector
        det = BotDetector()
        assert det.model is not None
        assert det.scaler is not None

    @skip_if_no_torch
    def test_missing_model_raises(self, tmp_path):
        from src.detection.bot_detector import BotDetector
        with pytest.raises(FileNotFoundError, match="Bot RNN model not found"):
            BotDetector(model_path=str(tmp_path / "no_model.pth"))

    @skip_if_no_torch
    def test_missing_scaler_raises(self, tmp_path):
        from src.detection.bot_detector import BotDetector
        with pytest.raises(FileNotFoundError, match="Bot feature scaler not found"):
            BotDetector(scaler_path=str(tmp_path / "no_scaler.joblib"))


class TestBotDetectorPredictions:
    """Verify inference correctness and output schema."""

    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls):
        if not torch_available():
            pytest.skip("torch not installed")
        from src.detection.bot_detector import BotDetector
        return BotDetector()

    def test_human_prediction(self, detector):
        result = detector.predict(HUMAN_SAMPLE)
        assert result["prediction"] == "human", (
            f"Expected 'human', got '{result['prediction']}'. Full result: {result}"
        )

    def test_bot_prediction(self, detector):
        result = detector.predict(BOT_SAMPLE)
        assert result["prediction"] == "bot", (
            f"Expected 'bot', got '{result['prediction']}'. Full result: {result}"
        )

    def test_human_schema(self, detector):
        result = detector.predict(HUMAN_SAMPLE)
        assert_canonical_schema(
            result,
            expected_agent="bot_detection",
            expected_predictions=("human", "bot"),
            expected_score_keys=("human", "bot"),
        )

    def test_bot_schema(self, detector):
        result = detector.predict(BOT_SAMPLE)
        assert_canonical_schema(
            result,
            expected_agent="bot_detection",
            expected_predictions=("human", "bot"),
            expected_score_keys=("human", "bot"),
        )

    def test_legacy_keys_present(self, detector):
        """Legacy keys must still be present for backwards compatibility."""
        result = detector.predict(HUMAN_SAMPLE)
        assert "bot_score" in result
        assert "bot_class" in result
        assert "bot_label" in result
        assert result["bot_label"] in ("HUMAN", "BOT")
        assert 0.0 <= result["bot_score"] <= 1.0
        assert result["bot_class"] in (0, 1)

    def test_scores_sum_to_one(self, detector):
        result = detector.predict(HUMAN_SAMPLE)
        total = result["scores"]["human"] + result["scores"]["bot"]
        assert abs(total - 1.0) < 1e-4, f"Scores must sum to 1.0, got {total}"

    def test_missing_fields_handled_gracefully(self, detector):
        """Minimal input (no optional fields) must not raise an exception."""
        minimal = {
            "followers_count": 100,
            "friends_count": 50,
            "statuses_count": 500,
        }
        result = detector.predict(minimal)
        assert_canonical_schema(
            result,
            expected_agent="bot_detection",
            expected_predictions=("human", "bot"),
            expected_score_keys=("human", "bot"),
        )

    def test_confidence_matches_predicted_class_score(self, detector):
        for sample in (HUMAN_SAMPLE, BOT_SAMPLE):
            result = detector.predict(sample)
            pred = result["prediction"]
            assert abs(result["confidence"] - result["scores"][pred]) < 1e-4, (
                f"confidence must equal scores['{pred}']"
            )


class TestBotDetectorBatch:
    """Verify batch inference for Agent 2."""

    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls):
        if not torch_available():
            pytest.skip("torch not installed")
        from src.detection.bot_detector import BotDetector
        return BotDetector()

    def test_batch_returns_list(self, detector):
        df = pd.DataFrame([HUMAN_SAMPLE, BOT_SAMPLE])
        results = detector.predict_batch(df)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_schema_each_item(self, detector):
        df = pd.DataFrame([HUMAN_SAMPLE, BOT_SAMPLE])
        for result in detector.predict_batch(df):
            assert_canonical_schema(
                result,
                expected_agent="bot_detection",
                expected_predictions=("human", "bot"),
                expected_score_keys=("human", "bot"),
            )

    def test_batch_matches_single(self, detector):
        df = pd.DataFrame([HUMAN_SAMPLE, BOT_SAMPLE])
        batch = detector.predict_batch(df)
        for sample, batch_res in zip([HUMAN_SAMPLE, BOT_SAMPLE], batch):
            single = detector.predict(sample)
            assert batch_res["prediction"] == single["prediction"]
            assert abs(batch_res["confidence"] - single["confidence"]) < 1e-4

    def test_batch_human_predicted_human(self, detector):
        df = pd.DataFrame([HUMAN_SAMPLE, HUMAN_SAMPLE])
        results = detector.predict_batch(df)
        for r in results:
            assert r["prediction"] == "human"

    def test_batch_bot_predicted_bot(self, detector):
        df = pd.DataFrame([BOT_SAMPLE, BOT_SAMPLE])
        results = detector.predict_batch(df)
        for r in results:
            assert r["prediction"] == "bot"

    def test_empty_batch(self, detector):
        assert detector.predict_batch(pd.DataFrame()) == []

    def test_non_dict_sample_raises(self, detector):
        with pytest.raises(TypeError, match="must be a dict"):
            detector.predict(["not", "a", "dict"])

    def test_batch_with_missing_optional_fields_matches_single(self, detector):
        minimal = {
            "followers_count": 120,
            "friends_count": 95,
            "statuses_count": 300,
        }
        df = pd.DataFrame([HUMAN_SAMPLE, minimal])
        batch = detector.predict_batch(df)
        for sample, batch_res in zip([HUMAN_SAMPLE, minimal], batch):
            single = detector.predict(sample)
            assert batch_res["prediction"] == single["prediction"]
            assert abs(batch_res["confidence"] - single["confidence"]) < 1e-4


class TestBotFeaturePipeline:
    """Unit tests for the feature engineering pipeline (no torch needed)."""

    def test_account_features_count(self):
        from src.data.preprocessing import ACCOUNT_FEATURES
        assert len(ACCOUNT_FEATURES) == 15

    def test_engineer_account_features_smoke(self):
        from src.data.preprocessing import engineer_account_features
        df = pd.DataFrame([HUMAN_SAMPLE])
        result = engineer_account_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_all_account_features_present(self):
        from src.data.preprocessing import ACCOUNT_FEATURES, engineer_account_features
        df = pd.DataFrame([HUMAN_SAMPLE])
        result = engineer_account_features(df)
        for col in ACCOUNT_FEATURES:
            assert col in result.columns, f"Missing feature: {col}"

    def test_build_sequences_shape(self):
        from src.features.temporal_features import build_sequences, SEQ_LEN
        import numpy as np
        X = np.random.randn(10, 15).astype("float32")
        X_seq = build_sequences(X, seq_len=SEQ_LEN)
        assert X_seq.shape == (10, 5, 3)

    def test_no_nans_in_engineered_features(self):
        from src.data.preprocessing import ACCOUNT_FEATURES, engineer_account_features
        df = pd.DataFrame([HUMAN_SAMPLE, BOT_SAMPLE])
        result = engineer_account_features(df)
        for col in ACCOUNT_FEATURES:
            assert result[col].notna().all(), f"NaN in feature '{col}'"
