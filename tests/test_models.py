"""
tests/test_models.py
---------------------
Unit tests for Agent 2 — Twitter Bot Detection (RNN).

Tests cover:
  1. Data loading and preprocessing.
  2. Temporal & behavioral feature extraction.
  3. Sequence building for PyTorch GRU.
  4. BotRNN model forward pass and probabilities.
  5. BotDetector inference interface and integration contract format.

Run with:
  pytest tests/test_models.py -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.loader import load_bot_dataset, resolve_bot_dataset_path
from src.data.preprocessing import (
    preprocess,
    engineer_account_features,
    get_features_and_labels,
    ACCOUNT_FEATURES,
    clean_text,
)
from src.features.temporal_features import (
    build_sequences,
    prepare_rnn_input,
    SEQ_LEN,
)
from src.models.rnn import BotRNN, predict_proba
from src.detection.bot_detector import BotDetector
from src.moderation.pipeline import SocialThreatPipeline
from src.utils.helpers import resolve_project_path


@pytest.fixture(scope="module")
def sample_df():
    """Load a small slice of the dataset for testing."""
    df = load_bot_dataset()
    return df.head(100)


@pytest.fixture(scope="module")
def preprocessed_df(sample_df):
    return preprocess(sample_df)


# ─── 1. Preprocessing tests ───────────────────────────────────────────────────

class TestPreprocessing:

    def test_clean_text_removes_urls(self):
        text = "Visit https://t.co/xyz now for free bots!"
        cleaned = clean_text(text)
        assert "http" not in cleaned

    def test_clean_text_removes_mentions(self):
        text = "Hey @elonmusk check this out!"
        cleaned = clean_text(text)
        assert "@" not in cleaned

    def test_preprocess_returns_dataframe(self, preprocessed_df):
        assert isinstance(preprocessed_df, pd.DataFrame)
        assert len(preprocessed_df) > 0

    def test_preprocess_has_bot_label(self, preprocessed_df):
        assert "bot_label" in preprocessed_df.columns
        assert set(preprocessed_df["bot_label"].unique()).issubset({0, 1})

    def test_account_features_extracted(self, preprocessed_df):
        for col in ACCOUNT_FEATURES:
            assert col in preprocessed_df.columns, f"Missing feature: {col}"

    def test_get_features_and_labels(self, preprocessed_df):
        X, y = get_features_and_labels(preprocessed_df)
        assert len(X) == len(y)
        assert X.shape[1] == len(ACCOUNT_FEATURES)


# ─── 2. Temporal & sequence tests ─────────────────────────────────────────────

class TestTemporalSequences:

    def test_build_sequences_shape(self):
        X = np.random.randn(20, len(ACCOUNT_FEATURES)).astype(np.float32)
        X_seq = build_sequences(X, seq_len=SEQ_LEN)
        assert X_seq.ndim == 3
        assert X_seq.shape[0] == 20
        assert X_seq.shape[1] == SEQ_LEN
        assert X_seq.shape[2] == int(np.ceil(len(ACCOUNT_FEATURES) / SEQ_LEN))

    def test_prepare_rnn_input(self, preprocessed_df):
        X_seq, y, scaler = prepare_rnn_input(preprocessed_df, seq_len=SEQ_LEN)
        assert X_seq.ndim == 3
        assert len(X_seq) == len(preprocessed_df)
        assert y is not None


# ─── 3. BotRNN model architecture tests ───────────────────────────────────────

class TestBotRNN:

    def test_forward_shape(self):
        step_size = int(np.ceil(len(ACCOUNT_FEATURES) / SEQ_LEN))
        model = BotRNN(input_size=step_size, hidden_size=32, num_layers=2)
        x = torch.randn(8, SEQ_LEN, step_size)
        logits = model(x)
        assert logits.shape == (8, 2)

    def test_predict_proba_sums_to_one(self):
        step_size = int(np.ceil(len(ACCOUNT_FEATURES) / SEQ_LEN))
        model = BotRNN(input_size=step_size, hidden_size=32, num_layers=2)
        x = torch.randn(8, SEQ_LEN, step_size)
        proba = predict_proba(model, x)
        sums = proba.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)


# ─── 4. Inference & Integration contract tests ────────────────────────────────

class TestBotDetectorIntegration:

    def test_detector_inference(self):
        model_path = resolve_project_path("models/saved/bot_rnn_model.pth")
        assert os.path.exists(model_path), "Model weights missing at models/saved/bot_rnn_model.pth"

        detector = BotDetector()
        sample = {
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
        }
        res = detector.predict(sample)

        assert "bot_score" in res
        assert "bot_class" in res
        assert "bot_label" in res
        assert 0.0 <= res["bot_score"] <= 1.0
        assert res["bot_class"] in (0, 1)
        assert res["bot_label"] in ("HUMAN", "BOT")

    def test_load_dataset_independent_of_cwd(self, monkeypatch, tmp_path):
        """Dataset resolution must succeed regardless of current working directory."""
        monkeypatch.chdir(tmp_path)
        resolved = resolve_bot_dataset_path()
        assert resolved.is_file()
        df = load_bot_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "account_type" in df.columns

    def test_bot_inference_does_not_load_training_csv(self, monkeypatch, tmp_path):
        """BotDetector and SocialThreatPipeline inference must never read the training CSV."""
        import src.data.loader as loader_mod

        def _forbid_dataset_load(*args, **kwargs):
            raise AssertionError("Bot inference must not load the training CSV dataset.")

        monkeypatch.setattr(loader_mod, "load_bot_dataset", _forbid_dataset_load)
        monkeypatch.setattr(loader_mod, "resolve_bot_dataset_path", _forbid_dataset_load)
        monkeypatch.chdir(tmp_path)

        pipeline = SocialThreatPipeline()
        sample_account = {
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
        out = pipeline.run(
            "BREAKING: Secret government microchip inside tap water exposed!! "
            "Share before they delete this! Mainstream media refuses to report "
            "this Big Pharma conspiracy cure they are hiding from the public!",
            sample_account,
        )
        assert out["bot_detection"]["prediction"] == "bot"
        assert out["moderation"]["decision"] == "RESTRICT"
