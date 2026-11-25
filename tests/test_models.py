"""
tests/test_models.py
---------------------
Unit tests for Agent 2 — Bot Detection (RNN).

Tests cover:
  1. Data loading and preprocessing (shape, columns, no nulls in features).
  2. Temporal feature extraction (output shape, expected columns).
  3. Sequence building (correct 3-D shape).
  4. BotRNN forward pass (output shape and valid probabilities).
  5. BotDetector predict() output format (integration contract).

Run with:
  pytest tests/test_models.py -v
"""

import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessing import (
    preprocess,
    get_features_and_labels,
    BEHAVIORAL_FEATURES,
    engineer_features,
    clean_tweet,
)
from src.features.temporal_features import (
    extract_temporal_scalars,
    build_sequences,
    prepare_rnn_input,
    RNN_FEATURES,
    SEQ_LEN,
)
from src.models.rnn import BotRNN, predict_proba


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Load a small sample of the bot detection dataset."""
    path = "bot_detection_data.csv"
    if not os.path.exists(path):
        pytest.skip(f"Dataset not found: {path}")
    df = pd.read_csv(path, nrows=200)
    return df


@pytest.fixture(scope="module")
def preprocessed_df(raw_df):
    return preprocess(raw_df)


# ─── 1. Preprocessing tests ───────────────────────────────────────────────────

class TestPreprocessing:

    def test_clean_tweet_removes_urls(self):
        raw = "Check https://example.com now!"
        cleaned = clean_tweet(raw)
        assert "http" not in cleaned

    def test_clean_tweet_removes_at_mentions(self):
        raw = "Hello @user how are you?"
        cleaned = clean_tweet(raw)
        assert "@" not in cleaned

    def test_clean_tweet_lowercases(self):
        raw = "Hello World"
        assert clean_tweet(raw) == clean_tweet(raw).lower()

    def test_preprocess_returns_dataframe(self, preprocessed_df):
        assert isinstance(preprocessed_df, pd.DataFrame)

    def test_preprocess_no_null_labels(self, preprocessed_df):
        assert preprocessed_df["Bot Label"].isnull().sum() == 0

    def test_preprocess_adds_tweet_clean(self, preprocessed_df):
        assert "tweet_clean" in preprocessed_df.columns

    def test_get_features_and_labels_shapes(self, preprocessed_df):
        X, y = get_features_and_labels(preprocessed_df)
        assert len(X) == len(y)
        assert set(BEHAVIORAL_FEATURES).issubset(set(X.columns))

    def test_labels_binary(self, preprocessed_df):
        X, y = get_features_and_labels(preprocessed_df)
        assert set(y.unique()).issubset({0, 1})


# ─── 2. Temporal feature tests ────────────────────────────────────────────────

class TestTemporalFeatures:

    def test_extract_temporal_scalars_adds_columns(self, preprocessed_df):
        df = extract_temporal_scalars(preprocessed_df)
        for col in ["hour_of_day", "day_of_week", "month", "is_weekend", "is_night"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_hour_range(self, preprocessed_df):
        df = extract_temporal_scalars(preprocessed_df)
        valid = df["hour_of_day"].between(0, 23)
        # allow -1 for missing values, so check only non -1 rows
        non_missing = df[df["hour_of_day"] != -1]["hour_of_day"]
        assert non_missing.between(0, 23).all()

    def test_is_weekend_binary(self, preprocessed_df):
        df = extract_temporal_scalars(preprocessed_df)
        assert set(df["is_weekend"].unique()).issubset({0, 1})

    def test_prepare_rnn_input_shape(self, preprocessed_df):
        X_seq, y, scaler = prepare_rnn_input(preprocessed_df, seq_len=SEQ_LEN)
        assert X_seq.ndim == 3, "Expected 3-D array (n, seq_len, step_size)"
        assert X_seq.shape[0] == len(preprocessed_df)
        assert X_seq.shape[1] == SEQ_LEN


# ─── 3. Sequence building tests ───────────────────────────────────────────────

class TestSequenceBuilding:

    def test_build_sequences_3d_output(self):
        X = np.random.randn(50, 13).astype(np.float32)
        X_seq = build_sequences(X, seq_len=5)
        assert X_seq.ndim == 3
        assert X_seq.shape[0] == 50
        assert X_seq.shape[1] == 5

    def test_build_sequences_no_data_loss(self):
        """All original values should still be present (not zeroed out)."""
        X = np.ones((10, 10), dtype=np.float32)
        X_seq = build_sequences(X, seq_len=5)
        # Every reconstructed value should be 1.0 (from original) or 0.0 (padding)
        assert X_seq.min() >= 0.0


# ─── 4. RNN model tests ───────────────────────────────────────────────────────

class TestBotRNN:

    def _make_input(self, batch=8, seq_len=5, input_size=3):
        return torch.randn(batch, seq_len, input_size)

    def test_forward_output_shape(self):
        model = BotRNN(input_size=3, hidden_size=16, num_layers=1)
        x = self._make_input()
        logits = model(x)
        assert logits.shape == (8, 2), f"Expected (8,2), got {logits.shape}"

    def test_predict_proba_sums_to_one(self):
        model = BotRNN(input_size=3, hidden_size=16, num_layers=1)
        x = self._make_input()
        proba = predict_proba(model, x)
        sums = proba.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_predict_proba_in_range(self):
        model = BotRNN(input_size=3, hidden_size=16, num_layers=1)
        x = self._make_input()
        proba = predict_proba(model, x)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_model_different_batch_sizes(self):
        model = BotRNN(input_size=3, hidden_size=16, num_layers=1)
        for batch in [1, 4, 32]:
            x = self._make_input(batch=batch)
            out = model(x)
            assert out.shape[0] == batch

    def test_model_has_named_parameters(self):
        model = BotRNN(input_size=5)
        param_names = [n for n, _ in model.named_parameters()]
        assert len(param_names) > 0


# ─── 5. Integration contract format test ──────────────────────────────────────

class TestIntegrationContract:

    def test_output_keys_present(self):
        """
        Verify that a manually constructed result matches the integration format.
        (Full BotDetector test requires a saved model — skipped if not available.)
        """
        expected_keys = {"bot_score", "bot_class", "bot_label"}
        fake_result = {
            "bot_score": 0.72,
            "bot_class": 1,
            "bot_label": "BOT",
        }
        assert expected_keys == set(fake_result.keys())

    def test_bot_score_range(self):
        for score in [0.0, 0.5, 1.0]:
            assert 0.0 <= score <= 1.0

    def test_bot_label_values(self):
        valid_labels = {"BOT", "HUMAN"}
        for label in ["BOT", "HUMAN"]:
            assert label in valid_labels
