"""
src/detection/bot_detector.py
-------------------------------
High-level inference interface for Twitter Bot Detection (Agent 2).

Integration contract
--------------------
{
    "agent":      "bot_detection",
    "prediction": "human" | "bot",
    "confidence": float,            # P(predicted class) in [0.0, 1.0]
    "scores":     {"human": float, "bot": float},
    "raw_class":  int               # 0 = human, 1 = bot
}

Legacy keys (bot_score, bot_class, bot_label) are retained for backwards
compatibility but the canonical output is the dict above.
"""

from __future__ import annotations

import os
import math
import warnings
import numpy as np
import pandas as pd
import joblib

try:
    import torch
    from src.models.rnn import BotRNN, predict_proba
    _TORCH_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    BotRNN = None
    predict_proba = None
    _TORCH_IMPORT_ERROR = exc

from src.data.preprocessing import (
    ACCOUNT_FEATURES,
    engineer_account_features,
)
from src.features.temporal_features import (
    build_sequences,
    apply_scaler,
    SEQ_LEN,
)


DEFAULT_MODEL_PATH  = "models/saved/bot_rnn_model.pth"
DEFAULT_SCALER_PATH = "models/vectorizers/bot_scaler.joblib"
DEFAULT_META_PATH   = "models/vectorizers/bot_input_meta.joblib"


class BotDetector:
    """
    Loads the trained BotRNN model and handles single-sample or batch inference.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
        meta_path: str = DEFAULT_META_PATH,
        device: str = None,
    ):
        if torch is None:
            raise ImportError(
                "PyTorch ('torch') is required for BotDetector (Agent 2). "
                "Install it with: pip install torch"
            ) from _TORCH_IMPORT_ERROR

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Bot RNN model not found at '{model_path}'.\n"
                "Please train the model first using the Colab notebook or train script."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Bot feature scaler not found at '{scaler_path}'.\n"
                "Please ensure the fitted StandardScaler file is present."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scaler = joblib.load(scaler_path)

        if os.path.exists(meta_path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta = joblib.load(meta_path)
            self.input_size = meta.get("input_size", 3)
            self.seq_len = meta.get("seq_len", SEQ_LEN)
            self.features = meta.get("features", ACCOUNT_FEATURES)
        else:
            self.seq_len = SEQ_LEN
            self.features = ACCOUNT_FEATURES
            self.input_size = math.ceil(len(self.features) / self.seq_len)

        n_expected = getattr(self.scaler, "n_features_in_", len(self.features))
        if n_expected != len(self.features):
            raise ValueError(
                f"Feature count mismatch: scaler expects {n_expected} features, "
                f"but metadata specifies {len(self.features)}."
            )

        self.model = BotRNN(input_size=self.input_size).to(self.device)
        # weights_only=True is used when PyTorch supports it (>=2.0) to avoid
        # deserialising arbitrary Python objects from untrusted checkpoints.
        try:
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
        except TypeError:
            # Older PyTorch versions (<2.0) do not support weights_only.
            state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[BotDetector] Successfully loaded BotRNN from '{model_path}' on {self.device}.")

    def _prepare_dataframe_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        """Transform raw account DataFrame into (n_samples, seq_len, input_size) tensor."""
        df_feat = engineer_account_features(df)

        # Ensure all ACCOUNT_FEATURES are present
        for col in self.features:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        X = df_feat[self.features].values.astype(np.float32)
        X_scaled = apply_scaler(X, self.scaler)
        X_seq = build_sequences(X_scaled, seq_len=self.seq_len)
        return torch.tensor(X_seq, dtype=torch.float32)

    def _prepare_sample_tensor(self, sample: dict) -> torch.Tensor:
        """Transform raw account dict into (1, seq_len, input_size) tensor."""
        return self._prepare_dataframe_tensor(pd.DataFrame([sample]))

    @staticmethod
    def _format_result(proba_row: np.ndarray) -> dict:
        """Format softmax probabilities [P(human), P(bot)] into canonical schema."""
        human_score = float(proba_row[0])
        bot_score = float(proba_row[1])
        raw_class = int(np.argmax(proba_row))
        prediction = "bot" if raw_class == 1 else "human"
        confidence = bot_score if raw_class == 1 else human_score

        return {
            # Canonical schema
            "agent": "bot_detection",
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "scores": {
                "human": round(human_score, 4),
                "bot": round(bot_score, 4),
            },
            "raw_class": raw_class,
            # Legacy keys – kept for backwards compatibility
            "bot_score": round(bot_score, 4),
            "bot_class": raw_class,
            "bot_label": "BOT" if raw_class == 1 else "HUMAN",
        }

    def predict(self, sample: dict) -> dict:
        """
        Predict whether an account is human or bot.

        Parameters
        ----------
        sample : dict
            Account attributes. Required for best accuracy:
            followers_count, friends_count, statuses_count, favourites_count,
            average_tweets_per_day, account_age_days, verified,
            default_profile, default_profile_image, geo_enabled.
            Optional: created_at, description, screen_name.
            Missing fields default to 0.0.

        Returns
        -------
        dict
            {
                "agent":      "bot_detection",
                "prediction": "human" | "bot",
                "confidence": float,   # P(predicted class) in [0.0, 1.0]
                "scores":     {"human": float, "bot": float},
                "raw_class":  int,     # 0 = human, 1 = bot
                # Legacy keys (backwards compatibility):
                "bot_score":  float,   # P(bot)
                "bot_class":  int,     # 0 = human, 1 = bot
                "bot_label":  str,     # "HUMAN" or "BOT"
            }
        """
        if not isinstance(sample, dict):
            raise TypeError(
                f"Input sample must be a dict, got {type(sample).__name__}."
            )

        X_tensor = self._prepare_sample_tensor(sample)
        proba = predict_proba(self.model, X_tensor, device=self.device)
        proba_np = proba.cpu().numpy()[0]  # [P(human), P(bot)]
        return self._format_result(proba_np)

    def predict_batch(self, df: pd.DataFrame) -> list:
        """
        Batch inference on a DataFrame of account records.

        Parameters
        ----------
        df : pd.DataFrame
            Each row is one account; same field conventions as predict().

        Returns
        -------
        list[dict]
            One result dict per row, following the canonical schema.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Input df must be a pandas DataFrame, got {type(df).__name__}."
            )
        if df.empty:
            return []

        X_tensor = self._prepare_dataframe_tensor(df)
        proba = predict_proba(self.model, X_tensor, device=self.device).cpu().numpy()
        return [self._format_result(p) for p in proba]


def detect_bot(sample: dict, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Convenience helper for one-shot bot prediction."""
    detector = BotDetector(model_path=model_path)
    return detector.predict(sample)
