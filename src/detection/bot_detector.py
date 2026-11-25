"""
src/detection/bot_detector.py
-------------------------------
High-level inference interface for Twitter Bot Detection (Agent 2).

Integration contract:
  {
      "bot_score": float,   # Probability of being a bot [0.0, 1.0]
      "bot_class": int,     # 0 = human, 1 = bot
      "bot_label": str      # "HUMAN" or "BOT"
  }
"""

import os
import math
import numpy as np
import pandas as pd
import joblib
import torch

from src.data.preprocessing import (
    ACCOUNT_FEATURES,
    engineer_account_features,
)
from src.features.temporal_features import (
    build_sequences,
    apply_scaler,
    SEQ_LEN,
)
from src.models.rnn import BotRNN, predict_proba


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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Bot RNN model not found at '{model_path}'.\n"
                "Please train the model first using the Colab notebook or train script."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at '{scaler_path}'.")

        self.scaler = joblib.load(scaler_path)

        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.input_size = meta.get("input_size", 3)
            self.seq_len = meta.get("seq_len", SEQ_LEN)
            self.features = meta.get("features", ACCOUNT_FEATURES)
        else:
            self.seq_len = SEQ_LEN
            self.features = ACCOUNT_FEATURES
            self.input_size = math.ceil(len(self.features) / self.seq_len)

        self.model = BotRNN(input_size=self.input_size).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        print(f"[BotDetector] Successfully loaded BotRNN from '{model_path}' on {self.device}.")

    def _prepare_sample_tensor(self, sample: dict) -> torch.Tensor:
        """Transform raw account dict into (1, seq_len, input_size) tensor."""
        df_tmp = pd.DataFrame([sample])
        df_feat = engineer_account_features(df_tmp)

        # Ensure all ACCOUNT_FEATURES are present
        for col in self.features:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        X = df_feat[self.features].values.astype(np.float32)
        X_scaled = apply_scaler(X, self.scaler)
        X_seq = build_sequences(X_scaled, seq_len=self.seq_len)
        return torch.tensor(X_seq, dtype=torch.float32)

    def predict(self, sample: dict) -> dict:
        """
        Predict whether an account is human or bot.

        Parameters
        ----------
        sample : dict
            Account attributes, e.g.:
            {
                "followers_count": 120,
                "friends_count": 250,
                "statuses_count": 890,
                "favourites_count": 340,
                "verified": False,
                "default_profile": True,
                "default_profile_image": False,
                "geo_enabled": False,
                "average_tweets_per_day": 5.2,
                "account_age_days": 450,
                "description": "Just another tech enthusiast.",
                "screen_name": "tech_user_99"
            }

        Returns
        -------
        dict
            {
                "bot_score": float,  # P(bot) in [0, 1]
                "bot_class": int,    # 0 = human, 1 = bot
                "bot_label": str     # "HUMAN" or "BOT"
            }
        """
        X_tensor = self._prepare_sample_tensor(sample)
        proba = predict_proba(self.model, X_tensor, device=self.device)
        proba_np = proba.cpu().numpy()[0]  # [P(human), P(bot)]

        bot_score = float(proba_np[1])
        bot_class = int(np.argmax(proba_np))
        bot_label = "BOT" if bot_class == 1 else "HUMAN"

        return {
            "bot_score": round(bot_score, 4),
            "bot_class": bot_class,
            "bot_label": bot_label,
        }

    def predict_batch(self, df: pd.DataFrame) -> list:
        """Batch inference on a DataFrame."""
        df_feat = engineer_account_features(df)
        for col in self.features:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        X = df_feat[self.features].values.astype(np.float32)
        X_scaled = apply_scaler(X, self.scaler)
        X_seq = build_sequences(X_scaled, seq_len=self.seq_len)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        proba = predict_proba(self.model, X_tensor, device=self.device).cpu().numpy()
        results = []
        for p in proba:
            bot_score = float(p[1])
            bot_class = int(np.argmax(p))
            results.append({
                "bot_score": round(bot_score, 4),
                "bot_class": bot_class,
                "bot_label": "BOT" if bot_class == 1 else "HUMAN",
            })
        return results


def detect_bot(sample: dict, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Convenience helper for one-shot bot prediction."""
    detector = BotDetector(model_path=model_path)
    return detector.predict(sample)
