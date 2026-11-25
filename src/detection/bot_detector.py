"""
src/detection/bot_detector.py
-------------------------------
High-level interface for bot detection.

This module exposes:
  - BotDetector   : loads the saved RNN + scaler and runs inference.
  - detect_bot()  : convenience function that returns the standard
                    integration dict expected by Agent 3.

Integration contract (see section 5 of the project specification):
  {
      "bot_score":  float,   # P(bot)  in [0, 1]
      "bot_class":  int,     # 0=human, 1=bot
      "bot_label":  str,     # "HUMAN" or "BOT"
  }
"""

import os
import numpy as np
import joblib
import torch

from src.data.preprocessing import (
    preprocess,
    BEHAVIORAL_FEATURES,
    engineer_features,
)
from src.features.temporal_features import (
    extract_temporal_scalars,
    build_sequences,
    apply_scaler,
    RNN_FEATURES,
    SEQ_LEN,
)
from src.models.rnn import BotRNN, predict_proba


# Default saved-model paths (relative to project root)
DEFAULT_MODEL_PATH   = "models/saved/bot_rnn_model.pth"
DEFAULT_SCALER_PATH  = "models/vectorizers/bot_scaler.joblib"
DEFAULT_INPUT_META   = "models/vectorizers/bot_input_meta.joblib"


class BotDetector:
    """
    Loads a trained BotRNN model from disk and exposes a predict() method.

    Parameters
    ----------
    model_path  : path to the saved .pth file
    scaler_path : path to the saved StandardScaler joblib file
    meta_path   : path to a dict containing input_size and seq_len
    device      : 'cuda' if available, else 'cpu'
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
        meta_path: str = DEFAULT_INPUT_META,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Bot RNN model not found: {model_path}\n"
                "Train the model first using the Colab notebook "
                "(notebooks/03_model_experiments.ipynb)."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        self.scaler = joblib.load(scaler_path)

        # Load metadata (input_size, seq_len)
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.input_size = meta["input_size"]
            self.seq_len    = meta.get("seq_len", SEQ_LEN)
        else:
            # Fallback: derive from RNN_FEATURES and SEQ_LEN
            self.seq_len    = SEQ_LEN
            import math
            step_size = math.ceil(len(RNN_FEATURES) / self.seq_len)
            self.input_size = step_size

        # Build model architecture and load weights
        self.model = BotRNN(input_size=self.input_size).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        print(f"[BotDetector] Model loaded from '{model_path}' on {self.device}.")

    def _build_input(self, sample: dict) -> torch.Tensor:
        """
        Convert a raw sample dict into a 3-D tensor (1, seq_len, input_size).

        The sample dict must contain the same keys as RNN_FEATURES.
        Missing keys default to 0.
        """
        import pandas as pd

        row = {feat: sample.get(feat, 0) for feat in RNN_FEATURES}
        df_tmp = pd.DataFrame([row])
        X = df_tmp.values.astype(np.float32)
        X_scaled = apply_scaler(X, self.scaler)
        X_seq = build_sequences(X_scaled, seq_len=self.seq_len)
        return torch.tensor(X_seq, dtype=torch.float32)

    def predict(self, sample: dict) -> dict:
        """
        Predict whether a single account/tweet sample is a bot.

        Parameters
        ----------
        sample : dict with keys matching RNN_FEATURES (see temporal_features.py)
                 e.g. {"Retweet Count": 50, "Mention Count": 3, ...}

        Returns
        -------
        dict:
          {
              "bot_score":  float,   # P(bot) in [0, 1]
              "bot_class":  int,     # 0=human, 1=bot
              "bot_label":  str,     # "HUMAN" or "BOT"
          }
        """
        X_tensor = self._build_input(sample)
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

    def predict_batch(self, df) -> list:
        """
        Run prediction on a full DataFrame that has been preprocessed.

        Parameters
        ----------
        df : preprocessed DataFrame (output of preprocessing.preprocess())

        Returns
        -------
        list of dicts — one result dict per row
        """
        import pandas as pd

        df = extract_temporal_scalars(df)
        available = [c for c in RNN_FEATURES if c in df.columns]
        X = df[available].fillna(0).values.astype(np.float32)
        X_scaled = apply_scaler(X, self.scaler)
        X_seq = build_sequences(X_scaled, seq_len=self.seq_len)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        proba = predict_proba(self.model, X_tensor, device=self.device)
        proba_np = proba.cpu().numpy()  # (n, 2)

        results = []
        for p in proba_np:
            bot_score = float(p[1])
            bot_class = int(np.argmax(p))
            results.append({
                "bot_score": round(bot_score, 4),
                "bot_class": bot_class,
                "bot_label": "BOT" if bot_class == 1 else "HUMAN",
            })
        return results


# ─── Convenience function ─────────────────────────────────────────────────────

def detect_bot(sample: dict,
               model_path: str = DEFAULT_MODEL_PATH,
               scaler_path: str = DEFAULT_SCALER_PATH) -> dict:
    """
    One-shot bot detection for a single sample.

    Returns the integration contract dict:
      {"bot_score": float, "bot_class": int, "bot_label": str}
    """
    detector = BotDetector(model_path=model_path, scaler_path=scaler_path)
    return detector.predict(sample)
