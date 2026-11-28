"""
src/features/temporal_features.py
-----------------------------------
Temporal and behavioral feature extraction and sequence modeling for BotRNN.

Features:
- Behavioral activity rates: followers, friends, statuses, favourites, tweets per day
- Profile signals: default profile, profile image, geo enabled, verified
- Temporal dynamics: account age, creation hour, day of week, year
- Linguistic metadata: description length, digits in username

Reshapes the feature vector into temporal steps (seq_len=5, step_size=3)
for PyTorch GRU input.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import math

from src.data.preprocessing import ACCOUNT_FEATURES, engineer_account_features

SEQ_LEN = 5


def extract_temporal_scalars(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all behavioral, temporal, and account features are engineered."""
    return engineer_account_features(df)


def build_sequences(X: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Reshape a 2-D feature matrix (n_samples, n_features) into a 3-D tensor
    (n_samples, seq_len, step_size) for RNN / GRU processing.
    """
    n_samples, n_features = X.shape
    step_size = math.ceil(n_features / seq_len)
    pad_width = step_size * seq_len - n_features
    if pad_width > 0:
        X = np.hstack([X, np.zeros((n_samples, pad_width), dtype=np.float32)])

    X_seq = X.reshape(n_samples, seq_len, step_size).astype(np.float32)
    return X_seq


def fit_scaler(X: np.ndarray, save_path: str = None) -> StandardScaler:
    """Fit a StandardScaler on 2-D array and optionally save it."""
    scaler = StandardScaler()
    scaler.fit(X)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"[temporal_features] Scaler saved to '{save_path}'.")
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply fitted StandardScaler to X with cross-version fallback."""
    try:
        return scaler.transform(X).astype(np.float32)
    except AttributeError:
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        safe_scale = np.where(scale == 0.0, 1.0, scale)
        return ((np.asarray(X, dtype=np.float64) - mean) / safe_scale).astype(np.float32)


def prepare_rnn_input(df: pd.DataFrame, scaler: StandardScaler = None,
                      seq_len: int = SEQ_LEN, fit: bool = True):
    """
    Extract features, scale with StandardScaler, and reshape into 3D RNN tensor.
    """
    df = engineer_account_features(df)
    X = df[ACCOUNT_FEATURES].copy().fillna(0).values.astype(np.float32)
    y = df["bot_label"].astype(int).values if "bot_label" in df.columns else None

    if fit or scaler is None:
        scaler = fit_scaler(X)
    X_scaled = apply_scaler(X, scaler)
    X_seq = build_sequences(X_scaled, seq_len=seq_len)

    return X_seq, y, scaler
