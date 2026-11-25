"""
src/features/temporal_features.py
-----------------------------------
Temporal and behavioural feature engineering for bot detection.

DATASET NOTE
------------
The bot_detection_data.csv contains a single 'Created At' timestamp per row
(one tweet per user), not a full publication history. Therefore a true
time-series RNN with per-user activity sequences cannot be built from raw
timestamps alone.

Approach adopted (documented transparently):
  1. We extract scalar temporal features from the single timestamp
     (hour, day, etc.) and combine them with the available behavioural
     columns (retweet/mention/follower counts, verified flag).
  2. We then *reshape* each sample into a fixed-length sequence of shape
     (seq_len, n_features) so the data can be fed into an RNN.
     The sequence is built by sliding a small window over the feature
     vector — effectively treating the feature dimensions as a "sequence"
     of observations.  This is a common fallback when no genuine temporal
     history is available.
  3. This limitation is documented here and in the technical report so
     the team can substitute real per-user histories if the dataset is
     extended.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


# ─── Scalar temporal feature extraction ──────────────────────────────────────

def extract_temporal_scalars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract scalar temporal signals from the 'Created At' column.

    Columns added:
      hour_of_day  (0–23)
      day_of_week  (0=Mon … 6=Sun)
      month        (1–12)
      year         (e.g. 2020)
      is_weekend   (1 if Sat/Sun, else 0)
      is_night     (1 if 22:00–06:00, else 0)
    """
    df = df.copy()
    dt = pd.to_datetime(df["Created At"], errors="coerce")
    df["hour_of_day"] = dt.dt.hour.fillna(-1).astype(int)
    df["day_of_week"] = dt.dt.dayofweek.fillna(-1).astype(int)
    df["month"] = dt.dt.month.fillna(-1).astype(int)
    df["year"] = dt.dt.year.fillna(-1).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = (
        (df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)
    ).astype(int)
    return df


# ─── Feature list used by the RNN ────────────────────────────────────────────

RNN_FEATURES = [
    "Retweet Count",
    "Mention Count",
    "Follower Count",
    "verified_int",
    "tweet_length",
    "word_count",
    "hashtag_count",
    "has_hashtag",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
]

SEQ_LEN = 5   # Sequence length fed to the RNN


# ─── Sequence construction ────────────────────────────────────────────────────

def build_sequences(X: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Reshape a 2-D feature matrix (n_samples, n_features) into a 3-D tensor
    (n_samples, seq_len, n_features // seq_len + padding) suitable for an RNN.

    Strategy: pad the feature vector to be divisible by seq_len, then split
    it into seq_len chunks, each representing one "time step".

    Parameters
    ----------
    X        : np.ndarray — shape (n_samples, n_features)
    seq_len  : int        — number of time steps per sample

    Returns
    -------
    np.ndarray — shape (n_samples, seq_len, step_size)
    """
    n_samples, n_features = X.shape

    # Pad features so they split evenly
    step_size = int(np.ceil(n_features / seq_len))
    pad_width = step_size * seq_len - n_features
    if pad_width > 0:
        X = np.hstack([X, np.zeros((n_samples, pad_width))])

    # Reshape → (n_samples, seq_len, step_size)
    X_seq = X.reshape(n_samples, seq_len, step_size)
    return X_seq


# ─── Scaling ─────────────────────────────────────────────────────────────────

def fit_scaler(X: np.ndarray, save_path: str = None) -> StandardScaler:
    """Fit a StandardScaler on X (2-D) and optionally save it."""
    scaler = StandardScaler()
    scaler.fit(X)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"[temporal_features] Scaler saved to '{save_path}'.")
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted scaler to X (2-D)."""
    return scaler.transform(X)


# ─── Full pipeline ────────────────────────────────────────────────────────────

def prepare_rnn_input(df: pd.DataFrame, scaler: StandardScaler = None,
                      seq_len: int = SEQ_LEN, fit: bool = True):
    """
    Full pipeline: extract features → scale → build sequences.

    Parameters
    ----------
    df      : preprocessed DataFrame (output of preprocessing.preprocess())
    scaler  : existing StandardScaler (if None and fit=True, a new one is fitted)
    seq_len : RNN sequence length
    fit     : if True, fit a new scaler on df; otherwise use the provided one

    Returns
    -------
    X_seq   : np.ndarray — shape (n_samples, seq_len, step_size)
    y       : np.ndarray — shape (n_samples,) — 0=human, 1=bot
    scaler  : fitted StandardScaler
    """
    # Ensure temporal scalars are present
    df = extract_temporal_scalars(df)

    # Align available columns (some may already exist from preprocessing.py)
    available = [c for c in RNN_FEATURES if c in df.columns]
    X = df[available].fillna(0).values.astype(np.float32)
    y = df["Bot Label"].astype(int).values

    if fit or scaler is None:
        scaler = fit_scaler(X)
    X_scaled = apply_scaler(X, scaler)

    X_seq = build_sequences(X_scaled, seq_len=seq_len)

    print(f"[temporal_features] X_seq shape: {X_seq.shape}, y shape: {y.shape}")
    return X_seq, y, scaler
