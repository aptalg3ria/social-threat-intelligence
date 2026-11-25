"""
src/data/preprocessing.py
--------------------------
Cleans and preprocesses the bot detection dataset.

Dataset columns (bot_detection_data.csv):
  User ID, Username, Tweet, Retweet Count, Mention Count,
  Follower Count, Verified, Bot Label, Location, Created At, Hashtags
"""

import pandas as pd
import numpy as np
import re


# ─── Text cleaning ───────────────────────────────────────────────────────────

def clean_tweet(text: str) -> str:
    """Remove URLs, mentions, hashtag symbols and extra whitespace from a tweet."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)                  # remove @mentions
    text = re.sub(r"#", "", text)                     # remove # symbol (keep word)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)      # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ─── Feature engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features from existing columns.

    New columns added:
      - tweet_length     : character count of the cleaned tweet
      - word_count       : word count of the cleaned tweet
      - hashtag_count    : number of hashtags in the Hashtags column
      - has_hashtag      : binary flag (1 if at least one hashtag)
      - verified_int     : Verified boolean cast to int (0/1)
      - hour_of_day      : hour extracted from Created At
      - day_of_week      : day of week (0=Monday … 6=Sunday) from Created At
    """
    df = df.copy()

    # Clean tweet text
    df["tweet_clean"] = df["Tweet"].apply(clean_tweet)
    df["tweet_length"] = df["tweet_clean"].apply(len)
    df["word_count"] = df["tweet_clean"].apply(lambda x: len(x.split()))

    # Hashtags
    df["Hashtags"] = df["Hashtags"].fillna("")
    df["hashtag_count"] = df["Hashtags"].apply(
        lambda x: len(x.split()) if x.strip() else 0
    )
    df["has_hashtag"] = (df["hashtag_count"] > 0).astype(int)

    # Verified
    df["verified_int"] = df["Verified"].astype(int)

    # Temporal features from Created At
    df["created_at_dt"] = pd.to_datetime(df["Created At"], errors="coerce")
    df["hour_of_day"] = df["created_at_dt"].dt.hour.fillna(-1).astype(int)
    df["day_of_week"] = df["created_at_dt"].dt.dayofweek.fillna(-1).astype(int)

    return df


# ─── Full preprocessing pipeline ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the bot detection dataset.

    1. Drop duplicate rows.
    2. Drop rows with missing Bot Label.
    3. Engineer all derived features.

    Returns the cleaned, feature-enriched DataFrame.
    """
    initial = len(df)

    df = df.drop_duplicates()
    df = df.dropna(subset=["Bot Label"])
    df = engineer_features(df)

    print(f"[preprocessing] Rows kept: {len(df):,} / {initial:,}")
    return df


# ─── Feature matrix & labels ─────────────────────────────────────────────────

BEHAVIORAL_FEATURES = [
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
]


def get_features_and_labels(df: pd.DataFrame):
    """
    Extract the feature matrix X and label vector y.

    Returns
    -------
    X : pd.DataFrame  — shape (n_samples, n_features)
    y : pd.Series     — 0 = human, 1 = bot
    """
    X = df[BEHAVIORAL_FEATURES].copy()
    y = df["Bot Label"].astype(int)
    return X, y
