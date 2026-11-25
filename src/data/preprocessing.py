"""
src/data/preprocessing.py
--------------------------
Cleans and preprocesses Twitter bot datasets.
Primary dataset: twitter_human_bots_dataset.csv
"""

import pandas as pd
import numpy as np
import re


def clean_text(text: str) -> str:
    """Clean text by removing URLs, mentions, punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def engineer_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer behavioral, temporal, and metadata features from Twitter accounts.
    """
    df = df.copy()

    # Target label mapping (0 = human, 1 = bot)
    if "account_type" in df.columns:
        df["bot_label"] = (df["account_type"].astype(str).str.lower() == "bot").astype(int)
    elif "Bot Label" in df.columns:
        df["bot_label"] = df["Bot Label"].astype(int)
    elif "bot" in df.columns:
        df["bot_label"] = df["bot"].astype(int)

    # Boolean flags
    for col in ["default_profile", "default_profile_image", "geo_enabled", "verified", "Verified"]:
        if col in df.columns:
            target_name = col.lower()
            df[target_name] = df[col].astype(bool).astype(int)

    # Numeric counts with log1p scaling to handle heavy tails
    for col in ["followers_count", "Follower Count"]:
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
            df["log_followers"] = np.log1p(val)

    for col in ["friends_count", "Mention Count"]:
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
            df["log_friends"] = np.log1p(val)

    for col in ["statuses_count", "Retweet Count"]:
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
            df["log_statuses"] = np.log1p(val)

    if "favourites_count" in df.columns:
        val = pd.to_numeric(df["favourites_count"], errors="coerce").fillna(0).clip(lower=0)
        df["log_favourites"] = np.log1p(val)
    else:
        df["log_favourites"] = 0.0

    # Temporal & activity rate features
    time_col = "created_at" if "created_at" in df.columns else ("Created At" if "Created At" in df.columns else None)
    if time_col:
        dt = pd.to_datetime(df[time_col], errors="coerce")
        df["hour_of_day"] = dt.dt.hour.fillna(-1).astype(int)
        df["day_of_week"] = dt.dt.dayofweek.fillna(-1).astype(int)
        df["year"] = dt.dt.year.fillna(2015).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int)
    else:
        df["hour_of_day"] = -1
        df["day_of_week"] = -1
        df["year"] = 2015
        df["is_weekend"] = 0
        df["is_night"] = 0

    if "account_age_days" in df.columns:
        df["account_age_days"] = pd.to_numeric(df["account_age_days"], errors="coerce").fillna(0)
    else:
        df["account_age_days"] = 0.0

    if "average_tweets_per_day" in df.columns:
        df["log_tweets_per_day"] = np.log1p(pd.to_numeric(df["average_tweets_per_day"], errors="coerce").fillna(0).clip(lower=0))
    else:
        df["log_tweets_per_day"] = 0.0

    # Profile metadata & text signals
    if "description" in df.columns:
        df["desc_len"] = df["description"].fillna("").apply(len)
        df["has_desc"] = (df["desc_len"] > 0).astype(int)
    else:
        df["desc_len"] = 0
        df["has_desc"] = 0

    name_col = "screen_name" if "screen_name" in df.columns else ("Username" if "Username" in df.columns else None)
    if name_col:
        df["digits_in_name"] = df[name_col].fillna("").apply(lambda s: sum(c.isdigit() for c in str(s)))
        df["name_len"] = df[name_col].fillna("").apply(lambda s: len(str(s)))
    else:
        df["digits_in_name"] = 0
        df["name_len"] = 0

    # Fallback default flags if not present
    for req in ["default_profile", "default_profile_image", "geo_enabled", "verified"]:
        if req not in df.columns:
            df[req] = 0

    return df


ACCOUNT_FEATURES = [
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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for Twitter bot detection dataset.
    """
    initial_len = len(df)
    df = df.drop_duplicates()
    df = engineer_account_features(df)
    if "bot_label" in df.columns:
        df = df.dropna(subset=["bot_label"])
    print(f"[preprocessing] Processed {len(df):,} / {initial_len:,} rows.")
    return df


def get_features_and_labels(df: pd.DataFrame):
    """
    Extract feature matrix X and target y.
    """
    X = df[ACCOUNT_FEATURES].copy().fillna(0)
    y = df["bot_label"].astype(int) if "bot_label" in df.columns else None
    return X, y
