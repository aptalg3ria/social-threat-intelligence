"""
src/data/meta_dataset.py
------------------------
Schema, validation utilities, and human-annotation template generator for
the Part 3 Logistic Regression Meta-Dataset.

IMPORTANT:
- No paired Meta-Dataset exists in the repository by default.
- Synthetic bootstrap samples must remain strictly segregated from genuine
  human-annotated moderation labels via the 'label_source' column.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

from src.models.logistic_regression import META_CLASSES, META_FEATURE_COLUMNS


VALID_LABEL_SOURCES: tuple[str, ...] = (
    "human_annotated",
    "synthetic_bootstrap",
)

SYNTHETIC_METADATA_COLUMNS: tuple[str, ...] = (
    "is_synthetic",
    "data_origin",
)

VALID_DATA_ORIGINS: tuple[str, ...] = (
    "human",
    "synthetic",
)

# Minimum columns required in a raw paired input file before frozen-agent
# feature extraction (account fields align with src/data/preprocessing.py).
RAW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "case_id",
    "text",
    "followers_count",
    "friends_count",
    "statuses_count",
)

RAW_OPTIONAL_ACCOUNT_COLUMNS: tuple[str, ...] = (
    "favourites_count",
    "average_tweets_per_day",
    "account_age_days",
    "default_profile",
    "default_profile_image",
    "geo_enabled",
    "verified",
    "created_at",
    "description",
    "screen_name",
)

PROVENANCE_AND_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "source_text_id",
    "source_account_id",
    "moderation_label",
    "label_source",
    "annotator_id",
    "annotation_notes",
)

ANNOTATION_TEMPLATE_COLUMNS: tuple[str, ...] = (
    RAW_REQUIRED_COLUMNS
    + RAW_OPTIONAL_ACCOUNT_COLUMNS
    + PROVENANCE_AND_ANNOTATION_COLUMNS
)

EXTRACTED_META_DATASET_COLUMNS: tuple[str, ...] = (
    "case_id",
    "source_text_id",
    "source_account_id",
    "fake_news_prediction",
    "bot_prediction",
) + META_FEATURE_COLUMNS + (
    "moderation_label",
    "label_source",
    "annotator_id",
)


def _check_missing_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _coerce_bool_series(series: pd.Series, column_name: str) -> pd.Series:
    """Parse a boolean metadata column strictly without silent NaN coercion."""
    if series.isna().any():
        raise ValueError(f"Column '{column_name}' contains missing values.")
    parsed = []
    for val in series:
        if isinstance(val, (bool, np.bool_)):
            parsed.append(bool(val))
        elif isinstance(val, str):
            norm = val.strip().lower()
            if norm in ("true", "1", "yes"):
                parsed.append(True)
            elif norm in ("false", "0", "no"):
                parsed.append(False)
            else:
                raise ValueError(f"Invalid boolean value {val!r} in '{column_name}'.")
        elif isinstance(val, (int, np.integer)) and int(val) in (0, 1):
            parsed.append(bool(int(val)))
        else:
            raise ValueError(f"Unsupported value {val!r} in '{column_name}'.")
    return pd.Series(parsed, index=series.index, dtype=bool)


def _validate_labels_and_provenance(
    df: pd.DataFrame,
    require_human_annotated: bool,
    context: str,
) -> pd.Series:
    """Validate moderation labels, label_source, and optional synthetic metadata."""
    _check_missing_columns(df, ("moderation_label", "label_source"), context)

    labels = df["moderation_label"].astype(str).str.strip().str.upper()
    invalid_mask = df["moderation_label"].isna() | (~labels.isin(META_CLASSES))
    if invalid_mask.any():
        bad_vals = df.loc[invalid_mask, "moderation_label"].unique().tolist()[:5]
        raise ValueError(
            f"Invalid or missing moderation_label values: {bad_vals}. "
            f"Expected one of {META_CLASSES}."
        )

    sources = df["label_source"].astype(str).str.strip().str.lower()
    invalid_src = df["label_source"].isna() | (~sources.isin(VALID_LABEL_SOURCES))
    if invalid_src.any():
        bad_src = df.loc[invalid_src, "label_source"].unique().tolist()[:5]
        raise ValueError(
            f"Invalid 'label_source' values: {bad_src}. "
            f"Expected one of {VALID_LABEL_SOURCES}."
        )

    is_synth_series = None
    if "is_synthetic" in df.columns:
        is_synth_series = _coerce_bool_series(df["is_synthetic"], "is_synthetic")

    origin_series = None
    if "data_origin" in df.columns:
        origin_series = df["data_origin"].astype(str).str.strip().str.lower()
        invalid_origin = df["data_origin"].isna() | (~origin_series.isin(VALID_DATA_ORIGINS))
        if invalid_origin.any():
            bad_org = df.loc[invalid_origin, "data_origin"].unique().tolist()[:5]
            raise ValueError(
                f"Invalid 'data_origin' values: {bad_org}. "
                f"Expected one of {VALID_DATA_ORIGINS}."
            )

    if is_synth_series is not None and origin_series is not None:
        mismatch = (is_synth_series & (origin_series != "synthetic")) | (
            (~is_synth_series) & (origin_series == "synthetic")
        )
        if mismatch.any():
            raise ValueError(
                "Inconsistent synthetic metadata: 'is_synthetic' and 'data_origin' disagree."
            )

    if is_synth_series is not None and (is_synth_series & (sources == "human_annotated")).any():
        raise ValueError(
            "Inconsistent provenance: 'is_synthetic=True' cannot be marked as 'human_annotated'."
        )
    if origin_series is not None and ((origin_series == "synthetic") & (sources == "human_annotated")).any():
        raise ValueError(
            "Inconsistent provenance: data_origin='synthetic' cannot be marked as 'human_annotated'."
        )

    if require_human_annotated:
        if (sources != "human_annotated").any():
            raise ValueError(
                "Training rejected: dataset contains non-human-annotated rows "
                "('synthetic_bootstrap'). Keep synthetic bootstrap data separate "
                "from genuine human-reviewed labels."
            )
        if is_synth_series is not None and is_synth_series.any():
            raise ValueError(
                "Training rejected: dataset has is_synthetic=True rows. "
                "Genuine human-reviewed labels are required."
            )
        if origin_series is not None and (origin_series == "synthetic").any():
            raise ValueError(
                "Training rejected: dataset has data_origin='synthetic' rows. "
                "Genuine human-reviewed labels are required."
            )

    return labels


def validate_raw_meta_dataset(
    df: pd.DataFrame,
    require_labels: bool = True,
    require_human_annotated: bool = True,
) -> pd.DataFrame:
    """
    Validate a raw paired (text + account metadata) Meta-Dataset DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Paired input rows.
    require_labels : bool
        When True, verifies 'moderation_label' and 'label_source' are present
        and contain valid values without blanks.
    require_human_annotated : bool
        When True (default for genuine training), rejects any row whose
        'label_source' is not 'human_annotated' or has synthetic metadata flags.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("Raw Meta-Dataset cannot be empty.")

    _check_missing_columns(df, RAW_REQUIRED_COLUMNS, "Raw Meta-Dataset")

    if df["case_id"].isna().any() or (df["case_id"].astype(str).str.strip() == "").any():
        raise ValueError("Raw Meta-Dataset contains empty or missing 'case_id' values.")
    if df["case_id"].duplicated().any():
        dupes = df.loc[df["case_id"].duplicated(), "case_id"].tolist()[:5]
        raise ValueError(f"Duplicate 'case_id' entries detected: {dupes}")
    if df["text"].isna().any():
        raise ValueError("Raw Meta-Dataset contains missing 'text' values.")

    if require_labels:
        _validate_labels_and_provenance(
            df,
            require_human_annotated=require_human_annotated,
            context="Labeled Meta-Dataset",
        )

    return df


def validate_meta_features_dataset(
    df: pd.DataFrame,
    require_labels: bool = True,
    require_human_annotated: bool = True,
    require_all_classes: bool = False,
) -> pd.DataFrame:
    """
    Validate an extracted Meta-Feature dataset prior to training or evaluation.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("Extracted Meta-Dataset cannot be empty.")

    _check_missing_columns(df, ("case_id",) + META_FEATURE_COLUMNS, "Meta-Feature Dataset")

    feat_values = df[list(META_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if np.isnan(feat_values).any() or np.isinf(feat_values).any():
        raise ValueError("Meta-feature columns contain NaN or infinite values.")
    if (feat_values < 0.0).any() or (feat_values > 1.0).any():
        raise ValueError("All meta-feature probabilities/confidences must lie in [0.0, 1.0].")

    if require_labels:
        labels = _validate_labels_and_provenance(
            df,
            require_human_annotated=require_human_annotated,
            context="Labeled Meta-Feature Dataset",
        )

        if require_all_classes:
            present = set(labels.unique())
            missing_classes = [c for c in META_CLASSES if c not in present]
            if missing_classes:
                raise ValueError(
                    f"Dataset does not contain all 3 required classes; missing: {missing_classes}."
                )

    return df


def create_annotation_template(output_path: str | None = None) -> pd.DataFrame:
    """
    Create an empty human-annotation template DataFrame with all documented
    schema columns (and optionally write it to CSV).
    """
    template_df = pd.DataFrame(columns=list(ANNOTATION_TEMPLATE_COLUMNS))
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        template_df.to_csv(output_path, index=False)
    return template_df
