"""
src/features/meta_features.py
-----------------------------
Frozen-agent inference pipeline that extracts Part 3 Meta-Dataset features
from paired (text + account metadata) inputs using Agent 1 and Agent 2 without
retraining or modifying either base model.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from src.data.meta_dataset import (
    EXTRACTED_META_DATASET_COLUMNS,
    validate_raw_meta_dataset,
)
from src.models.logistic_regression import extract_meta_features_from_outputs


def build_meta_features_dataframe(
    raw_df: pd.DataFrame,
    fake_news_detector: Any = None,
    bot_detector: Any = None,
    require_labels: bool = False,
    require_human_annotated: bool = True,
) -> pd.DataFrame:
    """
    Run frozen Agent 1 (FakeNewsDetector) and Agent 2 (BotDetector) on a
    validated paired DataFrame and return the extracted meta-feature DataFrame.
    """
    validate_raw_meta_dataset(
        raw_df,
        require_labels=require_labels,
        require_human_annotated=require_human_annotated,
    )

    if fake_news_detector is None:
        from src.detection.fake_news_detector import FakeNewsDetector
        fake_news_detector = FakeNewsDetector()
    if bot_detector is None:
        from src.detection.bot_detector import BotDetector
        bot_detector = BotDetector()

    texts = raw_df["text"].astype(str).tolist()
    fn_outputs = fake_news_detector.predict_batch(texts)
    bot_outputs = bot_detector.predict_batch(raw_df)

    rows = []
    for idx, (fn_out, bot_out) in enumerate(zip(fn_outputs, bot_outputs)):
        src_row = raw_df.iloc[idx]
        feats = extract_meta_features_from_outputs(fn_out, bot_out)
        record = {
            "case_id": src_row["case_id"],
            "source_text_id": src_row["source_text_id"] if "source_text_id" in raw_df.columns else src_row["case_id"],
            "source_account_id": src_row["source_account_id"] if "source_account_id" in raw_df.columns else src_row["case_id"],
            "fake_news_prediction": fn_out["prediction"],
            "bot_prediction": bot_out["prediction"],
            **feats,
            "moderation_label": (
                str(src_row["moderation_label"]).strip().upper()
                if "moderation_label" in raw_df.columns and pd.notna(src_row["moderation_label"])
                else ""
            ),
            "label_source": (
                str(src_row["label_source"]).strip().lower()
                if "label_source" in raw_df.columns and pd.notna(src_row["label_source"])
                else ""
            ),
            "annotator_id": (
                str(src_row["annotator_id"]).strip()
                if "annotator_id" in raw_df.columns and pd.notna(src_row["annotator_id"])
                else ""
            ),
        }
        rows.append(record)

    return pd.DataFrame(rows, columns=list(EXTRACTED_META_DATASET_COLUMNS))


def extract_and_save_meta_features(
    input_csv_path: str,
    output_csv_path: str,
    fake_news_detector: Any = None,
    bot_detector: Any = None,
    require_labels: bool = True,
    require_human_annotated: bool = True,
) -> pd.DataFrame:
    """
    Load a raw paired CSV, extract frozen-agent meta-features, and write
    the resulting Meta-Dataset CSV to output_csv_path.
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Raw paired dataset not found at: '{input_csv_path}'")

    raw_df = pd.read_csv(input_csv_path)
    meta_df = build_meta_features_dataframe(
        raw_df,
        fake_news_detector=fake_news_detector,
        bot_detector=bot_detector,
        require_labels=require_labels,
        require_human_annotated=require_human_annotated,
    )
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    meta_df.to_csv(output_csv_path, index=False)
    return meta_df
