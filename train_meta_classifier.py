"""
train_meta_classifier.py
------------------------
Reproducible training and evaluation entrypoint for Agent 3's Logistic
Regression Meta-Classifier.

Safety rules enforced by this script:
  1. Refuses to run when no valid labeled Meta-Dataset CSV is provided.
  2. Refuses to train on 'synthetic_bootstrap' labels unless --allow-synthetic
     is explicitly passed for experimental dry runs.
  3. Uses leakage-safe grouped stratified splitting when source_text_id /
     source_account_id are available, and fits StandardScaler + LogisticRegression
     strictly on the training split.
  4. Never modifies or retrains Agent 1 or Agent 2.

Usage:
  python train_meta_classifier.py --input Data/processed/meta_dataset_labeled.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from src.data.meta_dataset import (
    validate_meta_features_dataset,
    validate_raw_meta_dataset,
)
from src.evaluation.compare_models import compare_rule_vs_logreg, evaluate_meta_classifier
from src.features.meta_features import build_meta_features_dataframe
from src.models.logistic_regression import (
    DEFAULT_META_MODEL_PATH,
    META_FEATURE_COLUMNS,
    MetaLogisticRegression,
)
from src.moderation.decision_engine import SupervisorAgent


def _split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Meta-Dataset into train and test without source-entity leakage.
    """
    y = df["moderation_label"].astype(str).str.strip().str.upper().to_numpy()

    # If distinct provenance group IDs exist, prefer StratifiedGroupKFold so no
    # source article or account appears in both train and test.
    if "source_text_id" in df.columns and "source_account_id" in df.columns:
        groups = (
            df["source_text_id"].astype(str) + "::" + df["source_account_id"].astype(str)
        ).to_numpy()
        n_groups = len(np.unique(groups))
        n_splits = max(2, int(round(1.0 / test_size)))
        if n_groups >= n_splits and n_groups < len(df):
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            train_idx, test_idx = next(sgkf.split(df, y, groups))
            return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, y))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _evaluate_split(
    model: MetaLogisticRegression,
    split_df: pd.DataFrame,
    rule_supervisor: SupervisorAgent,
) -> tuple[dict, dict]:
    """Evaluate fitted MetaLogisticRegression and Rule-Based baseline on one split."""
    X_split = split_df[list(META_FEATURE_COLUMNS)]
    y_true = split_df["moderation_label"].astype(str).str.strip().str.upper().tolist()

    y_pred = model.predict(X_split).tolist()
    y_proba = model.predict_proba(X_split)

    rule_preds = []
    for _, row in split_df.iterrows():
        p_fake = float(row["fake_probability"])
        p_bot = float(row["bot_probability"])
        fn_out = {
            "agent": "fake_news",
            "prediction": "fake" if p_fake >= 0.5 else "real",
            "confidence": max(p_fake, 1.0 - p_fake),
            "scores": {"fake": p_fake, "real": 1.0 - p_fake},
            "raw_class": 0 if p_fake >= 0.5 else 1,
        }
        bot_out = {
            "agent": "bot_detection",
            "prediction": "bot" if p_bot >= 0.5 else "human",
            "confidence": max(p_bot, 1.0 - p_bot),
            "scores": {"bot": p_bot, "human": 1.0 - p_bot},
            "raw_class": 1 if p_bot >= 0.5 else 0,
        }
        rule_preds.append(rule_supervisor.evaluate(fn_out, bot_out)["decision"])

    metrics = evaluate_meta_classifier(y_true, y_pred, y_proba=y_proba)
    comparison = compare_rule_vs_logreg(
        y_true=y_true,
        rule_preds=rule_preds,
        logreg_preds=y_pred,
        logreg_proba=y_proba,
    )
    return metrics, comparison


def train_from_dataframe(
    df: pd.DataFrame,
    output_model_path: str = DEFAULT_META_MODEL_PATH,
    test_size: float = 0.20,
    val_size: float = 0.0,
    random_state: int = 42,
    allow_synthetic: bool = False,
    C: float = 1.0,
    fake_news_detector=None,
    bot_detector=None,
    save_artifact: bool = True,
) -> dict:
    """
    Validate a labeled Meta-Dataset (either already extracted meta-features or
    raw paired records), train MetaLogisticRegression on Train only, evaluate
    on optional Validation and held-out Test splits against the Rule-Based
    baseline, and optionally save the artifact.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
    if not (0.0 <= val_size < 1.0) or (test_size + val_size) >= 1.0:
        raise ValueError(
            f"val_size and test_size must be non-negative and sum to < 1.0 "
            f"(got val_size={val_size}, test_size={test_size})."
        )

    has_meta_cols = all(col in df.columns for col in META_FEATURE_COLUMNS)
    require_human = not allow_synthetic

    if has_meta_cols:
        meta_df = validate_meta_features_dataset(
            df,
            require_labels=True,
            require_human_annotated=require_human,
            require_all_classes=True,
        )
    else:
        validate_raw_meta_dataset(
            df,
            require_labels=True,
            require_human_annotated=require_human,
        )
        meta_df = build_meta_features_dataframe(
            df,
            fake_news_detector=fake_news_detector,
            bot_detector=bot_detector,
            require_labels=True,
            require_human_annotated=require_human,
        )
        validate_meta_features_dataset(
            meta_df,
            require_labels=True,
            require_human_annotated=require_human,
            require_all_classes=True,
        )

    train_val_df, test_df = _split_train_test(
        meta_df, test_size=test_size, random_state=random_state
    )

    val_df = None
    if val_size > 0.0:
        rel_val_size = val_size / (1.0 - test_size)
        train_df, val_df = _split_train_test(
            train_val_df, test_size=rel_val_size, random_state=random_state + 1
        )
    else:
        train_df = train_val_df

    X_train = train_df[list(META_FEATURE_COLUMNS)]
    y_train = train_df["moderation_label"].astype(str).str.strip().str.upper().tolist()

    model = MetaLogisticRegression(C=C, random_state=random_state)
    model.fit(X_train, y_train)

    rule_supervisor = SupervisorAgent()
    test_metrics, test_comparison = _evaluate_split(model, test_df, rule_supervisor)

    val_metrics = None
    val_comparison = None
    if val_df is not None:
        val_metrics, val_comparison = _evaluate_split(model, val_df, rule_supervisor)

    if save_artifact:
        model.save(output_model_path)

    is_synth = bool(
        allow_synthetic
        or (meta_df["label_source"].astype(str).str.strip().str.lower() != "human_annotated").any()
        or ("is_synthetic" in meta_df.columns and meta_df["is_synthetic"].astype(bool).any())
        or (
            "data_origin" in meta_df.columns
            and (meta_df["data_origin"].astype(str).str.strip().str.lower() == "synthetic").any()
        )
    )

    return {
        "n_total": len(meta_df),
        "n_train": len(train_df),
        "n_val": len(val_df) if val_df is not None else 0,
        "n_test": len(test_df),
        "is_synthetic": is_synth,
        "evaluation_scope": "SYNTHETIC" if is_synth else "HUMAN_ANNOTATED",
        "label_sources": sorted(meta_df["label_source"].astype(str).unique().tolist()),
        "data_origins": (
            sorted(meta_df["data_origin"].astype(str).unique().tolist())
            if "data_origin" in meta_df.columns
            else []
        ),
        "val_metrics": val_metrics,
        "val_comparison": val_comparison,
        "metrics": test_metrics,
        "comparison": test_comparison,
        "model_path": output_model_path if save_artifact else None,
    }


def train_from_csv(
    input_csv_path: str,
    output_model_path: str = DEFAULT_META_MODEL_PATH,
    test_size: float = 0.20,
    val_size: float = 0.0,
    random_state: int = 42,
    allow_synthetic: bool = False,
    C: float = 1.0,
) -> dict:
    """Load CSV from disk, validate strictly, and run train_from_dataframe()."""
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(
            f"Meta-Dataset CSV not found at '{input_csv_path}'. "
            "No valid labeled Meta-Dataset exists yet; annotate paired cases "
            "before running training."
        )
    df = pd.read_csv(input_csv_path)
    return train_from_dataframe(
        df,
        output_model_path=output_model_path,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        allow_synthetic=allow_synthetic,
        C=C,
        save_artifact=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Part 3 Logistic Regression Meta-Classifier on labeled Meta-Dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to human-annotated Meta-Dataset CSV (raw paired or extracted meta-features).",
    )
    parser.add_argument(
        "--output-model",
        default=DEFAULT_META_MODEL_PATH,
        help=f"Destination path for trained joblib model (default: {DEFAULT_META_MODEL_PATH}).",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Explicit opt-in for experimental runs on synthetic_bootstrap data only.",
    )
    args = parser.parse_args()

    try:
        report = train_from_csv(
            input_csv_path=args.input,
            output_model_path=args.output_model,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            allow_synthetic=args.allow_synthetic,
            C=args.C,
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"[train_meta_classifier] Aborted: {exc}", file=sys.stderr)
        return 1

    scope_tag = "[SYNTHETIC EVALUATION - PROOF OF CONCEPT ONLY]" if report["is_synthetic"] else "[LABELED EVALUATION]"
    print(
        f"[train_meta_classifier] {scope_tag} Trained on {report['n_train']} samples, "
        f"validated on {report['n_val']}, evaluated on {report['n_test']} unseen test samples."
    )
    print(f"  Accuracy : {report['metrics']['accuracy']:.4f}")
    print(f"  Macro F1 : {report['metrics']['macro_f1']:.4f}")
    if report["metrics"].get("roc_auc_ovr_macro") is not None:
        print(f"  ROC-AUC  : {report['metrics']['roc_auc_ovr_macro']:.4f}")
    print(f"  Saved to : {report['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
