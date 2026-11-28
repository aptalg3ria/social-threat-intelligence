"""
src/evaluation/compare_models.py
--------------------------------
Evaluation and model-comparison utilities for Part 3 (Meta-Classifier vs
Rule-Based Supervisor baseline).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.logistic_regression import META_CLASSES


def evaluate_meta_classifier(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_proba: np.ndarray | None = None,
    labels: tuple[str, ...] = META_CLASSES,
) -> dict:
    """
    Compute classification and safety metrics for a 3-class moderation model.
    """
    y_true_arr = [str(v).strip().upper() for v in y_true]
    y_pred_arr = [str(v).strip().upper() for v in y_pred]

    results = {
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
        "macro_f1": round(
            float(f1_score(y_true_arr, y_pred_arr, labels=list(labels), average="macro", zero_division=0)),
            4,
        ),
        "weighted_f1": round(
            float(f1_score(y_true_arr, y_pred_arr, labels=list(labels), average="weighted", zero_division=0)),
            4,
        ),
        "macro_precision": round(
            float(precision_score(y_true_arr, y_pred_arr, labels=list(labels), average="macro", zero_division=0)),
            4,
        ),
        "macro_recall": round(
            float(recall_score(y_true_arr, y_pred_arr, labels=list(labels), average="macro", zero_division=0)),
            4,
        ),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=list(labels)).tolist(),
        "labels": list(labels),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=list(labels),
            zero_division=0,
            output_dict=True,
        ),
    }

    # Safety check: how often true REVIEW/RESTRICT cases were wrongly APPROVEd.
    threat_indices = [i for i, y in enumerate(y_true_arr) if y in ("REVIEW", "RESTRICT")]
    if threat_indices:
        false_approve = sum(1 for i in threat_indices if y_pred_arr[i] == "APPROVE")
        results["false_approve_rate"] = round(false_approve / len(threat_indices), 4)
    else:
        results["false_approve_rate"] = 0.0

    if y_proba is not None:
        sorted_labels = sorted(labels)
        col_order = [labels.index(lbl) for lbl in sorted_labels]
        y_proba_sorted = np.asarray(y_proba)[:, col_order]
        results["log_loss"] = round(
            float(log_loss(y_true_arr, y_proba_sorted, labels=sorted_labels)), 4
        )
        try:
            y_proba_norm = np.asarray(y_proba_sorted, dtype=np.float64)
            row_sums = y_proba_norm.sum(axis=1, keepdims=True)
            y_proba_norm = np.where(row_sums > 0, y_proba_norm / row_sums, 1.0 / len(sorted_labels))
            results["roc_auc_ovr_macro"] = round(
                float(
                    roc_auc_score(
                        y_true_arr,
                        y_proba_norm,
                        labels=sorted_labels,
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                4,
            )
            results["roc_auc_ovr_weighted"] = round(
                float(
                    roc_auc_score(
                        y_true_arr,
                        y_proba_norm,
                        labels=sorted_labels,
                        multi_class="ovr",
                        average="weighted",
                    )
                ),
                4,
            )
        except ValueError:
            results["roc_auc_ovr_macro"] = None
            results["roc_auc_ovr_weighted"] = None

    return results


def compare_rule_vs_logreg(
    y_true: Sequence[str],
    rule_preds: Sequence[str],
    logreg_preds: Sequence[str],
    logreg_proba: np.ndarray | None = None,
) -> dict:
    """
    Compare the Rule-Based Supervisor baseline against the Logistic Regression
    Meta-Classifier on the exact same held-out test cases.
    """
    if not (len(y_true) == len(rule_preds) == len(logreg_preds)):
        raise ValueError("y_true, rule_preds, and logreg_preds must have equal length.")

    rule_metrics = evaluate_meta_classifier(y_true, rule_preds)
    logreg_metrics = evaluate_meta_classifier(y_true, logreg_preds, y_proba=logreg_proba)
    agreement = sum(1 for a, b in zip(rule_preds, logreg_preds) if a == b) / len(y_true)

    return {
        "n_samples": len(y_true),
        "rule_based": rule_metrics,
        "logistic_regression": logreg_metrics,
        "decision_agreement_rate": round(float(agreement), 4),
    }
