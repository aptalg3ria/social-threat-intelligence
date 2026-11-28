"""
src/models/logistic_regression.py
---------------------------------
Agent 3 Meta-Classifier using Scikit-learn Logistic Regression.

Combines frozen outputs from Agent 1 (Fake News Detection) and Agent 2
(Bot Detection) into a 3-class moderation decision:
    - APPROVE
    - REVIEW
    - RESTRICT

Uses a Scikit-learn Pipeline(StandardScaler -> LogisticRegression) with
non-redundant meta-features derived from both agents' predicted probabilities.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.helpers import load_model, save_model


META_CLASSES: tuple[str, ...] = ("APPROVE", "REVIEW", "RESTRICT")
CLASS_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(META_CLASSES)}
ID_TO_CLASS: dict[int, str] = {idx: label for idx, label in enumerate(META_CLASSES)}

# Non-redundant meta-features derived from Agent 1 and Agent 2 probabilities:
# - real_probability (= 1 - fake_probability) and human_probability (= 1 - bot_probability)
#   are omitted from the linear matrix to avoid exact multicollinearity.
# - min_agent_confidence and threat_interaction allow a linear decision boundary
#   to distinguish uncertain REVIEW cases and dual high-confidence RESTRICT cases.
META_FEATURE_COLUMNS: tuple[str, ...] = (
    "fake_probability",
    "bot_probability",
    "fake_news_confidence",
    "bot_confidence",
    "min_agent_confidence",
    "threat_interaction",
)

DEFAULT_META_MODEL_PATH = "models/saved/meta_logreg_model.joblib"


def extract_meta_features_from_outputs(
    fake_news_output: dict[str, Any],
    bot_output: dict[str, Any],
) -> dict[str, float]:
    """
    Build the 6 non-redundant meta-features from frozen Agent 1 and Agent 2 outputs.
    """
    if not isinstance(fake_news_output, dict):
        raise TypeError("fake_news_output must be a dict.")
    if not isinstance(bot_output, dict):
        raise TypeError("bot_output must be a dict.")

    fn_scores = fake_news_output.get("scores") or {}
    if "fake" in fn_scores:
        p_fake = float(fn_scores["fake"])
    else:
        fn_pred = str(fake_news_output.get("prediction", "")).lower()
        fn_conf = float(fake_news_output["confidence"])
        p_fake = fn_conf if fn_pred == "fake" else (1.0 - fn_conf)

    bot_scores = bot_output.get("scores") or {}
    if "bot" in bot_scores:
        p_bot = float(bot_scores["bot"])
    elif "bot_score" in bot_output:
        p_bot = float(bot_output["bot_score"])
    else:
        bot_pred = str(bot_output.get("prediction", "")).lower()
        bot_conf = float(bot_output["confidence"])
        p_bot = bot_conf if bot_pred == "bot" else (1.0 - bot_conf)

    p_fake = float(np.clip(p_fake, 0.0, 1.0))
    p_bot = float(np.clip(p_bot, 0.0, 1.0))

    fn_conf = float(fake_news_output.get("confidence", max(p_fake, 1.0 - p_fake)))
    bot_conf = float(bot_output.get("confidence", max(p_bot, 1.0 - p_bot)))
    fn_conf = float(np.clip(fn_conf, 0.0, 1.0))
    bot_conf = float(np.clip(bot_conf, 0.0, 1.0))

    return {
        "fake_probability": round(p_fake, 6),
        "bot_probability": round(p_bot, 6),
        "fake_news_confidence": round(fn_conf, 6),
        "bot_confidence": round(bot_conf, 6),
        "min_agent_confidence": round(min(fn_conf, bot_conf), 6),
        "threat_interaction": round(p_fake * p_bot, 6),
    }


def _normalize_labels(y: Sequence[str | int]) -> np.ndarray:
    """Convert string or integer moderation labels to integer class IDs (0, 1, 2)."""
    normalized = []
    for item in y:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            idx = int(item)
            if idx not in ID_TO_CLASS:
                raise ValueError(f"Invalid integer label {idx}; expected 0, 1, or 2.")
            normalized.append(idx)
        elif isinstance(item, str):
            key = item.strip().upper()
            if key not in CLASS_TO_ID:
                raise ValueError(
                    f"Invalid string label {item!r}; expected one of {META_CLASSES}."
                )
            normalized.append(CLASS_TO_ID[key])
        else:
            raise ValueError(f"Unsupported label type: {type(item).__name__}.")
    return np.asarray(normalized, dtype=np.int64)


class MetaLogisticRegression:
    """
    Scikit-learn Pipeline (StandardScaler + LogisticRegression) meta-classifier
    for Agent 3 moderation decisions: APPROVE, REVIEW, RESTRICT.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | dict | None = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
        feature_names: tuple[str, ...] = META_FEATURE_COLUMNS,
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.feature_names = tuple(feature_names)

        self.pipeline: Pipeline = self._build_pipeline()
        self.classes_: tuple[str, ...] = META_CLASSES
        self.is_fitted_: bool = False

    def _build_pipeline(self) -> Pipeline:
        # Compatible with scikit-learn >= 1.6 through 1.9+ (uses default L2
        # regularization via l1_ratio=0.0 and multinomial loss via solver='lbfgs').
        clf = LogisticRegression(
            C=self.C,
            l1_ratio=0.0,
            solver="lbfgs",
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

    def _validate_feature_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [col for col in self.feature_names if col not in X.columns]
            if missing:
                raise ValueError(f"Missing required meta-feature columns: {missing}")
            arr = X[list(self.feature_names)].to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D feature array, got shape {arr.shape}.")
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features {self.feature_names}, "
                    f"got {arr.shape[1]}."
                )

        if arr.shape[0] == 0:
            raise ValueError("Feature matrix X cannot be empty.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("Feature matrix X contains NaN or infinite values.")
        return arr

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[str | int],
    ) -> "MetaLogisticRegression":
        """Fit the StandardScaler + LogisticRegression pipeline."""
        X_arr = self._validate_feature_matrix(X)
        y_arr = _normalize_labels(y)
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X_arr.shape[0]} vs {len(y_arr)}."
            )

        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            raise ValueError(
                "At least 2 distinct target classes are required to fit LogisticRegression."
            )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return class probabilities aligned to META_CLASSES = ('APPROVE', 'REVIEW', 'RESTRICT'),
        with shape (n_samples, 3).
        """
        if not self.is_fitted_:
            raise RuntimeError("MetaLogisticRegression must be fitted before calling predict_proba.")

        X_arr = self._validate_feature_matrix(X)
        raw_proba = self.pipeline.predict_proba(X_arr)
        clf_classes = self.pipeline.named_steps["classifier"].classes_

        if len(clf_classes) == len(META_CLASSES) and np.array_equal(clf_classes, np.array([0, 1, 2])):
            return raw_proba

        aligned = np.zeros((X_arr.shape[0], len(META_CLASSES)), dtype=np.float64)
        for col_idx, cls_id in enumerate(clf_classes):
            aligned[:, int(cls_id)] = raw_proba[:, col_idx]
        return aligned

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predicted moderation decision strings ('APPROVE', 'REVIEW', 'RESTRICT')."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.asarray([ID_TO_CLASS[int(idx)] for idx in indices], dtype=object)

    def predict_from_outputs(
        self,
        fake_news_output: dict[str, Any],
        bot_output: dict[str, Any],
    ) -> dict[str, Any]:
        """Convenience helper for single-case prediction directly from Agent 1 & Agent 2 dicts."""
        features = extract_meta_features_from_outputs(fake_news_output, bot_output)
        X_df = pd.DataFrame([features], columns=list(self.feature_names))
        proba = self.predict_proba(X_df)[0]
        pred_idx = int(np.argmax(proba))
        prediction = ID_TO_CLASS[pred_idx]
        return {
            "prediction": prediction,
            "confidence": round(float(proba[pred_idx]), 4),
            "probabilities": {
                label: round(float(proba[idx]), 4)
                for idx, label in enumerate(META_CLASSES)
            },
            "features": features,
        }

    def save(self, path: str = DEFAULT_META_MODEL_PATH) -> None:
        """Persist fitted pipeline and metadata using joblib."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted MetaLogisticRegression model.")
        payload = {
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "classes": self.classes_,
            "params": {
                "C": self.C,
                "class_weight": self.class_weight,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
            },
            "sklearn_version": sklearn.__version__,
        }
        save_model(payload, path)

    @classmethod
    def load(cls, path: str = DEFAULT_META_MODEL_PATH) -> "MetaLogisticRegression":
        """Load a saved MetaLogisticRegression model bundle."""
        payload = load_model(path)
        if not isinstance(payload, dict) or "pipeline" not in payload:
            raise ValueError(f"Invalid MetaLogisticRegression artifact at '{path}'.")

        params = payload.get("params", {})
        instance = cls(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=params.get("max_iter", 1000),
            random_state=params.get("random_state", 42),
            feature_names=tuple(payload.get("feature_names", META_FEATURE_COLUMNS)),
        )
        instance.pipeline = payload["pipeline"]
        instance.classes_ = tuple(payload.get("classes", META_CLASSES))
        instance.is_fitted_ = True
        return instance


def create_logistic_regression_model(**kwargs) -> MetaLogisticRegression:
    """Factory helper matching src/models/ module conventions."""
    return MetaLogisticRegression(**kwargs)
