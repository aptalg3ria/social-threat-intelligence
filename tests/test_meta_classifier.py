"""
tests/test_meta_classifier.py
-----------------------------
Unit and integration tests for Part 3 (Logistic Regression Meta-Classifier,
Meta-Dataset schema/validation, frozen-agent extraction, guarded training
pipeline, model comparison, and MetaSupervisorAgent).

Note: Any labeled training fixtures constructed inside tests use temporary
in-memory / tmp_path data explicitly marked as 'synthetic_bootstrap' to test
pipeline mechanics without fabricating real model training or human labels.

Run with:
  pytest tests/test_meta_classifier.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.meta_dataset import (
    ANNOTATION_TEMPLATE_COLUMNS,
    create_annotation_template,
    validate_meta_features_dataset,
    validate_raw_meta_dataset,
)
from src.data.synthetic_meta_dataset import (
    SYNTHETIC_DATASET_DEFAULT_PATH,
    TARGET_LABEL_COUNTS,
    assign_synthetic_moderation_label,
    generate_synthetic_meta_dataset,
)
from src.evaluation.compare_models import (
    compare_rule_vs_logreg,
    evaluate_meta_classifier,
)
from src.features.meta_features import build_meta_features_dataframe
from src.models.logistic_regression import (
    META_CLASSES,
    META_FEATURE_COLUMNS,
    MetaLogisticRegression,
    extract_meta_features_from_outputs,
)
from src.moderation import (
    MetaSupervisorAgent,
    ReviewQueue,
    SupervisorAgent,
)
from train_meta_classifier import train_from_csv, train_from_dataframe


# ---------------------------------------------------------------------------
# Helpers for pipeline mechanics testing (explicitly synthetic_bootstrap)
# ---------------------------------------------------------------------------

def _make_meta_feature_fixture(n_per_class: int = 12, label_source: str = "synthetic_bootstrap") -> pd.DataFrame:
    """
    Build a small separable 3-class meta-feature DataFrame strictly for unit
    testing pipeline mechanics (not real evaluation data).
    """
    rng = np.random.default_rng(42)
    rows = []

    def add_row(idx: int, p_fake: float, p_bot: float, label: str, group_id: str) -> None:
        fn_conf = max(p_fake, 1.0 - p_fake)
        bot_conf = max(p_bot, 1.0 - p_bot)
        rows.append({
            "case_id": f"{label.lower()}-{idx}",
            "source_text_id": f"txt-{group_id}",
            "source_account_id": f"acc-{group_id}",
            "fake_news_prediction": "fake" if p_fake >= 0.5 else "real",
            "bot_prediction": "bot" if p_bot >= 0.5 else "human",
            "fake_probability": round(p_fake, 4),
            "bot_probability": round(p_bot, 4),
            "fake_news_confidence": round(fn_conf, 4),
            "bot_confidence": round(bot_conf, 4),
            "min_agent_confidence": round(min(fn_conf, bot_conf), 4),
            "threat_interaction": round(p_fake * p_bot, 4),
            "moderation_label": label,
            "label_source": label_source,
            "annotator_id": "unit_test_fixture",
        })

    for i in range(n_per_class):
        # APPROVE cluster: low p_fake, low p_bot
        add_row(i, float(rng.uniform(0.02, 0.12)), float(rng.uniform(0.02, 0.12)), "APPROVE", f"A{i // 2}")
        # REVIEW cluster: single threat OR uncertain
        if i % 2 == 0:
            add_row(i, float(rng.uniform(0.86, 0.96)), float(rng.uniform(0.03, 0.12)), "REVIEW", f"R{i // 2}")
        else:
            add_row(i, float(rng.uniform(0.45, 0.58)), float(rng.uniform(0.45, 0.58)), "REVIEW", f"R{i // 2}")
        # RESTRICT cluster: high p_fake, high p_bot
        add_row(i, float(rng.uniform(0.88, 0.98)), float(rng.uniform(0.88, 0.98)), "RESTRICT", f"X{i // 2}")

    return pd.DataFrame(rows)


def _fit_demo_meta_model() -> MetaLogisticRegression:
    df = _make_meta_feature_fixture(n_per_class=15, label_source="synthetic_bootstrap")
    model = MetaLogisticRegression()
    model.fit(df[list(META_FEATURE_COLUMNS)], df["moderation_label"].tolist())
    return model


# ===========================================================================
# 1. MetaLogisticRegression Unit Tests
# ===========================================================================

class TestMetaLogisticRegression:

    def test_pipeline_uses_standard_scaler_and_logistic_regression(self):
        model = MetaLogisticRegression()
        assert isinstance(model.pipeline, Pipeline)
        assert isinstance(model.pipeline.named_steps["scaler"], StandardScaler)
        assert isinstance(model.pipeline.named_steps["classifier"], LogisticRegression)

    def test_fit_predict_predict_proba_three_classes(self):
        df = _make_meta_feature_fixture(n_per_class=12)
        model = MetaLogisticRegression()
        model.fit(df[list(META_FEATURE_COLUMNS)], df["moderation_label"].tolist())
        assert model.is_fitted_ is True

        preds = model.predict(df[list(META_FEATURE_COLUMNS)])
        proba = model.predict_proba(df[list(META_FEATURE_COLUMNS)])

        assert len(preds) == len(df)
        assert set(preds).issubset(set(META_CLASSES))
        assert proba.shape == (len(df), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_and_load_roundtrip(self, tmp_path):
        model = _fit_demo_meta_model()
        save_path = str(tmp_path / "meta_logreg.joblib")
        model.save(save_path)
        assert os.path.exists(save_path)

        loaded = MetaLogisticRegression.load(save_path)
        df = _make_meta_feature_fixture(n_per_class=4)
        X = df[list(META_FEATURE_COLUMNS)]
        assert np.array_equal(model.predict(X), loaded.predict(X))
        assert np.allclose(model.predict_proba(X), loaded.predict_proba(X))

    def test_unfitted_and_invalid_inputs_raise(self):
        model = MetaLogisticRegression()
        df = _make_meta_feature_fixture(n_per_class=3)

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(df[list(META_FEATURE_COLUMNS)])

        with pytest.raises(ValueError, match="Missing required meta-feature columns"):
            model.fit(df[["fake_probability"]], df["moderation_label"].tolist())

        bad_arr = np.array([[0.1, 0.2, np.nan, 0.8, 0.8, 0.02]])
        with pytest.raises(ValueError, match="NaN"):
            model.fit(bad_arr, ["APPROVE"])

    def test_extract_meta_features_non_redundant(self):
        fn_out = {"prediction": "fake", "confidence": 0.92, "scores": {"fake": 0.92, "real": 0.08}}
        bot_out = {"prediction": "bot", "confidence": 0.88, "scores": {"human": 0.12, "bot": 0.88}}
        feats = extract_meta_features_from_outputs(fn_out, bot_out)

        assert tuple(feats.keys()) == META_FEATURE_COLUMNS
        assert feats["fake_probability"] == pytest.approx(0.92)
        assert feats["bot_probability"] == pytest.approx(0.88)
        assert feats["min_agent_confidence"] == pytest.approx(0.88)
        assert feats["threat_interaction"] == pytest.approx(0.92 * 0.88, abs=1e-4)


# ===========================================================================
# 2. Meta-Dataset Schema, Validation & Template Tests
# ===========================================================================

class TestMetaDatasetValidation:

    def test_annotation_template_columns_and_csv(self, tmp_path):
        out_csv = str(tmp_path / "template.csv")
        df_tpl = create_annotation_template(out_csv)
        assert os.path.exists(out_csv)
        assert tuple(df_tpl.columns) == ANNOTATION_TEMPLATE_COLUMNS
        assert df_tpl.empty

    def test_rejects_synthetic_bootstrap_by_default(self):
        df = _make_meta_feature_fixture(n_per_class=4, label_source="synthetic_bootstrap")
        with pytest.raises(ValueError, match="synthetic_bootstrap"):
            validate_meta_features_dataset(df, require_labels=True, require_human_annotated=True)

        # Explicitly opting in succeeds
        validated = validate_meta_features_dataset(
            df, require_labels=True, require_human_annotated=False, require_all_classes=True
        )
        assert len(validated) == len(df)

    def test_rejects_missing_columns_duplicates_and_bad_labels(self):
        raw_df = pd.DataFrame([
            {
                "case_id": "c1",
                "text": "Sample text",
                "followers_count": 100,
                "friends_count": 50,
                "statuses_count": 200,
                "moderation_label": "APPROVE",
                "label_source": "human_annotated",
            },
            {
                "case_id": "c1",  # duplicate
                "text": "Another text",
                "followers_count": 10,
                "friends_count": 500,
                "statuses_count": 1000,
                "moderation_label": "REVIEW",
                "label_source": "human_annotated",
            },
        ])
        with pytest.raises(ValueError, match="Duplicate 'case_id'"):
            validate_raw_meta_dataset(raw_df)

        raw_df.loc[1, "case_id"] = "c2"
        raw_df.loc[1, "moderation_label"] = "DELETE_FOREVER"
        with pytest.raises(ValueError, match="Invalid or missing moderation_label"):
            validate_raw_meta_dataset(raw_df)


# ===========================================================================
# 3. Frozen-Agent Extraction Pipeline Tests
# ===========================================================================

class TestFrozenAgentFeatureExtraction:

    def test_extracts_meta_features_using_real_frozen_agents(self):
        raw_df = pd.DataFrame([
            {
                "case_id": "live-1",
                "text": "The central bank maintained interest rates following quarterly inflation data.",
                "followers_count": 640,
                "friends_count": 410,
                "statuses_count": 1850,
                "favourites_count": 1200,
                "average_tweets_per_day": 1.8,
                "account_age_days": 2400,
                "default_profile": False,
                "default_profile_image": False,
                "geo_enabled": True,
                "verified": False,
                "created_at": "2018-05-14 14:22:10",
                "description": "Journalist covering economics.",
                "screen_name": "econ_reporter",
            },
            {
                "case_id": "live-2",
                "text": "BREAKING: Secret cure doctors hide from you!! Share before deletion!",
                "followers_count": 15,
                "friends_count": 4800,
                "statuses_count": 45000,
                "favourites_count": 1,
                "average_tweets_per_day": 180.0,
                "account_age_days": 250,
                "default_profile": True,
                "default_profile_image": True,
                "geo_enabled": False,
                "verified": False,
                "created_at": "2023-11-02 03:15:00",
                "description": "",
                "screen_name": "bot_9948271",
            },
        ])

        meta_df = build_meta_features_dataframe(raw_df, require_labels=False)
        assert len(meta_df) == 2
        assert meta_df.loc[0, "fake_news_prediction"] == "real"
        assert meta_df.loc[0, "bot_prediction"] == "human"
        assert meta_df.loc[1, "fake_news_prediction"] == "fake"
        assert meta_df.loc[1, "bot_prediction"] == "bot"
        for col in META_FEATURE_COLUMNS:
            assert (meta_df[col] >= 0.0).all() and (meta_df[col] <= 1.0).all()


# ===========================================================================
# 4. Guarded Training & Evaluation Tests
# ===========================================================================

class TestGuardedTrainingAndEvaluation:

    def test_refuses_missing_csv_and_refuses_synthetic_by_default(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Meta-Dataset CSV not found"):
            train_from_csv(str(tmp_path / "does_not_exist.csv"))

        synth_df = _make_meta_feature_fixture(n_per_class=8, label_source="synthetic_bootstrap")
        with pytest.raises(ValueError, match="synthetic_bootstrap"):
            train_from_dataframe(synth_df, save_artifact=False)

    def test_training_pipeline_works_when_explicitly_permitted_in_tmp_path(self, tmp_path):
        synth_df = _make_meta_feature_fixture(n_per_class=10, label_source="synthetic_bootstrap")
        out_model = str(tmp_path / "experimental_meta.joblib")

        report = train_from_dataframe(
            synth_df,
            output_model_path=out_model,
            test_size=0.25,
            allow_synthetic=True,
            save_artifact=True,
        )
        assert os.path.exists(out_model)
        assert report["n_train"] > 0 and report["n_test"] > 0
        assert 0.0 <= report["metrics"]["macro_f1"] <= 1.0
        assert "rule_based" in report["comparison"]
        assert "logistic_regression" in report["comparison"]

    def test_evaluate_and_compare_models(self):
        y_true = ["APPROVE", "REVIEW", "RESTRICT", "REVIEW"]
        rule_preds = ["APPROVE", "REVIEW", "RESTRICT", "APPROVE"]
        lr_preds = ["APPROVE", "REVIEW", "RESTRICT", "REVIEW"]
        comp = compare_rule_vs_logreg(y_true, rule_preds, lr_preds)
        assert comp["n_samples"] == 4
        assert comp["logistic_regression"]["accuracy"] == 1.0
        assert comp["rule_based"]["false_approve_rate"] == pytest.approx(1 / 3, abs=1e-3)
        eval_only = evaluate_meta_classifier(y_true, lr_preds)
        assert eval_only["macro_f1"] == 1.0


# ===========================================================================
# 5. MetaSupervisorAgent Integration & Safety Guardrails
# ===========================================================================

class TestMetaSupervisorIntegration:

    def test_requires_fitted_or_saved_model(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MetaSupervisorAgent(model_path=str(tmp_path / "missing_meta_logreg.joblib"))

        unfitted = MetaLogisticRegression()
        with pytest.raises(ValueError, match="not fitted"):
            MetaSupervisorAgent(meta_classifier=unfitted)

    def test_safety_guardrail_bot_real_news_never_restricts(self):
        class _ForcedPredictModel:
            is_fitted_ = True

            def __init__(self, forced_label: str):
                self.forced_label = forced_label

            def predict_from_outputs(self, fn_out, bot_out):
                return {
                    "prediction": self.forced_label,
                    "confidence": 0.95,
                    "probabilities": {
                        lbl: (0.95 if lbl == self.forced_label else 0.025)
                        for lbl in META_CLASSES
                    },
                    "features": extract_meta_features_from_outputs(fn_out, bot_out),
                }

        real_bot_fn = {"agent": "fake_news", "prediction": "real", "confidence": 0.94, "scores": {"fake": 0.06, "real": 0.94}}
        real_bot_bd = {"agent": "bot_detection", "prediction": "bot", "confidence": 0.96, "scores": {"human": 0.04, "bot": 0.96}}

        # Even if LR predicts RESTRICT or APPROVE on real+bot, guardrails enforce REVIEW.
        for forced in ("RESTRICT", "APPROVE"):
            supervisor = MetaSupervisorAgent(meta_classifier=_ForcedPredictModel(forced))
            res = supervisor.evaluate(real_bot_fn, real_bot_bd)
            assert res["decision"] == "REVIEW"
            assert res["human_review_required"] is True
            assert res["meta_classifier"]["guardrail_applied"] is True

    def test_safety_guardrail_uncertain_base_prediction_routes_to_review(self):
        lr_model = _fit_demo_meta_model()
        supervisor = MetaSupervisorAgent(meta_classifier=lr_model)

        uncertain_fn = {"agent": "fake_news", "prediction": "real", "confidence": 0.58, "scores": {"fake": 0.42, "real": 0.58}}
        high_human = {"agent": "bot_detection", "prediction": "human", "confidence": 0.95, "scores": {"human": 0.95, "bot": 0.05}}

        res = supervisor.evaluate(uncertain_fn, high_human)
        assert res["decision"] == "REVIEW"
        assert res["human_review_required"] is True

    def test_end_to_end_with_frozen_agents_and_review_queue(self):
        lr_model = _fit_demo_meta_model()
        supervisor = MetaSupervisorAgent(meta_classifier=lr_model)
        queue = ReviewQueue()

        text = (
            "BREAKING: Secret government microchip inside tap water exposed!! "
            "Share before they delete this! Mainstream media refuses to report!"
        )
        bot_account = {
            "followers_count": 15,
            "friends_count": 4800,
            "statuses_count": 45000,
            "favourites_count": 1,
            "average_tweets_per_day": 180.0,
            "account_age_days": 250,
            "default_profile": True,
            "default_profile_image": True,
            "geo_enabled": False,
            "verified": False,
            "created_at": "2023-11-02 03:15:00",
            "description": "",
            "screen_name": "news_bot_9948271",
        }
        res = supervisor.analyze(text, bot_account)
        assert res["agent"] == "supervisor"
        assert res["decision"] == "RESTRICT"
        assert res["human_review_required"] is True
        assert res["meta_classifier"]["model"] == "LogisticRegression"

        case = queue.submit(res, case_id="meta-case-1")
        assert case is not None
        assert case.ai_decision == "RESTRICT"

        # Rule-Based baseline still works alongside MetaSupervisorAgent
        baseline = SupervisorAgent().analyze(text, bot_account)
        assert baseline["decision"] == "RESTRICT"


# ===========================================================================
# 6. 5,000-Row Synthetic Meta-Dataset & Proof-of-Concept Training Tests
# ===========================================================================

class TestSyntheticMetaDatasetAndTraining:

    def test_generate_synthetic_dataset_exact_shape_and_label_counts(self):
        df = generate_synthetic_meta_dataset(n_samples=5000, random_state=42)
        assert len(df) == 5000
        assert df["case_id"].nunique() == 5000
        assert df["source_text_id"].nunique() == 5000
        assert df["source_account_id"].nunique() == 5000

        counts = df["moderation_label"].value_counts().to_dict()
        assert counts == TARGET_LABEL_COUNTS
        assert counts["APPROVE"] == 2250
        assert counts["REVIEW"] == 1750
        assert counts["RESTRICT"] == 1000

    def test_synthetic_dataset_feature_ranges_confidences_and_scenarios(self):
        df = generate_synthetic_meta_dataset(n_samples=5000, random_state=42)

        for col in META_FEATURE_COLUMNS:
            assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all()

        assert (df["fake_news_confidence"] >= 0.5).all()
        assert (df["bot_confidence"] >= 0.5).all()

        expected_fn_conf = np.maximum(df["fake_probability"], 1.0 - df["fake_probability"]).round(6)
        expected_bot_conf = np.maximum(df["bot_probability"], 1.0 - df["bot_probability"]).round(6)
        expected_min_conf = np.minimum(expected_fn_conf, expected_bot_conf).round(6)
        expected_interaction = (df["fake_probability"] * df["bot_probability"]).round(6)

        assert np.allclose(df["fake_news_confidence"], expected_fn_conf, atol=1e-6)
        assert np.allclose(df["bot_confidence"], expected_bot_conf, atol=1e-6)
        assert np.allclose(df["min_agent_confidence"], expected_min_conf, atol=1e-6)
        assert np.allclose(df["threat_interaction"], expected_interaction, atol=1e-6)

        # Verify diverse scenario combinations are present
        bot_real = ((df["bot_probability"] >= 0.5) & (df["fake_probability"] < 0.5)).sum()
        human_fake = ((df["bot_probability"] < 0.5) & (df["fake_probability"] >= 0.5)).sum()
        bot_fake = ((df["bot_probability"] >= 0.5) & (df["fake_probability"] >= 0.5)).sum()
        human_real = ((df["bot_probability"] < 0.5) & (df["fake_probability"] < 0.5)).sum()
        low_conf = (df["min_agent_confidence"] < 0.70).sum()

        assert bot_real > 0 and human_fake > 0 and bot_fake > 0 and human_real > 0 and low_conf > 0
        # Bot + Real News must never receive RESTRICT under the documented policy
        assert (df.loc[(df["bot_probability"] >= 0.5) & (df["fake_probability"] < 0.5), "moderation_label"] != "RESTRICT").all()

        # Verify every row adheres strictly to assign_synthetic_moderation_label
        policy_labels = [
            assign_synthetic_moderation_label(pf, pb)
            for pf, pb in zip(df["fake_probability"], df["bot_probability"])
        ]
        assert list(df["moderation_label"]) == policy_labels

    def test_synthetic_metadata_and_strict_human_rejection(self):
        df = generate_synthetic_meta_dataset(n_samples=5000, random_state=42)
        assert (df["is_synthetic"] == True).all()
        assert (df["data_origin"] == "synthetic").all()
        assert (df["label_source"] == "synthetic_bootstrap").all()

        # Default human-only validation must reject synthetic data
        with pytest.raises(ValueError, match="synthetic_bootstrap"):
            validate_meta_features_dataset(df, require_labels=True, require_human_annotated=True)

        # Even if label_source were forged, is_synthetic/data_origin mismatch is rejected
        tampered = df.head(5).copy()
        tampered["label_source"] = "human_annotated"
        with pytest.raises(ValueError, match="Inconsistent provenance"):
            validate_meta_features_dataset(tampered, require_labels=True, require_human_annotated=True)

    def test_saved_synthetic_dataset_and_three_way_training_load(self, tmp_path):
        assert os.path.exists(SYNTHETIC_DATASET_DEFAULT_PATH), (
            f"Expected saved synthetic CSV at '{SYNTHETIC_DATASET_DEFAULT_PATH}'."
        )
        disk_df = pd.read_csv(SYNTHETIC_DATASET_DEFAULT_PATH)
        assert len(disk_df) == 5000
        assert disk_df["moderation_label"].value_counts().to_dict() == TARGET_LABEL_COUNTS
        assert (disk_df["is_synthetic"] == True).all()
        assert (disk_df["data_origin"] == "synthetic").all()

        model_out = str(tmp_path / "synthetic_meta_logreg_test.joblib")
        report = train_from_csv(
            SYNTHETIC_DATASET_DEFAULT_PATH,
            output_model_path=model_out,
            test_size=0.15,
            val_size=0.15,
            random_state=42,
            allow_synthetic=True,
        )
        assert report["n_total"] == 5000
        assert report["n_train"] == 3500
        assert report["n_val"] == 750
        assert report["n_test"] == 750
        assert report["is_synthetic"] is True
        assert report["evaluation_scope"] == "SYNTHETIC"
        assert report["metrics"]["accuracy"] >= 0.90
        assert report["metrics"]["roc_auc_ovr_macro"] is not None
        assert report["val_metrics"]["roc_auc_ovr_macro"] is not None
        assert os.path.exists(model_out)

        loaded = MetaLogisticRegression.load(model_out)
        preds = loaded.predict(disk_df[list(META_FEATURE_COLUMNS)].head(10))
        assert len(preds) == 10
        assert set(preds).issubset(set(META_CLASSES))

        # Verify persisted synthetic artifact loads cleanly as well
        assert os.path.exists("models/saved/meta_logreg_synthetic.joblib")
        persisted = MetaLogisticRegression.load("models/saved/meta_logreg_synthetic.joblib")
        assert persisted.is_fitted_ is True
