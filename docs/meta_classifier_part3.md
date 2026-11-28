# Part 3 Documentation — Logistic Regression Meta-Classifier & Meta-Dataset

## 1. Overview & Preserved Architecture
Part 3 extends the Social Threat Intelligence project without altering or retraining Agent 1 (`FakeNewsDetector`) or Agent 2 (`BotDetector`), and while preserving the Rule-Based Supervisor baseline ([`decision_engine.py`](../src/moderation/decision_engine.py)).

- **Agent 1 (Frozen):** Multinomial Naive Bayes + TF-IDF (`fake` vs `real`).
- **Agent 2 (Frozen):** PyTorch GRU (`bot` vs `human`).
- **Agent 3 (Logistic Regression Meta-Classifier):** [`src/models/logistic_regression.py`](../src/models/logistic_regression.py) (`MetaLogisticRegression`) using `Pipeline([StandardScaler(), LogisticRegression(...)])` to predict `APPROVE`, `REVIEW`, or `RESTRICT`.
- **Moderation Integration:** [`src/moderation/meta_supervisor.py`](../src/moderation/meta_supervisor.py) (`MetaSupervisorAgent`) wraps `MetaLogisticRegression` with safety guardrails, explainability, and [`ReviewQueue`](../src/moderation/human_review.py) support.

---

## 2. Non-Redundant Meta-Features

Instead of passing collinear complements (`1 - p`), `MetaLogisticRegression` uses 6 non-redundant features extracted by `extract_meta_features_from_outputs()`:

1. `fake_probability`: `P(fake)` from Agent 1 (`[0, 1]`).
2. `bot_probability`: `P(bot)` from Agent 2 (`[0, 1]`).
3. `fake_news_confidence`: `max(P(fake), P(real))` (`[0.5, 1.0]`).
4. `bot_confidence`: `max(P(bot), P(human))` (`[0.5, 1.0]`).
5. `min_agent_confidence`: `min(fake_news_confidence, bot_confidence)` — surfaces weakest-link uncertainty to separate uncertain `REVIEW` cases from high-confidence `APPROVE` / `RESTRICT`.
6. `threat_interaction`: `fake_probability * bot_probability` — separates dual-threat (`fake` + `bot`) cases from single-signal `REVIEW` cases.

---

## 3. Meta-Dataset Schema & Human Annotation Guidelines

No paired Meta-Dataset exists in the repository yet. Before training the final Meta-Classifier:

1. Use the empty CSV template [`meta_annotation_template.csv`](meta_annotation_template.csv) (or `create_annotation_template()` in [`src/data/meta_dataset.py`](../src/data/meta_dataset.py)).
2. Populate paired cases (`case_id`, `text`, account fields, and provenance IDs `source_text_id`, `source_account_id` from **unseen** posts/accounts not used to train Agent 1 or Agent 2).
3. Assign human moderation labels independently:
   - `APPROVE`: Credible content + authentic human account + low ambiguity.
   - `REVIEW`: Uncertain content/account, human posting potential fake news, or bot posting factual/real news.
   - `RESTRICT`: Clear high-severity misinformation amplified by automated bot activity (temporary restriction pending review).
4. Set `label_source = "human_annotated"` and record `annotator_id`.
   - Any experimental `synthetic_bootstrap` rows are strictly rejected by `train_meta_classifier.py` unless `--allow-synthetic` is explicitly passed.

---

## 4. Training & Evaluation Command (Once Labeled Data Exists)

1. Extract meta-features from frozen Agent 1 & Agent 2:
   - Use `extract_and_save_meta_features()` in [`src/features/meta_features.py`](../src/features/meta_features.py), or pass raw paired CSV directly to `train_meta_classifier.py`.
2. Train & evaluate (`StratifiedGroupKFold` / stratified split, scaler + classifier fitted on Train only):
   ```bash
   python train_meta_classifier.py --input Data/processed/meta_dataset_labeled.csv
   ```
3. Model artifact will be saved to `models/saved/meta_logreg_model.joblib`, enabling `MetaSupervisorAgent()` and [`src/evaluation/compare_models.py`](../src/evaluation/compare_models.py) comparisons against the Rule-Based Supervisor baseline.
