# Social Threat Intelligence

A multi-agent system designed to detect fake news, automated social bot accounts, and coordinated manipulation patterns on social media, combining independent content and account signals into structured moderation recommendations with human-in-the-loop review.

> **Important:** This system provides **threat scores and moderation recommendations only**. It does not automatically ban real users. Flagged cases (`REVIEW` and `RESTRICT`) route to the **Human Review Queue** for human verification and accountability.

---

## Project Overview

The system integrates three specialized agents through `SocialThreatPipeline`:

| Agent | Role | Model / Implementation | Status | Author |
|---|---|---|---|---|
| **Agent 1** | Fake News Detection | Multinomial Naive Bayes + TF-IDF | Completed (`94.38%` benchmark test accuracy) | Ahlam |
| **Agent 2** | Bot Detection | PyTorch 2-layer GRU RNN + `StandardScaler` | Completed (`86.61%` benchmark test accuracy) | Zakaria |
| **Agent 3** | Decision & Moderation | Logistic Regression Meta-Classifier (`MetaSupervisorAgent`) + Rule-Based Baseline (`SupervisorAgent`) + Human Review Queue | Integrated (Meta-Classifier trained on synthetic proof-of-concept data; Rule-Based baseline preserved) | Collaborative |

---

## Agent 1 — Fake News Detection

- **Objective:** Classify news articles and social media posts as `fake` or `real` from textual content.
- **Training Dataset:** Balanced Fake and Real News Dataset (`44,058` cleaned/deduplicated articles; `80/20` split — raw dataset not redistributed in repository).
- **Pipeline:** Text cleaning (`clean_text`) → TF-IDF vectorization → Multinomial Naive Bayes (`MultinomialNB`).
- **Benchmark Accuracy:** `94.38%`.
- **Saved Artifacts:**
  - `models/saved/fake_news_model.joblib`
  - `models/vectorizers/tfidf_vectorizer.joblib`

---

## Agent 2 — Bot Detection (RNN)

- **Objective:** Identify automated accounts using network counts, behavioral frequency, profile signals, metadata, and temporal creation features (`15` extracted features).
- **Training Dataset:** `twitter_human_bots_dataset.csv` (`37,438` accounts: `25,013` Human / `12,425` Bot — raw CSV excluded from repository).
- **Architecture:** `log1p` scaling + `StandardScaler` → temporal sequence shaping `(batch_size, seq_len=5, input_size=3)` → 2-layer PyTorch GRU (`hidden_size=64`, `dropout=0.2`) → dense classifier head (`64 → 32 → 2`).
- **Benchmark Metrics:** `86.61%` accuracy (`Human F1: 0.90`, `Bot F1: 0.80`).
- **Saved Artifacts:**
  - `models/saved/bot_rnn_model.pth`
  - `models/vectorizers/bot_scaler.joblib`
  - `models/vectorizers/bot_input_meta.joblib`

---

## Agent 3 — Decision, Meta-Classifier & Human Review

Agent 3 combines the probability and confidence outputs of frozen Agent 1 and Agent 2 into three moderation actions (`APPROVE`, `REVIEW`, `RESTRICT`) **without conflating bot accounts with fake news**.

- **6 Non-Redundant Meta-Features (`src/models/logistic_regression.py`):**
  `fake_probability`, `bot_probability`, `fake_news_confidence`, `bot_confidence`, `min_agent_confidence`, `threat_interaction`.
- **Meta-Classifier (`MetaLogisticRegression` + `MetaSupervisorAgent`):**
  `Pipeline([StandardScaler(), LogisticRegression(...)])` wrapped with 5 deterministic safety guardrails (including uncertainty escalation and preventing `RESTRICT` on legitimate news posted by a bot).
- **Training Provenance Note:** No human-annotated paired Meta-Dataset exists in the repository yet. The active artifact (`models/saved/meta_logreg_synthetic.joblib`) is a **proof-of-concept** model trained on `5,000` synthetic samples generated from the documented policy in `src/data/synthetic_meta_dataset.py`. Synthetic evaluation metrics must not be presented as real-world human moderation accuracy. Once real cases are annotated using `docs/meta_annotation_template.csv`, retrain via `train_meta_classifier.py`.
- **Rule-Based Baseline (`SupervisorAgent`):** Preserved intact in `src/moderation/decision_engine.py` alongside `SupervisorPolicy` (`src/moderation/policy.py`).
- **Human Review Queue (`ReviewQueue`):** Implemented in `src/moderation/human_review.py`; queues `REVIEW` and `RESTRICT` cases and records human dispositions separately while preserving the original AI decision.

---

## Project Structure

```text
social-threat-intelligence/
├── app/
│   └── streamlit_app.py                  # Interactive 5-page Streamlit dashboard
├── docs/
│   ├── meta_annotation_template.csv      # Empty CSV schema template for human annotation
│   ├── meta_classifier_part3.md          # Agent 3 Meta-Classifier & Meta-Dataset docs
│   └── supervisor_agent.md               # Rule-based Supervisor & Human Review docs
├── models/
│   ├── saved/
│   │   ├── fake_news_model.joblib        # Agent 1: Multinomial NB weights
│   │   ├── bot_rnn_model.pth             # Agent 2: PyTorch GRU weights
│   │   └── meta_logreg_synthetic.joblib  # Agent 3: Meta-Classifier (synthetic POC)
│   └── vectorizers/
│       ├── tfidf_vectorizer.joblib       # Agent 1: TF-IDF vectorizer
│       ├── bot_scaler.joblib             # Agent 2: StandardScaler (15 features)
│       └── bot_input_meta.joblib         # Agent 2: Input metadata & feature order
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_experiments.ipynb        # Agent 2 GRU experiments & training
├── reports/
│   ├── final_system_audit.md             # Full system & UI audit report
│   ├── final_verification_report.md      # Integrity & scenario verification report
│   ├── github_release_readiness.md       # Cleanup & GitHub release checklist
│   └── technical_report.md
├── src/
│   ├── data/
│   │   ├── loader.py                     # Raw bot dataset loader (local CSV)
│   │   ├── preprocessing.py              # Text cleaning & 15 account features
│   │   ├── meta_dataset.py               # Meta-Dataset validation & schema helpers
│   │   └── synthetic_meta_dataset.py     # 5,000-row synthetic meta-dataset generator
│   ├── detection/
│   │   ├── fake_news_detector.py         # Agent 1 inference module
│   │   └── bot_detector.py               # Agent 2 inference module
│   ├── evaluation/
│   │   ├── compare_models.py             # Rule-based vs Meta-Classifier comparison
│   │   └── metrics.py                    # Classification metric helpers
│   ├── features/
│   │   ├── meta_features.py              # Frozen-agent meta-feature extraction
│   │   ├── temporal_features.py          # GRU sequence builder (seq_len=5)
│   │   └── text_features.py              # Text feature helpers
│   ├── models/
│   │   ├── logistic_regression.py        # Agent 3 MetaLogisticRegression pipeline
│   │   ├── naive_bayes.py                # Naive Bayes wrapper
│   │   └── rnn.py                        # PyTorch BotRNN (GRU) architecture
│   ├── moderation/
│   │   ├── __init__.py
│   │   ├── decision_engine.py            # Rule-based SupervisorAgent baseline
│   │   ├── human_review.py               # In-memory ReviewQueue & ReviewCase
│   │   ├── meta_supervisor.py            # MetaSupervisorAgent + safety guardrails
│   │   ├── pipeline.py                   # End-to-End SocialThreatPipeline
│   │   └── policy.py                     # Configurable SupervisorPolicy thresholds
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_agents.py                    # Agent 1 & Agent 2 unit/contract tests
│   ├── test_end_to_end_pipeline.py       # End-to-End pipeline & validation tests
│   ├── test_meta_classifier.py           # Agent 3, Meta-Dataset & training tests
│   ├── test_models.py                    # Preprocessing, GRU & dataset tests
│   ├── test_preprocessing.py             # Reserved preprocessing test file
│   ├── test_streamlit_app.py             # Streamlit AppTest UI & E2E tests
│   └── test_supervisor.py                # Rule-based supervisor & HRQ tests
├── main.py                               # Agent 1 standalone CLI demo
├── train_bot_rnn.py                      # Agent 2 training script (requires raw CSV)
├── train_meta_classifier.py              # Agent 3 training script
├── requirements.txt                      # Python dependencies
├── LICENSE
└── README.md
```

---

## Installation & Usage

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/aptalg3ria/social-threat-intelligence.git
   cd social-threat-intelligence
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit Dashboard:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Run the Agent 1 CLI Demo:**
   ```bash
   python main.py
   ```

4. **Run the Automated Test Suites:**
   ```bash
   pytest tests/test_agents.py tests/test_supervisor.py tests/test_end_to_end_pipeline.py tests/test_streamlit_app.py -v
   ```
   - To run `tests/test_meta_classifier.py`, generate the local gitignored synthetic dataset first if not already present:
     ```bash
     python -m src.data.synthetic_meta_dataset
     pytest tests/test_meta_classifier.py -v
     ```
   - Note: `tests/test_models.py` includes 5 dataset-loading tests that require placing `twitter_human_bots_dataset.csv` in the project root (not included in the repository).

5. **Train Agent 3 on Human-Annotated Data (Optional Future Step):**
   ```bash
   python train_meta_classifier.py --input Data/processed/meta_dataset_labeled.csv
   ```

---

## Academic Project

Developed as part of a university project in Software Engineering / Data Intelligence.
