# Social Threat Intelligence
### Multi-Agent System for Fake News Detection, Bot Detection, and Automated Content Moderation

## 1. Project Overview

Social Threat Intelligence is an intelligent multi-agent system designed to identify potential threats and suspicious activities on social media platforms.

The system combines multiple machine learning agents to analyze social media content and account behavior, then integrates their outputs to support automated moderation decisions.

Instead of relying on a single detection model, the platform uses three specialized agents:

- **Agent 1 — Fake News Detection:** Analyzes textual content to identify potentially misleading or false information.
- **Agent 2 — Bot Detection:** Analyzes account behavior and metadata to identify potentially automated accounts.
- **Agent 3 — Decision & Moderation:** Combines the outputs of the first two agents to determine an appropriate moderation action.

The system also includes a human review mechanism for cases requiring additional verification.

## 2. Project Objectives

The main objectives are:

- Detect potentially fake or misleading news.
- Identify suspicious automated social media accounts.
- Combine independent detection results into a unified decision.
- Reduce the risk of incorrect automated moderation.
- Escalate uncertain cases to human reviewers.
- Provide an integrated dashboard for analyzing posts and monitoring moderation decisions.
- Build a modular architecture that allows individual agents to be improved independently.

## 3. System Architecture

The project follows a multi-agent architecture.

```text
                  Social Media Post
                          |
                          v
                Input Validation
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Agent 1                  Agent 2
    Fake News Detection       Bot Detection
              |                       |
              v                       v
       Fake Probability         Bot Probability
              |                       |
              +-----------+-----------+
                          |
                          v
                       Agent 3
                Decision & Moderation
                          |
                          v
                 Decision Engine
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
           APPROVE      REVIEW     RESTRICT
                          |
                          v
                 Human Review Queue
```

### Main Components

1. **Input Validation:** Checks submitted text and account metadata before analysis.
2. **Detection Agents:** Analyze the content and account characteristics independently.
3. **Meta-Decision Agent:** Integrates the detection outputs.
4. **Moderation Engine:** Applies decision policies and safety constraints.
5. **Human Review Queue:** Stores cases that require manual intervention.
6. **Streamlit Dashboard:** Provides the user interface.

## 4. Intelligent Agents

### Agent 1 — Fake News Detection

**Objective:** Analyze textual content and estimate whether a submitted statement is potentially fake or real.

**Technique:**
- Multinomial Naive Bayes
- TF-IDF (Term Frequency–Inverse Document Frequency)

**Workflow:**

1. Receive the submitted text.
2. Transform the text using the TF-IDF vectorizer.
3. Pass the resulting representation to the Naive Bayes classifier.
4. Generate class probabilities.
5. Return the predicted class and confidence information.

**Outputs:**
- Fake News Probability
- Real News Probability
- Predicted Class
- Confidence Score

### Agent 2 — Bot Detection

**Objective:** Analyze social media account characteristics to estimate whether an account is human-operated or automated.

**Technique:**
- GRU (Gated Recurrent Unit)
- PyTorch
- StandardScaler
- Feature Engineering

The agent uses 15 engineered features derived from account metadata and activity information.

**Feature Categories:**
- Followers and friends statistics.
- Account activity indicators.
- Account age.
- Profile configuration.
- Verification and geographic settings.
- Temporal activity information.
- Username and description characteristics.

**Workflow:**

1. Receive account metadata.
2. Extract the required features.
3. Apply the saved scaler.
4. Pass the feature vector to the GRU model.
5. Generate bot and human probabilities.
6. Return the predicted class and confidence score.

**Outputs:**
- Bot Probability
- Human Probability
- Predicted Class
- Confidence Score

### Agent 3 — Decision & Moderation Agent

**Objective:** Integrate the results of Agent 1 and Agent 2 to determine an appropriate moderation decision.

**Technique:**
- Logistic Regression
- Meta-Feature Engineering
- Multi-class Classification

Agent 3 does not independently analyze the original post. Instead, it uses information produced by the two detection agents.

**Meta-Features:**

1. `fake_probability`
2. `bot_probability`
3. `fake_news_confidence`
4. `bot_confidence`
5. `min_agent_confidence`
6. `threat_interaction`

These features represent the individual detection probabilities, their confidence levels, and the interaction between the two threat signals.

**Possible Decisions:**

| Decision | Description |
|---|---|
| APPROVE | The content can proceed without additional moderation. |
| REVIEW | The case requires human verification. |
| RESTRICT | The case meets the conditions for a restriction decision. |

Agent 3 works alongside moderation guardrails to prevent inappropriate decisions in uncertain or conflicting situations.

## 5. Moderation and Safety Mechanisms

The system does not assume that every bot publishes fake news or that every human account publishes real information.

It distinguishes between:

- Bot + Real News
- Human + Fake News
- Bot + Fake News
- Human + Real News

The moderation system also considers confidence levels and threat interactions.

Additional mechanisms include:

- Confidence-based escalation.
- Rule-based moderation policies.
- Human review for uncertain cases.
- Protection against automatic restriction in conflicting detection scenarios.

A Rule-Based Supervisor is retained as a separate decision baseline.

## 6. Human Review System

The Human Review Queue handles cases requiring human intervention.

It supports:

- Adding cases requiring review.
- Preserving the outputs of all detection agents.
- Recording the original AI moderation decision.
- Allowing a reviewer to resolve a pending case.
- Keeping the human decision separate from the original AI decision.

This mechanism provides an additional layer of control over automated moderation.

## 7. Project Structure

```text
social-threat-intelligence/
│
├── app/
│   └── streamlit_app.py
│
├── Data/
│   └── processed/
│       └── meta_dataset_synthetic.csv
│
├── models/
│   ├── saved/
│   │   ├── fake_news_model.joblib
│   │   ├── bot_rnn_model.pth
│   │   └── meta_logreg_synthetic.joblib
│   │
│   └── vectorizers/
│       ├── tfidf_vectorizer.joblib
│       ├── bot_scaler.joblib
│       └── bot_input_meta.joblib
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── meta_dataset.py
│   │   └── synthetic_meta_dataset.py
│   │
│   ├── detection/
│   │   ├── fake_news_detector.py
│   │   └── bot_detector.py
│   │
│   ├── features/
│   │   ├── temporal_features.py
│   │   └── text_features.py
│   │
│   ├── models/
│   │   ├── naive_bayes.py
│   │   ├── rnn.py
│   │   └── logistic_regression.py
│   │
│   ├── moderation/
│   │   ├── pipeline.py
│   │   ├── decision_engine.py
│   │   └── policy.py
│   │
│   └── evaluation/
│       └── compare_models.py
│
├── tests/
│   ├── test_agents.py
│   ├── test_meta_classifier.py
│   ├── test_end_to_end_pipeline.py
│   ├── test_supervisor.py
│   ├── test_models.py
│   └── test_streamlit_app.py
│
├── reports/
│
├── train_bot_rnn.py
├── train_meta_classifier.py
├── main.py
├── requirements.txt
└── README.md
```

## 8. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Scikit-learn | Traditional machine learning models |
| PyTorch | GRU-based bot detection |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Streamlit | Interactive dashboard |
| Joblib | Saving and loading trained models |
| Pytest | Automated testing |
| Git & GitHub | Version control |

## 9. Installation and Execution

### Requirements

- Python 3.10 or later, subject to the installed dependency versions.
- pip
- Git

### Step 1 — Clone the Repository

```bash
git clone <REPOSITORY_URL>
cd social-threat-intelligence
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard will be available at the local URL displayed in the terminal.

### Step 4 — Run the Tests

```bash
pytest tests/
```

To run the main integration tests separately:

```bash
pytest tests/test_end_to_end_pipeline.py
pytest tests/test_meta_classifier.py
pytest tests/test_supervisor.py
pytest tests/test_agents.py
pytest tests/test_streamlit_app.py
```

## 10. Dashboard Features

The Streamlit interface provides:

- **Overview Dashboard:** General system status and activity.
- **Analyze a Post:** Submit text and account metadata for analysis.
- **Analysis Results:** View the outputs of all three agents and the final moderation decision.
- **Human Review Queue:** Inspect and resolve pending cases.
- **System Information:** View system components and operational status.

## 11. Current Implementation Status

The project currently includes:

- Three integrated detection and moderation agents.
- A complete end-to-end processing pipeline.
- Input validation for post text and account metadata.
- Confidence-based moderation guardrails.
- A rule-based supervisor baseline.
- A human review queue.
- A Streamlit dashboard.
- Automated integration and component tests.

The current implementation is a working proof of concept. Further validation using representative, independently annotated real-world data is required before making production-level performance claims.

## 12. Future Improvements

Potential future development includes:

1. Expand and improve the datasets used by the detection agents.
2. Evaluate each agent on independent and representative test datasets.
3. Improve the meta-decision model using independently reviewed moderation examples.
4. Add multilingual fake news detection.
5. Improve bot detection using richer behavioral signals.
6. Introduce explainable AI to show why a moderation decision was made.
7. Add database persistence for posts, decisions, and review history.
8. Integrate social media APIs where permitted.
9. Improve dashboard monitoring and reporting.
10. Evaluate the system under real-world conditions.

## 13. Project Goal

Social Threat Intelligence aims to provide a modular and extensible multi-agent framework for social media threat analysis.

By combining fake news detection, bot detection, and intelligent moderation, the system provides a unified approach to identifying suspicious content and supporting safer social media environments.

---

**Project:** Social Threat Intelligence  
**Architecture:** Multi-Agent Machine Learning System  

