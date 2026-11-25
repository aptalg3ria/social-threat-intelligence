# Social Threat Intelligence

A university project designed to detect social bots, fake news, and coordinated manipulation patterns on social media.

> **Important:** This system provides **scores and moderation recommendations only**. It does not automatically ban real users. Any real ban action must remain subject to human validation and platform rules.

---

## Project Overview

The system is composed of three main agents:

| Agent | Role | Model | Status | Author |
|-------|------|-------|--------|--------|
| **Agent 1** | Fake News Detection | Multinomial Naive Bayes + TF-IDF | Completed (94.38% Accuracy) | Ahlam |
| **Agent 2** | Bot Detection | GRU-based RNN (PyTorch) | Completed (86.61% Accuracy) | Zakaria |
| **Agent 3** | Moderation & Integration | Multi-signal Scoring & Decision Matrix | Integration Phase | Collaborative |

---

## Agent 1 – Fake News Detection

### Objective
Classify news articles and social media posts as Fake News or Real News based on textual content analysis.

### Dataset
- **Dataset:** Fake and Real News Dataset (balanced collection of real and manipulated articles).
- **Size:** **44,058 articles** after comprehensive cleaning and deduplication.
- **Split:** 80% training, 20% test set.

### Methodology
1. **Text Cleaning & Preprocessing:** Noise removal, tokenization, lowercase normalization, and stopword removal.
2. **Feature Extraction:** TF-IDF (*Term Frequency – Inverse Document Frequency*) vectorizer.
3. **Classifier:** Multinomial Naive Bayes (`MultinomialNB`).
4. **Validation:** Standard classification metrics (Precision, Recall, F1-score, and ROC curve analysis).

### Technologies
- Python, Scikit-learn, Pandas, NumPy, Joblib

### Performance
- **Test Accuracy:** **94.38%**
- *Note:* Performance reflects evaluation on the benchmark test split; deployment incorporates continuous recalibration on emerging narrative patterns.

### Model Files
- Model Weights: `models/saved/fake_news_model.joblib`
- Vectorizer: `models/vectorizers/tfidf_vectorizer.joblib`

### Execution (Agent 1)
```bash
python main.py
```

---

## Agent 2 — Bot Detection (RNN)

### Objective
Identify automated (bot) social media accounts by analyzing account metadata, activity frequency, profile signals, and temporal dynamics.

### Dataset
- **Dataset:** `twitter_human_bots_dataset.csv`
- **Size:** **37,438 real accounts** (25,013 Humans, 12,425 Bots — ~2:1 natural ratio).
- **Features Extracted:**
  - **Activity & Network:** `followers_count`, `friends_count`, `statuses_count`, `favourites_count`
  - **Behavioral Frequency:** `average_tweets_per_day`
  - **Profile Signals:** `default_profile`, `default_profile_image`, `geo_enabled`, `verified`
  - **Temporal & Lifespan:** `created_at` (hour, day of week, year), `account_age_days`
  - **Metadata:** `description` length, numerical digits in `screen_name`

### Architecture & Pipeline
1. **Preprocessing & Scaling:** `log1p` transformation on heavy-tailed activity counts, boolean signal mapping, and `StandardScaler` fitted strictly on training data (no data leakage).
2. **Temporal Sequence Representation:** Sliding sequence decomposition structured into 3D tensors `(batch_size, seq_len=5, step_size=3)`.
3. **Neural Network Architecture:** 2-layer Gated Recurrent Unit (**GRU**) with hidden dimension 64, dropout 0.2, feeding a non-linear dense classification head (`64 -> 32 -> 2`).
4. **Loss Function:** Class-weighted Cross-Entropy Loss to handle class imbalance without artificial distortion.

### Performance Metrics (Test Set)
- **Overall Accuracy:** **86.61%**
- **Human Class (0):** Precision: `0.91` | Recall: `0.89` | F1-score: **`0.90`**
- **Bot Class (1):** Precision: `0.79` | Recall: `0.81` | F1-score: **`0.80`**
- **Confusion Matrix:**
  - True Humans correctly classified: **3,345**
  - True Bots correctly detected: **1,519**

### Model Files
- Weights: `models/saved/bot_rnn_model.pth`
- Scaler: `models/vectorizers/bot_scaler.joblib`
- Input Metadata: `models/vectorizers/bot_input_meta.joblib`

### Usage Example (Agent 2)
```python
from src.detection.bot_detector import BotDetector

detector = BotDetector()
result = detector.predict({
    "followers_count": 12,
    "friends_count": 500,
    "statuses_count": 15000,
    "favourites_count": 2,
    "verified": False,
    "default_profile": True,
    "default_profile_image": True,
    "geo_enabled": False,
    "average_tweets_per_day": 85.0,
    "account_age_days": 150,
    "description": "",
    "screen_name": "bot94830218"
})

print(result)
# Output:
# {
#     "bot_score": 0.8942,
#     "bot_class": 1,
#     "bot_label": "BOT"
# }
```

### Execution & Testing (Agent 2)
```bash
# Run unit tests
pytest tests/test_models.py -v

# Train locally
python train_bot_rnn.py

# Train on Google Colab:
# Open notebooks/03_model_experiments.ipynb, upload twitter_human_bots_dataset.csv, and execute cells.
```

---

## Agent 3 — Moderation & Integration (In Progress)

Combines probability outputs from **Agent 1** (content credibility) and **Agent 2** (account authenticity) to generate unified threat scores and triaged moderation actions (Approve, Review, Restrict).

---

## Project Structure

```text
social-threat-intelligence/
├── app/
│   └── streamlit_app.py                # Dashboard application
├── data/                               # Dataset storage (local/cached)
├── models/
│   ├── saved/
│   │   ├── fake_news_model.joblib      # Agent 1: Naive Bayes weights
│   │   └── bot_rnn_model.pth           # Agent 2: PyTorch GRU RNN weights
│   └── vectorizers/
│       ├── tfidf_vectorizer.joblib     # Agent 1: TF-IDF feature extractor
│       ├── bot_scaler.joblib           # Agent 2: Feature standardizer
│       └── bot_input_meta.joblib       # Agent 2: Input feature mapping metadata
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Exploratory analysis
│   ├── 02_data_preprocessing.ipynb     # Preprocessing experiments
│   └── 03_model_experiments.ipynb     # Agent 2 GRU training & Colab workflow
├── reports/
│   └── technical_report.md             # Project documentation & metrics
├── src/
│   ├── data/
│   │   ├── loader.py                   # Data loading utilities
│   │   └── preprocessing.py            # Feature transformation pipelines
│   ├── detection/
│   │   ├── fake_news_detector.py       # Agent 1 inference module
│   │   └── bot_detector.py             # Agent 2 inference module
│   ├── evaluation/                     # Metric evaluation & plotting
│   ├── features/
│   │   └── temporal_features.py        # Sequence & temporal feature creation
│   ├── models/
│   │   └── rnn.py                      # PyTorch GRU neural architecture
│   ├── moderation/                     # Agent 3 integration logic
│   └── utils/
├── tests/
│   └── test_models.py                  # Unit and integration test suite
├── twitter_human_bots_dataset.csv      # Bot detection benchmark dataset
├── train_bot_rnn.py                    # Standalone Bot RNN training script
├── main.py                             # Agent 1 demo entrypoint
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aptalg3ria/social-threat-intelligence.git
   cd social-threat-intelligence
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Academic Project
Developed as part of a university project in Software Engineering / Data Intelligence.
