# Social Threat Intelligence

A university project designed to detect social bots, fake news, and coordinated manipulation patterns on social media.

> **Important:** This system provides **scores and moderation recommendations only**. It does not automatically ban real users. Any real ban action must remain subject to human validation and platform rules.

---

## Project Overview

The system is composed of three main agents:

| Agent | Role | Model | Status |
|-------|------|-------|--------|
| **Agent 1** | Fake News Detection | Multinomial Naive Bayes + TF-IDF | Completed (94.4% Accuracy) |
| **Agent 2** | Bot Detection | GRU-based RNN (PyTorch) | Completed (86.6% Accuracy) |
| **Agent 3** | Moderation & Integration | Logistic Regression | Integration Phase |

---

## Agent 2 — Bot Detection (RNN)

### Objective
Identify automated (bot) social media accounts by analyzing account metadata, behavioral rates, and temporal signals.

### Dataset
`twitter_human_bots_dataset.csv` — **37,438 real accounts** (25,013 Humans, 12,425 Bots).

Columns utilized:
- **Activity & Network:** `followers_count`, `friends_count`, `statuses_count`, `favourites_count`
- **Behavioral Frequency:** `average_tweets_per_day`
- **Profile Signals:** `default_profile`, `default_profile_image`, `geo_enabled`, `verified`
- **Temporal & Lifespan:** `created_at` (hour, day of week, year), `account_age_days`
- **Metadata:** `description` length, numerical digits in `screen_name`

### Architecture & Pipeline
1. **Preprocessing:** Log1p transformations on skewed activity counts, boolean flags conversion, temporal timestamp decomposition.
2. **Standardization:** `StandardScaler` fitted strictly on training data (no data leakage).
3. **Temporal Sequence Representation:** Reshaped into a 3D tensor `(batch_size, seq_len=5, step_size=3)`.
4. **Model:** 2-layer Gated Recurrent Unit (GRU) with hidden size 64, dropout 0.2, followed by a non-linear dense head (`64 -> 32 -> 2`).
5. **Loss:** Weighted Cross-Entropy Loss to handle class balance (2:1 human-to-bot ratio).

### Performance Metrics (Test Set)
- **Accuracy:** **86.61%**
- **Human Class:** Precision: 0.91 | Recall: 0.89 | F1-score: **0.90**
- **Bot Class:** Precision: 0.79 | Recall: 0.81 | F1-score: **0.80**
- **Confusion Matrix:**
  - True Humans correctly detected: **3,345**
  - True Bots correctly detected: **1,519**

### Model Files
- Weights: `models/saved/bot_rnn_model.pth`
- Scaler: `models/vectorizers/bot_scaler.joblib`
- Input Metadata: `models/vectorizers/bot_input_meta.joblib`

---

## Integration Contract Output (Agent 2)

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

---

## Testing

Run the full Agent 2 test suite:

```bash
pytest tests/test_models.py -v
```

---

## Training Options

1. **Locally:**
   ```bash
   python train_bot_rnn.py
   ```
2. **On Google Colab:**
   Open `notebooks/03_model_experiments.ipynb`, upload `twitter_human_bots_dataset.csv`, and run all cells.
