# Social Threat Intelligence

A university project designed to detect social bots, fake news, and coordinated manipulation patterns on social media.

> **Important:** This system provides **scores and moderation recommendations only**. It does not automatically ban real users. Any real ban action must remain subject to human validation and platform rules.

---

## Project Overview

The system is composed of three main agents:

| Agent | Role | Model |
|-------|------|-------|
| **Agent 1** | Fake News Detection | Multinomial Naive Bayes + TF-IDF |
| **Agent 2** | Bot Detection | GRU-based RNN (PyTorch) |
| **Agent 3** | Moderation & Integration | Logistic Regression |

---

## Agent 1 — Fake News Detection

### Objective
Classify news articles as **Fake News** or **Real News** based on textual content.

### Dataset
Fake and Real News Dataset — 44,058 articles after cleaning (80/20 train/test split).

### Methodology
1. Data cleaning and preprocessing
2. TF-IDF feature extraction
3. Multinomial Naive Bayes classification
4. Standard evaluation metrics

### Model Performance
Accuracy: **94.38%** on the test set.

### Model Files
- `models/saved/fake_news_model.joblib`
- `models/vectorizers/tfidf_vectorizer.joblib`

---

## Agent 2 — Bot Detection

### Objective
Identify automated (bot) social media accounts using behavioural and temporal features.

### Dataset
`bot_detection_data.csv` — **50,000 tweets** with binary bot labels (0 = human, 1 = bot).  
Columns: `User ID`, `Username`, `Tweet`, `Retweet Count`, `Mention Count`, `Follower Count`, `Verified`, `Bot Label`, `Location`, `Created At`, `Hashtags`.

Place the CSV in the **project root directory** before running.

### Dataset Note
The CSV contains one tweet per user (no sequential history per account).  
A true temporal RNN would require a full per-user publication history.  
**Approach adopted:** extract 13 scalar behavioural + temporal features, then reshape into fixed-length sequences (seq_len=5) for GRU input. This limitation is documented in `src/features/temporal_features.py`.

### Features Used
| Category | Features |
|----------|----------|
| Behavioural | Retweet Count, Mention Count, Follower Count, Verified |
| Text | tweet_length, word_count, hashtag_count, has_hashtag |
| Temporal | hour_of_day, day_of_week, month, is_weekend, is_night |

### Model Architecture
- **GRU** (2 layers, hidden_size=64, dropout=0.3)
- Input shape: `(batch, seq_len=5, step_size=3)`
- Output: 2 classes (human / bot)

### Training (Google Colab)
See **[How to Train on Google Colab](#how-to-train-on-google-colab)** below.

### Integration Contract Output
```python
{
    "bot_score": 0.87,   # P(bot) in [0, 1]
    "bot_class": 1,      # 0=human, 1=bot
    "bot_label": "BOT"   # "HUMAN" or "BOT"
}
```

### Model Files (generated after training)
- `models/saved/bot_rnn_model.pth`
- `models/vectorizers/bot_scaler.joblib`
- `models/vectorizers/bot_input_meta.joblib`

---

## Agent 3 — Moderation & Integration

Receives outputs from Agent 1 and Agent 2 and produces a final risk recommendation using Logistic Regression.

Risk levels: `NORMAL` | `REVIEW` | `HIGH_RISK`

---

## Integration Contract (between all agents)

```python
{
    "text_spam_score": 0.0,   # from Agent 1 (Naive Bayes)
    "fake_news_score": 0.0,   # from Agent 1
    "bot_score":       0.0,   # from Agent 2 (RNN)
    "risk_score":      0.0,   # from Agent 3 (Logistic Regression)
    "decision":        "REVIEW"
}
```

---

## How to Train on Google Colab

> This is the recommended way to train Agent 2's RNN model (requires GPU for speed).

### Step-by-step

1. **Open the notebook**  
   Go to [Google Colab](https://colab.research.google.com/) → File → Upload notebook → select `notebooks/03_model_experiments.ipynb`.

2. **Enable GPU**  
   Runtime → Change runtime type → Hardware accelerator → **GPU** → Save.

3. **Upload the dataset**  
   In the left sidebar click the 📁 Files icon → Upload → select `bot_detection_data.csv`.

4. **Run all cells**  
   Runtime → Run all (Ctrl+F9).  
   Training takes ~3–5 minutes on a GPU with 50,000 samples.

5. **Download the model files**  
   The last cell will auto-download:
   - `bot_rnn_model.pth`
   - `bot_scaler.joblib`
   - `bot_input_meta.joblib`

6. **Place files locally**  
   Copy downloaded files to their correct folders:
   ```
   models/saved/bot_rnn_model.pth
   models/vectorizers/bot_scaler.joblib
   models/vectorizers/bot_input_meta.joblib
   ```

7. **Test the detector locally**
   ```bash
   python -c "
   from src.detection.bot_detector import BotDetector
   d = BotDetector()
   result = d.predict({'Retweet Count': 999, 'Mention Count': 50,
                        'Follower Count': 10, 'verified_int': 0,
                        'tweet_length': 15, 'word_count': 3,
                        'hashtag_count': 10, 'has_hashtag': 1,
                        'hour_of_day': 3, 'day_of_week': 6,
                        'month': 11, 'is_weekend': 1, 'is_night': 1})
   print(result)
   "
   ```

---

## Project Structure

```text
social-threat-intelligence/
├── app/
│   └── streamlit_app.py
├── models/
│   ├── saved/
│   │   ├── fake_news_model.joblib
│   │   └── bot_rnn_model.pth          ← generated after Colab training
│   └── vectorizers/
│       ├── tfidf_vectorizer.joblib
│       ├── bot_scaler.joblib           ← generated after Colab training
│       └── bot_input_meta.joblib       ← generated after Colab training
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_experiments.ipynb     ← Agent 2 training notebook (Colab)
├── reports/
│   └── technical_report.md
├── src/
│   ├── data/
│   │   ├── loader.py                  ← Agent 2: CSV loader
│   │   └── preprocessing.py           ← Agent 2: feature engineering
│   ├── detection/
│   │   ├── fake_news_detector.py      ← Agent 1
│   │   └── bot_detector.py            ← Agent 2: main inference interface
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── compare_models.py
│   ├── features/
│   │   ├── text_features.py           ← Agent 1
│   │   └── temporal_features.py       ← Agent 2: RNN feature pipeline
│   ├── models/
│   │   ├── naive_bayes.py             ← Agent 1
│   │   ├── rnn.py                     ← Agent 2: GRU model definition
│   │   └── logistic_regression.py     ← Agent 3
│   ├── moderation/
│   │   └── decision_engine.py         ← Agent 3
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py                 ← Agent 2 unit tests
├── bot_detection_data.csv             ← place dataset here (not versioned)
├── main.py
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `torch` (PyTorch) is included in requirements. For local CPU training, `pip install torch` is sufficient. For GPU training, use Google Colab.

---

## Run Unit Tests (Agent 2)

```bash
pytest tests/test_models.py -v
```

---

## Run the Full Pipeline

```bash
python main.py
```

---

## Academic Project

Developed as part of a university project in Software Engineering.
