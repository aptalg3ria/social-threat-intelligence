# Social Threat Intelligence

A university project designed to detect social bots, fake news, and coordinated manipulation patterns on social media.

## Project Overview

The system is composed of three main agents:

- **Agent 1 – Fake News Detection:** Identifies potentially misleading news using Natural Language Processing and Machine Learning.
- **Agent 2 – Bot Detection:** Focuses on identifying automated social media accounts.
- **Agent 3 – Moderation and Integration:** Combines detection results to support content moderation.

## Agent 1 – Fake News Detection

### Objective

Classify news articles as Fake News or Real News based on their textual content.

### Dataset

The model uses the Fake and Real News Dataset, containing fake and real news articles.

After data cleaning and preprocessing, the final dataset contains **44,058 articles**.

The dataset is divided into training and testing sets.

### Methodology

1. Data cleaning and preprocessing
2. Text feature extraction using TF-IDF
3. Classification using Multinomial Naive Bayes
4. Model evaluation using standard classification metrics

### Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Joblib

### Model Performance

The model achieved **94.38% accuracy** on the test dataset (20% of the data).

This result reflects performance on this specific dataset and does not guarantee perfect detection on real-world news.

### Model Files

- `models/saved/fake_news_model.joblib`
- `models/vectorizers/tfidf_vectorizer.joblib`

### Execution

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the detection program:

```bash
python main.py
```

## Project Structure

```text
social-threat-intelligence/
├── app/
├── models/
│   ├── saved/
│   └── vectorizers/
├── notebooks/
├── reports/
├── src/
│   ├── data/
│   ├── detection/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── moderation/
│   └── utils/
├── tests/
├── main.py
└── requirements.txt
```

## Academic Project

Developed as part of a university project in Software Engineering.
