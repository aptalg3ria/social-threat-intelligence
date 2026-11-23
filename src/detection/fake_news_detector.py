
import joblib


class FakeNewsDetector:

    def __init__(self):
        self.vectorizer = joblib.load(
            "models/vectorizers/tfidf_vectorizer.joblib"
        )

        self.model = joblib.load(
            "models/saved/fake_news_model.joblib"
        )

    def predict(self, text):
        X = self.vectorizer.transform([text])

        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]

        return prediction, probability
