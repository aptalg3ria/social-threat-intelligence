from src.features.text_features import create_tfidf_vectorizer
from src.models.naive_bayes import create_naive_bayes_model


class FakeNewsDetector:

    def __init__(self):
        self.vectorizer = create_tfidf_vectorizer()
        self.model = create_naive_bayes_model()

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]

        return prediction, probability