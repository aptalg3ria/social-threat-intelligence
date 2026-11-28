"""
src/detection/fake_news_detector.py
------------------------------------
High-level inference interface for Fake News Detection (Agent 1).

Preprocessing note
------------------
The TF-IDF vectorizer was trained with its default token_pattern
(r'(?u)\\b\\w\\w+\\b'), which already lowercases input and strips
punctuation / special characters internally.  Applying clean_text()
before vectorization is therefore safe and strictly additive: it removes
URL fragments and @mentions that the vectorizer would otherwise tokenise
into noise tokens (e.g. 'https', 'co', usernames).

Integration contract
--------------------
{
    "agent":      "fake_news",
    "prediction": "fake" | "real",
    "confidence": float,            # P(predicted class) in [0.0, 1.0]
    "scores":     {"fake": float, "real": float},
    "raw_class":  int               # 0 = fake, 1 = real
}
"""

import os
import warnings
import joblib

from src.data.preprocessing import clean_text


# ---------------------------------------------------------------------------
# Default artifact paths (relative to project root)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH     = "models/saved/fake_news_model.joblib"
DEFAULT_VECTORIZER_PATH = "models/vectorizers/tfidf_vectorizer.joblib"


class FakeNewsDetector:
    """
    Loads the trained Multinomial Naive Bayes model and TF-IDF vectorizer,
    then provides single-text and batch inference for fake news detection.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        vectorizer_path: str = DEFAULT_VECTORIZER_PATH,
    ):
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at '{vectorizer_path}'.\n"
                "Please ensure the trained vectorizer file is present."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Fake news model not found at '{model_path}'.\n"
                "Please ensure the trained model file is present."
            )

        try:
            with warnings.catch_warnings():
                # Suppress sklearn version mismatch warnings on load; artefacts
                # produced under sklearn 1.6.x remain compatible with 1.9.x.
                warnings.simplefilter("ignore")
                self.vectorizer = joblib.load(vectorizer_path)
                self.model      = joblib.load(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FakeNewsDetector artifacts from "
                f"'{model_path}' and '{vectorizer_path}': {exc}"
            ) from exc

        print(
            f"[FakeNewsDetector] Loaded model from '{model_path}' "
            f"and vectorizer from '{vectorizer_path}'."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> str:
        """
        Apply clean_text() to strip URLs and @mentions before vectorisation.

        The TF-IDF token_pattern already handles lowercasing and punctuation,
        so clean_text() is a safe, additive hygiene step that prevents noise
        tokens (e.g. 'https', 'co', usernames) from influencing scores.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Input text must be a str, got {type(text).__name__}."
            )
        return clean_text(text)

    @staticmethod
    def _format_result(raw_class: int, proba) -> dict:
        """Format model outputs into the canonical integration schema."""
        raw_class = int(raw_class)
        fake_score = float(proba[0])
        real_score = float(proba[1])
        confidence = float(proba[raw_class])
        prediction = "real" if raw_class == 1 else "fake"
        return {
            "agent": "fake_news",
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "scores": {
                "fake": round(fake_score, 4),
                "real": round(real_score, 4),
            },
            "raw_class": raw_class,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict:
        """
        Classify a news article or social post as fake or real.

        Parameters
        ----------
        text : str
            Raw article text, headline, or social media post.

        Returns
        -------
        dict
            {
                "agent":      "fake_news",
                "prediction": "fake" | "real",
                "confidence": float,   # P(predicted class) in [0.0, 1.0]
                "scores":     {"fake": float, "real": float},
                "raw_class":  int      # 0 = fake, 1 = real
            }
        """
        cleaned = self._preprocess(text)
        X = self.vectorizer.transform([cleaned])

        raw_class = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]  # [P(fake), P(real)]
        return self._format_result(raw_class, proba)

    def predict_batch(self, texts: list) -> list:
        """
        Classify a list of texts. Returns one result dict per input.

        Parameters
        ----------
        texts : list[str]

        Returns
        -------
        list[dict]
        """
        if isinstance(texts, str):
            raise TypeError(
                "predict_batch() expects a sequence of strings, got a single str. "
                "Use predict() for a single text."
            )
        if not texts:
            return []

        cleaned_texts = [self._preprocess(t) for t in texts]
        X = self.vectorizer.transform(cleaned_texts)

        raw_classes = self.model.predict(X)
        probas = self.model.predict_proba(X)

        return [
            self._format_result(raw_class, proba)
            for raw_class, proba in zip(raw_classes, probas)
        ]


def detect_fake_news(text: str, model_path: str = DEFAULT_MODEL_PATH,
                     vectorizer_path: str = DEFAULT_VECTORIZER_PATH) -> dict:
    """Convenience one-shot helper for fake news prediction."""
    detector = FakeNewsDetector(
        model_path=model_path,
        vectorizer_path=vectorizer_path,
    )
    return detector.predict(text)

