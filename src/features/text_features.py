from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer():
    """
    Create a TF-IDF vectorizer for fake news detection.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )

    return vectorizer