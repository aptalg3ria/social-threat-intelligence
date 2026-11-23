from sklearn.naive_bayes import MultinomialNB


def create_naive_bayes_model():
    """
    Create a Multinomial Naive Bayes model.
    """
    model = MultinomialNB(alpha=1.0)

    return model