
from src.detection.fake_news_detector import FakeNewsDetector


def main():
    detector = FakeNewsDetector()

    article = """
    The government announced a new economic policy
    to improve employment and support small businesses.
    """

    prediction, probability = detector.predict(article)

    if prediction == 0:
        print("Prediction: FAKE NEWS")
    else:
        print("Prediction: REAL NEWS")

    print(f"Fake News probability: {probability[0]:.2%}")
    print(f"Real News probability: {probability[1]:.2%}")


if __name__ == "__main__":
    main()
