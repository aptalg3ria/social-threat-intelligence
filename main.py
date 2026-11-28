
from src.detection.fake_news_detector import FakeNewsDetector


def main():
    detector = FakeNewsDetector()

    article = """
    The government announced a new economic policy
    to improve employment and support small businesses.
    """

    result = detector.predict(article)

    print(f"Prediction : {result['prediction'].upper()}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"P(fake)    : {result['scores']['fake']:.4f}")
    print(f"P(real)    : {result['scores']['real']:.4f}")
    print(f"Raw class  : {result['raw_class']}  (0=fake, 1=real)")


if __name__ == "__main__":
    main()
