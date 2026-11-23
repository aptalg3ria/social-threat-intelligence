import joblib
import os


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved successfully: {path}")


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)