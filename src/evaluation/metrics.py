
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred):
    """
    Evaluate the fake news detection model.
    """

    results = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),

        "recall": recall_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),

        "f1_score": f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

    return results


def print_classification_report(y_true, y_pred):
    """
    Print detailed classification metrics for Fake and Real News.
    """

    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    ))
