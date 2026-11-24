
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
)


def threshold_f1(y, p):
    """
    Select threshold using validation data only.
    """

    thresholds = np.linspace(0.05, 0.95, 91)

    scores = []

    for threshold in thresholds:
        predictions = (p >= threshold).astype(int)

        score = f1_score(
            y,
            predictions,
            zero_division=0
        )

        scores.append(score)

    best_index = int(np.argmax(scores))

    return float(thresholds[best_index])


def metrics(y, p, threshold=0.5):

    y = np.asarray(y)
    p = np.asarray(p)

    predictions = (p >= threshold).astype(int)

    has_both_classes = len(np.unique(y)) > 1

    return {
        "threshold": float(threshold),

        "pr_auc": (
            float(average_precision_score(y, p))
            if has_both_classes else None
        ),

        "roc_auc": (
            float(roc_auc_score(y, p))
            if has_both_classes else None
        ),

        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),

        "brier": float(
            brier_score_loss(y, p)
        ),

        "confusion_matrix": confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        ).tolist()
    }
