from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, average_precision_score)


def evaluate(name, y_true, probabilities, output_dir="outputs", threshold=0.5):
    y_pred = (probabilities >= threshold).astype(int)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1],
                                                         target_names=["genuine", "potentially_fraudulent"],
                                                         output_dict=True, zero_division=0),
    }
    root = Path(output_dir)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "plots").mkdir(parents=True, exist_ok=True)
    with (root / "metrics" / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    display = ConfusionMatrixDisplay(confusion_matrix=np.asarray(metrics["confusion_matrix"], dtype=int),
                                     display_labels=["Genuine", "Potentially fraudulent"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{name}: test confusion matrix")
    plt.tight_layout()
    plt.savefig(root / "plots" / f"{name}_confusion_matrix.png", dpi=160)
    plt.close()
    return metrics

