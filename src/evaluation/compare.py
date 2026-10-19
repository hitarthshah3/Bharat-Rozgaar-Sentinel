"""Build a comparison table from metric files that actually exist."""
import json
from pathlib import Path
import pandas as pd


def main():
    files = sorted(Path("outputs/metrics").glob("*.json"))
    if not files:
        raise SystemExit("No evaluation reports found. Train and evaluate at least one model first.")
    rows = []
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"Model": report["model"], "Accuracy": report["accuracy"],
                     "Precision": report["precision"], "Recall": report["recall"],
                     "F1": report["f1"], "ROC-AUC": report["roc_auc"],
                     "PR-AUC": report.get("pr_auc", float("nan")),
                     "Threshold": report.get("threshold", 0.5)})
    table = pd.DataFrame(rows).sort_values("F1", ascending=False)
    out = Path("outputs/metrics/model_comparison.csv")
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

