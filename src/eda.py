"""Create descriptive dataset summaries and plots from the supplied CSV."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from src.preprocessing.data import TARGET, load_dataset


def main():
    frame = load_dataset()
    output = Path("outputs/eda")
    plots = Path("outputs/plots")
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    summary = {
        "records": int(len(frame)), "columns": list(frame.columns),
        "target": TARGET,
        "class_distribution": {str(k): int(v) for k, v in frame[TARGET].value_counts().sort_index().items()},
        "blank_or_missing_per_column": {str(k): int(v) for k, v in frame.isna().sum().add(
            frame.select_dtypes(include="object").apply(lambda col: col.str.strip().eq("").sum()), fill_value=0).items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.Series(summary["class_distribution"], name="records").rename_axis("fraudulent").to_csv(output / "class_distribution.csv")
    missing = pd.Series(summary["blank_or_missing_per_column"]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    missing[missing > 0].plot.barh(ax=ax, color="#4d8065")
    ax.set_title("Blank or missing values by field")
    ax.set_xlabel("Records")
    fig.tight_layout()
    fig.savefig(plots / "missing_values.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

