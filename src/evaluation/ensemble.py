"""Evaluate the equal-weight three-model ensemble and save its inference config."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve

from src.evaluation.metrics import evaluate
from src.models.bilstm.model import BiLSTMClassifier
from src.preprocessing.data import TARGET, combine_text, load_dataset
from src.train import partition

ROOT = Path(__file__).resolve().parents[2]
TEXT_TOKEN = re.compile(r"(?u)\b[\w']+\b")
MEMBERS = ("baseline", "bilstm", "distilbert")
WEIGHTS = {name: 1 / len(MEMBERS) for name in MEMBERS}


def _baseline_probabilities(texts: list[str]) -> np.ndarray:
    from joblib import load

    model = load(ROOT / "models" / "tfidf_lr" / "model.joblib")
    return model.predict_proba(texts)[:, 1]


def _bilstm_probabilities(texts: list[str], batch_size: int = 64) -> np.ndarray:
    payload = torch.load(ROOT / "models" / "word2vec_bilstm" / "model.pt",
                         map_location="cpu", weights_only=False)
    cfg, vocab = payload["config"], payload["vocab"]
    model = BiLSTMClassifier(len(vocab), cfg["embedding_dim"], cfg["hidden_dim"],
                             dropout=cfg["dropout"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    max_length = int(cfg["max_length"])
    encoded = [([vocab.get(t, 1) for t in TEXT_TOKEN.findall(text.lower())[:max_length]] or [1])
               for text in texts]
    ids = np.zeros((len(encoded), max_length), dtype=np.int64)
    lengths = np.empty(len(encoded), dtype=np.int64)
    for i, sequence in enumerate(encoded):
        clipped = sequence[:max_length]
        ids[i, :len(clipped)] = clipped
        lengths[i] = len(clipped)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            logits = model(torch.from_numpy(ids[start:end]), torch.from_numpy(lengths[start:end]))
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def _distilbert_probabilities(texts: list[str], batch_size: int = 48) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = ROOT / "models" / "distilbert"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    metadata_path = path / "training_metadata.json"
    max_length = json.loads(metadata_path.read_text(encoding="utf-8")).get("max_length", 64)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, truncation=True, padding="max_length",
                                max_length=max_length, return_tensors="pt")
            logits = model(**encoded).logits
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(texts):
                print(f"DistilBERT scored {min(start + batch_size, len(texts))}/{len(texts)} rows", flush=True)
    return np.asarray(scores, dtype=np.float64)


def _select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        threshold = 0.5
    else:
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best = np.flatnonzero(np.isclose(f1, f1.max()))
        threshold = float(thresholds[best[-1]])
    return threshold, float(f1_score(y_true, probabilities >= threshold, zero_division=0))


def main() -> None:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    frame = load_dataset()
    train_idx, val_idx, test_idx = partition(frame)
    all_texts = combine_text(frame).tolist()
    y = frame[TARGET].to_numpy()
    scored_indices = np.concatenate((val_idx, test_idx))
    scored_texts = [all_texts[i] for i in scored_indices]
    split_at = len(val_idx)

    predictions = {}
    for name, predict in (
        ("baseline", _baseline_probabilities),
        ("bilstm", _bilstm_probabilities),
        ("distilbert", _distilbert_probabilities),
    ):
        print(f"Scoring {name} on validation and held-out test rows…", flush=True)
        all_probabilities = predict(scored_texts)
        predictions[name] = (all_probabilities[:split_at], all_probabilities[split_at:])

    val_probability = sum(WEIGHTS[name] * predictions[name][0] for name in MEMBERS)
    test_probability = sum(WEIGHTS[name] * predictions[name][1] for name in MEMBERS)
    threshold, validation_f1 = _select_f1_threshold(y[val_idx], val_probability)

    config = {
        "model": "ensemble",
        "members": list(MEMBERS),
        "weights": WEIGHTS,
        "method": "equal-weight soft voting",
        "decision_threshold": threshold,
        "threshold_selection": "maximum validation F1; held-out test split untouched",
        "validation_f1": validation_f1,
        "seed": 42,
    }
    config_path = ROOT / "models" / "ensemble" / "ensemble_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    report = evaluate("ensemble", y[test_idx], test_probability, threshold=threshold)
    report["ensemble_method"] = config["method"]
    report["members"] = list(MEMBERS)
    report["weights"] = WEIGHTS
    report["validation_f1_at_selected_threshold"] = validation_f1
    report_path = ROOT / "outputs" / "metrics" / "ensemble.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    for index, row_index in enumerate(test_idx):
        rows.append({
            "job_id": int(frame.iloc[row_index]["job_id"]),
            "fraudulent": int(y[row_index]),
            **{f"{name}_probability": float(predictions[name][1][index]) for name in MEMBERS},
            "ensemble_probability": float(test_probability[index]),
        })
    output = ROOT / "outputs" / "predictions" / "ensemble_test_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(output, index=False)

    print(json.dumps({
        "model": "ensemble",
        "method": config["method"],
        "weights": WEIGHTS,
        "decision_threshold": threshold,
        "validation_f1": validation_f1,
        "test_accuracy": report["accuracy"],
        "test_precision": report["precision"],
        "test_recall": report["recall"],
        "test_f1": report["f1"],
        "test_roc_auc": report["roc_auc"],
        "test_pr_auc": report["pr_auc"],
    }, indent=2))
    print(f"Saved inference config: {config_path}")
    print(f"Saved held-out metrics: {report_path}")
    print(f"Saved held-out component scores: {output}")


if __name__ == "__main__":
    main()
