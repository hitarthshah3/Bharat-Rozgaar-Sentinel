from __future__ import annotations
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from src.preprocessing.data import TEXT_FIELDS, combine_text
from src.explainability.indicators import risk_indicators

MODEL = os.getenv("BRS_MODEL", "baseline").lower()
DEFAULT_PATHS = {
    "baseline": Path("models/tfidf_lr/model.joblib"),
    "bilstm": Path("models/word2vec_bilstm/model.pt"),
    "hybrid_bilstm": Path("models/hybrid_bilstm/model.pt"),
    "distilbert": Path("models/distilbert"),
}


def model_path():
    return Path(os.getenv("BRS_MODEL_PATH", str(DEFAULT_PATHS.get(MODEL, ""))))


def ready():
    path = model_path()
    if MODEL == "distilbert":
        return path.is_dir() and (path / "config.json").is_file() and any(
            (path / weight_name).is_file() for weight_name in ("model.safetensors", "pytorch_model.bin")
        )
    return path.is_file()


def _text(posting):
    row = {field: str(posting.get(field) or "") for field in TEXT_FIELDS}
    return combine_text(pd.DataFrame([row])).iloc[0]


@lru_cache(maxsize=1)
def _load():
    path = model_path()
    if MODEL == "baseline":
        from joblib import load
        return load(path)
    if MODEL in {"bilstm", "hybrid_bilstm"}:
        import torch
        from src.models.bilstm.model import BiLSTMClassifier
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg, vocab = payload["config"], payload["vocab"]
        model = BiLSTMClassifier(len(vocab), cfg["embedding_dim"], cfg["hidden_dim"],
                                 dropout=cfg["dropout"], metadata_dim=cfg.get("metadata_dim", 0))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, vocab, cfg, payload.get("metadata_schema")
    if MODEL == "distilbert":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        return tokenizer, model
    raise ValueError(f"Unknown model '{MODEL}'. Choose baseline, bilstm, hybrid_bilstm, or distilbert.")


def _bilstm_probability(model, vocab, cfg, text):
    import torch
    tokens = re.findall(r"(?u)\b[\w']+\b", text.lower())[:cfg["max_length"]] or ["<unk>"]
    ids = torch.tensor([[vocab.get(token, 1) for token in tokens]], dtype=torch.long)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    with torch.no_grad():
        return float(torch.sigmoid(model(ids, lengths))[0])


def _hybrid_probability(model, vocab, cfg, schema, text, posting):
    import numpy as np
    import torch
    tokens = re.findall(r"(?u)\b[\w']+\b", text.lower())[:cfg["max_length"]] or ["<unk>"]
    ids = torch.tensor([[vocab.get(token, 1) for token in tokens]], dtype=torch.long)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    columns = []
    for field, levels in schema["categorical"].items():
        value = str(posting.get(field) or "").strip() or "<MISSING>"
        if value not in levels:
            value = "<UNK>"
        columns.extend(float(value == level) for level in levels)
    for field in schema["binary"]:
        value = str(posting.get(field) or "0").strip().lower()
        columns.append(float(value in {"1", "true", "yes"}))
    metadata = torch.tensor(np.asarray([columns], dtype="float32"))
    with torch.no_grad():
        return float(torch.sigmoid(model(ids, lengths, metadata))[0])


def _saved_threshold():
    import json
    if MODEL == "baseline":
        return 0.5
    metadata_path = (model_path() / "training_metadata.json" if MODEL == "distilbert"
                     else model_path().parent / "metadata.json")
    if not metadata_path.is_file():
        return 0.5
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return float(metadata.get("validation_selected_threshold", metadata.get("threshold", 0.5)))


def predict(posting: dict) -> dict:
    if not ready():
        raise FileNotFoundError(f"Trained artifact not found for '{MODEL}': {model_path()}")
    text = _text(posting)
    artifact = _load()
    token_evidence = []
    if MODEL == "baseline":
        probability = float(artifact.predict_proba([text])[0, 1])
        from src.explainability.tfidf import explain_tfidf
        token_evidence = explain_tfidf(artifact, text)
    elif MODEL == "bilstm":
        model, vocab, cfg, _ = artifact
        probability = _bilstm_probability(model, vocab, cfg, text)
    elif MODEL == "hybrid_bilstm":
        model, vocab, cfg, schema = artifact
        probability = _hybrid_probability(model, vocab, cfg, schema, text, posting)
    else:
        import torch
        tokenizer, model = artifact
        metadata = model_path() / "training_metadata.json"
        max_length = 256
        if metadata.is_file():
            import json
            max_length = json.loads(metadata.read_text(encoding="utf-8")).get("max_length", max_length)
        model.eval()
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            probability = float(torch.softmax(model(**encoded).logits, dim=1)[0, 1])
    risk = "high" if probability >= .70 else "moderate" if probability >= .35 else "low"
    threshold = _saved_threshold()
    return {
        "assessment": "potentially_fraudulent" if probability >= threshold else "genuine",
        "decision_threshold": round(threshold, 6),
        "fraud_probability": round(probability, 6),
        "genuine_probability": round(1 - probability, 6),
        "risk_level": risk,
        "model_used": MODEL,
        "assessment_type": "AI-based risk assessment; this is not definitive proof of fraud.",
        "risk_indicators": risk_indicators(posting),
        "model_token_evidence": token_evidence,
    }

