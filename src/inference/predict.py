from __future__ import annotations
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from src.preprocessing.data import TEXT_FIELDS, combine_text
from src.explainability.indicators import risk_indicators

MODEL = os.getenv("BRS_MODEL", "ensemble").lower()
ENSEMBLE_MEMBERS = ("baseline", "bilstm", "distilbert")
MODEL_LABELS = {
    "baseline": "TF-IDF + Logistic Regression",
    "bilstm": "Word2Vec + BiLSTM",
    "distilbert": "DistilBERT",
}
DEFAULT_PATHS = {
    "baseline": Path("models/tfidf_lr/model.joblib"),
    "bilstm": Path("models/word2vec_bilstm/model.pt"),
    "hybrid_bilstm": Path("models/hybrid_bilstm/model.pt"),
    "distilbert": Path("models/distilbert"),
}


def model_path(model: str | None = None):
    name = model or MODEL
    if name == "ensemble":
        return {**{member: model_path(member) for member in ENSEMBLE_MEMBERS},
                "config": Path(os.getenv("BRS_ENSEMBLE_CONFIG_PATH", "models/ensemble/ensemble_config.json"))}
    configured = os.getenv(f"BRS_{name.upper()}_MODEL_PATH")
    if not configured and name == MODEL:
        configured = os.getenv("BRS_MODEL_PATH")
    return Path(configured or DEFAULT_PATHS.get(name, ""))


def _artifact_ready(name: str):
    path = model_path(name)
    if name == "distilbert":
        return path.is_dir() and (path / "config.json").is_file() and any(
            (path / weight_name).is_file() for weight_name in ("model.safetensors", "pytorch_model.bin")
        )
    return path.is_file()


def ready():
    if MODEL == "ensemble":
        config_path = model_path("ensemble")["config"]
        return config_path.is_file() and all(_artifact_ready(name) for name in ENSEMBLE_MEMBERS)
    return MODEL in {*ENSEMBLE_MEMBERS, "hybrid_bilstm"} and _artifact_ready(MODEL)


def _text(posting):
    row = {field: str(posting.get(field) or "") for field in TEXT_FIELDS}
    return combine_text(pd.DataFrame([row])).iloc[0]


@lru_cache(maxsize=5)
def _load(model_name: str | None = None):
    name = model_name or MODEL
    path = model_path(name)
    if name == "baseline":
        from joblib import load
        return load(path)
    if name in {"bilstm", "hybrid_bilstm"}:
        import torch
        from src.models.bilstm.model import BiLSTMClassifier
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg, vocab = payload["config"], payload["vocab"]
        model = BiLSTMClassifier(len(vocab), cfg["embedding_dim"], cfg["hidden_dim"],
                                 dropout=cfg["dropout"], metadata_dim=cfg.get("metadata_dim", 0))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, vocab, cfg, payload.get("metadata_schema")
    if name == "distilbert":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        return tokenizer, model
    raise ValueError(f"Unknown model '{name}'. Choose ensemble, baseline, bilstm, hybrid_bilstm, or distilbert.")


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


def _ensemble_config():
    import json
    path = model_path("ensemble")["config"]
    if not path.is_file():
        raise FileNotFoundError(f"Ensemble configuration not found: {path}. Run python -m src.evaluation.ensemble first.")
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config.get("members", ())) != ENSEMBLE_MEMBERS:
        raise ValueError("Ensemble config does not match the required baseline, BiLSTM, and DistilBERT members.")
    return config


def _saved_threshold(model_name: str | None = None):
    name = model_name or MODEL
    if name == "ensemble":
        return float(_ensemble_config()["decision_threshold"])
    if name == "baseline":
        return 0.5
    path = model_path(name)
    metadata_path = (path / "training_metadata.json" if name == "distilbert"
                     else path.parent / "metadata.json")
    if not metadata_path.is_file():
        return 0.5
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return float(metadata.get("validation_selected_threshold", metadata.get("threshold", 0.5)))


def _model_probability(model_name: str, text: str, posting: dict) -> float:
    artifact = _load(model_name)
    if model_name == "baseline":
        return float(artifact.predict_proba([text])[0, 1])
    if model_name == "bilstm":
        model, vocab, cfg, _ = artifact
        return _bilstm_probability(model, vocab, cfg, text)
    if model_name == "hybrid_bilstm":
        model, vocab, cfg, schema = artifact
        return _hybrid_probability(model, vocab, cfg, schema, text, posting)
    if model_name == "distilbert":
        import torch
        tokenizer, model = artifact
        metadata = model_path(model_name) / "training_metadata.json"
        max_length = 256
        if metadata.is_file():
            import json
            max_length = json.loads(metadata.read_text(encoding="utf-8")).get("max_length", max_length)
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            return float(torch.softmax(model(**encoded).logits, dim=1)[0, 1])
    raise ValueError(f"Unsupported model '{model_name}'.")


def predict(posting: dict) -> dict:
    if not ready():
        if MODEL == "ensemble":
            missing = [name for name in ENSEMBLE_MEMBERS if not _artifact_ready(name)]
            config_path = model_path("ensemble")["config"]
            if not config_path.is_file():
                missing.append("ensemble validation config (run python -m src.evaluation.ensemble)")
            raise FileNotFoundError(f"The three-model ensemble is not ready. Missing: {', '.join(missing)}")
        raise FileNotFoundError(f"Trained artifact not found for '{MODEL}': {model_path()}")
    text = _text(posting)
    token_evidence = []
    component_probabilities = None
    if MODEL == "ensemble":
        config = _ensemble_config()
        component_probabilities = {
            name: _model_probability(name, text, posting) for name in ENSEMBLE_MEMBERS
        }
        weights = config["weights"]
        probability = sum(float(weights[name]) * value for name, value in component_probabilities.items())
        from joblib import load
        from src.explainability.tfidf import explain_tfidf
        token_evidence = explain_tfidf(load(model_path("baseline")), text)
    else:
        probability = _model_probability(MODEL, text, posting)
        if MODEL == "baseline":
            from src.explainability.tfidf import explain_tfidf
            token_evidence = explain_tfidf(_load("baseline"), text)
    risk = "high" if probability >= .70 else "moderate" if probability >= .35 else "low"
    threshold = _saved_threshold()
    result = {
        "assessment": "potentially_fraudulent" if probability >= threshold else "genuine",
        "decision_threshold": round(threshold, 6),
        "fraud_probability": round(probability, 6),
        "genuine_probability": round(1 - probability, 6),
        "risk_level": risk,
        "model_used": MODEL,
        "assessment_type": ("Equal-weight ensemble; threshold selected on validation data. AI-based risk assessment, not definitive proof of fraud."
                            if MODEL == "ensemble" else
                            "AI-based risk assessment; this is not definitive proof of fraud."),
        "risk_indicators": risk_indicators(posting),
        "model_token_evidence": token_evidence,
    }
    if component_probabilities is not None:
        result["ensemble_method"] = config["method"]
        result["ensemble_weights"] = weights
        result["model_probabilities"] = {
            name: round(value, 6) for name, value in component_probabilities.items()
        }
        result["model_labels"] = MODEL_LABELS
    return result

