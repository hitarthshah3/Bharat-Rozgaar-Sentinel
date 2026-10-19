"""Train a project model on the one approved dataset."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing.data import TARGET, combine_text, load_dataset
from src.evaluation.metrics import evaluate

SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def partition(frame):
    """Same 72/8/20 stratified row partition for every model."""
    indices = np.arange(len(frame))
    y = frame[TARGET].to_numpy()
    train_val, test = train_test_split(indices, test_size=.20, random_state=SEED, stratify=y)
    train, val = train_test_split(train_val, test_size=.10, random_state=SEED,
                                  stratify=y[train_val])
    return train, val, test


def train_baseline(frame, train_idx, test_idx, args):
    from joblib import dump
    from src.models.baseline.model import build_model
    texts = combine_text(frame)
    y = frame[TARGET].to_numpy()
    model = build_model(args.max_features, args.min_df, args.ngram_max)
    model.fit(texts.iloc[train_idx], y[train_idx])
    probabilities = model.predict_proba(texts.iloc[test_idx])[:, 1]
    artifact_dir = Path("models/tfidf_lr")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dump(model, artifact_dir / "model.joblib")
    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as f:
        from src.preprocessing.data import TEXT_FIELDS
        json.dump({"model": "baseline", "threshold": .5, "seed": SEED,
                   "text_fields": TEXT_FIELDS, "max_features": args.max_features,
                   "min_df": args.min_df, "ngram_range": [1, args.ngram_max]}, f, indent=2)
    return evaluate("tfidf_lr", y[test_idx], probabilities)


TOKEN_RE = re.compile(r"(?u)\b[\w']+\b")


def simple_tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def build_vocab(sentences, min_count=2, max_vocab=60_000):
    from collections import Counter
    counts = Counter(token for sent in sentences for token in sent)
    words = [word for word, count in counts.most_common(max_vocab - 2) if count >= min_count]
    return {"<pad>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(words)}}


def encode_sentences(sentences, vocab, max_length):
    from torch.nn.utils.rnn import pad_sequence
    import torch
    encoded = [torch.tensor([vocab.get(token, 1) for token in sent[:max_length]] or [1]) for sent in sentences]
    lengths = torch.tensor([min(len(seq), max_length) for seq in encoded], dtype=torch.long)
    padded = pad_sequence(encoded, batch_first=True, padding_value=0)
    if padded.size(1) < max_length:
        padded = torch.nn.functional.pad(padded, (0, max_length - padded.size(1)))
    return padded[:, :max_length], lengths


METADATA_CATEGORICAL = ["employment_type", "required_experience", "required_education", "industry", "function"]
METADATA_BINARY = ["telecommuting", "has_company_logo", "has_questions"]


def fit_metadata_encoder(frame, train_idx):
    """Build a train-only one-hot schema, with explicit missing and unknown buckets."""
    categories = {}
    for column in METADATA_CATEGORICAL:
        values = frame[column].fillna("").astype(str).str.strip().replace("", "<MISSING>")
        categories[column] = ["<UNK>", *sorted(set(values.iloc[train_idx]) - {"<UNK>"})]
    return {"categorical": categories, "binary": METADATA_BINARY}


def encode_metadata(frame, schema):
    # Deterministic manual one-hot encoding avoids version-specific encoder APIs.
    columns = []
    for column, levels in schema["categorical"].items():
        values = frame[column].fillna("").astype(str).str.strip().replace("", "<MISSING>")
        allowed = set(levels)
        values = values.where(values.isin(allowed), "<UNK>")
        columns.extend((values.to_numpy() == level).astype("float32") for level in levels)
    for column in schema["binary"]:
        values = frame[column].fillna(0).astype(str).str.strip().str.lower()
        columns.append(values.isin({"1", "true", "yes"}).to_numpy(dtype="float32"))
    return np.column_stack(columns).astype("float32")


def select_f1_threshold(y_true, probabilities):
    from sklearn.metrics import f1_score
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def batched_distilbert_probabilities(model, input_ids, attention_mask, batch_size, device):
    """Run evaluation in small batches to bound CPU and accelerator memory."""
    import torch
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size
            logits = model(input_ids[start:end].to(device),
                           attention_mask=attention_mask[start:end].to(device)).logits
            chunks.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(chunks)


def train_hybrid_bilstm(frame, train_idx, val_idx, test_idx, args):
    """BiLSTM over text fused with separately encoded structured posting metadata."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from gensim.models import Word2Vec
    from sklearn.metrics import f1_score
    from src.models.bilstm.model import BiLSTMClassifier

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    texts = combine_text(frame).tolist()
    labels = frame[TARGET].to_numpy()
    train_sentences = [simple_tokenize(texts[i]) for i in train_idx]
    vocab = build_vocab(train_sentences, args.min_token_count, args.max_vocab)
    w2v = Word2Vec(sentences=train_sentences, vector_size=args.embedding_dim,
                   window=5, min_count=args.min_token_count, workers=args.workers,
                   sg=1, seed=SEED, epochs=10)
    weights = np.random.default_rng(SEED).normal(0, .05, (len(vocab), args.embedding_dim)).astype("float32")
    weights[0] = 0
    for word, idx in vocab.items():
        if word in w2v.wv:
            weights[idx] = w2v.wv[word]
    encoded, lengths = encode_sentences([simple_tokenize(t) for t in texts], vocab, args.max_length)
    schema = fit_metadata_encoder(frame, train_idx)
    metadata = torch.tensor(encode_metadata(frame, schema), dtype=torch.float32)
    y_train = torch.tensor(labels[train_idx], dtype=torch.float32)
    loader = DataLoader(TensorDataset(encoded[train_idx], lengths[train_idx], metadata[train_idx], y_train),
                        batch_size=args.batch_size, shuffle=True)
    model = BiLSTMClassifier(len(vocab), args.embedding_dim, args.hidden_dim,
                             dropout=args.dropout, pretrained=weights,
                             metadata_dim=metadata.shape[1]).to(device)
    pos_weight = torch.tensor([(len(train_idx) - y_train.sum().item()) / max(y_train.sum().item(), 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state, best_score, best_epoch = None, -1.0, 0
    x_val, len_val, meta_val = encoded[val_idx].to(device), lengths[val_idx].to(device), metadata[val_idx].to(device)
    for epoch in range(args.epochs):
        model.train()
        for xb, lb, mb, yb in loader:
            xb, lb, mb, yb = xb.to(device), lb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, lb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(x_val, len_val, meta_val)).cpu().numpy()
        score = f1_score(labels[val_idx], val_prob >= .5, zero_division=0)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch + 1} val_f1_at_0.5={score:.4f} best_epoch={best_epoch}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(x_val, len_val, meta_val)).cpu().numpy()
        test_prob = torch.sigmoid(model(encoded[test_idx].to(device), lengths[test_idx].to(device),
                                        metadata[test_idx].to(device))).cpu().numpy()
    threshold, selected_val_f1 = select_f1_threshold(labels[val_idx], val_prob)
    artifact_dir = Path("models/hybrid_bilstm")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "metadata_schema": schema,
                "config": {"embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim,
                           "max_length": args.max_length, "dropout": args.dropout,
                           "metadata_dim": int(metadata.shape[1])}}, artifact_dir / "model.pt")
    w2v.save(str(artifact_dir / "word2vec.model"))
    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"model": "hybrid_bilstm", "seed": SEED, "epochs_run": args.epochs,
                   "best_epoch_by_validation_f1_at_0.5": best_epoch,
                   "validation_f1_at_0.5": best_score, "validation_selected_threshold": threshold,
                   "validation_f1_at_selected_threshold": selected_val_f1,
                   "class_weight_positive": float(pos_weight.item()), "device": str(device),
                   "metadata_categorical": METADATA_CATEGORICAL, "metadata_binary": METADATA_BINARY}, f, indent=2)
    # Save both fixed-threshold and validation-selected-threshold test reports.
    evaluate("hybrid_bilstm_0_5", labels[test_idx], test_prob)
    return evaluate("hybrid_bilstm", labels[test_idx], test_prob, threshold=threshold)


def train_bilstm(frame, train_idx, val_idx, test_idx, args):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from gensim.models import Word2Vec
    from sklearn.metrics import f1_score, roc_auc_score
    from src.models.bilstm.model import BiLSTMClassifier

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    texts = combine_text(frame).tolist()
    labels = frame[TARGET].to_numpy()
    train_sentences = [simple_tokenize(texts[i]) for i in train_idx]
    vocab = build_vocab(train_sentences, args.min_token_count, args.max_vocab)
    w2v = Word2Vec(sentences=train_sentences, vector_size=args.embedding_dim,
                   window=5, min_count=args.min_token_count, workers=args.workers,
                   sg=1, seed=SEED, epochs=10)
    weights = np.random.default_rng(SEED).normal(0, .05, (len(vocab), args.embedding_dim)).astype("float32")
    weights[0] = 0
    for word, idx in vocab.items():
        if word in w2v.wv:
            weights[idx] = w2v.wv[word]
    encoded, lengths = encode_sentences([simple_tokenize(t) for t in texts], vocab, args.max_length)
    x_train, x_val, x_test = encoded[train_idx], encoded[val_idx], encoded[test_idx]
    len_train, len_val, len_test = lengths[train_idx], lengths[val_idx], lengths[test_idx]
    y_train = torch.tensor(labels[train_idx], dtype=torch.float32)
    y_val = torch.tensor(labels[val_idx], dtype=torch.float32)
    y_test = torch.tensor(labels[test_idx], dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_train, len_train, y_train), batch_size=args.batch_size, shuffle=True)
    model = BiLSTMClassifier(len(vocab), args.embedding_dim, args.hidden_dim,
                             dropout=args.dropout, pretrained=weights).to(device)
    pos_weight = torch.tensor([(len(train_idx) - y_train.sum().item()) / max(y_train.sum().item(), 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state, best_score, best_epoch = None, -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        for xb, lb, yb in loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(x_val.to(device), len_val.to(device))).cpu().numpy()
        score = f1_score(labels[val_idx], val_prob >= .5, zero_division=0)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch + 1} val_f1={score:.4f} best_epoch={best_epoch}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(model(x_test.to(device), len_test.to(device))).cpu().numpy()
    artifact_name = getattr(args, "artifact_name", "word2vec_bilstm")
    artifact_dir = Path("models") / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                "config": {"embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim,
                           "max_length": args.max_length, "dropout": args.dropout}}, artifact_dir / "model.pt")
    w2v.save(str(artifact_dir / "word2vec.model"))
    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"model": "bilstm", "seed": SEED, "epochs_run": args.epochs,
                   "best_epoch_by_validation_f1": best_epoch, "validation_f1": best_score,
                   "class_weight_positive": float(pos_weight.item()), "device": str(device)}, f, indent=2)
    return evaluate(artifact_name, labels[test_idx], test_prob)


def train_distilbert(frame, train_idx, val_idx, test_idx, args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import f1_score

    torch.manual_seed(SEED)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    texts = combine_text(frame).tolist()
    labels = frame[TARGET].to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=args.max_length,
                        return_tensors="pt")
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], y)
    actual_train_idx = train_idx
    if args.train_limit and args.train_limit < len(train_idx):
        # Keep a representative subset while preserving both classes. Validation
        # and held-out test data remain the full, shared partitions.
        actual_train_idx, _ = train_test_split(
            train_idx, train_size=args.train_limit, random_state=SEED,
            stratify=labels[train_idx])
        actual_train_idx = np.sort(actual_train_idx)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, actual_train_idx.tolist()),
                              batch_size=args.batch_size, shuffle=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=2).to(device)
    counts = np.bincount(labels[actual_train_idx], minlength=2)
    class_weights = torch.tensor(len(actual_train_idx) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_state, best_score, best_epoch = None, -1.0, 0
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_number, (input_ids, attention_mask, batch_y) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            output = model(input_ids.to(device), attention_mask=attention_mask.to(device))
            loss = torch.nn.functional.cross_entropy(output.logits, batch_y.to(device), weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            if batch_number % 20 == 0 or batch_number == len(train_loader):
                print(f"epoch={epoch + 1} batch={batch_number}/{len(train_loader)} "
                      f"mean_loss={epoch_loss / batch_number:.4f}", flush=True)
        model.eval()
        val_prob = batched_distilbert_probabilities(
            model, encoded["input_ids"][val_idx_t], encoded["attention_mask"][val_idx_t],
            args.batch_size, device)
        score = f1_score(labels[val_idx], val_prob >= .5, zero_division=0)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch + 1} val_f1={score:.4f} best_epoch={best_epoch}")
    model.load_state_dict(best_state)
    model.eval()
    val_prob = batched_distilbert_probabilities(
        model, encoded["input_ids"][val_idx_t], encoded["attention_mask"][val_idx_t],
        args.batch_size, device)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long)
    test_prob = batched_distilbert_probabilities(
        model, encoded["input_ids"][test_idx_t], encoded["attention_mask"][test_idx_t],
        args.batch_size, device)
    threshold, selected_val_f1 = select_f1_threshold(labels[val_idx], val_prob)
    artifact_dir = Path("models/distilbert")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact_dir)
    tokenizer.save_pretrained(artifact_dir)
    with (artifact_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"model": "distilbert", "checkpoint": args.checkpoint, "seed": SEED,
                   "max_length": args.max_length, "epochs_run": args.epochs,
                   "best_epoch_by_validation_f1": best_epoch, "validation_f1": best_score,
                   "validation_selected_threshold": threshold,
                   "validation_f1_at_selected_threshold": selected_val_f1,
                   "class_weights": class_weights.cpu().tolist(), "device": str(device),
                   "full_train_rows": int(len(train_idx)),
                   "actual_train_rows": int(len(actual_train_idx)),
                   "training_subset_stratified": bool(len(actual_train_idx) < len(train_idx)),
                   "batch_size": args.batch_size, "cpu_threads": args.cpu_threads}, f, indent=2)
    evaluate("distilbert_0_5", labels[test_idx], test_prob)
    return evaluate("distilbert", labels[test_idx], test_prob, threshold=threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "bilstm", "hybrid_bilstm", "distilbert"], required=True)
    parser.add_argument("--cpu", action="store_true", help="Force CPU for neural models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=0,
                        help="Optional stratified cap on DistilBERT training rows; validation/test stay full")
    parser.add_argument("--cpu-threads", type=int, default=2,
                        help="PyTorch CPU thread count for DistilBERT; set 0 to leave unchanged")
    parser.add_argument("--checkpoint", default="distilbert-base-uncased")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.35)
    parser.add_argument("--max-vocab", type=int, default=60_000)
    parser.add_argument("--min-token-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--artifact-name", default="word2vec_bilstm",
                        help="Output folder/report name for a text-only BiLSTM run")
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2)
    args = parser.parse_args()
    seed_everything()
    frame = load_dataset()
    train_idx, val_idx, test_idx = partition(frame)
    print(f"rows={len(frame)} train={len(train_idx)} validation={len(val_idx)} test={len(test_idx)}")
    if args.model == "baseline":
        metrics = train_baseline(frame, train_idx, test_idx, args)
    elif args.model == "bilstm":
        metrics = train_bilstm(frame, train_idx, val_idx, test_idx, args)
    elif args.model == "hybrid_bilstm":
        metrics = train_hybrid_bilstm(frame, train_idx, val_idx, test_idx, args)
    else:
        metrics = train_distilbert(frame, train_idx, val_idx, test_idx, args)
    print(json.dumps({k: metrics[k] for k in
                      ("model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold")},
                     indent=2))


if __name__ == "__main__":
    main()

