from __future__ import annotations

from pathlib import Path
import pandas as pd

TARGET = "fraudulent"
TEXT_FIELDS = [
    "title", "location", "department", "salary_range", "company_profile",
    "description", "requirements", "benefits", "employment_type",
    "required_experience", "required_education", "industry", "function",
]
REQUIRED_COLUMNS = {"job_id", TARGET, *TEXT_FIELDS, "telecommuting", "has_company_logo", "has_questions"}


def discover_dataset(data_dir: str | Path = "data") -> Path:
    """Return the only CSV under data_dir, with actionable missing/ambiguity errors."""
    root = Path(data_dir)
    files = sorted(root.glob("*.csv")) if root.exists() else []
    if not files:
        raise FileNotFoundError(
            f"Dataset not found. Place the official Employment Scam Aegean CSV in {root.resolve()} "
            "(for example, fake_job_postings.csv). No dataset is downloaded automatically."
        )
    if len(files) != 1:
        names = ", ".join(p.name for p in files)
        raise RuntimeError(f"Expected exactly one CSV in {root.resolve()}, found {len(files)}: {names}")
    return files[0]


def load_dataset() -> pd.DataFrame:
    """Load and validate the provided dataset without silently dropping records."""
    csv_path = discover_dataset()
    frame = pd.read_csv(csv_path, keep_default_na=True)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset schema is missing required columns: {', '.join(sorted(missing))}")
    if frame[TARGET].isna().any():
        raise ValueError(f"Target column '{TARGET}' contains missing labels.")
    labels = set(pd.to_numeric(frame[TARGET], errors="raise").astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Expected binary {TARGET} labels 0/1; found {sorted(labels)}")
    frame[TARGET] = pd.to_numeric(frame[TARGET]).astype("int64")
    for field in TEXT_FIELDS:
        frame[field] = frame[field].fillna("").astype(str)
    for field in ("telecommuting", "has_company_logo", "has_questions"):
        frame[field] = frame[field].fillna(0).astype(str)
    return frame


def combine_text(frame: pd.DataFrame) -> pd.Series:
    """Preserve field boundaries and missingness while making a single NLP input."""
    parts = []
    for field in TEXT_FIELDS:
        values = frame[field].fillna("").astype(str).str.strip()
        parts.append(values.where(values.eq(""), field.replace("_", " ") + ": " + values))
    return pd.concat(parts, axis=1).apply(lambda row: " ".join(x for x in row if x), axis=1).str.strip()

