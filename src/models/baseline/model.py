from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model(max_features: int = 100_000, min_df: int = 2, ngram_max: int = 2) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", sublinear_tf=True,
            max_features=max_features, min_df=min_df,
            ngram_range=(1, ngram_max), token_pattern=r"(?u)\b\w\w+\b",
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1500, solver="liblinear", random_state=42,
        )),
    ])

