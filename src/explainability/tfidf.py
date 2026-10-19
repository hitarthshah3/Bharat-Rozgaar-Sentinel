import numpy as np


def explain_tfidf(pipeline, text: str, top_n: int = 6) -> list[dict]:
    """Return signed linear-model token contributions for the TF-IDF model."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    row = vectorizer.transform([text])
    contributions = row.multiply(classifier.coef_[0]).tocoo()
    features = vectorizer.get_feature_names_out()
    indices = np.argsort(np.abs(contributions.data))[::-1][:top_n]
    return [{"term": str(features[contributions.col[i]]),
             "direction": "fraud risk" if contributions.data[i] > 0 else "genuine signal",
             "contribution": float(contributions.data[i])} for i in indices]

