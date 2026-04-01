"""
train_model.py
--------------
Train a simple text-classification model to predict compliance risk level
based on clause text extracted from the AIS-175 rules JSON.

The trained model is saved to risk_model.pkl so predict_risk.py can load it.

Usage:
    python train_model.py
"""

import json
import os

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except ImportError as exc:
    raise ImportError(
        "scikit-learn and joblib are required. Run: pip install scikit-learn joblib"
    ) from exc

RULES_PATH = os.path.join(os.path.dirname(__file__), "ais175_rules.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "risk_model.pkl")

# Keywords that suggest a high-risk clause
HIGH_RISK_KEYWORDS = [
    "shall not", "prohibited", "penalty", "fine", "offence", "offense",
    "mandatory", "failure", "non-compliance", "recall", "unsafe",
]


def label_risk(text: str) -> int:
    """Return 1 (high-risk) or 0 (low-risk) based on keyword presence."""
    text_lower = text.lower()
    return int(any(kw in text_lower for kw in HIGH_RISK_KEYWORDS))


def train(rules_path: str = RULES_PATH, model_path: str = MODEL_PATH) -> None:
    with open(rules_path, "r", encoding="utf-8") as fh:
        rules = json.load(fh)

    if not rules:
        print("[train_model] No rules found. Run 'python save_clauses.py' first.")
        return

    texts = [r["text"] for r in rules]
    labels = [label_risk(t) for t in texts]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, model_path)
    print(f"[train_model] Model saved to {model_path}")
    high = sum(labels)
    print(f"[train_model] Training complete: {len(texts)} clauses, {high} high-risk.")


if __name__ == "__main__":
    train()
