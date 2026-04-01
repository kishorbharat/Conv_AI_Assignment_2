import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = "risk_model.pkl"


def build_training_data() -> tuple[list[str], list[int]]:
    """
    Build a small synthetic training dataset for risk prediction.
    Labels: 0 = low risk, 1 = high risk.
    """
    texts = [
        "shall ensure safety of personnel at all times",
        "must comply with fire prevention standards",
        "equipment shall be inspected regularly",
        "mandatory safety drills are required",
        "personnel protective equipment is optional",
        "no specific requirement for this item",
        "general guidance only, not mandatory",
        "informational note for reference purposes",
        "failure to comply may result in serious injury",
        "non-compliance can lead to loss of life",
        "critical safety systems must be maintained",
        "emergency procedures are mandatory",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    return texts, labels


def train_and_save_model(output_path: str = MODEL_PATH) -> Pipeline:
    """Train a TF-IDF + Logistic Regression pipeline and save it to disk."""
    texts, labels = build_training_data()
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, output_path)
    return pipeline


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    """Load the trained model from disk, training it first if necessary."""
    if not os.path.exists(model_path):
        return train_and_save_model(model_path)
    return joblib.load(model_path)


if __name__ == "__main__":
    model = train_and_save_model()
    print(f"Model trained and saved to {MODEL_PATH}")
