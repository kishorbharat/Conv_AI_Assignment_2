"""
predict_risk.py
---------------
Predict a compliance risk score for a given text clause using a trained
scikit-learn model (see train_model.py).

Usage:
    python predict_risk.py "Clause text to evaluate..."
"""

import os
import sys

try:
    import joblib
except ImportError:
    joblib = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "risk_model.pkl")


def load_model(model_path: str = MODEL_PATH):
    """Load the trained risk model from *model_path*, or return None."""
    if joblib is None:
        raise ImportError("joblib is required. Run: pip install joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def predict_risk(clause_text: str, model=None) -> dict:
    """
    Predict the compliance risk level for *clause_text*.

    Returns a dict with keys:
      - ``"risk_score"``  (float 0-1)
      - ``"risk_level"``  ("Low" | "Medium" | "High")
      - ``"message"``     (str)
    """
    if model is None:
        model = load_model()

    if model is None:
        return {
            "risk_score": 0.0,
            "risk_level": "Unknown",
            "message": "No trained model found. Run 'python train_model.py' first.",
        }

    score = float(model.predict_proba([clause_text])[0][1])
    if score < 0.33:
        level = "Low"
    elif score < 0.66:
        level = "Medium"
    else:
        level = "High"

    return {
        "risk_score": round(score, 4),
        "risk_level": level,
        "message": f"Predicted risk level: {level} (score={score:.4f})",
    }


if __name__ == "__main__":
    clause = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Sample clause text."
    result = predict_risk(clause)
    print(result["message"])
