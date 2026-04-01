from typing import Callable
from train_model import load_model


def predict_risk(
    clauses: list[str],
    stop_flag: Callable[[], bool] | None = None,
) -> list[dict]:
    """
    Predict the risk level for each clause using the trained model.

    Args:
        clauses: List of text clauses.
        stop_flag: Optional callable that returns True when processing should stop.

    Returns:
        List of dicts with 'clause' and 'risk' (0 = low, 1 = high) keys.
    """
    model = load_model()
    results = []
    for clause in clauses:
        if stop_flag is not None and stop_flag():
            break
        proba = model.predict_proba([clause])[0]
        risk_label = int(proba.argmax())
        risk_score = float(proba[risk_label])
        results.append(
            {
                "clause": clause,
                "risk": risk_label,
                "risk_label": "High" if risk_label == 1 else "Low",
                "confidence": round(risk_score, 3),
            }
        )
    return results
