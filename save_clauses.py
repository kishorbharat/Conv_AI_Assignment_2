import json
import os


def save_clauses(clauses: list[str], output_path: str = "clauses.json") -> None:
    """Save extracted clauses to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"clauses": clauses}, f, indent=2, ensure_ascii=False)


def load_clauses(input_path: str = "clauses.json") -> list[str]:
    """Load clauses from a JSON file."""
    if not os.path.exists(input_path):
        return []
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("clauses", [])
