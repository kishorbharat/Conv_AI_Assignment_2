"""Compliance engine: loads AIS-175 rules and checks queries against them."""

import json
import re
import os


_RULES_PATH = os.path.join(os.path.dirname(__file__), "ais175_rules.json")


def load_rules(path: str = _RULES_PATH) -> list[dict]:
    """Load compliance rules from the JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Rules file not found: {path}. "
            "Please ensure 'ais175_rules.json' exists in the project directory."
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse rules file '{path}': {exc}"
        ) from exc


def check_query(query: str, rules: list[dict] | None = None) -> list[dict]:
    """Return every rule whose keywords appear in *query*.

    Each rule is expected to have at minimum the fields:
        - ``id``      (str)  – rule identifier
        - ``title``   (str)  – short description
        - ``keywords``(list) – list of keyword strings to match
        - ``text``    (str)  – full rule text (optional)

    Returns a (possibly empty) list of matching rule dicts.
    """
    if rules is None:
        rules = load_rules()

    query_lower = query.lower()
    matches = []
    for rule in rules:
        keywords = rule.get("keywords", [])
        if any(re.search(re.escape(kw.lower()), query_lower) for kw in keywords):
            matches.append(rule)
    return matches
