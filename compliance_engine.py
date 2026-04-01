import json
import re
from typing import Callable


def load_rules(rules_path: str = "ais175_rules.json") -> list[dict]:
    """Load AIS-175 compliance rules from a JSON file."""
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        if isinstance(rules, list):
            return rules
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def check_clause(clause: str, rules: list[dict]) -> list[dict]:
    """Check a single clause against all rules and return any violations."""
    violations = []
    for rule in rules:
        keyword = rule.get("keyword", "")
        if keyword and re.search(re.escape(keyword), clause, re.IGNORECASE):
            violations.append(
                {
                    "rule_id": rule.get("id", ""),
                    "rule_desc": rule.get("description", ""),
                    "keyword": keyword,
                    "clause": clause,
                }
            )
    return violations


def run_compliance_check(
    clauses: list[str],
    rules: list[dict],
    stop_flag: Callable[[], bool] | None = None,
) -> list[dict]:
    """
    Run compliance check on all clauses.

    Args:
        clauses: List of text clauses to check.
        rules: List of AIS-175 rule dictionaries.
        stop_flag: Optional callable that returns True when processing should stop.

    Returns:
        List of violation dictionaries.
    """
    all_violations = []
    for clause in clauses:
        if stop_flag is not None and stop_flag():
            break
        violations = check_clause(clause, rules)
        all_violations.extend(violations)
    return all_violations
