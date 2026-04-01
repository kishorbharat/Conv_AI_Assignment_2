"""
compliance_engine.py
--------------------
Core query engine for AIS-175 regulatory compliance.

Usage (Python API):
    from compliance_engine import ComplianceEngine
    engine = ComplianceEngine()
    results = engine.query("What are the lighting requirements?")
    for r in results:
        print(r["id"], r["text"])
"""

import json
import os
import re

RULES_PATH = os.path.join(os.path.dirname(__file__), "ais175_rules.json")


class ComplianceEngine:
    """
    Load AIS-175 rules from *rules_path* and answer free-text queries by
    returning the clauses whose text contains the query keywords.
    """

    def __init__(self, rules_path: str = RULES_PATH):
        self.rules_path = rules_path
        self.rules: list = []
        self._load_rules()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_rules(self) -> None:
        """Load rules from the JSON file (or leave empty if the file is empty)."""
        if not os.path.exists(self.rules_path):
            return
        with open(self.rules_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            self.rules = data

    def _tokenize(self, text: str) -> set:
        """Return a set of lower-case alphabetic tokens from *text*."""
        return set(re.findall(r"[a-z]+", text.lower()))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> list:
        """
        Return the *top_k* most relevant rules for *question*.

        Relevance is measured by the number of query keywords that appear
        in the clause text (simple keyword overlap).

        Parameters
        ----------
        question : str
            Free-text question, e.g. "What are the lighting requirements?"
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list of dict
            Each dict has keys ``"id"`` and ``"text"``.
        """
        if not self.rules:
            return []

        query_tokens = self._tokenize(question)
        # Remove very common English stop-words so they don't dominate scoring
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "what", "which", "who", "when", "where", "how", "and", "or",
            "not", "no", "that", "this", "these", "those",
        }
        query_tokens -= stop_words

        if not query_tokens:
            # Return first top_k rules when query has no meaningful tokens
            return self.rules[:top_k]

        scored = []
        for rule in self.rules:
            rule_tokens = self._tokenize(rule.get("text", ""))
            score = len(query_tokens & rule_tokens)
            if score > 0:
                scored.append((score, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rule for _, rule in scored[:top_k]]

    def check_compliance(self, clause_text: str) -> dict:
        """
        Check whether *clause_text* matches any known AIS-175 rule.

        Returns a dict with:
          - ``"compliant"`` (bool)
          - ``"matched_rule"`` (dict or None)
          - ``"message"`` (str)
        """
        results = self.query(clause_text, top_k=1)
        if results:
            return {
                "compliant": True,
                "matched_rule": results[0],
                "message": f"Matches rule {results[0]['id']}.",
            }
        return {
            "compliant": False,
            "matched_rule": None,
            "message": "No matching AIS-175 rule found.",
        }


if __name__ == "__main__":
    engine = ComplianceEngine()
    print(f"Loaded {len(engine.rules)} rules from {engine.rules_path}")
    sample = engine.query("safety requirements")
    for item in sample:
        print(f"[{item['id']}] {item['text'][:120]}")
