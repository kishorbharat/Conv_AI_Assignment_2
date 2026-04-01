"""
query.py
--------
Interactive terminal interface for the AIS-175 Compliance Query Engine.

Usage:
    python query.py

You will be prompted to enter a question. Type 'exit' or press Ctrl+C to quit.

Non-interactive (single query):
    python query.py "What are the lighting requirements for vehicles?"
"""

import sys
from compliance_engine import ComplianceEngine


BANNER = """
╔══════════════════════════════════════════════════════╗
║   AIS-175 Regulatory Compliance Query Engine         ║
║   Type your question and press Enter.                ║
║   Type 'exit' or press Ctrl+C to quit.               ║
╚══════════════════════════════════════════════════════╝
"""


def print_results(results: list) -> None:
    if not results:
        print("\n  [No matching clauses found. Try different keywords.]\n")
        return
    print(f"\n  Found {len(results)} matching clause(s):\n")
    for i, rule in enumerate(results, 1):
        clause_id = rule.get("id", "?")
        text = rule.get("text", "")
        print(f"  {i}. Clause {clause_id}")
        print(f"     {text[:300]}{'...' if len(text) > 300 else ''}")
        print()


def send_query(question: str, engine: ComplianceEngine) -> list:
    """
    Send *question* to the compliance engine and return matching rules.

    This is the programmatic entry point you can import from other modules:

        from query import send_query
        from compliance_engine import ComplianceEngine
        engine = ComplianceEngine()
        results = send_query("lighting requirements", engine)
    """
    return engine.query(question)


def run_interactive(engine: ComplianceEngine) -> None:
    """Run a REPL-style interactive query session."""
    print(BANNER)
    while True:
        try:
            question = input("  Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("  Goodbye!")
            break
        results = send_query(question, engine)
        print_results(results)


if __name__ == "__main__":
    engine = ComplianceEngine()
    if len(engine.rules) == 0:
        print(
            "[query] No rules loaded. Run 'python save_clauses.py' first to ingest the PDF."
        )

    # Allow a single query to be passed as a command-line argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n  Query: {question}")
        results = send_query(question, engine)
        print_results(results)
    else:
        run_interactive(engine)
