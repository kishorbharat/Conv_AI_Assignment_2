"""Terminal query interface for the AIS-175 Compliance Engine.

Usage
-----
    python query.py

Type a natural-language question or a clause fragment and press Enter.
The engine will return every AIS-175 rule whose keywords match your input.
Type ``exit`` or ``quit`` (or press Ctrl-C / Ctrl-D) to leave.
"""

from compliance_engine import check_query, load_rules


BANNER = """
╔══════════════════════════════════════════════════════╗
║       AIS-175 Regulatory Compliance Query Tool       ║
╚══════════════════════════════════════════════════════╝
Type a query to check against AIS-175 rules.
Type 'list' to show all rules.  Type 'exit' to quit.
"""


def _print_rule(rule: dict, index: int | None = None) -> None:
    prefix = f"[{index}] " if index is not None else ""
    print(f"\n  {prefix}{rule.get('id', '?')} – {rule.get('title', 'Untitled')}")
    print(f"  {rule.get('text', '')}")


def main() -> None:
    print(BANNER)
    rules = load_rules()

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if query.lower() == "list":
            print(f"\n{len(rules)} rule(s) loaded:")
            for i, rule in enumerate(rules, 1):
                _print_rule(rule, i)
            print()
            continue

        matches = check_query(query, rules)
        if matches:
            print(f"\n✔  {len(matches)} matching rule(s) found:")
            for i, rule in enumerate(matches, 1):
                _print_rule(rule, i)
        else:
            print("\n✘  No matching rules found for that query.")
        print()


if __name__ == "__main__":
    main()
