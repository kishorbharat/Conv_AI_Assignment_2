"""
save_clauses.py
---------------
Convenience script: ingest the AIS-175 PDF and save the extracted clauses
to ais175_rules.json.

Usage:
    python save_clauses.py
"""

from pdf_ingestion import ingest

if __name__ == "__main__":
    clauses = ingest()
    print(f"Done – {len(clauses)} clauses saved to ais175_rules.json")
