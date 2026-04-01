"""
pdf_ingestion.py
----------------
Extract clauses and rules from the AIS-175 regulatory PDF and store them
in ais175_rules.json so the compliance engine can query them.
"""

import json
import re
import os

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


PDF_PATH = os.path.join(os.path.dirname(__file__), "AIS 175_Final Draft_MARCH_2025.pdf")
RULES_PATH = os.path.join(os.path.dirname(__file__), "ais175_rules.json")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Return all text extracted from *pdf_path*."""
    if pdfplumber is None:
        raise ImportError("pdfplumber is required. Run: pip install pdfplumber")
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def parse_clauses(raw_text: str) -> list:
    """
    Split the raw PDF text into individual clauses / rules.

    The heuristic looks for lines that start with a numbered clause pattern
    such as "1.", "1.1", "2.3.4", etc. and groups the following lines as the
    body of that clause.
    """
    clauses = []
    clause_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)")

    current_id = None
    current_lines = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = clause_pattern.match(line)
        if match:
            if current_id is not None:
                clauses.append(
                    {"id": current_id, "text": " ".join(current_lines).strip()}
                )
            current_id = match.group(1)
            current_lines = [match.group(2)]
        else:
            if current_id is not None:
                current_lines.append(line)

    if current_id is not None:
        clauses.append({"id": current_id, "text": " ".join(current_lines).strip()})

    return clauses


def ingest(pdf_path: str = PDF_PATH, rules_path: str = RULES_PATH) -> list:
    """
    Main entry point – extract clauses from *pdf_path* and save them to
    *rules_path*.  Returns the list of extracted clause dicts.
    """
    print(f"[pdf_ingestion] Reading PDF: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)
    clauses = parse_clauses(raw_text)
    print(f"[pdf_ingestion] Extracted {len(clauses)} clauses.")
    with open(rules_path, "w", encoding="utf-8") as fh:
        json.dump(clauses, fh, indent=2, ensure_ascii=False)
    print(f"[pdf_ingestion] Saved to {rules_path}")
    return clauses


if __name__ == "__main__":
    ingest()
