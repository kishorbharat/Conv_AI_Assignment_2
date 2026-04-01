import pdfplumber
import re


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_into_clauses(text: str) -> list[str]:
    """Split extracted text into individual clauses/sentences."""
    # Split on sentence boundaries or numbered clauses
    clauses = re.split(r"(?<=[.?!])\s+|\n{2,}", text)
    return [c.strip() for c in clauses if c.strip()]
