from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber


@dataclass
class DocumentChunk:
	chunk_id: str
	page: int
	text: str


def extract_pages(pdf_path: str | Path) -> List[str]:
	"""Extract text page-by-page from a PDF."""
	pages: List[str] = []
	with pdfplumber.open(str(pdf_path)) as pdf:
		for page in pdf.pages:
			pages.append((page.extract_text() or "").strip())
	return pages


def chunk_pages(pages: List[str], chunk_size: int = 1200, overlap: int = 150) -> List[DocumentChunk]:
	"""Split page text into overlapping fixed-size chunks for retrieval."""
	if chunk_size <= 0:
		raise ValueError("chunk_size must be > 0")
	if overlap < 0 or overlap >= chunk_size:
		raise ValueError("overlap must be >= 0 and < chunk_size")

	chunks: List[DocumentChunk] = []
	step = chunk_size - overlap

	for page_idx, page_text in enumerate(pages, start=1):
		if not page_text:
			continue

		start = 0
		chunk_number = 0
		while start < len(page_text):
			end = min(start + chunk_size, len(page_text))
			text = page_text[start:end].strip()
			if text:
				chunks.append(
					DocumentChunk(
						chunk_id=f"p{page_idx}_c{chunk_number}",
						page=page_idx,
						text=text,
					)
				)
			chunk_number += 1
			start += step

	return chunks


def ingest_pdf(pdf_path: str | Path, chunk_size: int = 1200, overlap: int = 150) -> List[DocumentChunk]:
	"""Read a PDF and return retrieval-ready text chunks."""
	pages = extract_pages(pdf_path)
	return chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
