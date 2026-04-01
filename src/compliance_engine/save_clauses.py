from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .pdf_ingestion import DocumentChunk


def save_chunks(chunks: List[DocumentChunk], output_path: str | Path) -> None:
	"""Persist ingested chunks to JSON."""
	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)

	payload = [
		{"chunk_id": c.chunk_id, "page": c.page, "text": c.text}
		for c in chunks
	]
	out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_chunks(input_path: str | Path) -> List[DocumentChunk]:
	"""Load previously saved chunks from JSON."""
	path = Path(input_path)
	if not path.exists():
		return []

	data = json.loads(path.read_text(encoding="utf-8"))
	return [
		DocumentChunk(
			chunk_id=item["chunk_id"],
			page=int(item["page"]),
			text=item["text"],
		)
		for item in data
	]
