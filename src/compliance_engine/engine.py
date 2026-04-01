from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .pdf_ingestion import DocumentChunk, ingest_pdf
from .save_clauses import load_chunks, save_chunks

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text: str) -> set[str]:
	return {t.lower() for t in TOKEN_PATTERN.findall(text)}


@dataclass
class QueryResult:
	answer: str
	sources: List[dict]


class ComplianceEngine:
	"""Simple in-memory retrieval engine for PDF-based compliance Q&A."""

	def __init__(self, rules_path: str | Path, clauses_path: str | Path) -> None:
		self.rules_path = Path(rules_path)
		self.clauses_path = Path(clauses_path)
		self.rules = self._load_rules()
		self.chunks: List[DocumentChunk] = load_chunks(self.clauses_path)
		self._vectorizer: TfidfVectorizer | None = None
		self._chunk_matrix = None
		self._rebuild_index()

	def _rebuild_index(self) -> None:
		if not self.chunks:
			self._vectorizer = None
			self._chunk_matrix = None
			return

		self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
		self._chunk_matrix = self._vectorizer.fit_transform([chunk.text for chunk in self.chunks])

	def _load_rules(self) -> list:
		if not self.rules_path.exists():
			return []
		try:
			data = json.loads(self.rules_path.read_text(encoding="utf-8"))
			return data if isinstance(data, list) else []
		except json.JSONDecodeError:
			return []

	def ingest(self, pdf_path: str | Path) -> int:
		self.chunks = ingest_pdf(pdf_path)
		save_chunks(self.chunks, self.clauses_path)
		self._rebuild_index()
		return len(self.chunks)

	def _rank_chunks(self, question: str, top_k: int) -> List[tuple[float, DocumentChunk]]:
		q_tokens = _tokens(question)
		if not q_tokens or not self.chunks:
			return []

		semantic_scores = []
		if self._vectorizer is not None and self._chunk_matrix is not None:
			q_vec = self._vectorizer.transform([question])
			semantic_scores = cosine_similarity(q_vec, self._chunk_matrix).flatten().tolist()
		else:
			semantic_scores = [0.0] * len(self.chunks)

		ranked: List[tuple[float, DocumentChunk]] = []
		for idx, chunk in enumerate(self.chunks):
			chunk_tokens = _tokens(chunk.text)
			lexical_score = len(q_tokens.intersection(chunk_tokens)) / max(len(q_tokens), 1)
			combined = (0.7 * float(semantic_scores[idx])) + (0.3 * lexical_score)
			if combined > 0:
				ranked.append((combined, chunk))

		ranked.sort(key=lambda item: item[0], reverse=True)
		return ranked[:top_k]

	def ask(self, question: str, top_k: int = 3) -> QueryResult:
		if not self.chunks:
			return QueryResult(
				answer="No document has been ingested yet. Ingest a PDF first.",
				sources=[],
			)

		ranked = self._rank_chunks(question, top_k=top_k)
		best = [chunk for _, chunk in ranked]

		if not best:
			return QueryResult(
				answer="I could not find relevant text for that query in the ingested PDF.",
				sources=[],
			)

		sources = [
			{
				"chunk_id": chunk.chunk_id,
				"page": chunk.page,
				"snippet": chunk.text[:320],
				"score": round(score, 4),
			}
			for score, chunk in ranked
		]
		lines = [
			"Top matching excerpts from the ingested PDF:",
		]
		for idx, source in enumerate(sources, start=1):
			lines.append(
				f"{idx}. Page {source['page']} ({source['chunk_id']}, score={source['score']}): {source['snippet']}"
			)
		answer = "\n".join(lines)

		return QueryResult(answer=answer, sources=sources)
