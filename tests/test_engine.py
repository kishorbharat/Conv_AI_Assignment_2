from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.compliance_engine.engine import ComplianceEngine
from src.compliance_engine.pdf_ingestion import DocumentChunk
from src.compliance_engine.save_clauses import save_chunks


class ComplianceEngineTests(unittest.TestCase):
    def test_ask_requires_ingest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            rules = base / "rules.json"
            rules.write_text("[]", encoding="utf-8")

            engine = ComplianceEngine(rules_path=rules, clauses_path=base / "clauses.json")
            result = engine.ask("What are braking requirements?")

            self.assertIn("No document has been ingested yet", result.answer)
            self.assertEqual(result.sources, [])

    def test_semantic_ranking_prefers_relevant_chunk(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            rules = base / "rules.json"
            rules.write_text("[]", encoding="utf-8")
            clauses = base / "clauses.json"

            chunks = [
                DocumentChunk(
                    chunk_id="p1_c0",
                    page=1,
                    text="The braking performance requirements include emergency braking and stopping distance checks.",
                ),
                DocumentChunk(
                    chunk_id="p2_c0",
                    page=2,
                    text="Lighting equipment and indicator lamps must remain visible under all weather conditions.",
                ),
            ]
            save_chunks(chunks, clauses)

            engine = ComplianceEngine(rules_path=rules, clauses_path=clauses)
            result = engine.ask("What does the document say about emergency braking performance?", top_k=1)

            self.assertEqual(len(result.sources), 1)
            self.assertEqual(result.sources[0]["chunk_id"], "p1_c0")
            self.assertIn("Top matching excerpts", result.answer)


if __name__ == "__main__":
    unittest.main()
