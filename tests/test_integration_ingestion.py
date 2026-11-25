"""Isolated sandbox integration tests for ingestion, edge cases, corruption, updates, and persistence."""

from __future__ import annotations

import gc
import shutil
import tempfile
import unittest
from pathlib import Path

import openpyxl
import pymupdf

from src.ingestion.indexer import chunk_file, get_chroma_client, get_or_create_collection, ingest_directory
from src.ingestion.parsers import parse_file
from src.retrieval.hybrid_retriever import hybrid_retrieve_and_rerank


class TestIsolatedIngestionAndEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path(tempfile.mkdtemp(prefix="ragquest_audit_"))
        self.data_dir = self.temp_root / "data"
        self.chroma_dir = self.temp_root / "chroma_db"
        self.corpus_path = self.chroma_dir / "bm25_corpus.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        gc.collect()
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_empty_files_across_all_formats(self) -> None:
        (self.data_dir / "empty.txt").write_text("", encoding="utf-8")
        (self.data_dir / "whitespace.md").write_text("   \n\n  ", encoding="utf-8")
        (self.data_dir / "empty.csv").write_text("", encoding="utf-8")
        # Blank 1-page PDF with no text
        doc = pymupdf.open()
        doc.new_page()
        doc.save(self.data_dir / "blank.pdf")
        doc.close()

        report = ingest_directory(
            data_dir=self.data_dir,
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
            reset_index=True,
        )
        self.assertEqual(len(report.files_processed), 4)
        self.assertEqual(report.total_chunks_indexed, 0)
        self.assertEqual(report.chroma_count, 0)
        self.assertEqual(report.bm25_doc_count, 0)

        # Querying an empty index should return [] cleanly without crashing
        hits = hybrid_retrieve_and_rerank(
            "anything",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        self.assertEqual(hits, [])

    def test_unsupported_file_extensions_rejected_or_ignored(self) -> None:
        bad_file = self.data_dir / "malware.exe"
        bad_file.write_bytes(b"MZ\x90\x00")
        with self.assertRaises(ValueError):
            parse_file(bad_file)

        # ingest_directory ignores unsupported extensions while indexing valid ones
        (self.data_dir / "valid.txt").write_text("Policy ABC-999 allows 15 days.", encoding="utf-8")
        report = ingest_directory(
            data_dir=self.data_dir,
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
            reset_index=True,
        )
        self.assertEqual(report.files_processed, ["valid.txt"])
        self.assertEqual(report.total_chunks_indexed, 1)

    def test_corrupted_pdf_isolated_without_aborting_healthy_files(self) -> None:
        (self.data_dir / "good.txt").write_text("Valid company rule XYZ-777.", encoding="utf-8")
        (self.data_dir / "corrupt.pdf").write_bytes(b"%PDF-1.4 NOT A REAL PDF CORRUPTED BYTES")

        report = ingest_directory(
            data_dir=self.data_dir,
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
            reset_index=True,
        )
        self.assertEqual(report.files_processed, ["good.txt"])
        self.assertIn("corrupt.pdf", report.failed_files)
        self.assertEqual(report.chroma_count, 1)
        self.assertEqual(report.bm25_doc_count, 1)

    def test_special_chars_tables_codeblocks_and_unicode(self) -> None:
        md_content = (
            "# Engineering Handbook — Café & Zürich Office 🚀\n\n"
            "| Tier | Max Latency | Budget |\n"
            "|------|-------------|--------|\n"
            "| P0   | <= 50ms     | €12,500 |\n\n"
            "```python\ndef verify_signature(tok: str) -> bool:\n    return tok.startswith('nv_live_')\n```\n"
        )
        md_file = self.data_dir / "special.md"
        md_file.write_text(md_content, encoding="utf-8")

        report = ingest_directory(
            data_dir=self.data_dir,
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
            reset_index=True,
        )
        self.assertGreaterEqual(report.total_chunks_indexed, 1)
        hits = hybrid_retrieve_and_rerank(
            "nv_live_ signature or P0 budget in Euros",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("12,500", hits[0].text)

    def test_duplicate_reingestion_with_reset_true_vs_false(self) -> None:
        doc_path = self.data_dir / "policy.md"
        doc_path.write_text("# Policy\nInitial allowance is $100 USD.", encoding="utf-8")

        r1 = ingest_directory(self.data_dir, self.chroma_dir, self.corpus_path, reset_index=True)
        self.assertEqual(r1.chroma_count, 1)

        # Exact same content re-ingested with reset_index=True -> no duplicates
        r2 = ingest_directory(self.data_dir, self.chroma_dir, self.corpus_path, reset_index=True)
        self.assertEqual(r2.chroma_count, 1)
        self.assertEqual(r2.bm25_doc_count, 1)

        # Modify file and run with reset_index=False -> verify ChromaDB & BM25 stay 1:1 synced!
        doc_path.write_text("# Policy\nUpdated allowance is $250 USD.", encoding="utf-8")
        r3 = ingest_directory(self.data_dir, self.chroma_dir, self.corpus_path, reset_index=False)
        self.assertEqual(r3.bm25_doc_count, 1)
        self.assertEqual(r3.chroma_count, 1)
        hits = hybrid_retrieve_and_rerank(
            "Updated allowance",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        self.assertIn("$250 USD", hits[0].text)
        self.assertNotIn("$100 USD", hits[0].text)

    def test_persistence_across_new_client_instance(self) -> None:
        (self.data_dir / "persist.txt").write_text(
            "Secret project code OMEGA-772 launches in Q4.", encoding="utf-8"
        )
        ingest_directory(self.data_dir, self.chroma_dir, self.corpus_path, reset_index=True)

        # Open a brand-new PersistentClient simulating process restart
        fresh_client = get_chroma_client(self.chroma_dir)
        fresh_col = get_or_create_collection(fresh_client)
        self.assertEqual(fresh_col.count(), 1)

        hits = hybrid_retrieve_and_rerank(
            "OMEGA-772",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        self.assertEqual(len(hits), 1)
        self.assertIn("OMEGA-772", hits[0].text)

    def test_multi_sheet_excel_workbook_ingestion_and_retrieval(self) -> None:
        xlsx_path = self.data_dir / "company_budgets.xlsx"
        wb = openpyxl.Workbook()

        # Sheet 1: Regional Budgets
        ws1 = wb.active
        ws1.title = "Regional Budgets"
        ws1.append(["cost_center", "hub", "annual_budget_usd", "owner"])
        ws1.append(["CC-ZUR-90", "Zurich Lab", 450000, "Hannah Lindqvist"])
        ws1.append(["CC-AUS-12", "Austin HQ", 620000, "Marcus Vance"])

        # Sheet 2: Empty sheet (should be skipped cleanly)
        wb.create_sheet("Blank Notes")

        # Sheet 3: Hardware Inventory
        ws3 = wb.create_sheet("Hardware Inventory")
        ws3.append(["asset_tag", "model", "status"])
        ws3.append(["AMR-HELIOS-88", "Helios-9", "Active Deployment"])

        wb.save(xlsx_path)
        wb.close()

        report = ingest_directory(
            data_dir=self.data_dir,
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
            reset_index=True,
        )
        self.assertEqual(report.files_processed, ["company_budgets.xlsx"])
        self.assertEqual(report.failed_files, {})
        self.assertGreaterEqual(report.total_chunks_indexed, 2)

        hits = hybrid_retrieve_and_rerank(
            "What is the annual budget for cost center CC-ZUR-90 in Zurich Lab?",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("450000", hits[0].text)
        self.assertIn("Sheet 'Regional Budgets'", hits[0].page_or_section)

        hits_sheet3 = hybrid_retrieve_and_rerank(
            "What model is asset_tag AMR-HELIOS-88?",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        self.assertGreaterEqual(len(hits_sheet3), 1)
        self.assertIn("Helios-9", hits_sheet3[0].text)
        self.assertIn("Sheet 'Hardware Inventory'", hits_sheet3[0].page_or_section)


if __name__ == "__main__":
    unittest.main()
