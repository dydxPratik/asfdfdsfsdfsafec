"""Regression tests for hardening fixes: BM25 caching, file signatures, scanned-PDF OCR, upload guards."""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from src.ingestion.indexer import ingest_directory, load_bm25_index, save_bm25_corpus
from src.ingestion.parsers import parse_file, validate_file_signature


def _record(cid: str, text: str) -> dict:
    return {
        "chunk_id": cid,
        "text": text,
        "source_file": "t.txt",
        "page_or_section": "Text Document",
        "tokens": text.lower().split(),
    }


class TestHardeningFixes(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ragquest_hardening_"))
        self.data_dir = self.tmp / "data"
        self.data_dir.mkdir()
        self.chroma_dir = self.tmp / "chroma"
        self.corpus_path = self.chroma_dir / "bm25_corpus.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------- BM25 cache ----------
    def test_bm25_index_cached_until_corpus_changes(self) -> None:
        save_bm25_corpus([_record("a", "alpha beta"), _record("b", "gamma delta")], self.corpus_path)
        idx1, recs1 = load_bm25_index(self.corpus_path)
        idx2, recs2 = load_bm25_index(self.corpus_path)
        self.assertIsNotNone(idx1)
        self.assertIs(idx1, idx2, "Second load should reuse the cached BM25Okapi object")
        self.assertIs(recs1, recs2)

        save_bm25_corpus(
            [_record("a", "alpha"), _record("b", "beta"), _record("c", "gamma")], self.corpus_path
        )
        idx3, recs3 = load_bm25_index(self.corpus_path)
        self.assertIsNot(idx3, idx1, "Rewriting the corpus must invalidate the cache")
        self.assertEqual(len(recs3), 3)
        self.assertFalse((self.chroma_dir / "bm25_corpus.json.tmp").exists(), "Atomic write leaves no temp file")

    def test_bm25_empty_corpus_returns_none(self) -> None:
        save_bm25_corpus([], self.corpus_path)
        idx, recs = load_bm25_index(self.corpus_path)
        self.assertIsNone(idx)
        self.assertEqual(recs, [])

    # ---------- File signature checks ----------
    def test_fake_pdf_and_excel_rejected_by_signature(self) -> None:
        fake_pdf = self.data_dir / "fake.pdf"
        fake_pdf.write_text("just plain text pretending to be a pdf", encoding="utf-8")
        fake_xlsx = self.data_dir / "fake.xlsx"
        fake_xlsx.write_text("col_a,col_b\n1,2", encoding="utf-8")
        (self.data_dir / "good.txt").write_text("Valid rule QRS-321 applies.", encoding="utf-8")

        with self.assertRaises(ValueError):
            validate_file_signature(fake_pdf)
        with self.assertRaises(ValueError):
            parse_file(fake_xlsx)

        report = ingest_directory(
            data_dir=self.data_dir,
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
            reset_index=True,
        )
        self.assertEqual(report.files_processed, ["good.txt"])
        self.assertIn("fake.pdf", report.failed_files)
        self.assertIn("signature", report.failed_files["fake.pdf"].lower())
        self.assertIn("fake.xlsx", report.failed_files)

    def test_valid_pdf_signature_accepted(self) -> None:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Signature check policy SIG-100.")
        pdf_path = self.data_dir / "ok.pdf"
        doc.save(pdf_path)
        doc.close()
        validate_file_signature(pdf_path)  # must not raise
        sections = parse_file(pdf_path)
        self.assertEqual(sections[0].page_or_section, "Page 1")

    # ---------- Scanned PDF OCR fallback ----------
    def test_scanned_pdf_page_is_ocr_extracted(self) -> None:
        img = Image.new("RGB", (1400, 360), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=64)
        draw.text((40, 60), "SCANNED POLICY SCN-778", fill="black", font=font)
        draw.text((40, 190), "Parking fee is 45 USD", fill="black", font=font)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        doc = pymupdf.open()
        page = doc.new_page(width=700, height=180)
        page.insert_image(page.rect, stream=buf.getvalue())
        pdf_path = self.data_dir / "scanned.pdf"
        doc.save(pdf_path)
        doc.close()

        sections = parse_file(pdf_path)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].page_or_section, "Page 1 (OCR)")
        text = sections[0].text.lower()
        self.assertIn("scanned", text)
        self.assertIn("778", text)
        self.assertIn("45", text)


class TestUploadGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.api import server

        cls.server = server
        cls.client = TestClient(server.app)

    def test_concurrent_ingestion_rejected_with_409(self) -> None:
        lock = self.server._INGEST_LOCK
        self.assertTrue(lock.acquire(blocking=False))
        try:
            res = self.client.post("/api/ingest/start", data={"reset_index": "true"})
            self.assertEqual(res.status_code, 409)
        finally:
            lock.release()

    def test_oversized_upload_rejected_and_lock_released(self) -> None:
        name = "zz_oversized_upload_test.txt"
        target = self.server.DATA_DIR / name
        with patch.object(self.server, "MAX_UPLOAD_BYTES", 10):
            res = self.client.post(
                "/api/ingest/start",
                data={"reset_index": "true"},
                files=[("files", (name, b"x" * 50, "text/plain"))],
            )
        self.assertEqual(res.status_code, 413)
        self.assertFalse(target.exists(), "Rejected upload must not be written to ./data")
        # Lock must be released after rejection
        self.assertTrue(self.server._INGEST_LOCK.acquire(blocking=False))
        self.server._INGEST_LOCK.release()

    def test_unsupported_upload_extension_rejected(self) -> None:
        res = self.client.post(
            "/api/ingest/start",
            data={"reset_index": "true"},
            files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))],
        )
        self.assertEqual(res.status_code, 400)
        self.assertFalse((self.server.DATA_DIR / "evil.exe").exists())


if __name__ == "__main__":
    unittest.main()
