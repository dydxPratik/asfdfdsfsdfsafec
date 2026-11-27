"""End-to-end tests for Image Ingestion (.png/.jpg/.jpeg/.webp), OCR Spatial Layout, Grounding & Regression."""

from __future__ import annotations

import gc
import shutil
import tempfile
import unittest
from pathlib import Path

import openpyxl
from PIL import Image, ImageDraw, ImageFont

from src.generation.generator import generate_grounded_answer
from src.ingestion.indexer import get_chroma_client, get_or_create_collection, ingest_directory
from src.ingestion.parsers import parse_file
from src.retrieval.hybrid_retriever import hybrid_retrieve_and_rerank


def _load_font(size: int = 24) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _create_document_screenshot_png(path: Path) -> None:
    img = Image.new("RGB", (920, 320), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_lg = _load_font(28)
    font_md = _load_font(22)
    draw.rectangle([20, 20, 900, 300], outline=(30, 60, 120), width=3)
    draw.text((45, 45), "NEXAVAULT VISITOR ACCESS NOTICE", fill=(15, 23, 42), font=font_lg)
    draw.text((45, 110), "Badge Replacement Fee: $35 USD", fill=(15, 23, 42), font=font_md)
    draw.text((45, 160), "Maximum Visitor Pass Validity: 14 Days", fill=(15, 23, 42), font=font_md)
    draw.text((45, 210), "Escort Policy Code: VIS-ESC-204", fill=(15, 23, 42), font=font_md)
    img.save(path, format="PNG")


def _create_table_chart_jpg(path: Path) -> None:
    img = Image.new("RGB", (960, 360), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)
    font_hdr = _load_font(24)
    font_cell = _load_font(22)
    # Draw crisp 3-column table with wide spacing
    draw.text((40, 30), "Q2 DEPARTMENT TRAVEL BUDGET", fill=(10, 10, 10), font=font_hdr)
    rows = [
        ("Department", "Budget Cap", "Approver"),
        ("Robotics", "$48,000", "E. Rostova"),
        ("Security", "$22,500", "M. Vance"),
        ("PeopleOps", "$15,000", "P. Nair"),
    ]
    y = 95
    for idx, (c1, c2, c3) in enumerate(rows):
        color = (20, 40, 90) if idx == 0 else (20, 20, 20)
        draw.text((40, y), c1, fill=color, font=font_cell)
        draw.text((340, y), c2, fill=color, font=font_cell)
        draw.text((620, y), c3, fill=color, font=font_cell)
        draw.line([(35, y + 36), (910, y + 36)], fill=(180, 180, 190), width=2)
        y += 55
    img.save(path, format="JPEG", quality=95)


def _create_text_free_diagram_webp(path: Path) -> None:
    img = Image.new("RGB", (400, 260), color=(240, 245, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 60, 160, 180], fill=(59, 130, 246), outline=(30, 58, 138), width=3)
    draw.ellipse([240, 60, 360, 180], fill=(16, 185, 129), outline=(6, 95, 70), width=3)
    draw.line([(160, 120), (240, 120)], fill=(30, 41, 59), width=4)
    img.save(path, format="WEBP")


class TestImageIngestionAndEndToEndRAG(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ragquest_img_e2e_"))
        self.data_dir = self.tmp / "data"
        self.chroma_dir = self.tmp / "chroma_db"
        self.corpus_path = self.chroma_dir / "bm25_corpus.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        gc.collect()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_corrupted_renamed_and_empty_images_fail_gracefully(self) -> None:
        (self.data_dir / "empty.png").write_bytes(b"")
        (self.data_dir / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0CORRUPT_NOT_JPEG")
        (self.data_dir / "fake_renamed.webp").write_text("This is plain text pretending to be webp", encoding="utf-8")
        _create_document_screenshot_png(self.data_dir / "valid_notice.png")

        for bad_name in ("empty.png", "corrupt.jpg", "fake_renamed.webp"):
            with self.assertRaises(ValueError):
                parse_file(self.data_dir / bad_name)

        report = ingest_directory(self.data_dir, self.chroma_dir, self.corpus_path, reset_index=True)
        self.assertEqual(report.files_processed, ["valid_notice.png"])
        self.assertEqual(set(report.failed_files.keys()), {"empty.png", "corrupt.jpg", "fake_renamed.webp"})
        self.assertEqual(report.chroma_count, 1)

    def test_e2e_image_text_table_textfree_unanswerable_and_regression(self) -> None:
        # 1. Screenshot image with text + numbers
        _create_document_screenshot_png(self.data_dir / "visitor_notice.png")
        # 2. Table/chart image (.jpg)
        _create_table_chart_jpg(self.data_dir / "q2_budget_table.jpg")
        # 3. Text-free diagram (.webp)
        _create_text_free_diagram_webp(self.data_dir / "pure_diagram.webp")
        # 4. Existing Text & Excel regression files
        (self.data_dir / "hr_note.txt").write_text(
            "NexaVault Coffee Stipend: Employees receive $40 USD monthly for espresso beans.",
            encoding="utf-8",
        )
        wb = openpyxl.Workbook()
        wb.active.title = "Assets"
        wb.active.append(["asset_id", "location", "owner"])
        wb.active.append(["LAB-SCOPE-9", "Zurich Lab", "Dr. Elena Rostova"])
        wb.save(self.data_dir / "lab_assets.xlsx")
        wb.close()

        report = ingest_directory(self.data_dir, self.chroma_dir, self.corpus_path, reset_index=True)
        self.assertEqual(report.failed_files, {})
        self.assertEqual(len(report.files_processed), 5)

        # Verify metadata: source, source_file, and file_type='image' vs 'document'
        col = get_or_create_collection(get_chroma_client(self.chroma_dir))
        stored = col.get(include=["metadatas", "documents"])
        meta_by_file = {m["source_file"]: m for m in stored["metadatas"]}
        doc_by_file = {m["source_file"]: d for m, d in zip(stored["metadatas"], stored["documents"])}

        self.assertEqual(meta_by_file["visitor_notice.png"]["file_type"], "image")
        self.assertEqual(meta_by_file["visitor_notice.png"]["source"], "visitor_notice.png")
        self.assertEqual(meta_by_file["q2_budget_table.jpg"]["file_type"], "image")
        self.assertEqual(meta_by_file["pure_diagram.webp"]["file_type"], "image")
        self.assertEqual(meta_by_file["lab_assets.xlsx"]["file_type"], "document")
        self.assertIn("No readable text detected", doc_by_file["pure_diagram.webp"])

        # Duplicate image re-ingestion check (reset_index=False)
        rep_dup = ingest_directory(self.data_dir, self.chroma_dir, self.corpus_path, reset_index=False)
        self.assertEqual(rep_dup.chroma_count, report.chroma_count)
        self.assertEqual(rep_dup.bm25_doc_count, report.bm25_doc_count)

        # Live Check 1: Screenshot/document image QA
        c1 = hybrid_retrieve_and_rerank(
            "What is the visitor badge replacement fee and escort policy code?",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        r1 = generate_grounded_answer("What is the visitor badge replacement fee and escort policy code?", c1)
        print(f"\n[IMG-E2E-1 Screenshot] supported={r1.is_supported} | ans={r1.answer}")
        self.assertTrue(r1.is_supported)
        self.assertIn("35", r1.answer)
        self.assertIn("VIS-ESC-204", r1.answer)
        self.assertTrue(any(cit.source_file == "visitor_notice.png" for cit in r1.citations))

        # Live Check 2: Table image QA
        c2 = hybrid_retrieve_and_rerank(
            "In the Q2 Department Travel Budget table image, what is the Robotics budget cap and approver?",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        r2 = generate_grounded_answer(
            "In the Q2 Department Travel Budget table image, what is the Robotics budget cap and approver?", c2
        )
        print(f"[IMG-E2E-2 Table JPG] supported={r2.is_supported} | ans={r2.answer}")
        self.assertTrue(r2.is_supported)
        self.assertIn("48,000", r2.answer)
        self.assertTrue(any(cit.source_file == "q2_budget_table.jpg" for cit in r2.citations))

        # Live Check 4: Unanswerable query about indexed image
        c4 = hybrid_retrieve_and_rerank(
            "What is the visitor parking garage height limit in visitor_notice.png?",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        r4 = generate_grounded_answer("What is the visitor parking garage height limit in visitor_notice.png?", c4)
        print(f"[IMG-E2E-4 Unanswerable] supported={r4.is_supported} | ans={r4.answer}")
        self.assertFalse(r4.is_supported)
        self.assertEqual(r4.citations, [])

        # Live Check 5: Regression on TXT & Excel
        c5_txt = hybrid_retrieve_and_rerank(
            "How much is the NexaVault Coffee Stipend?",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        r5_txt = generate_grounded_answer("How much is the NexaVault Coffee Stipend?", c5_txt)
        print(f"[IMG-E2E-5a TXT Regression] supported={r5_txt.is_supported} | ans={r5_txt.answer}")
        self.assertTrue(r5_txt.is_supported)
        self.assertIn("40", r5_txt.answer)

        c5_xls = hybrid_retrieve_and_rerank(
            "Where is asset LAB-SCOPE-9 located and who owns it?",
            chroma_dir=self.chroma_dir,
            corpus_path=self.corpus_path,
        )
        r5_xls = generate_grounded_answer("Where is asset LAB-SCOPE-9 located and who owns it?", c5_xls)
        print(f"[IMG-E2E-5b Excel Regression] supported={r5_xls.is_supported} | ans={r5_xls.answer}")
        self.assertTrue(r5_xls.is_supported)
        self.assertIn("Zurich Lab", r5_xls.answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
