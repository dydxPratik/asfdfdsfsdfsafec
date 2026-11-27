"""Comprehensive offline pytest unit & integration suite for RAGQuest."""

from __future__ import annotations

import gc
import io
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openpyxl
import pymupdf
import pytest
from groq import BadRequestError
from PIL import Image, ImageDraw, ImageFont

from src.generation.generator import (
    _normalize_for_quote_match,
    answer_question,
    format_chunks_for_prompt,
    generate_grounded_answer,
)
from src.ingestion.chunker import make_chunk_id, recursive_split_text
from src.ingestion.indexer import chunk_file, ingest_directory, tokenize_for_bm25
from src.ingestion.parsers import (
    _reconstruct_spatial_lines,
    parse_csv,
    parse_excel,
    parse_file,
    parse_markdown,
    parse_txt,
)
from src.retrieval.hybrid_retriever import hybrid_retrieve_and_rerank, reciprocal_rank_fusion
from src.schemas import Citation, RAGResponse, RetrievedChunk


@pytest.fixture()
def isolated_rag_env():
    """Provide an isolated temporary data_dir, chroma_dir, and bm25 corpus path."""
    root = Path(tempfile.mkdtemp(prefix="ragquest_pytest_"))
    data_dir = root / "data"
    chroma_dir = root / "chroma_db"
    corpus_path = chroma_dir / "bm25_corpus.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield data_dir, chroma_dir, corpus_path
    finally:
        gc.collect()
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Parsing & Chunking Unit Tests
# ---------------------------------------------------------------------------


def test_parse_markdown_preamble_and_headings(tmp_path: Path) -> None:
    md_path = tmp_path / "handbook.md"
    md_path.write_text(
        "Introductory preamble text.\n\n# First Heading\nBody A.\n\n## Second Heading\nBody B.",
        encoding="utf-8",
    )
    sections = parse_markdown(md_path)
    assert [s.page_or_section for s in sections] == [
        "Preamble",
        "Section: First Heading",
        "Section: Second Heading",
    ]
    assert "# First Heading" in sections[1].text


def test_parse_txt_with_and_without_bracket_sections(tmp_path: Path) -> None:
    plain_path = tmp_path / "plain.txt"
    plain_path.write_text("Single unstructured memo.", encoding="utf-8")
    plain_secs = parse_txt(plain_path)
    assert len(plain_secs) == 1
    assert plain_secs[0].page_or_section == "Text Document"

    structured_path = tmp_path / "runbook.txt"
    structured_path.write_text(
        "Doc Title Banner\n\n[SEC-1: AUTH]\nUse YubiKey.\n\n[SEC-2: LOGS]\nRetain 90 days.",
        encoding="utf-8",
    )
    sec_list = parse_txt(structured_path)
    assert [s.page_or_section for s in sec_list] == ["Document Header", "SEC-1: AUTH", "SEC-2: LOGS"]


def test_parse_csv_utf8_bom_and_row_batching(tmp_path: Path) -> None:
    csv_path = tmp_path / "staff.csv"
    csv_path.write_bytes(
        "\ufeffid,name,hub\nE1,Alice,Zurich\nE2,Bob,Austin\nE3,Cara,Bengaluru\nE4,Dan,Tokyo\n".encode("utf-8")
    )
    sections = parse_csv(csv_path, rows_per_section=3)
    assert len(sections) == 2
    assert sections[0].page_or_section == "Rows 1-3"
    assert "id: E1 | name: Alice | hub: Zurich" in sections[0].text
    assert sections[1].page_or_section == "Row 4"
    assert "id: E4 | name: Dan | hub: Tokyo" in sections[1].text


def test_parse_excel_skips_blank_rows_and_preserves_sheet_headers(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "matrix.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Caps"
    ws.append([None, None])  # Leading empty row before actual header
    ws.append(["tier", "cap_usd"])
    ws.append(["Tier-1", 340])
    ws.append([None, None])  # Mid-sheet blank row
    ws.append(["Tier-2", 230])
    wb.save(xlsx_path)
    wb.close()

    sections = parse_excel(xlsx_path, rows_per_section=3)
    assert len(sections) == 1
    assert "Sheet 'Caps'" in sections[0].page_or_section
    assert "tier: Tier-1 | cap_usd: 340" in sections[0].text
    assert "tier: Tier-2 | cap_usd: 230" in sections[0].text


def test_spatial_ocr_line_and_column_reconstruction() -> None:
    # Two rows; first row has wide horizontal gap (>= 24px) -> ' | ', second row has narrow gap -> ' '
    synthetic_ocr = [
        ([[10, 10], [90, 10], [90, 30], [10, 30]], "Dept", 0.99),
        ([[200, 11], [290, 11], [290, 31], [200, 31]], "$48,000", 0.98),
        ([[10, 60], [60, 60], [60, 80], [10, 80]], "Note:", 0.99),
        ([[68, 60], [150, 60], [150, 80], [68, 80]], "Approved", 0.97),
    ]
    reconstructed = _reconstruct_spatial_lines(synthetic_ocr)
    assert reconstructed == "Dept | $48,000\nNote: Approved"


def test_chunk_file_sets_image_vs_document_metadata(tmp_path: Path) -> None:
    img_path = tmp_path / "diagram.webp"
    img = Image.new("RGB", (240, 160), color=(240, 244, 248))
    img.save(img_path, format="WEBP")

    img_chunks = chunk_file(img_path)
    assert len(img_chunks) == 1
    assert img_chunks[0].metadata.file_type == "image"
    assert "No readable text detected" in img_chunks[0].text

    txt_path = tmp_path / "note.txt"
    txt_path.write_text("Standard policy text.", encoding="utf-8")
    doc_chunks = chunk_file(txt_path)
    assert doc_chunks[0].metadata.file_type == "document"


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(200, 40), (350, 70), (700, 120)],
)
def test_recursive_split_respects_max_length_and_determinism(chunk_size: int, chunk_overlap: int) -> None:
    sample = ("Paragraph with policy POL-EXP-402 and receipt threshold $25 USD. " * 25).strip()
    pieces_a = recursive_split_text(sample, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces_b = recursive_split_text(sample, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert pieces_a == pieces_b
    assert all(len(p) <= chunk_size for p in pieces_a)
    ids = [make_chunk_id("doc.md", idx, p) for idx, p in enumerate(pieces_a)]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 2. Hybrid Retrieval & FlashRank Reranking Integration Tests
# ---------------------------------------------------------------------------


def test_hybrid_retrieval_multiformat_and_rerank_ordering(isolated_rag_env) -> None:
    data_dir, chroma_dir, corpus_path = isolated_rag_env

    # 1. Markdown policy
    (data_dir / "travel.md").write_text(
        "# Travel Caps\nZurich hotel lodging cap is $340 USD per night. Submit in VaultSpend within 21 days.",
        encoding="utf-8",
    )
    # 2. CSV directory
    (data_dir / "staff.csv").write_text(
        "employee_id,full_name,project_code\nNV-EMP-1042,Priya Nair,OPS-HYBRID-26\n",
        encoding="utf-8",
    )
    # 3. Image notice (.png)
    img = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=42)
    draw.text((30, 40), "ESCORT CODE VIS-ESC-204", fill="black", font=font)
    draw.text((30, 115), "BADGE FEE 35 USD", fill="black", font=font)
    img.save(data_dir / "badge.png", format="PNG")

    report = ingest_directory(data_dir, chroma_dir, corpus_path, reset_index=True)
    assert report.failed_files == {}
    assert report.chroma_count == report.bm25_doc_count == 3

    # Exact hyphenated employee code lookup -> staff.csv ranked #1
    emp_hits = hybrid_retrieve_and_rerank(
        "Who is employee NV-EMP-1042?",
        top_n=2,
        chroma_dir=chroma_dir,
        corpus_path=corpus_path,
    )
    assert len(emp_hits) == 2
    assert emp_hits[0].source_file == "staff.csv"
    assert emp_hits[0].bm25_rank == 1
    assert emp_hits[0].rerank_score >= emp_hits[1].rerank_score

    # Image OCR code lookup -> badge.png ranked #1
    img_hits = hybrid_retrieve_and_rerank(
        "VIS-ESC-204 badge fee",
        top_n=2,
        chroma_dir=chroma_dir,
        corpus_path=corpus_path,
    )
    assert img_hits[0].source_file == "badge.png"

    # Empty / whitespace query -> []
    assert hybrid_retrieve_and_rerank("   ", chroma_dir=chroma_dir, corpus_path=corpus_path) == []


# ---------------------------------------------------------------------------
# 3. Grounded Generation, Fallback, Quote Verification & Error Handling Tests
# ---------------------------------------------------------------------------


def test_generator_falls_back_to_json_object_when_json_schema_unsupported() -> None:
    chunk = RetrievedChunk(
        chunk_id="travel.md::0",
        text="Lodging cap in Zurich is $340 USD / night.",
        source_file="travel.md",
        page_or_section="Section: Caps",
        rerank_score=0.98,
    )
    valid_payload = RAGResponse(
        answer="The Zurich lodging cap is $340 USD per night.",
        citations=[Citation(source_file="travel.md", quote="Lodging cap in Zurich is $340 USD / night.")],
        is_supported=True,
        confidence="high",
    ).model_dump_json()

    fake_http = MagicMock()
    fake_http.status_code = 400
    schema_unsupported_err = BadRequestError(
        message="Error 400: response_format json_schema is not supported by this model",
        response=fake_http,
        body={"error": {"message": "json_schema not supported"}},
    )

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = [
        schema_unsupported_err,
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=valid_payload))]),
    ]

    with patch("src.generation.generator.get_groq_client", return_value=mock_groq):
        resp = generate_grounded_answer("What is the Zurich cap?", [chunk], model="custom-model")

    assert resp.is_supported is True
    assert resp.confidence == "high"
    assert len(resp.citations) == 1
    assert mock_groq.chat.completions.create.call_count == 2
    second_call_kwargs = mock_groq.chat.completions.create.call_args_list[1].kwargs
    assert second_call_kwargs["response_format"] == {"type": "json_object"}


def test_generator_non_recoverable_bad_request_propagates() -> None:
    chunk = RetrievedChunk(
        chunk_id="a.md::0",
        text="Fact.",
        source_file="a.md",
        page_or_section="S1",
        rerank_score=0.9,
    )
    fake_http = MagicMock()
    fake_http.status_code = 400
    fatal_err = BadRequestError(
        message="Error 400: model_decommissioned",
        response=fake_http,
        body={"error": {"code": "model_decommissioned"}},
    )
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = fatal_err

    with patch("src.generation.generator.get_groq_client", return_value=mock_groq):
        with pytest.raises(BadRequestError):
            generate_grounded_answer("Question?", [chunk])
    assert mock_groq.chat.completions.create.call_count == 1


def test_quote_verifier_normalizes_typography_and_filters_fabricated_quotes() -> None:
    chunk = RetrievedChunk(
        chunk_id="pol.md::0",
        text="Employees accrue **22\u00a0days** during years 1\u20133 under `POL\u2011PTO\u2011305`.",
        source_file="pol.md",
        page_or_section="Section 3",
        rerank_score=0.95,
    )
    llm_payload = RAGResponse(
        answer="Employees accrue 22 days during years 1-3.",
        citations=[
            # Legitimate quote with plain ASCII hyphens/spaces and stripped markdown bold
            Citation(source_file="pol.md", quote="Employees accrue 22 days during years 1-3 under POL-PTO-305."),
            # Fabricated quote from same real file -> must be stripped!
            Citation(source_file="pol.md", quote="Employees also get 99 days of paid space travel."),
        ],
        is_supported=True,
        confidence="high",
    ).model_dump_json()

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=llm_payload))]
    )

    with patch("src.generation.generator.get_groq_client", return_value=mock_groq):
        resp = generate_grounded_answer("How many PTO days in years 1-3?", [chunk])

    assert resp.is_supported is True
    assert len(resp.citations) == 1
    assert "22 days" in resp.citations[0].quote


def test_end_to_end_answer_question_offline_mock(isolated_rag_env) -> None:
    data_dir, chroma_dir, corpus_path = isolated_rag_env
    (data_dir / "sec.txt").write_text(
        "[SEC-1]\nVaultKeeper KMS API keys rotate every 45 days.",
        encoding="utf-8",
    )
    ingest_directory(data_dir, chroma_dir, corpus_path, reset_index=True)

    mock_json = RAGResponse(
        answer="VaultKeeper KMS API keys must rotate every 45 days.",
        citations=[Citation(source_file="sec.txt", quote="VaultKeeper KMS API keys rotate every 45 days.")],
        is_supported=True,
        confidence="high",
    ).model_dump_json()
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=mock_json))]
    )

    with (
        patch(
            "src.generation.generator.hybrid_retrieve_and_rerank",
            side_effect=lambda query, top_n=6: hybrid_retrieve_and_rerank(
                query=query,
                top_n=top_n,
                chroma_dir=chroma_dir,
                corpus_path=corpus_path,
            ),
        ),
        patch("src.generation.generator.get_groq_client", return_value=mock_groq),
    ):
        resp, chunks = answer_question("How often do VaultKeeper KMS keys rotate?", top_n=3)

    assert resp.is_supported is True
    assert "45 days" in resp.answer
    assert len(chunks) == 1
    assert chunks[0].source_file == "sec.txt"
