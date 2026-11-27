"""Tests for Multilingual RAG support: Unicode BM25 tokenization, embedding profiles, and cross-lingual generation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import EMBEDDING_PROFILES, SUPPORTED_RESPONSE_LANGUAGES
from src.generation.generator import generate_grounded_answer
from src.ingestion.indexer import tokenize_for_bm25
from src.schemas import Citation, RAGResponse, RetrievedChunk


def test_bm25_tokenizer_handles_unicode_scripts_and_preserves_codes() -> None:
    tokens = tokenize_for_bm25("Zürich café POL-EXP-402 नीति दिल्ली")
    assert "zürich" in tokens
    assert "café" in tokens
    assert "pol-exp-402" in tokens
    assert "pol" in tokens
    assert "exp" in tokens
    assert "402" in tokens
    assert "नीति" in tokens
    assert "दिल्ली" in tokens
    assert tokenize_for_bm25("??? !!! ---") == []


def test_embedding_profiles_and_response_languages_configured() -> None:
    assert "english" in EMBEDDING_PROFILES
    assert "multilingual" in EMBEDDING_PROFILES
    assert EMBEDDING_PROFILES["multilingual"]["model_name"] == "intfloat/multilingual-e5-small"
    assert "Hindi" in SUPPORTED_RESPONSE_LANGUAGES
    assert "Auto (Match Query)" in SUPPORTED_RESPONSE_LANGUAGES


def test_cross_lingual_generation_keeps_verbatim_english_citation() -> None:
    chunk = RetrievedChunk(
        chunk_id="policy.md::0",
        text="Lodging cap in Zurich is $340 USD / night.",
        source_file="policy.md",
        page_or_section="Section 2.1",
        rerank_score=0.95,
    )
    hindi_resp = RAGResponse(
        answer="ज्यूरिख में होटल की सीमा $340 USD प्रति रात है।",
        citations=[Citation(source_file="policy.md", quote="Lodging cap in Zurich is $340 USD / night.")],
        is_supported=True,
        confidence="high",
    ).model_dump_json()

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=hindi_resp))]
    )

    with patch("src.generation.generator.get_groq_client", return_value=mock_groq):
        resp = generate_grounded_answer(
            query="ज्यूरिख में होटल की सीमा क्या है?",
            retrieved_chunks=[chunk],
            response_language="Hindi",
        )

    assert resp.is_supported is True
    assert len(resp.citations) == 1
    assert "Hindi" in mock_groq.chat.completions.create.call_args.kwargs["messages"][1]["content"]
