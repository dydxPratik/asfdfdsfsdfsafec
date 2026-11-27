"""Pydantic data models shared across ingestion, retrieval, generation, and UI."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Required metadata stored alongside every indexed chunk."""

    source_file: str = Field(..., description="Filename of the source company document")
    source: str = Field(default="", description="Source filename alias for citation provenance")
    file_type: str = Field(default="document", description="Source category: 'document' or 'image'")
    page_or_section: str = Field(..., description="Page number (PDF), section header, or CSV/Excel/Image info")
    chunk_id: str = Field(..., description="Deterministic unique identifier for this chunk")


class DocumentChunk(BaseModel):
    """A parsed and chunked unit ready for embedding and indexing."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    """A candidate chunk scored through dense search, BM25, RRF, and FlashRank."""

    chunk_id: str
    text: str
    source_file: str
    page_or_section: str
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class Citation(BaseModel):
    """Exact verbatim supporting quote from a retrieved company document."""

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(..., description="Exact source_file name where the quote appears")
    quote: str = Field(..., description="Exact verbatim substring or sentence from the retrieved chunk")


class RAGResponse(BaseModel):
    """Strict structured output returned by the grounded Groq generator."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        description=(
            "Direct, grounded answer based strictly on retrieved chunks. "
            "If unsupported, explicitly state that the answer is not in the company documents."
        ),
    )
    citations: list[Citation] = Field(
        ...,
        description="Supporting source file citations and exact quotes; empty list [] if unsupported.",
    )
    is_supported: bool = Field(
        ...,
        description="True ONLY if the retrieved company chunks directly answer the question; False otherwise.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence level grounded in retrieval evidence ('low' whenever is_supported is False).",
    )
