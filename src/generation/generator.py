"""Grounded RAG generation with Groq SDK and strict Pydantic schema enforcement."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from groq import BadRequestError, Groq

from src.config import DEFAULT_GROQ_MODEL, PROJECT_ROOT, RERANK_TOP_N
from src.retrieval.hybrid_retriever import hybrid_retrieve_and_rerank
from src.schemas import Citation, RAGResponse, RetrievedChunk

SYSTEM_PROMPT = """You are a strict, zero-hallucination Enterprise Policy & Knowledge Assistant.
You must answer the user's question ONLY using facts explicitly stated inside <retrieved_company_chunks>.

STRICT GROUNDING RULES:
1. NEVER use external knowledge, prior training assumptions, or guesswork.
2. If NONE of the requested information is present in <retrieved_company_chunks>:
   - Set "is_supported" to false.
   - Set "confidence" to "low".
   - Set "citations" to [].
   - Explicitly state in "answer" that this information is not found in the provided company documents.
3. If the answer IS fully supported by <retrieved_company_chunks>:
   - Set "is_supported" to true and "confidence" to "high".
   - Provide concise, accurate details matching exact numbers, codes, dates, and conditions in the text.
   - Populate "citations" with entries where "source_file" is the exact filename from the chunk header and "quote" is an exact verbatim excerpt copied from that chunk.
4. If the user asks a multi-part question and some parts ARE explicitly answered in <retrieved_company_chunks> while another part is missing:
   - Answer every supported part accurately with exact "citations", and explicitly note which sub-question is not covered in the retrieved chunks (never guess it).
   - Set "is_supported" to true and "confidence" to "medium".
5. Output valid JSON matching the required schema with keys: "answer", "citations", "is_supported", "confidence"."""


def get_groq_client() -> Groq:
    """Load GROQ_API_KEY from .env and return an authenticated Groq client."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing or empty. Add GROQ_API_KEY=... to your .env file."
        )
    return Groq(api_key=api_key)


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Format reranked chunks with clear XML provenance boundaries for the LLM."""
    blocks: list[str] = []
    for idx, ch in enumerate(chunks, start=1):
        blocks.append(
            f'<chunk index="{idx}" source_file="{ch.source_file}" '
            f'location="{ch.page_or_section}" rerank_score="{ch.rerank_score:.4f}">\n'
            f"{ch.text}\n"
            f"</chunk>"
        )
    return "<retrieved_company_chunks>\n" + "\n\n".join(blocks) + "\n</retrieved_company_chunks>"


def generate_grounded_answer(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    model: str = DEFAULT_GROQ_MODEL,
) -> RAGResponse:
    """Generate a strictly grounded, Pydantic-validated RAGResponse from retrieved chunks."""
    if not retrieved_chunks:
        return RAGResponse(
            answer="This information is not found in the provided company documents (no relevant chunks were retrieved).",
            citations=[],
            is_supported=False,
            confidence="low",
        )

    client = get_groq_client()
    context_xml = format_chunks_for_prompt(retrieved_chunks)
    schema_dict = RAGResponse.model_json_schema()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{context_xml}\n\n"
                f"User Question: {query}\n\n"
                "Respond strictly in JSON matching the RAGResponse schema."
            ),
        },
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=2048,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_response",
                    "strict": True,
                    "schema": schema_dict,
                },
            },
        )
    except BadRequestError as exc:
        # Fallback to json_object mode ONLY if a user-selected model does not support json_schema
        if "json_schema" not in str(exc) and "response_format" not in str(exc):
            raise
        schema_prompt = (
            SYSTEM_PROMPT
            + "\n\nJSON SCHEMA TO FOLLOW EXACTLY:\n"
            + json.dumps(schema_dict, indent=2)
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": schema_prompt},
                messages[1],
            ],
            temperature=0.0,
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
        )

    raw_content = completion.choices[0].message.content or ""
    parsed = RAGResponse.model_validate_json(raw_content)

    # Post-generation provenance enforcement
    valid_sources = {c.source_file for c in retrieved_chunks}
    verified_citations: list[Citation] = [
        cit for cit in parsed.citations if cit.source_file in valid_sources and cit.quote.strip()
    ]

    if not parsed.is_supported:
        return RAGResponse(
            answer=parsed.answer,
            citations=[],
            is_supported=False,
            confidence="low",
        )

    return RAGResponse(
        answer=parsed.answer,
        citations=verified_citations,
        is_supported=parsed.is_supported,
        confidence=parsed.confidence,
    )


def answer_question(
    query: str,
    top_n: int = RERANK_TOP_N,
    model: str = DEFAULT_GROQ_MODEL,
) -> tuple[RAGResponse, list[RetrievedChunk]]:
    """End-to-end RAG pipeline: Hybrid Retrieve + FlashRank Rerank + Grounded Groq Answer."""
    chunks = hybrid_retrieve_and_rerank(query=query, top_n=top_n)
    response = generate_grounded_answer(query=query, retrieved_chunks=chunks, model=model)
    return response, chunks
