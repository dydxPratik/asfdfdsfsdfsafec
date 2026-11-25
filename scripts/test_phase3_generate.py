"""Phase 3 CLI Verification: Grounded Groq Generation + Strict Pydantic Citations & Abstention."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_GROQ_MODEL, RERANK_TOP_N
from src.generation.generator import answer_question


def run_test_case(label: str, query: str, expect_supported: bool) -> None:
    print("\n" + "=" * 80)
    print(f"[{label}]")
    print(f"QUESTION: {query}")
    print("=" * 80)

    response, chunks = answer_question(query=query, top_n=RERANK_TOP_N, model=DEFAULT_GROQ_MODEL)

    print("\nValidated Pydantic RAGResponse JSON:")
    print(response.model_dump_json(indent=2))

    print(f"\nTop Retrieved Sources Passed to Groq ({len(chunks)} chunks):")
    for idx, ch in enumerate(chunks[:3], start=1):
        print(f"  {idx}. {ch.source_file} ({ch.page_or_section}) — rerank={ch.rerank_score:.4f}")

    assert response.is_supported is expect_supported, (
        f"Expected is_supported={expect_supported}, got {response.is_supported}"
    )
    if expect_supported:
        assert len(response.citations) >= 1, "Expected at least 1 exact source citation!"
    else:
        assert len(response.citations) == 0, "Unsupported answer must have zero citations!"
        assert response.confidence == "low", "Unsupported answer must have confidence='low'!"


def main() -> None:
    print(f"Using Verified Groq Model: {DEFAULT_GROQ_MODEL}")

    # Test 1: Grounded HR & Expense policy question
    run_test_case(
        label="TEST 1: IN-SCOPE POLICY QUESTION (Multi-Chunk Grounded Synthesis)",
        query=(
            "What is the nightly lodging cap in Zurich under POL-EXP-402, "
            "how many days do employees have to submit VaultSpend expenses, "
            "and how much is the new-hire ergonomic stipend?"
        ),
        expect_supported=True,
    )

    # Test 2: Cross-document CSV + PDF lookup
    run_test_case(
        label="TEST 2: CROSS-DOCUMENT CSV + PDF LOOKUP",
        query=(
            "Who is employee NV-EMP-1001, what project code are they assigned to, "
            "and how many months is the standard Helios-9 hardware warranty?"
        ),
        expect_supported=True,
    )

    # Test 3: Out-of-scope hallucination trap (must strictly abstain!)
    run_test_case(
        label="TEST 3: OUT-OF-SCOPE HALLUCINATION TRAP (Strict Refusal Verification)",
        query="What is NexaVault Robotics' 401(k) cryptocurrency match percentage and NYSE stock ticker?",
        expect_supported=False,
    )

    print("\nPhase 3 Status: PASSED (Strict Pydantic JSON, exact citations, and abstention verified).")


if __name__ == "__main__":
    main()
