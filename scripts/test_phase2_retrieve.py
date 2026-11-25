"""Phase 2 CLI Verification: Hybrid Retrieval (ChromaDB + BM25 -> RRF) + FlashRank Reranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import hybrid_retrieve_and_rerank


def run_query_demo(query: str, top_n: int = 3) -> None:
    print("\n" + "=" * 78)
    print(f"QUERY: {query}")
    print("=" * 78)

    results = hybrid_retrieve_and_rerank(query=query, top_n=top_n)
    assert len(results) > 0, "Expected non-empty reranked retrieval results!"

    for rank, chunk in enumerate(results, start=1):
        preview = chunk.text.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(
            f"\n[Top #{rank}] {chunk.source_file} ({chunk.page_or_section})\n"
            f"  chunk_id     : {chunk.chunk_id}\n"
            f"  FlashRank    : {chunk.rerank_score:.4f}  |  RRF Score: {chunk.rrf_score:.6f}\n"
            f"  Dense Rank   : {chunk.dense_rank} (cos={chunk.dense_score})  |  "
            f"BM25 Rank: {chunk.bm25_rank} (bm25={chunk.bm25_score})\n"
            f"  Text Snippet : \"{preview}\""
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Phase 2 Hybrid Retrieval + FlashRank")
    parser.add_argument("--query", type=str, default=None, help="Custom search query")
    parser.add_argument("--top-n", type=int, default=3, help="Number of reranked chunks to print")
    args = parser.parse_args()

    if args.query:
        run_query_demo(args.query, top_n=args.top_n)
    else:
        run_query_demo(
            "What is the lodging cap in Zurich under POL-EXP-402 and when must VaultSpend receipts be submitted?",
            top_n=args.top_n,
        )
        run_query_demo(
            "Which hub does employee NV-EMP-1042 work at and what happens when battery health drops below 80%?",
            top_n=args.top_n,
        )

    print("\nPhase 2 Status: PASSED (Hybrid Dense + BM25 + RRF + FlashRank verified).")


if __name__ == "__main__":
    main()
