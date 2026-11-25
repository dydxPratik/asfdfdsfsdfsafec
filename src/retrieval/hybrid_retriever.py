"""Hybrid Retrieval (ChromaDB Dense + BM25 Sparse -> RRF) + FlashRank CPU Reranker."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from flashrank import Ranker, RerankRequest

from src.config import (
    BGE_QUERY_PREFIX,
    BM25_CORPUS_PATH,
    BM25_TOP_K,
    CHROMA_DIR,
    DENSE_TOP_K,
    FLASHRANK_CACHE_DIR,
    FLASHRANK_MODEL_NAME,
    RERANK_TOP_N,
    RRF_K,
)
from src.ingestion.indexer import (
    get_chroma_client,
    get_embedding_model,
    get_or_create_collection,
    load_bm25_index,
    tokenize_for_bm25,
)
from src.schemas import RetrievedChunk


@lru_cache(maxsize=1)
def get_reranker(
    model_name: str = FLASHRANK_MODEL_NAME,
    cache_dir: str = str(FLASHRANK_CACHE_DIR),
) -> Ranker:
    """Load and cache the local ONNX CPU cross-encoder reranker via FlashRank."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Ranker(
        model_name=model_name,
        cache_dir=cache_dir,
        max_length=512,
        log_level="WARNING",
    )


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    rrf_k: int = RRF_K,
) -> list[RetrievedChunk]:
    """Fuse dense and BM25 ranked lists using Reciprocal Rank Fusion: sum(1 / (k + rank))."""
    merged: dict[str, RetrievedChunk] = {}

    for rank_idx, hit in enumerate(dense_hits, start=1):
        cid = hit["chunk_id"]
        rrf_contrib = 1.0 / (rrf_k + rank_idx)
        merged[cid] = RetrievedChunk(
            chunk_id=cid,
            text=hit["text"],
            source_file=hit["source_file"],
            page_or_section=hit["page_or_section"],
            dense_rank=rank_idx,
            dense_score=round(float(hit["dense_score"]), 4),
            rrf_score=rrf_contrib,
        )

    for rank_idx, hit in enumerate(bm25_hits, start=1):
        cid = hit["chunk_id"]
        rrf_contrib = 1.0 / (rrf_k + rank_idx)
        if cid in merged:
            existing = merged[cid]
            existing.bm25_rank = rank_idx
            existing.bm25_score = round(float(hit["bm25_score"]), 4)
            existing.rrf_score += rrf_contrib
        else:
            merged[cid] = RetrievedChunk(
                chunk_id=cid,
                text=hit["text"],
                source_file=hit["source_file"],
                page_or_section=hit["page_or_section"],
                bm25_rank=rank_idx,
                bm25_score=round(float(hit["bm25_score"]), 4),
                rrf_score=rrf_contrib,
            )

    fused = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)
    for item in fused:
        item.rrf_score = round(item.rrf_score, 6)
    return fused


def hybrid_retrieve_and_rerank(
    query: str,
    top_n: int = RERANK_TOP_N,
    dense_top_k: int = DENSE_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    rrf_k: int = RRF_K,
    chroma_dir: Path = CHROMA_DIR,
    corpus_path: Path = BM25_CORPUS_PATH,
) -> list[RetrievedChunk]:
    """Run ChromaDB dense search + BM25 keyword search -> RRF fusion -> FlashRank reranking."""
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if dense_top_k <= 0 or bm25_top_k <= 0:
        raise ValueError("dense_top_k and bm25_top_k must be positive integers")

    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    client = get_chroma_client(chroma_dir)
    collection = get_or_create_collection(client)
    total_docs = collection.count()
    if total_docs == 0:
        return []

    # 1. Dense Vector Search (BGE query instruction prefix + cosine similarity)
    embedder = get_embedding_model()
    query_embedding = embedder.encode(
        BGE_QUERY_PREFIX + cleaned_query,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    n_dense = min(dense_top_k, total_docs)
    chroma_res = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_dense,
        include=["documents", "metadatas", "distances"],
    )

    dense_hits: list[dict] = []
    if chroma_res["ids"] and chroma_res["ids"][0]:
        for cid, doc_text, meta, dist in zip(
            chroma_res["ids"][0],
            chroma_res["documents"][0],
            chroma_res["metadatas"][0],
            chroma_res["distances"][0],
        ):
            dense_hits.append(
                {
                    "chunk_id": cid,
                    "text": doc_text,
                    "source_file": meta["source_file"],
                    "page_or_section": meta["page_or_section"],
                    "dense_score": 1.0 - float(dist),
                }
            )

    # 2. Sparse Keyword Search (BM25Okapi)
    bm25_hits: list[dict] = []
    bm25_index, corpus_records = load_bm25_index(corpus_path)
    if bm25_index is not None and corpus_records:
        query_tokens = tokenize_for_bm25(cleaned_query)
        if query_tokens:
            scores = bm25_index.get_scores(query_tokens)
            scored_indices = [
                (idx, float(score)) for idx, score in enumerate(scores) if float(score) > 0.0
            ]
            scored_indices.sort(key=lambda pair: pair[1], reverse=True)
            for idx, score in scored_indices[:bm25_top_k]:
                rec = corpus_records[idx]
                bm25_hits.append(
                    {
                        "chunk_id": rec["chunk_id"],
                        "text": rec["text"],
                        "source_file": rec["source_file"],
                        "page_or_section": rec["page_or_section"],
                        "bm25_score": score,
                    }
                )

    # 3. Reciprocal Rank Fusion (RRF)
    fused_candidates = reciprocal_rank_fusion(dense_hits, bm25_hits, rrf_k=rrf_k)
    if not fused_candidates:
        return []

    # 4. Local CPU Cross-Encoder Reranking via FlashRank
    ranker = get_reranker()
    candidate_by_id = {c.chunk_id: c for c in fused_candidates}
    passages = [{"id": c.chunk_id, "text": c.text} for c in fused_candidates]

    reranked_passages = ranker.rerank(RerankRequest(query=cleaned_query, passages=passages))

    final_results: list[RetrievedChunk] = []
    for item in reranked_passages[:top_n]:
        chunk = candidate_by_id[item["id"]]
        chunk.rerank_score = round(float(item["score"]), 4)
        final_results.append(chunk)

    return final_results
