"""Phase 3 & Phase 6 Empirical Audit: Retrieval Metrics (P@K, R@K, MRR, Hit@K, NDCG@K) & Chunking."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BGE_QUERY_PREFIX, BM25_CORPUS_PATH, CHROMA_DIR, DENSE_TOP_K, BM25_TOP_K, RRF_K
from src.ingestion.indexer import (
    get_chroma_client,
    get_embedding_model,
    get_or_create_collection,
    load_bm25_index,
    tokenize_for_bm25,
)
from src.retrieval.hybrid_retriever import hybrid_retrieve_and_rerank, reciprocal_rank_fusion


def dcg_at_k(relevances: list[int], k: int) -> float:
    score = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            score += float(rel) / math.log2(idx + 1)
    return score


def compute_ir_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float]:
    top_k = retrieved_ids[:k]
    rels = [1 if cid in relevant_ids else 0 for cid in top_k]
    hits = sum(rels)

    hit_rate = 1.0 if hits > 0 else 0.0
    precision = hits / float(k) if k > 0 else 0.0
    recall = hits / float(len(relevant_ids)) if relevant_ids else 0.0

    mrr = 0.0
    for rank_idx, rel in enumerate(rels, start=1):
        if rel > 0:
            mrr = 1.0 / float(rank_idx)
            break

    ideal_rels = [1] * min(len(relevant_ids), k) + [0] * max(0, k - len(relevant_ids))
    idcg = dcg_at_k(ideal_rels, k)
    ndcg = (dcg_at_k(rels, k) / idcg) if idcg > 0 else 0.0

    return {
        f"HitRate@{k}": hit_rate,
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        f"MRR@{k}": mrr,
        f"NDCG@{k}": ndcg,
    }


def run_ablation_retrievers(query: str) -> dict[str, list[str]]:
    """Return ranked chunk_id lists for Dense-only, BM25-only, Hybrid-RRF, and Full FlashRank."""
    client = get_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)
    total_docs = collection.count()

    embedder = get_embedding_model()
    q_emb = embedder.encode(
        BGE_QUERY_PREFIX + query.strip(),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    c_res = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=min(DENSE_TOP_K, total_docs),
        include=["documents", "metadatas", "distances"],
    )
    dense_hits = [
        {
            "chunk_id": cid,
            "text": doc,
            "source_file": meta["source_file"],
            "page_or_section": meta["page_or_section"],
            "dense_score": 1.0 - float(dist),
        }
        for cid, doc, meta, dist in zip(
            c_res["ids"][0], c_res["documents"][0], c_res["metadatas"][0], c_res["distances"][0]
        )
    ]

    bm25, records = load_bm25_index(BM25_CORPUS_PATH)
    bm25_hits = []
    q_toks = tokenize_for_bm25(query)
    if bm25 and q_toks:
        scores = bm25.get_scores(q_toks)
        scored = sorted(
            [(i, float(s)) for i, s in enumerate(scores) if float(s) > 0.0],
            key=lambda x: x[1],
            reverse=True,
        )
        for i, s in scored[:BM25_TOP_K]:
            r = records[i]
            bm25_hits.append(
                {
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "source_file": r["source_file"],
                    "page_or_section": r["page_or_section"],
                    "bm25_score": s,
                }
            )

    rrf_fused = reciprocal_rank_fusion(dense_hits, bm25_hits, rrf_k=RRF_K)
    full_reranked = hybrid_retrieve_and_rerank(query, top_n=6)

    return {
        "dense_only": [h["chunk_id"] for h in dense_hits],
        "bm25_only": [h["chunk_id"] for h in bm25_hits],
        "hybrid_rrf": [c.chunk_id for c in rrf_fused],
        "full_flashrank": [c.chunk_id for c in full_reranked],
    }


def audit_chunking_quality() -> dict:
    """Inspect actual persisted chunks for overlap loss, header loss, metadata accuracy, and encoding."""
    records = json.loads(BM25_CORPUS_PATH.read_text(encoding="utf-8"))
    lengths = [len(r["text"]) for r in records]

    consecutive_pairs = []
    zero_overlap_same_section = []
    for prev, curr in zip(records, records[1:]):
        if prev["source_file"] == curr["source_file"] and prev["page_or_section"] == curr["page_or_section"]:
            # Compute longest suffix of prev['text'] that is a prefix of curr['text']
            max_ov = 0
            limit = min(len(prev["text"]), len(curr["text"]), 250)
            for k in range(1, limit + 1):
                if prev["text"][-k:] == curr["text"][:k]:
                    max_ov = k
            pair_info = {
                "file": prev["source_file"],
                "section": prev["page_or_section"],
                "prev_id": prev["chunk_id"],
                "curr_id": curr["chunk_id"],
                "exact_overlap_chars": max_ov,
            }
            consecutive_pairs.append(pair_info)
            if max_ov == 0:
                zero_overlap_same_section.append(pair_info)

    encoding_artifacts = [
        {"chunk_id": r["chunk_id"], "snippet": r["text"][:90]}
        for r in records
        if " ? " in r["text"]
    ]

    csv_label_mismatches = []
    for r in records:
        if r["source_file"].endswith(".csv") and "Rows " in r["page_or_section"]:
            rec_count = r["text"].count("Record ")
            span = r["page_or_section"].replace("Rows ", "").split("-")
            expected_count = int(span[1]) - int(span[0]) + 1
            if rec_count != expected_count:
                csv_label_mismatches.append(
                    {
                        "chunk_id": r["chunk_id"],
                        "labeled_section": r["page_or_section"],
                        "actual_records_in_chunk": rec_count,
                    }
                )

    return {
        "total_chunks": len(records),
        "min_chars": min(lengths),
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "max_chars": max(lengths),
        "same_section_splits": consecutive_pairs,
        "zero_overlap_splits_count": len(zero_overlap_same_section),
        "zero_overlap_splits": zero_overlap_same_section,
        "pdf_encoding_artifacts": encoding_artifacts,
        "csv_metadata_row_range_mismatches": csv_label_mismatches,
    }


def main(output_json_path: str | None = None) -> None:
    dataset_path = PROJECT_ROOT / "tests" / "datasets" / "ground_truth_dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    ks = (1, 3, 6)
    modes = ("dense_only", "bm25_only", "hybrid_rrf", "full_flashrank")
    agg: dict[str, dict[str, list[float]]] = {m: {} for m in modes}

    per_query_details = []
    answerable_top1_scores = []
    unanswerable_top1_scores = []

    for item in dataset:
        q = item["query"]
        rel_chunks = set(item["relevant_chunks"])
        rel_docs = set(item["relevant_docs"])
        full_chunks = hybrid_retrieve_and_rerank(q, top_n=6)
        ablation_lists = run_ablation_retrievers(q)

        top1_score = full_chunks[0].rerank_score if full_chunks else 0.0
        if item["answerable"]:
            answerable_top1_scores.append(top1_score)
            for m in modes:
                for k in ks:
                    metrics = compute_ir_metrics(ablation_lists[m], rel_chunks, k)
                    for m_name, val in metrics.items():
                        agg[m].setdefault(m_name, []).append(val)

            retrieved_top6_ids = [c.chunk_id for c in full_chunks]
            retrieved_top6_docs = [c.source_file for c in full_chunks]
            hit_chunks = [cid for cid in retrieved_top6_ids if cid in rel_chunks]
            missed_chunks = [cid for cid in rel_chunks if cid not in retrieved_top6_ids]
            doc_hit = rel_docs.issubset(set(retrieved_top6_docs))
            status = "PASS" if len(missed_chunks) == 0 else ("PARTIAL" if hit_chunks else "FAIL")
        else:
            unanswerable_top1_scores.append(top1_score)
            retrieved_top6_ids = [c.chunk_id for c in full_chunks]
            hit_chunks = []
            missed_chunks = []
            doc_hit = True
            # Note: retriever always returns 6 chunks without a minimum score cutoff
            status = "PASS_NO_CUTOFF_OBSERVED"

        per_query_details.append(
            {
                "id": item["id"],
                "category": item["category"],
                "query": q,
                "answerable": item["answerable"],
                "expected_chunks": list(rel_chunks),
                "top3_retrieved": [
                    {
                        "chunk_id": c.chunk_id,
                        "source_file": c.source_file,
                        "rerank_score": c.rerank_score,
                        "dense_rank": c.dense_rank,
                        "bm25_rank": c.bm25_rank,
                    }
                    for c in full_chunks[:3]
                ],
                "top6_hit_chunks": hit_chunks,
                "top6_missed_chunks": missed_chunks,
                "all_relevant_docs_in_top6": doc_hit,
                "top1_rerank_score": top1_score,
                "status": status,
            }
        )

    summary_metrics = {
        m: {metric_name: round(sum(vals) / len(vals), 4) for metric_name, vals in m_dict.items()}
        for m, m_dict in agg.items()
    }

    chunking_audit = audit_chunking_quality()
    report = {
        "total_queries": len(dataset),
        "answerable_queries": sum(1 for d in dataset if d["answerable"]),
        "unanswerable_queries": sum(1 for d in dataset if not d["answerable"]),
        "summary_metrics_chunk_level": summary_metrics,
        "score_distribution": {
            "answerable_top1_min": round(min(answerable_top1_scores), 4),
            "answerable_top1_mean": round(sum(answerable_top1_scores) / len(answerable_top1_scores), 4),
            "answerable_top1_max": round(max(answerable_top1_scores), 4),
            "unanswerable_top1_min": round(min(unanswerable_top1_scores), 4),
            "unanswerable_top1_mean": round(sum(unanswerable_top1_scores) / len(unanswerable_top1_scores), 4),
            "unanswerable_top1_max": round(max(unanswerable_top1_scores), 4),
        },
        "chunking_audit": chunking_audit,
        "per_query_details": per_query_details,
    }

    print("=" * 80)
    print("PHASE 3 & 6 AUDIT: RETRIEVAL METRICS (24 ANSWERABLE + 3 UNANSWERABLE QUERIES)")
    print("=" * 80)
    for mode_name, m_vals in summary_metrics.items():
        print(f"\n[{mode_name.upper()}]")
        for k in ks:
            print(
                f"  K={k}: HitRate={m_vals[f'HitRate@{k}']:.4f} | "
                f"Recall={m_vals[f'Recall@{k}']:.4f} | "
                f"Precision={m_vals[f'Precision@{k}']:.4f} | "
                f"MRR={m_vals[f'MRR@{k}']:.4f} | "
                f"NDCG={m_vals[f'NDCG@{k}']:.4f}"
            )

    print("\n[SCORE SEPARATION (FlashRank Top-1)]")
    print(" ", report["score_distribution"])

    print("\n[QUERY FAILURES / PARTIALS IN TOP-6]")
    issues = [q for q in per_query_details if q["status"] not in ("PASS", "PASS_NO_CUTOFF_OBSERVED")]
    if not issues:
        print("  None! 100% of ground-truth chunks retrieved in Top-6.")
    else:
        for iss in issues:
            print(f"  - {iss['id']} ({iss['category']}): {iss['status']} | missed={iss['top6_missed_chunks']}")

    print("\n[CHUNKING & METADATA AUDIT FINDINGS]")
    print(f"  Total chunks: {chunking_audit['total_chunks']} | min={chunking_audit['min_chars']} | mean={chunking_audit['mean_chars']} | max={chunking_audit['max_chars']}")
    print(f"  Zero-overlap same-section splits: {chunking_audit['zero_overlap_splits_count']} / {len(chunking_audit['same_section_splits'])}")
    print(f"  CSV row-range label mismatches  : {len(chunking_audit['csv_metadata_row_range_mismatches'])}")
    print(f"  PDF font encoding '?' artifacts : {len(chunking_audit['pdf_encoding_artifacts'])}")

    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    out_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(out_arg)
