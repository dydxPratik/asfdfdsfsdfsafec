"""End-to-End RAG Evaluation Runner for RAGQuest.

Executes deterministic evaluation queries through the actual ingestion, hybrid retrieval,
FlashRank reranking, and Groq grounded generation pipeline, recording:
  - Retrieval metrics: DocHitRate@K, EvidenceRecall@K, ContextPrecision@K, MRR@K + 4-way ablation
  - Generation & Guardrail metrics: SupportAccuracy, AbstentionAccuracy, CitationFaithfulness,
    FactCoverage, ZeroHallucinationRate, and QueryAnswerTokenOverlap
  - Stage latencies (retrieve_rerank_ms, generation_ms, total_ms) and Groq token usage
  - Per-query failure diagnostics pinpointing the exact weak stage

Usage:
  uv run python tests/evaluate_rag_e2e.py --mode fast
  uv run python tests/evaluate_rag_e2e.py --mode full
  uv run python tests/evaluate_rag_e2e.py --mode fast --offline-mock
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import openpyxl
import pymupdf
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BGE_QUERY_PREFIX,
    BM25_TOP_K,
    DATA_DIR,
    DEFAULT_GROQ_MODEL,
    DENSE_TOP_K,
    RERANK_TOP_N,
    RRF_K,
    SUPPORTED_EXTENSIONS,
)
from src.generation.generator import (
    _normalize_for_quote_match,
    generate_grounded_answer,
    get_groq_client,
)
from src.ingestion.indexer import (
    get_chroma_client,
    get_embedding_model,
    get_or_create_collection,
    ingest_directory,
    load_bm25_index,
    tokenize_for_bm25,
)
from src.retrieval.hybrid_retriever import hybrid_retrieve_and_rerank, reciprocal_rank_fusion
from src.schemas import Citation, RAGResponse, RetrievedChunk

DATASET_PATH = PROJECT_ROOT / "tests" / "datasets" / "rag_e2e_eval_dataset.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "tests" / "reports" / "e2e_eval_report.json"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how",
    "in", "into", "is", "it", "many", "much", "must", "of", "on", "or", "per", "the",
    "their", "there", "to", "under", "what", "when", "where", "which", "who", "within",
}


def _load_font(size: int = 24) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default(size=size)


def _materialize_eval_corpus(staging_data_dir: Path) -> None:
    """Copy existing ./data files and generate deterministic Excel, PNG, JPG, and scanned-PDF fixtures."""
    staging_data_dir.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(DATA_DIR.iterdir()):
        if src_file.is_file() and src_file.suffix.lower() in SUPPORTED_EXTENSIONS:
            shutil.copy2(src_file, staging_data_dir / src_file.name)

    # 1. Multi-sheet Excel fixture (.xlsx)
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Regional Caps"
    ws1.append(["cost_center", "hub", "annual_budget_usd", "owner"])
    ws1.append(["CC-ZUR-90", "Zurich Lab", 450000, "Hannah Lindqvist"])
    ws1.append(["CC-AUS-12", "Austin HQ", 620000, "Marcus Vance"])
    ws2 = wb.create_sheet("Lab Equipment")
    ws2.append(["asset_tag", "model", "calibration_status"])
    ws2.append(["AMR-HELIOS-88", "Helios-9", "Calibrated-Q1"])
    wb.save(staging_data_dir / "ops_budget_matrix.xlsx")
    wb.close()

    # 2. PNG screenshot notice fixture
    png_img = Image.new("RGB", (920, 280), color=(255, 255, 255))
    draw_png = ImageDraw.Draw(png_img)
    draw_png.text((40, 35), "NEXAVAULT VISITOR BADGE POLICY", fill=(15, 23, 42), font=_load_font(28))
    draw_png.text((40, 105), "Badge Replacement Fee: $35 USD", fill=(15, 23, 42), font=_load_font(23))
    draw_png.text((40, 165), "Mandatory Escort Policy Code: VIS-ESC-204", fill=(15, 23, 42), font=_load_font(23))
    png_img.save(staging_data_dir / "visitor_badge_policy.png", format="PNG")

    # 3. JPG multi-column table fixture
    jpg_img = Image.new("RGB", (960, 320), color=(250, 250, 252))
    draw_jpg = ImageDraw.Draw(jpg_img)
    font_cell = _load_font(22)
    draw_jpg.text((40, 28), "Q2 DEPARTMENT BUDGET CAPS", fill=(10, 10, 10), font=_load_font(25))
    rows = [
        ("Department", "Budget Cap", "Approver"),
        ("Robotics", "$48,000", "E. Rostova"),
        ("Security", "$22,500", "M. Vance"),
    ]
    y = 95
    for c1, c2, c3 in rows:
        draw_jpg.text((40, y), c1, fill=(20, 20, 20), font=font_cell)
        draw_jpg.text((340, y), c2, fill=(20, 20, 20), font=font_cell)
        draw_jpg.text((620, y), c3, fill=(20, 20, 20), font=font_cell)
        y += 58
    jpg_img.save(staging_data_dir / "q2_dept_caps.jpg", format="JPEG", quality=95)

    # 4. Scanned PDF fixture (raster image page with no text layer)
    scan_img = Image.new("RGB", (1400, 340), "white")
    draw_scan = ImageDraw.Draw(scan_img)
    font_scan = _load_font(48)
    draw_scan.text((45, 55), "SCANNED WAREHOUSE ADDENDUM SCN-778", fill="black", font=font_scan)
    draw_scan.text((45, 165), "Monthly warehouse locker fee is 45 USD.", fill="black", font=font_scan)
    buf = io.BytesIO()
    scan_img.save(buf, format="PNG")
    pdf_doc = pymupdf.open()
    pdf_page = pdf_doc.new_page(width=700, height=170)
    pdf_page.insert_image(pdf_page.rect, stream=buf.getvalue())
    pdf_doc.save(staging_data_dir / "scanned_warehouse_addendum.pdf")
    pdf_doc.close()


def _chunk_is_relevant(chunk_text: str, source_file: str, expected_docs: set[str], snippets: list[str]) -> bool:
    if expected_docs and source_file not in expected_docs:
        return False
    norm_txt = _normalize_for_quote_match(chunk_text)
    return any(_normalize_for_quote_match(sn) in norm_txt for sn in snippets) if snippets else (source_file in expected_docs)


def _compute_ablation_doc_And_snippet_recall(
    query: str,
    chroma_dir: Path,
    corpus_path: Path,
    top_n: int,
    expected_docs: set[str],
    snippets: list[str],
) -> dict[str, float]:
    """Compute snippet evidence recall@top_n across dense_only, bm25_only, hybrid_rrf, and full_flashrank."""
    if not query.strip() or not snippets:
        return {}

    client = get_chroma_client(chroma_dir)
    col = get_or_create_collection(client)
    total_docs = col.count()
    embedder = get_embedding_model()
    q_emb = embedder.encode(
        BGE_QUERY_PREFIX + query.strip(),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    c_res = col.query(
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

    bm25_idx, records = load_bm25_index(corpus_path)
    bm25_hits = []
    q_toks = tokenize_for_bm25(query)
    if bm25_idx and q_toks:
        scores = bm25_idx.get_scores(q_toks)
        ranked = sorted([(i, float(s)) for i, s in enumerate(scores) if float(s) > 0.0], key=lambda x: x[1], reverse=True)
        for i, sc in ranked[:BM25_TOP_K]:
            r = records[i]
            bm25_hits.append(
                {
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "source_file": r["source_file"],
                    "page_or_section": r["page_or_section"],
                    "bm25_score": sc,
                }
            )

    rrf_list = reciprocal_rank_fusion(dense_hits, bm25_hits, rrf_k=RRF_K)
    reranked_list = hybrid_retrieve_and_rerank(
        query, top_n=top_n, chroma_dir=chroma_dir, corpus_path=corpus_path
    )

    def _snippet_recall(texts: list[str]) -> float:
        joined = " \n ".join(_normalize_for_quote_match(t) for t in texts)
        matched = sum(1 for sn in snippets if _normalize_for_quote_match(sn) in joined)
        return round(matched / len(snippets), 4)

    return {
        "dense_only": _snippet_recall([h["text"] for h in dense_hits[:top_n]]),
        "bm25_only": _snippet_recall([h["text"] for h in bm25_hits[:top_n]]),
        "hybrid_rrf": _snippet_recall([c.text for c in rrf_list[:top_n]]),
        "full_flashrank": _snippet_recall([c.text for c in reranked_list[:top_n]]),
    }


def _build_offline_mock_client(case: dict[str, Any], retrieved_chunks: list[RetrievedChunk]) -> MagicMock:
    """Deterministic mock Groq client used ONLY when --offline-mock is explicitly passed."""
    mock_client = MagicMock()
    if not case["expected_supported"] or not retrieved_chunks:
        payload = RAGResponse(
            answer="This information is not found in the provided company documents.",
            citations=[],
            is_supported=False,
            confidence="low",
        )
    else:
        cits: list[Citation] = []
        for ch in retrieved_chunks:
            if ch.source_file in case["expected_docs"]:
                first_line = next((ln.strip() for ln in ch.text.splitlines() if len(ln.strip()) >= 8), ch.text[:60])
                cits.append(Citation(source_file=ch.source_file, quote=first_line))
                if len(cits) >= 2:
                    break
        if not cits:
            first_ch = retrieved_chunks[0]
            cits.append(Citation(source_file=first_ch.source_file, quote=first_ch.text.splitlines()[0].strip()))
        facts_str = ", ".join(case["expected_answer_facts"])
        payload = RAGResponse(
            answer=f"Grounded facts from company records: {facts_str}.",
            citations=cits,
            is_supported=True,
            confidence="medium" if "partial" in case["category"] else "high",
        )

    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload.model_dump_json()))],
        usage=SimpleNamespace(prompt_tokens=950, completion_tokens=85, total_tokens=1035),
    )
    return mock_client


def _query_answer_relevance_overlap(query: str, answer: str) -> float:
    q_terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in _STOPWORDS and len(t) > 2}
    if not q_terms:
        return 1.0
    a_terms = set(re.findall(r"[a-z0-9]+", answer.lower()))
    return round(len(q_terms & a_terms) / len(q_terms), 4)


def run_evaluation(
    mode: str = "fast",
    offline_mock: bool = False,
    top_n: int = RERANK_TOP_N,
    model: str = DEFAULT_GROQ_MODEL,
    output_path: Path | None = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    all_cases: list[dict[str, Any]] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c.get("fast_subset")] if mode == "fast" else all_cases

    staging_root = Path(tempfile.mkdtemp(prefix="ragquest_e2e_eval_"))
    staging_data = staging_root / "data"
    staging_chroma = staging_root / "chroma_db"
    staging_corpus = staging_chroma / "bm25_corpus.json"

    try:
        t_ing0 = time.perf_counter()
        _materialize_eval_corpus(staging_data)
        ingest_report = ingest_directory(
            data_dir=staging_data,
            chroma_dir=staging_chroma,
            corpus_path=staging_corpus,
            reset_index=True,
        )
        ingest_ms = round((time.perf_counter() - t_ing0) * 1000.0, 2)

        per_query: list[dict[str, Any]] = []
        ablation_totals: dict[str, list[float]] = {
            "dense_only": [],
            "bm25_only": [],
            "hybrid_rrf": [],
            "full_flashrank": [],
        }

        for case in cases:
            qid = case["id"]
            query = case["query"]
            expected_docs = set(case["expected_docs"])
            snippets = case["evidence_snippets"]

            # 1. Retrieval + Reranking
            t_ret0 = time.perf_counter()
            chunks = hybrid_retrieve_and_rerank(
                query=query,
                top_n=top_n,
                chroma_dir=staging_chroma,
                corpus_path=staging_corpus,
            )
            retrieve_ms = round((time.perf_counter() - t_ret0) * 1000.0, 2)

            # Retrieval metrics (for answerable queries)
            if case["answerable"]:
                ret_docs = {c.source_file for c in chunks}
                doc_hit_rate = 1.0 if expected_docs.issubset(ret_docs) else 0.0
                joined_norm = " \n ".join(_normalize_for_quote_match(c.text) for c in chunks)
                matched_snips = [
                    sn for sn in snippets if _normalize_for_quote_match(sn) in joined_norm
                ]
                evidence_recall = round(len(matched_snips) / len(snippets), 4) if snippets else doc_hit_rate

                rel_flags = [
                    1 if _chunk_is_relevant(c.text, c.source_file, expected_docs, snippets) else 0
                    for c in chunks
                ]
                context_precision = round(sum(rel_flags) / len(chunks), 4) if chunks else 0.0
                mrr = 0.0
                for rank_i, flag in enumerate(rel_flags, start=1):
                    if flag:
                        mrr = round(1.0 / rank_i, 4)
                        break

                ablation = _compute_ablation_doc_And_snippet_recall(
                    query, staging_chroma, staging_corpus, top_n, expected_docs, snippets
                )
                for k_mode, val in ablation.items():
                    ablation_totals[k_mode].append(val)
            else:
                doc_hit_rate = None
                evidence_recall = None
                context_precision = None
                mrr = None

            # 2. Grounded Generation + Token Spy
            usage_spy = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0}
            t_gen0 = time.perf_counter()
            gen_error: str | None = None

            try:
                if offline_mock:
                    mock_groq = _build_offline_mock_client(case, chunks)

                    def _spy_mock_create(*args, **kwargs):
                        usage_spy["llm_calls"] += 1
                        res = mock_groq.chat.completions.create.return_value
                        usage_spy["prompt_tokens"] += res.usage.prompt_tokens
                        usage_spy["completion_tokens"] += res.usage.completion_tokens
                        usage_spy["total_tokens"] += res.usage.total_tokens
                        return res

                    mock_groq.chat.completions.create.side_effect = _spy_mock_create
                    with patch("src.generation.generator.get_groq_client", return_value=mock_groq):
                        resp = generate_grounded_answer(query=query, retrieved_chunks=chunks, model=model)
                else:
                    real_client = get_groq_client()
                    orig_create = real_client.chat.completions.create

                    def _spy_real_create(*args, **kwargs):
                        usage_spy["llm_calls"] += 1
                        res = orig_create(*args, **kwargs)
                        if getattr(res, "usage", None) is not None:
                            usage_spy["prompt_tokens"] += int(getattr(res.usage, "prompt_tokens", 0) or 0)
                            usage_spy["completion_tokens"] += int(getattr(res.usage, "completion_tokens", 0) or 0)
                            usage_spy["total_tokens"] += int(getattr(res.usage, "total_tokens", 0) or 0)
                        return res

                    with (
                        patch("src.generation.generator.get_groq_client", return_value=real_client),
                        patch.object(real_client.chat.completions, "create", side_effect=_spy_real_create),
                    ):
                        resp = generate_grounded_answer(query=query, retrieved_chunks=chunks, model=model)
            except Exception as exc:
                gen_error = f"{type(exc).__name__}: {exc}"
                resp = RAGResponse(
                    answer=f"ERROR: {gen_error}",
                    citations=[],
                    is_supported=False,
                    confidence="low",
                )

            gen_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)

            # 3. Generation & Faithfulness Checks
            verified_cit_count = 0
            for cit in resp.citations:
                src_texts = [_normalize_for_quote_match(c.text) for c in chunks if c.source_file == cit.source_file]
                norm_q = _normalize_for_quote_match(cit.quote)
                if norm_q and any(norm_q in st for st in src_texts):
                    verified_cit_count += 1
            citation_faithfulness = (
                round(verified_cit_count / len(resp.citations), 4) if resp.citations else 1.0
            )

            norm_ans = _normalize_for_quote_match(resp.answer)
            expected_facts = case["expected_answer_facts"]
            matched_facts = [f for f in expected_facts if _normalize_for_quote_match(f) in norm_ans]
            fact_coverage = round(len(matched_facts) / len(expected_facts), 4) if expected_facts else 1.0

            forbidden_hits = [
                bad for bad in case["forbidden_facts"] if _normalize_for_quote_match(bad) in norm_ans
            ]
            support_match = resp.is_supported == case["expected_supported"]
            relevance_overlap = (
                _query_answer_relevance_overlap(query, resp.answer) if resp.is_supported else None
            )

            # Determine pass/fail & pinpoint weak stage
            failed_stage: str | None = None
            failure_reason: str | None = None
            if gen_error:
                failed_stage = "GENERATION_API_ERROR"
                failure_reason = gen_error
            elif case["answerable"] and (evidence_recall is not None and evidence_recall < 1.0):
                failed_stage = "RETRIEVAL"
                failure_reason = f"EvidenceRecall@{top_n}={evidence_recall} (missed required snippet/doc in Top-{top_n})"
            elif not support_match:
                failed_stage = "GENERATION_SUPPORT_CLASSIFICATION"
                failure_reason = f"Expected is_supported={case['expected_supported']}, got {resp.is_supported}"
            elif citation_faithfulness < 1.0:
                failed_stage = "CITATION_VERIFICATION"
                failure_reason = f"Unverified citation leaked (faithfulness={citation_faithfulness})"
            elif forbidden_hits:
                failed_stage = "GENERATION_HALLUCINATION"
                failure_reason = f"Forbidden hallucinated content present: {forbidden_hits}"
            elif case["answerable"] and fact_coverage < 1.0:
                failed_stage = "GENERATION_COMPLETENESS"
                missing_f = [f for f in expected_facts if f not in matched_facts]
                failure_reason = f"Missing expected facts in answer: {missing_f}"

            per_query.append(
                {
                    "id": qid,
                    "category": case["category"],
                    "format": case["format"],
                    "query": query,
                    "answerable": case["answerable"],
                    "passed": failed_stage is None,
                    "failed_stage": failed_stage,
                    "failure_reason": failure_reason,
                    "retrieval": {
                        "retrieved_count": len(chunks),
                        "top_sources": [c.source_file for c in chunks[:3]],
                        "top1_rerank_score": chunks[0].rerank_score if chunks else 0.0,
                        f"doc_hit_rate@{top_n}": doc_hit_rate,
                        f"evidence_recall@{top_n}": evidence_recall,
                        f"context_precision@{top_n}": context_precision,
                        f"mrr@{top_n}": mrr,
                    },
                    "generation": {
                        "expected_supported": case["expected_supported"],
                        "actual_supported": resp.is_supported,
                        "confidence": resp.confidence,
                        "citations_count": len(resp.citations),
                        "citation_faithfulness": citation_faithfulness,
                        "fact_coverage": fact_coverage,
                        "query_answer_token_overlap": relevance_overlap,
                        "forbidden_hits": forbidden_hits,
                        "answer_preview": resp.answer[:220],
                    },
                    "latency_ms": {
                        "retrieve_rerank_ms": retrieve_ms,
                        "generation_ms": gen_ms,
                        "total_ms": round(retrieve_ms + gen_ms, 2),
                    },
                    "token_usage": usage_spy,
                }
            )

        # Aggregate summary
        ans_rows = [r for r in per_query if r["answerable"]]
        unans_rows = [r for r in per_query if not r["answerable"]]
        supported_rows = [r for r in per_query if r["generation"]["query_answer_token_overlap"] is not None]

        def _avg(vals: list[float]) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        total_prompt_toks = sum(r["token_usage"]["prompt_tokens"] for r in per_query)
        total_comp_toks = sum(r["token_usage"]["completion_tokens"] for r in per_query)
        total_toks = sum(r["token_usage"]["total_tokens"] for r in per_query)
        total_llm_calls = sum(r["token_usage"]["llm_calls"] for r in per_query)

        summary = {
            "mode": mode,
            "execution_backend": "offline_mock" if offline_mock else f"live_groq ({model})",
            "top_n": top_n,
            "ingestion": {
                "files_indexed": len(ingest_report.files_processed),
                "total_chunks": ingest_report.total_chunks_indexed,
                "failed_files": ingest_report.failed_files,
                "ingest_duration_ms": ingest_ms,
            },
            "counts": {
                "total_queries": len(per_query),
                "answerable_queries": len(ans_rows),
                "unanswerable_or_edge_queries": len(unans_rows),
                "passed_queries": sum(1 for r in per_query if r["passed"]),
                "failed_queries": sum(1 for r in per_query if not r["passed"]),
                "overall_pass_rate": _avg([1.0 if r["passed"] else 0.0 for r in per_query]),
            },
            "retrieval_metrics": {
                f"doc_hit_rate@{top_n}": _avg([r["retrieval"][f"doc_hit_rate@{top_n}"] for r in ans_rows]),
                f"evidence_recall@{top_n}": _avg([r["retrieval"][f"evidence_recall@{top_n}"] for r in ans_rows]),
                f"context_precision@{top_n}": _avg([r["retrieval"][f"context_precision@{top_n}"] for r in ans_rows]),
                f"mrr@{top_n}": _avg([r["retrieval"][f"mrr@{top_n}"] for r in ans_rows]),
                f"ablation_evidence_recall@{top_n}": {
                    m_key: _avg(m_vals) for m_key, m_vals in ablation_totals.items()
                },
            },
            "generation_metrics": {
                "support_classification_accuracy": _avg(
                    [1.0 if r["generation"]["expected_supported"] == r["generation"]["actual_supported"] else 0.0 for r in per_query]
                ),
                "abstention_accuracy_unanswerable": _avg(
                    [1.0 if (not r["generation"]["actual_supported"] and r["generation"]["citations_count"] == 0) else 0.0 for r in unans_rows]
                ),
                "citation_faithfulness": _avg([r["generation"]["citation_faithfulness"] for r in per_query]),
                "answer_fact_coverage": _avg([r["generation"]["fact_coverage"] for r in ans_rows]),
                "zero_hallucination_rate": _avg(
                    [1.0 if not r["generation"]["forbidden_hits"] else 0.0 for r in per_query]
                ),
                "mean_query_answer_token_overlap": _avg(
                    [r["generation"]["query_answer_token_overlap"] for r in supported_rows]
                ),
            },
            "latency_and_tokens": {
                "mean_retrieve_rerank_ms": round(_avg([r["latency_ms"]["retrieve_rerank_ms"] for r in per_query]), 2),
                "mean_generation_ms": round(_avg([r["latency_ms"]["generation_ms"] for r in per_query]), 2),
                "mean_total_ms": round(_avg([r["latency_ms"]["total_ms"] for r in per_query]), 2),
                "total_llm_api_calls": total_llm_calls,
                "total_prompt_tokens": total_prompt_toks,
                "total_completion_tokens": total_comp_toks,
                "total_tokens": total_toks,
                "cost_note": "Groq API returns token counts without USD billing rates; token totals are exact.",
            },
            "failures": [
                {
                    "id": r["id"],
                    "category": r["category"],
                    "failed_stage": r["failed_stage"],
                    "failure_reason": r["failure_reason"],
                    "query": r["query"],
                    "answer_preview": r["generation"]["answer_preview"],
                }
                for r in per_query
                if not r["passed"]
            ],
            "per_query": per_query,
        }

        _print_report(summary)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"\nSaved JSON evaluation report -> {output_path}")

        return summary
    finally:
        gc.collect()
        shutil.rmtree(staging_root, ignore_errors=True)


def _print_report(summary: dict[str, Any]) -> None:
    top_n = summary["top_n"]
    c = summary["counts"]
    ret = summary["retrieval_metrics"]
    gen = summary["generation_metrics"]
    lat = summary["latency_and_tokens"]
    ing = summary["ingestion"]

    print("=" * 84)
    print(f"RAGQUEST END-TO-END EVALUATION REPORT | mode={summary['mode']} | backend={summary['execution_backend']}")
    print("=" * 84)
    print(
        f"[INGESTION] Indexed {ing['files_indexed']} files ({ing['total_chunks']} chunks) "
        f"in {ing['ingest_duration_ms']:.1f} ms | failed_files={ing['failed_files']}"
    )
    print(
        f"[OVERALL]   Passed {c['passed_queries']}/{c['total_queries']} "
        f"({c['overall_pass_rate'] * 100:.1f}%) | "
        f"Answerable={c['answerable_queries']} | Unanswerable/Edge={c['unanswerable_or_edge_queries']}"
    )
    print("\n[RETRIEVAL METRICS (Answerable Queries)]")
    print(
        f"  DocHitRate@{top_n}      : {ret[f'doc_hit_rate@{top_n}']:.4f}\n"
        f"  EvidenceRecall@{top_n}  : {ret[f'evidence_recall@{top_n}']:.4f}\n"
        f"  ContextPrecision@{top_n}: {ret[f'context_precision@{top_n}']:.4f}\n"
        f"  MRR@{top_n}             : {ret[f'mrr@{top_n}']:.4f}"
    )
    abl = ret[f"ablation_evidence_recall@{top_n}"]
    print(
        f"  Ablation Recall@{top_n} : dense={abl['dense_only']:.4f} | bm25={abl['bm25_only']:.4f} | "
        f"rrf={abl['hybrid_rrf']:.4f} | flashrank={abl['full_flashrank']:.4f}"
    )
    print("\n[GENERATION & GUARDRAIL METRICS]")
    print(
        f"  SupportClassAccuracy : {gen['support_classification_accuracy']:.4f}\n"
        f"  AbstentionAccuracy   : {gen['abstention_accuracy_unanswerable']:.4f}\n"
        f"  CitationFaithfulness : {gen['citation_faithfulness']:.4f}\n"
        f"  AnswerFactCoverage   : {gen['answer_fact_coverage']:.4f}\n"
        f"  ZeroHallucinationRate: {gen['zero_hallucination_rate']:.4f}\n"
        f"  QueryAnswerOverlap   : {gen['mean_query_answer_token_overlap']:.4f}"
    )
    print("\n[LATENCY & TOKEN USAGE]")
    print(
        f"  Mean Retrieve+Rerank : {lat['mean_retrieve_rerank_ms']:.1f} ms\n"
        f"  Mean LLM Generation  : {lat['mean_generation_ms']:.1f} ms\n"
        f"  Mean End-to-End      : {lat['mean_total_ms']:.1f} ms\n"
        f"  Groq API Calls       : {lat['total_llm_api_calls']} | "
        f"Prompt={lat['total_prompt_tokens']} | Completion={lat['total_completion_tokens']} | Total={lat['total_tokens']} tokens"
    )
    print("\n[PER-QUERY FAILURE DIAGNOSTICS]")
    if not summary["failures"]:
        print("  None — all evaluated queries passed every retrieval, grounding, and citation check.")
    else:
        for f in summary["failures"]:
            print(f"  - [{f['id']}] ({f['category']}) Stage={f['failed_stage']}: {f['failure_reason']}")
            print(f"      Query : {f['query']}")
            print(f"      Answer: {f['answer_preview']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end RAG evaluation on RAGQuest.")
    parser.add_argument(
        "--mode",
        choices=("fast", "full"),
        default="fast",
        help="Run fast representative subset (8 queries) or full suite (18 queries).",
    )
    parser.add_argument(
        "--offline-mock",
        action="store_true",
        help="Mock Groq LLM calls for offline/CI execution while running real local ingestion & retrieval.",
    )
    parser.add_argument("--top-n", type=int, default=RERANK_TOP_N, help="Reranked Top-N chunks passed to generator.")
    parser.add_argument("--model", type=str, default=DEFAULT_GROQ_MODEL, help="Groq model identifier.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print evaluation report to stdout without writing a JSON file.",
    )
    args = parser.parse_args()
    out_p = None if args.no_save else Path(args.output)
    run_evaluation(
        mode=args.mode,
        offline_mock=args.offline_mock,
        top_n=args.top_n,
        model=args.model,
        output_path=out_p,
    )


if __name__ == "__main__":
    main()
