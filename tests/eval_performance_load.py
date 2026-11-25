"""Phase 8 Empirical Audit: Performance, Scaling (20 -> 200 -> 850 chunks), Concurrency, Memory & Tokens."""

from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_GROQ_MODEL
from src.generation.generator import format_chunks_for_prompt, get_groq_client
from src.ingestion.indexer import get_embedding_model, ingest_directory
from src.retrieval.hybrid_retriever import get_reranker, hybrid_retrieve_and_rerank
from src.schemas import RAGResponse


def build_synthetic_workload(data_dir: Path, num_md_files: int, include_large_pdf_pages: int = 0) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_md_files):
        sections = []
        for s in range(4):
            sections.append(
                f"## Synthetic Policy Module {i:03d}-{s}\n"
                f"Control Code SYN-{i:03d}-{s:02d} mandates automated telemetry verification every "
                f"{(i % 30) + 5} hours across regional robotics cluster zone-{i % 7}. "
                f"Escalation owners must archive audit checksums within {(s + 1) * 15} minutes "
                f"and maintain redundant LFP battery reserve above {75 + (i % 15)}% capacity."
            )
        (data_dir / f"synthetic_policy_{i:03d}.md").write_text("\n\n".join(sections), encoding="utf-8")

    if include_large_pdf_pages > 0:
        doc = pymupdf.open()
        for p in range(1, include_large_pdf_pages + 1):
            page = doc.new_page(width=612, height=792)
            txt = (
                f"LARGE ENTERPRISE ARCHITECTURE MANUAL - PAGE {p}\n\n"
                + (
                    f"Subsystem ARCH-PG-{p:03d} enforces deterministic CAN-bus motor heartbeats every 10ms "
                    f"with thermal shutdown at {65 + (p % 20)} Celsius across warehouse aisles. "
                )
                * 8
            )
            page.insert_textbox(pymupdf.Rect(45, 45, 567, 747), txt, fontsize=10, fontname="helv")
        doc.save(data_dir / "large_enterprise_manual.pdf")
        doc.close()


def benchmark_scale_tier(label: str, num_md: int, pdf_pages: int) -> dict:
    proc = psutil.Process(os.getpid())
    tmp = Path(tempfile.mkdtemp(prefix=f"ragquest_perf_{label}_"))
    try:
        d_dir = tmp / "data"
        c_dir = tmp / "chroma"
        b_path = c_dir / "bm25.json"
        build_synthetic_workload(d_dir, num_md_files=num_md, include_large_pdf_pages=pdf_pages)

        rss_before = proc.memory_info().rss / (1024 * 1024)
        t0 = time.perf_counter()
        report = ingest_directory(d_dir, c_dir, b_path, reset_index=True)
        ingest_ms = (time.perf_counter() - t0) * 1000.0
        rss_after = proc.memory_info().rss / (1024 * 1024)

        queries = [
            "SYN-002-01 telemetry verification hours",
            "How quickly must escalation owners archive audit checksums across regional robotics clusters?",
            "ARCH-PG-015 deterministic CAN-bus motor heartbeats thermal shutdown Celsius",
            "redundant LFP battery reserve capacity threshold",
            "warehouse robotics safety calibration schedule",
        ]
        latencies_ms = []
        for q in queries:
            qt0 = time.perf_counter()
            hits = hybrid_retrieve_and_rerank(q, top_n=6, chroma_dir=c_dir, corpus_path=b_path)
            latencies_ms.append((time.perf_counter() - qt0) * 1000.0)
            assert len(hits) > 0

        latencies_sorted = sorted(latencies_ms)
        return {
            "tier": label,
            "files": len(report.files_processed),
            "chunks": report.total_chunks_indexed,
            "ingest_total_ms": round(ingest_ms, 1),
            "chunks_per_sec": round(report.total_chunks_indexed / (ingest_ms / 1000.0), 1),
            "retrieve_p50_ms": round(statistics.median(latencies_sorted), 1),
            "retrieve_mean_ms": round(statistics.mean(latencies_sorted), 1),
            "retrieve_max_ms": round(max(latencies_sorted), 1),
            "rss_mb_after": round(rss_after, 1),
            "rss_delta_mb": round(rss_after - rss_before, 1),
        }
    finally:
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)


def benchmark_concurrency() -> dict:
    """Run 12 concurrent hybrid_retrieve_and_rerank calls across 4 worker threads on read-only index."""
    queries = [
        "POL-EXP-402 Zurich lodging cap",
        "NV-EMP-1042 hub location",
        "Clause BAT-80 battery 80%",
        "SEV-1 incident PagerDuty SLA",
    ] * 3

    errors = []
    latencies = []

    def _worker(q: str) -> float:
        t0 = time.perf_counter()
        res = hybrid_retrieve_and_rerank(q, top_n=6)
        if not res:
            raise RuntimeError("Empty result in concurrent worker")
        return (time.perf_counter() - t0) * 1000.0

    wall_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_worker, q) for q in queries]
        for f in concurrent.futures.as_completed(futs):
            try:
                latencies.append(f.result())
            except Exception as exc:
                errors.append(str(exc))
    wall_ms = (time.perf_counter() - wall_t0) * 1000.0

    return {
        "concurrent_requests": len(queries),
        "workers": 4,
        "errors_count": len(errors),
        "errors": errors,
        "total_wall_ms": round(wall_ms, 1),
        "per_req_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "per_req_max_ms": round(max(latencies), 1) if latencies else None,
    }


def benchmark_live_e2e_groq_tokens_and_latency() -> dict:
    """Measure end-to-end latency + exact Groq prompt/completion token usage on 3 representative queries."""
    client = get_groq_client()
    schema_dict = RAGResponse.model_json_schema()
    from src.generation.generator import SYSTEM_PROMPT

    samples = [
        ("short_query", "SAB-7YR"),
        ("medium_query", "What are the Tier 1 and Tier 2 lodging and meal caps in POL-EXP-402?"),
        (
            "complex_query",
            "Compare Gold Fleet Care vs Platinum 24/7 Autonomous Care and state the RMA-NV-77 return steps.",
        ),
    ]
    records = []
    for label, q in samples:
        t0 = time.perf_counter()
        chunks = hybrid_retrieve_and_rerank(q, top_n=6)
        t1 = time.perf_counter()
        ctx_xml = format_chunks_for_prompt(chunks)
        comp = client.chat.completions.create(
            model=DEFAULT_GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{ctx_xml}\n\nUser Question: {q}\n\nRespond strictly in JSON matching the RAGResponse schema.",
                },
            ],
            temperature=0.0,
            max_completion_tokens=2048,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "rag_response", "strict": True, "schema": schema_dict},
            },
        )
        t2 = time.perf_counter()
        usage = comp.usage
        records.append(
            {
                "type": label,
                "retrieve_ms": round((t1 - t0) * 1000.0, 1),
                "llm_ms": round((t2 - t1) * 1000.0, 1),
                "total_e2e_ms": round((t2 - t0) * 1000.0, 1),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        )
    return {"e2e_samples": records}


def main(output_json_path: str | None = None) -> None:
    print("=" * 80)
    print("PHASE 8 PERFORMANCE, SCALING, CONCURRENCY & TOKEN BENCHMARK")
    print("=" * 80)

    # Pre-warm models once so we measure steady-state pipeline performance separately from cold load
    cold_t0 = time.perf_counter()
    get_embedding_model()
    get_reranker()
    cold_warm_ms = round((time.perf_counter() - cold_t0) * 1000.0, 1)
    print(f"Cold Model Load Time (BGE-small + FlashRank ONNX): {cold_warm_ms} ms")

    tiers = [
        benchmark_scale_tier("small_20_chunks", num_md=5, pdf_pages=0),
        benchmark_scale_tier("medium_200_chunks", num_md=50, pdf_pages=0),
        benchmark_scale_tier("large_560_chunks_plus_40p_pdf", num_md=100, pdf_pages=40),
    ]
    print("\n[SCALING TIERS]")
    for t in tiers:
        print(" ", t)

    conc = benchmark_concurrency()
    print("\n[CONCURRENCY (4 threads x 12 requests)]")
    print(" ", conc)

    e2e = benchmark_live_e2e_groq_tokens_and_latency()
    print("\n[END-TO-END LATENCY & GROQ TOKEN USAGE]")
    for r in e2e["e2e_samples"]:
        print(" ", r)

    out = {
        "cold_model_load_ms": cold_warm_ms,
        "scaling_tiers": tiers,
        "concurrency": conc,
        "live_e2e": e2e["e2e_samples"],
    }
    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    out_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(out_arg)
