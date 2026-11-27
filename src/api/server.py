"""FastAPI + SSE Server powering the RAGQuest Real-Time AI Observability Platform."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from flashrank import RerankRequest
from pydantic import BaseModel, Field

from src.api.telemetry import INGESTION_STAGE_DEFS, QUERY_STAGE_DEFS, RunManager
from src.api.traced_pipeline import execute_traced_ingestion, execute_traced_query
from src.config import (
    AVAILABLE_GROQ_MODELS,
    BGE_QUERY_PREFIX,
    BM25_CORPUS_PATH,
    BM25_TOP_K,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    DEFAULT_GROQ_MODEL,
    DENSE_TOP_K,
    EMBEDDING_MODEL_NAME,
    FLASHRANK_MODEL_NAME,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    RERANK_TOP_N,
    RRF_K,
    SUPPORTED_EXTENSIONS,
)
from src.ingestion.indexer import (
    get_chroma_client,
    get_embedding_model,
    get_or_create_collection,
    ingest_directory,
    load_bm25_index,
    tokenize_for_bm25,
)
from src.retrieval.hybrid_retriever import get_reranker, reciprocal_rank_fusion

EVAL_CACHE_PATH: Path = CHROMA_DIR / "latest_eval_benchmark.json"
FRONTEND_DIST: Path = PROJECT_ROOT / "frontend" / "dist"

run_manager = RunManager()

# Mutable runtime settings initialized from src.config
RUNTIME_SETTINGS: dict[str, Any] = {
    "default_groq_model": DEFAULT_GROQ_MODEL,
    "dense_top_k": DENSE_TOP_K,
    "bm25_top_k": BM25_TOP_K,
    "rrf_k": RRF_K,
    "rerank_top_n": RERANK_TOP_N,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
}


class QueryStartRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: str | None = None
    top_n: int | None = Field(default=None, ge=1, le=20)
    dense_top_k: int | None = Field(default=None, ge=1, le=50)
    bm25_top_k: int | None = Field(default=None, ge=1, le=50)
    rrf_k: int | None = Field(default=None, ge=1, le=200)


class RetrievalExploreRequest(BaseModel):
    query: str = Field(..., min_length=1)
    dense_top_k: int = Field(default=12, ge=1, le=30)
    bm25_top_k: int = Field(default=12, ge=1, le=30)
    rrf_k: int = Field(default=60, ge=1, le=200)
    top_n: int = Field(default=6, ge=1, le=20)


class RuntimeSettingsUpdate(BaseModel):
    default_groq_model: str | None = None
    dense_top_k: int | None = Field(default=None, ge=1, le=50)
    bm25_top_k: int | None = Field(default=None, ge=1, le=50)
    rrf_k: int | None = Field(default=None, ge=1, le=200)
    rerank_top_n: int | None = Field(default=None, ge=1, le=20)
    chunk_size: int | None = Field(default=None, ge=200, le=2000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=500)


def _warm_and_ensure_index() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    get_embedding_model()
    get_reranker()
    files = [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    client = get_chroma_client(CHROMA_DIR)
    col = get_or_create_collection(client)
    if files and (col.count() == 0 or not BM25_CORPUS_PATH.exists()):
        ingest_directory(data_dir=DATA_DIR, chroma_dir=CHROMA_DIR, corpus_path=BM25_CORPUS_PATH, reset_index=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warm_and_ensure_index, daemon=True).start()
    yield


app = FastAPI(
    title="RAGQuest Observability API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _file_category(ext: str) -> str:
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in (".csv", ".xlsx", ".xlsm"):
        return "spreadsheet"
    return "document"


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    groq_configured = bool(os.environ.get("GROQ_API_KEY", "").strip())
    client = get_chroma_client(CHROMA_DIR)
    col = get_or_create_collection(client)
    _, bm25_recs = load_bm25_index(BM25_CORPUS_PATH)
    files = [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS] if DATA_DIR.exists() else []
    return {
        "status": "ok",
        "environment": "local-hybrid",
        "groq_api_configured": groq_configured,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": FLASHRANK_MODEL_NAME,
        "vector_collection": COLLECTION_NAME,
        "chroma_chunks": col.count(),
        "bm25_docs": len(bm25_recs),
        "source_files": len(files),
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "available_groq_models": list(AVAILABLE_GROQ_MODELS),
        "settings": RUNTIME_SETTINGS,
        "stage_definitions": {
            "ingestion": INGESTION_STAGE_DEFS,
            "query": QUERY_STAGE_DEFS,
        },
    }


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    health = api_health()
    docs = api_list_documents()["documents"]
    runs = run_manager.list_runs(limit=50)

    query_runs = [r for r in runs if r["pipeline_type"] == "query" and r["status"] == "completed"]
    durations = [r["total_duration_ms"] for r in query_runs if r["total_duration_ms"] is not None]
    supported_cnt = sum(
        1
        for r in query_runs
        if isinstance(r.get("result_preview"), dict) and r["result_preview"].get("is_supported") is True
    )
    abstained_cnt = len(query_runs) - supported_cnt

    by_format: dict[str, int] = {}
    for d in docs:
        by_format[d["extension"]] = by_format.get(d["extension"], 0) + 1

    return {
        "health": health,
        "documents_count": len(docs),
        "format_breakdown": by_format,
        "telemetry": {
            "total_runs": len(runs),
            "completed_query_runs": len(query_runs),
            "grounded_answers": supported_cnt,
            "strict_abstentions": abstained_cnt,
            "avg_e2e_ms": round(sum(durations) / len(durations), 1) if durations else None,
        },
        "recent_runs": runs[:12],
    }


@app.get("/api/documents")
def api_list_documents() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _, bm25_recs = load_bm25_index(BM25_CORPUS_PATH)
    chunks_by_file: dict[str, int] = {}
    for r in bm25_recs:
        sf = r.get("source_file", "")
        chunks_by_file[sf] = chunks_by_file.get(sf, 0) + 1

    docs = []
    for p in sorted(DATA_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stat = p.stat()
        ext = p.suffix.lower()
        docs.append(
            {
                "name": p.name,
                "extension": ext,
                "category": _file_category(ext),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "indexed_chunks": chunks_by_file.get(p.name, 0),
            }
        )
    return {"documents": docs, "total_chunks": len(bm25_recs)}


@app.delete("/api/documents/{filename}")
def api_delete_document(filename: str) -> dict[str, Any]:
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    target = DATA_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found")
    target.unlink()
    return {"deleted": safe_name}


@app.post("/api/ingest/start")
async def api_start_ingestion(
    reset_index: bool = Form(default=True),
    files: list[UploadFile] | None = File(default=None),
) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_names: list[str] = []
    if files:
        for uf in files:
            if not uf.filename:
                continue
            safe_name = Path(uf.filename).name
            ext = Path(safe_name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file extension '{ext}'")
            content = await uf.read()
            (DATA_DIR / safe_name).write_bytes(content)
            saved_names.append(safe_name)

    title = (
        f"Ingest {len(saved_names)} uploaded file(s)"
        if saved_names
        else ("Full Knowledge Base Re-Index" if reset_index else "Incremental Knowledge Base Sync")
    )
    run_rec = run_manager.create_run(
        pipeline_type="ingestion",
        title=title,
        params={
            "reset_index": reset_index,
            "uploaded_files": saved_names,
            "chunk_size": RUNTIME_SETTINGS["chunk_size"],
            "chunk_overlap": RUNTIME_SETTINGS["chunk_overlap"],
        },
    )
    run_id = run_rec["run_id"]

    threading.Thread(
        target=execute_traced_ingestion,
        kwargs={
            "run_id": run_id,
            "manager": run_manager,
            "data_dir": DATA_DIR,
            "chroma_dir": CHROMA_DIR,
            "corpus_path": BM25_CORPUS_PATH,
            "chunk_size": RUNTIME_SETTINGS["chunk_size"],
            "chunk_overlap": RUNTIME_SETTINGS["chunk_overlap"],
            "reset_index": reset_index,
            "uploaded_filenames": saved_names,
        },
        daemon=True,
    ).start()

    return {"run_id": run_id, "run": run_rec}


@app.post("/api/query/start")
async def api_start_query(req: QueryStartRequest) -> dict[str, Any]:
    model = req.model or RUNTIME_SETTINGS["default_groq_model"]
    top_n = req.top_n or RUNTIME_SETTINGS["rerank_top_n"]
    dense_k = req.dense_top_k or RUNTIME_SETTINGS["dense_top_k"]
    bm25_k = req.bm25_top_k or RUNTIME_SETTINGS["bm25_top_k"]
    rrf_k = req.rrf_k or RUNTIME_SETTINGS["rrf_k"]

    run_rec = run_manager.create_run(
        pipeline_type="query",
        title=req.query.strip()[:90],
        params={
            "query": req.query.strip(),
            "model": model,
            "top_n": top_n,
            "dense_top_k": dense_k,
            "bm25_top_k": bm25_k,
            "rrf_k": rrf_k,
        },
    )
    run_id = run_rec["run_id"]

    threading.Thread(
        target=execute_traced_query,
        kwargs={
            "run_id": run_id,
            "manager": run_manager,
            "query": req.query,
            "model": model,
            "top_n": top_n,
            "dense_top_k": dense_k,
            "bm25_top_k": bm25_k,
            "rrf_k": rrf_k,
        },
        daemon=True,
    ).start()

    return {"run_id": run_id, "run": run_rec}


@app.get("/api/runs")
def api_list_runs(limit: int = 30) -> dict[str, Any]:
    return {"runs": run_manager.list_runs(limit=limit)}


@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str) -> dict[str, Any]:
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@app.get("/api/runs/{run_id}/stream")
async def api_stream_run(run_id: str) -> StreamingResponse:
    initial_Check = run_manager.get_run(run_id)
    if not initial_Check:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    loop = asyncio.get_running_loop()
    queue = run_manager.subscribe(run_id, loop)

    async def event_generator():
        try:
            current = run_manager.get_run(run_id)
            yield f"data: {json.dumps({'type': 'snapshot', 'run': current})}\n\n"
            if current and current["status"] in ("completed", "failed"):
                return

            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("type") == "run_complete":
                        break
                except asyncio.TimeoutError:
                    # Keep-alive comment prevents proxy disconnects
                    yield ": keep-alive\n\n"
                    latest = run_manager.get_run(run_id)
                    if latest and latest["status"] in ("completed", "failed"):
                        yield f"data: {json.dumps({'type': 'run_complete', 'run': latest})}\n\n"
                        break
        finally:
            run_manager.unsubscribe(run_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/retrieval/explore")
def api_retrieval_explore(req: RetrievalExploreRequest) -> dict[str, Any]:
    t0 = time.perf_counter()
    q = req.query.strip()
    client = get_chroma_client(CHROMA_DIR)
    col = get_or_create_collection(client)
    total_docs = col.count()
    if total_docs == 0:
        return {"query": q, "total_indexed": 0, "dense": [], "bm25": [], "rrf": [], "reranked": []}

    embedder = get_embedding_model()
    q_vec = embedder.encode(
        BGE_QUERY_PREFIX + q,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    c_res = col.query(
        query_embeddings=[q_vec],
        n_results=min(req.dense_top_k, total_docs),
        include=["documents", "metadatas", "distances"],
    )
    dense_hits = [
        {
            "rank": idx,
            "chunk_id": cid,
            "text": doc,
            "source_file": meta["source_file"],
            "page_or_section": meta["page_or_section"],
            "dense_score": round(1.0 - float(dist), 4),
        }
        for idx, (cid, doc, meta, dist) in enumerate(
            zip(c_res["ids"][0], c_res["documents"][0], c_res["metadatas"][0], c_res["distances"][0]),
            start=1,
        )
    ]

    bm25_idx, recs = load_bm25_index(BM25_CORPUS_PATH)
    q_tokens = tokenize_for_bm25(q)
    bm25_hits = []
    if bm25_idx and recs and q_tokens:
        scores = bm25_idx.get_scores(q_tokens)
        ranked = sorted([(i, float(s)) for i, s in enumerate(scores) if float(s) > 0.0], key=lambda x: x[1], reverse=True)
        for rank_i, (doc_i, sc) in enumerate(ranked[: req.bm25_top_k], start=1):
            r = recs[doc_i]
            bm25_hits.append(
                {
                    "rank": rank_i,
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "source_file": r["source_file"],
                    "page_or_section": r["page_or_section"],
                    "bm25_score": round(sc, 4),
                }
            )

    fused = reciprocal_rank_fusion(dense_hits, bm25_hits, rrf_k=req.rrf_k)
    rrf_rank_map = {c.chunk_id: idx for idx, c in enumerate(fused, start=1)}

    ranker = get_reranker()
    cand_map = {c.chunk_id: c for c in fused}
    reranked_raw = ranker.rerank(
        RerankRequest(query=q, passages=[{"id": c.chunk_id, "text": c.text} for c in fused])
    )
    reranked_list = []
    for final_r, item in enumerate(reranked_raw[: req.top_n], start=1):
        ch = cand_map[item["id"]]
        prev_r = rrf_rank_map[ch.chunk_id]
        reranked_list.append(
            {
                "final_rank": final_r,
                "rrf_rank": prev_r,
                "rank_delta": prev_r - final_r,
                "chunk_id": ch.chunk_id,
                "source_file": ch.source_file,
                "page_or_section": ch.page_or_section,
                "rerank_score": round(float(item["score"]), 4),
                "rrf_score": ch.rrf_score,
                "dense_rank": ch.dense_rank,
                "dense_score": ch.dense_score,
                "bm25_rank": ch.bm25_rank,
                "bm25_score": ch.bm25_score,
                "text": ch.text,
            }
        )

    return {
        "query": q,
        "query_tokens": q_tokens,
        "total_indexed": total_docs,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "dense": dense_hits,
        "bm25": bm25_hits,
        "rrf": [c.model_dump() for c in fused],
        "reranked": reranked_list,
    }


@app.get("/api/evaluation")
def api_get_evaluation() -> dict[str, Any]:
    ds_path = PROJECT_ROOT / "tests" / "datasets" / "ground_truth_dataset.json"
    dataset = json.loads(ds_path.read_text(encoding="utf-8")) if ds_path.exists() else []
    cached = None
    if EVAL_CACHE_PATH.exists():
        try:
            cached = json.loads(EVAL_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            cached = None
    return {"dataset": dataset, "latest_report": cached}


@app.post("/api/evaluation/run")
def api_run_evaluation() -> dict[str, Any]:
    from tests.eval_retrieval import main as run_ir_eval

    EVAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_ir_eval(str(EVAL_CACHE_PATH))
    report = json.loads(EVAL_CACHE_PATH.read_text(encoding="utf-8"))
    return {"report": report}


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    return {
        "settings": RUNTIME_SETTINGS,
        "available_groq_models": list(AVAILABLE_GROQ_MODELS),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": FLASHRANK_MODEL_NAME,
    }


@app.put("/api/settings")
def api_update_settings(upd: RuntimeSettingsUpdate) -> dict[str, Any]:
    payload = upd.model_dump(exclude_none=True)
    if "chunk_size" in payload or "chunk_overlap" in payload:
        cs = payload.get("chunk_size", RUNTIME_SETTINGS["chunk_size"])
        co = payload.get("chunk_overlap", RUNTIME_SETTINGS["chunk_overlap"])
        if co >= cs:
            raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size")
    RUNTIME_SETTINGS.update(payload)
    return {"settings": RUNTIME_SETTINGS}


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
