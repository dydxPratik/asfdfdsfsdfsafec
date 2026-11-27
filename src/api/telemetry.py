"""Thread-safe multi-run telemetry store and asyncio SSE event broker."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.config import CHROMA_DIR

StageStatus = Literal["idle", "waiting", "running", "completed", "failed", "skipped"]
PipelineType = Literal["ingestion", "query"]

RUNS_HISTORY_PATH: Path = CHROMA_DIR / "observability_runs.json"

INGESTION_STAGE_DEFS: list[dict[str, str]] = [
    {"id": "file_upload", "label": "File Discovery & Upload", "group": "input"},
    {"id": "file_validation", "label": "Header & Format Validation", "group": "validation"},
    {"id": "document_parser", "label": "Document & Tabular Parser", "group": "parsing"},
    {"id": "ocr_extraction", "label": "Spatial Image OCR", "group": "parsing"},
    {"id": "recursive_chunking", "label": "Recursive Chunker", "group": "processing"},
    {"id": "embedding_generation", "label": "BGE-Small Vectorizer", "group": "indexing"},
    {"id": "vector_db_indexing", "label": "ChromaDB HNSW Upsert", "group": "indexing"},
    {"id": "bm25_indexing", "label": "BM25 Lexical Sync", "group": "indexing"},
]

QUERY_STAGE_DEFS: list[dict[str, str]] = [
    {"id": "query_input", "label": "Query Input & Config", "group": "input"},
    {"id": "tabular_sql", "label": "DuckDB SQL Router", "group": "routing"},
    {"id": "query_embedding", "label": "Query Embedding (384-d)", "group": "encoding"},
    {"id": "dense_retrieval", "label": "ChromaDB Dense Search", "group": "retrieval"},
    {"id": "bm25_retrieval", "label": "BM25 Keyword Search", "group": "retrieval"},
    {"id": "rrf_fusion", "label": "Reciprocal Rank Fusion", "group": "fusion"},
    {"id": "cross_encoder_rerank", "label": "FlashRank Reranker", "group": "reranking"},
    {"id": "context_assembly", "label": "XML Context Assembly", "group": "generation"},
    {"id": "llm_generation", "label": "Groq LLM Generation", "group": "generation"},
    {"id": "citation_verification", "label": "Pydantic & Quote Verifier", "group": "verification"},
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunManager:
    """Manages concurrent and historical RAG pipeline runs and broadcasts SSE events."""

    def __init__(self, persist_path: Path = RUNS_HISTORY_PATH, max_runs: int = 50) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._run_order: list[str] = []
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
        self._persist_path = persist_path
        self._max_runs = max_runs
        self._load_persisted()

    def _load_persisted(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for item in raw[-self._max_runs :]:
                    rid = item.get("run_id")
                    if rid:
                        self._runs[rid] = item
                        self._run_order.append(rid)
        except Exception:
            pass

    def _save_persisted(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            ordered = [self._runs[rid] for rid in self._run_order[-self._max_runs :] if rid in self._runs]
            self._persist_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
        except Exception:
            pass

    def create_run(
        self,
        pipeline_type: PipelineType,
        title: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prefix = "ing" if pipeline_type == "ingestion" else "qry"
        short_uid = uuid.uuid4().hex[:8]
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        run_id = f"{prefix}_{ts}_{short_uid}"
        now = utc_now_iso()

        stage_defs = INGESTION_STAGE_DEFS if pipeline_type == "ingestion" else QUERY_STAGE_DEFS
        stages: dict[str, dict[str, Any]] = {
            s["id"]: {
                "stage": s["id"],
                "label": s["label"],
                "group": s["group"],
                "status": "waiting",
                "started_at": None,
                "completed_at": None,
                "duration_ms": None,
                "message": "Waiting for upstream stage",
                "metadata": {},
            }
            for s in stage_defs
        }

        run_record: dict[str, Any] = {
            "run_id": run_id,
            "pipeline_type": pipeline_type,
            "title": title,
            "status": "running",
            "started_at": now,
            "completed_at": None,
            "total_duration_ms": None,
            "params": params or {},
            "stages": stages,
            "events": [],
            "result": None,
            "error": None,
        }

        with self._lock:
            self._runs[run_id] = run_record
            self._run_order.append(run_id)
            if len(self._run_order) > self._max_runs:
                evicted = self._run_order.pop(0)
                self._runs.pop(evicted, None)
            return json.loads(json.dumps(run_record))

    def emit(
        self,
        run_id: str,
        stage: str,
        status: StageStatus,
        message: str,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now_iso()
        event_payload = {
            "event_id": uuid.uuid4().hex[:10],
            "run_id": run_id,
            "stage": stage,
            "status": status,
            "message": message,
            "timestamp": now,
            "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
            "metadata": metadata or {},
        }

        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            st_obj = run["stages"].get(stage)
            if st_obj is not None:
                st_obj["status"] = status
                st_obj["message"] = message
                if status == "running" and st_obj["started_at"] is None:
                    st_obj["started_at"] = now
                if status in ("completed", "failed", "skipped"):
                    st_obj["completed_at"] = now
                if duration_ms is not None:
                    st_obj["duration_ms"] = round(duration_ms, 2)
                if metadata:
                    st_obj["metadata"] = {**st_obj.get("metadata", {}), **metadata}

            run["events"].append(event_payload)
            subs = list(self._subscribers.get(run_id, []))

        sse_msg = {"type": "stage_event", "event": event_payload}
        for loop, queue in subs:
            loop.call_soon_threadsafe(queue.put_nowait, sse_msg)

    def finish_run(
        self,
        run_id: str,
        status: Literal["completed", "failed"],
        total_duration_ms: float,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = utc_now_iso()
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            run["status"] = status
            run["completed_at"] = now
            run["total_duration_ms"] = round(total_duration_ms, 2)
            run["result"] = result
            run["error"] = error

            # Any stage still in 'waiting' when a run terminates was skipped
            for st_obj in run["stages"].values():
                if st_obj["status"] == "waiting":
                    st_obj["status"] = "skipped"
                    st_obj["completed_at"] = now
                    st_obj["message"] = "Skipped"

            snapshot = json.loads(json.dumps(run))
            self._save_persisted()
            subs = list(self._subscribers.get(run_id, []))

        sse_msg = {"type": "run_complete", "run": snapshot}
        for loop, queue in subs:
            loop.call_soon_threadsafe(queue.put_nowait, sse_msg)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            return json.loads(json.dumps(run)) if run else None

    def list_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            rids = list(reversed(self._run_order[-limit:]))
            summaries = []
            for rid in rids:
                r = self._runs.get(rid)
                if not r:
                    continue
                summaries.append(
                    {
                        "run_id": r["run_id"],
                        "pipeline_type": r["pipeline_type"],
                        "title": r["title"],
                        "status": r["status"],
                        "started_at": r["started_at"],
                        "completed_at": r["completed_at"],
                        "total_duration_ms": r["total_duration_ms"],
                        "params": r["params"],
                        "error": r["error"],
                        "result_preview": (
                            {
                                "is_supported": r["result"].get("response", {}).get("is_supported"),
                                "confidence": r["result"].get("response", {}).get("confidence"),
                                "citations_count": len(r["result"].get("response", {}).get("citations", [])),
                            }
                            if r["pipeline_type"] == "query" and isinstance(r.get("result"), dict)
                            else r.get("result")
                        ),
                    }
                )
            return summaries

    def subscribe(self, run_id: str, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(run_id, []).append((loop, queue))
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(run_id, [])
            self._subscribers[run_id] = [(lp, q) for lp, q in subs if q is not queue]
            if not self._subscribers[run_id]:
                self._subscribers.pop(run_id, None)
