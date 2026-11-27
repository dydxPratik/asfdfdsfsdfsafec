"""Real-time event-instrumented wrappers around the existing RAGQuest ingestion and query modules."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from flashrank import RerankRequest
from groq import BadRequestError
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from src.api.telemetry import RunManager
from src.config import (
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
    RERANK_TOP_N,
    RRF_K,
    SUPPORTED_EXTENSIONS,
)
from src.generation.generator import (
    SYSTEM_PROMPT,
    _normalize_for_quote_match,
    format_chunks_for_prompt,
    get_groq_client,
)
from src.ingestion.chunker import make_chunk_id, recursive_split_text
from src.ingestion.indexer import (
    get_chroma_client,
    get_embedding_model,
    get_or_create_collection,
    load_bm25_index,
    tokenize_for_bm25,
)
from src.ingestion.parsers import ParsedSection, parse_file
from src.retrieval.hybrid_retriever import get_reranker, reciprocal_rank_fusion
from src.schemas import ChunkMetadata, Citation, DocumentChunk, RAGResponse, RetrievedChunk

PARSER_LABELS: dict[str, str] = {
    ".pdf": "PyMuPDF (pymupdf 2.1)",
    ".md": "Markdown Heading Parser (stdlib re)",
    ".txt": "Section-Aware Text Parser (stdlib re)",
    ".csv": "Tabular CSV Row Serializer (stdlib csv)",
    ".xlsx": "Multi-Sheet Excel Streamer (openpyxl 3.1.5)",
    ".xlsm": "Multi-Sheet Excel Streamer (openpyxl 3.1.5)",
    ".png": "Pillow + RapidOCR Spatial Engine",
    ".jpg": "Pillow + RapidOCR Spatial Engine",
    ".jpeg": "Pillow + RapidOCR Spatial Engine",
    ".webp": "Pillow + RapidOCR Spatial Engine",
}


def execute_traced_ingestion(
    run_id: str,
    manager: RunManager,
    data_dir: Path = DATA_DIR,
    chroma_dir: Path = CHROMA_DIR,
    corpus_path: Path = BM25_CORPUS_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    reset_index: bool = True,
    uploaded_filenames: list[str] | None = None,
) -> None:
    """Execute the actual ingestion pipeline while emitting live stage telemetry events."""
    run_t0 = time.perf_counter()
    try:
        # 1. FILE DISCOVERY & UPLOAD
        t0 = time.perf_counter()
        manager.emit(run_id, "file_upload", "running", "Scanning knowledge base directory...")
        data_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        all_dir_files = sorted(p for p in data_dir.iterdir() if p.is_file())
        candidate_files = [p for p in all_dir_files if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        ignored_files = [p.name for p in all_dir_files if p.suffix.lower() not in SUPPORTED_EXTENSIONS]
        total_bytes = sum(p.stat().st_size for p in candidate_files)

        manager.emit(
            run_id,
            "file_upload",
            "completed",
            f"Discovered {len(candidate_files)} supported file(s) ({total_bytes / 1024:.1f} KB)",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            metadata={
                "data_dir": str(data_dir),
                "uploaded_now": uploaded_filenames or [],
                "discovered_files": [
                    {
                        "name": p.name,
                        "extension": p.suffix.lower(),
                        "size_bytes": p.stat().st_size,
                    }
                    for p in candidate_files
                ],
                "ignored_unsupported_files": ignored_files,
                "total_bytes": total_bytes,
            },
        )

        # 2. FILE & HEADER VALIDATION
        t0 = time.perf_counter()
        manager.emit(run_id, "file_validation", "running", "Validating file headers and non-empty payloads...")
        valid_doc_files: list[Path] = []
        valid_img_files: list[Path] = []
        failed_files: dict[str, str] = {}

        for fp in candidate_files:
            ext = fp.suffix.lower()
            size = fp.stat().st_size
            if size == 0:
                if ext in IMAGE_EXTENSIONS:
                    failed_files[fp.name] = "ValueError: Empty 0-byte image file"
                    continue
            if ext in IMAGE_EXTENSIONS:
                try:
                    with Image.open(fp) as probe:
                        fmt = (probe.format or "").upper()
                        probe.verify()
                    if fmt not in {"PNG", "JPEG", "WEBP"}:
                        failed_files[fp.name] = f"Unsupported image header '{fmt}'"
                        continue
                    valid_img_files.append(fp)
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    failed_files[fp.name] = f"{type(exc).__name__}: {exc}"
            else:
                valid_doc_files.append(fp)

        manager.emit(
            run_id,
            "file_validation",
            "completed",
            f"Validated {len(valid_doc_files)} document/tabular file(s) and {len(valid_img_files)} image(s)",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            metadata={
                "valid_documents": [p.name for p in valid_doc_files],
                "valid_images": [p.name for p in valid_img_files],
                "rejected_files": failed_files,
                "allowed_extensions": list(SUPPORTED_EXTENSIONS),
            },
        )

        # 3. DOCUMENT & TABULAR PARSER
        parsed_by_file: list[tuple[Path, list[ParsedSection]]] = []
        if not valid_doc_files:
            manager.emit(
                run_id,
                "document_parser",
                "skipped",
                "No text/PDF/spreadsheet files in batch",
                duration_ms=0.0,
                metadata={"files_parsed": []},
            )
        else:
            t0 = time.perf_counter()
            manager.emit(
                run_id,
                "document_parser",
                "running",
                f"Parsing {len(valid_doc_files)} document/spreadsheet file(s)...",
            )
            doc_summaries = []
            for fp in valid_doc_files:
                ft0 = time.perf_counter()
                try:
                    secs = parse_file(fp)
                    parsed_by_file.append((fp, secs))
                    preview_txt = secs[0].text[:240] if secs else ""
                    doc_summaries.append(
                        {
                            "source_file": fp.name,
                            "file_type": fp.suffix.lower(),
                            "parser": PARSER_LABELS.get(fp.suffix.lower(), "stdlib"),
                            "sections_extracted": len(secs),
                            "duration_ms": round((time.perf_counter() - ft0) * 1000.0, 2),
                            "preview": preview_txt,
                        }
                    )
                except Exception as exc:
                    failed_files[fp.name] = f"{type(exc).__name__}: {exc}"

            manager.emit(
                run_id,
                "document_parser",
                "completed",
                f"Extracted {sum(d['sections_extracted'] for d in doc_summaries)} section(s) from {len(doc_summaries)} file(s)",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={
                    "files_parsed": doc_summaries,
                    "parse_errors": failed_files,
                },
            )

        # 4. SPATIAL IMAGE OCR (WHEN REQUIRED)
        if not valid_img_files:
            manager.emit(
                run_id,
                "ocr_extraction",
                "skipped",
                "No image files (.png/.jpg/.jpeg/.webp) present — RapidOCR skipped",
                duration_ms=0.0,
                metadata={"engine": "RapidOCR (ONNX CPU)", "images_processed": []},
            )
        else:
            t0 = time.perf_counter()
            manager.emit(
                run_id,
                "ocr_extraction",
                "running",
                f"Running RapidOCR spatial extraction on {len(valid_img_files)} image(s)...",
            )
            ocr_summaries = []
            for fp in valid_img_files:
                it0 = time.perf_counter()
                try:
                    secs = parse_file(fp)
                    parsed_by_file.append((fp, secs))
                    preview_txt = secs[0].text[:280] if secs else ""
                    ocr_summaries.append(
                        {
                            "source_file": fp.name,
                            "location_info": secs[0].page_or_section if secs else "",
                            "duration_ms": round((time.perf_counter() - it0) * 1000.0, 2),
                            "extracted_preview": preview_txt,
                        }
                    )
                except Exception as exc:
                    failed_files[fp.name] = f"{type(exc).__name__}: {exc}"

            manager.emit(
                run_id,
                "ocr_extraction",
                "completed",
                f"Processed {len(ocr_summaries)} image(s) via RapidOCR ONNX",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={
                    "engine": "RapidOCR (ONNXRuntime CPU) + Pillow Header Guard",
                    "images_processed": ocr_summaries,
                },
            )

        # 5. RECURSIVE CHUNKING
        t0 = time.perf_counter()
        manager.emit(
            run_id,
            "recursive_chunking",
            "running",
            f"Splitting sections (chunk_size={chunk_size}, overlap={chunk_overlap})...",
        )
        all_chunks: list[DocumentChunk] = []
        chunks_per_file: dict[str, int] = {}
        succeeded_files: list[str] = []

        for fp, sections in parsed_by_file:
            file_type_tag = "image" if fp.suffix.lower() in IMAGE_EXTENSIONS else "document"
            chunk_counter = 0
            for sec in sections:
                pieces = recursive_split_text(sec.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                for piece in pieces:
                    cid = make_chunk_id(sec.source_file, chunk_counter, piece)
                    meta = ChunkMetadata(
                        source_file=sec.source_file,
                        source=sec.source_file,
                        file_type=file_type_tag,
                        page_or_section=sec.page_or_section,
                        chunk_id=cid,
                    )
                    all_chunks.append(DocumentChunk(chunk_id=cid, text=piece, metadata=meta))
                    chunk_counter += 1
            succeeded_files.append(fp.name)
            chunks_per_file[fp.name] = chunk_counter

        lengths = [len(c.text) for c in all_chunks]
        manager.emit(
            run_id,
            "recursive_chunking",
            "completed",
            f"Generated {len(all_chunks)} overlapping chunk(s) across {len(succeeded_files)} file(s)",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            metadata={
                "strategy": "Hierarchical Recursive Character Splitter (Word-Aligned Overlap)",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "total_chunks": len(all_chunks),
                "chunks_per_file": chunks_per_file,
                "min_chars": min(lengths) if lengths else 0,
                "avg_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
                "max_chars": max(lengths) if lengths else 0,
                "chunk_previews": [
                    {
                        "chunk_id": c.chunk_id,
                        "source_file": c.metadata.source_file,
                        "file_type": c.metadata.file_type,
                        "page_or_section": c.metadata.page_or_section,
                        "char_len": len(c.text),
                        "text": c.text[:320],
                    }
                    for c in all_chunks[:15]
                ],
            },
        )

        # 6. EMBEDDING GENERATION
        embeddings_list: list[list[float]] = []
        embedding_dim = 0
        if not all_chunks:
            manager.emit(
                run_id,
                "embedding_generation",
                "skipped",
                "No non-empty chunks generated — embedding skipped",
                duration_ms=0.0,
                metadata={"model": EMBEDDING_MODEL_NAME, "vectors_generated": 0},
            )
        else:
            t0 = time.perf_counter()
            manager.emit(
                run_id,
                "embedding_generation",
                "running",
                f"Encoding {len(all_chunks)} chunk(s) with {EMBEDDING_MODEL_NAME} on CPU...",
                metadata={"model": EMBEDDING_MODEL_NAME, "chunks_processed": 0, "total_chunks": len(all_chunks)},
            )
            embedder = get_embedding_model()
            texts = [c.text for c in all_chunks]
            batch_size = 32
            for start_idx in range(0, len(texts), batch_size):
                batch = texts[start_idx : start_idx + batch_size]
                batch_np = embedder.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                embedding_dim = int(batch_np.shape[1])
                embeddings_list.extend(batch_np.tolist())
                manager.emit(
                    run_id,
                    "embedding_generation",
                    "running",
                    f"Encoded {len(embeddings_list)} / {len(texts)} chunk vectors...",
                    metadata={
                        "model": EMBEDDING_MODEL_NAME,
                        "chunks_processed": len(embeddings_list),
                        "total_chunks": len(texts),
                        "dimensions": embedding_dim,
                    },
                )

            manager.emit(
                run_id,
                "embedding_generation",
                "completed",
                f"Generated {len(embeddings_list)} normalized {embedding_dim}-d vectors",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={
                    "model": EMBEDDING_MODEL_NAME,
                    "device": "cpu",
                    "normalize_embeddings": True,
                    "dimensions": embedding_dim,
                    "vectors_generated": len(embeddings_list),
                },
            )

        # 7. CHROMADB HNSW VECTOR INDEXING
        t0 = time.perf_counter()
        manager.emit(run_id, "vector_db_indexing", "running", "Upserting vectors & metadata to local ChromaDB...")
        client = get_chroma_client(chroma_dir)
        if reset_index:
            existing_names = [c.name for c in client.list_collections()]
            if COLLECTION_NAME in existing_names:
                client.delete_collection(COLLECTION_NAME)
        collection = get_or_create_collection(client)

        if not reset_index and succeeded_files:
            for fname in succeeded_files:
                collection.delete(where={"source_file": fname})

        if all_chunks:
            collection.upsert(
                ids=[c.chunk_id for c in all_chunks],
                documents=[c.text for c in all_chunks],
                embeddings=embeddings_list,
                metadatas=[c.metadata.model_dump() for c in all_chunks],
            )
        final_chroma_count = collection.count()
        manager.emit(
            run_id,
            "vector_db_indexing",
            "completed",
            f"Persisted {final_chroma_count} total vector(s) in '{COLLECTION_NAME}'",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            metadata={
                "engine": "ChromaDB PersistentClient (SQLite + HNSW)",
                "collection": COLLECTION_NAME,
                "distance_metric": "cosine",
                "reset_index": reset_index,
                "upserted_now": len(all_chunks),
                "total_persisted_vectors": final_chroma_count,
                "persist_directory": str(chroma_dir),
            },
        )

        # 8. BM25 LEXICAL CORPUS SYNC
        t0 = time.perf_counter()
        manager.emit(run_id, "bm25_indexing", "running", "Tokenizing hyphen/ID codes and syncing BM25 corpus...")
        if reset_index:
            bm25_records = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source_file": c.metadata.source_file,
                    "page_or_section": c.metadata.page_or_section,
                    "tokens": tokenize_for_bm25(c.text),
                }
                for c in all_chunks
            ]
        else:
            _, existing_records = load_bm25_index(corpus_path)
            updated_set = set(succeeded_files)
            kept = [r for r in existing_records if r["source_file"] not in updated_set]
            added = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source_file": c.metadata.source_file,
                    "page_or_section": c.metadata.page_or_section,
                    "tokens": tokenize_for_bm25(c.text),
                }
                for c in all_chunks
            ]
            bm25_records = kept + added

        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.write_text(json.dumps(bm25_records, indent=2), encoding="utf-8")
        bm25_idx, loaded_recs = load_bm25_index(corpus_path)
        bm25_count = len(loaded_recs) if bm25_idx is not None else 0
        total_tokens = sum(len(r["tokens"]) for r in loaded_recs)

        manager.emit(
            run_id,
            "bm25_indexing",
            "completed",
            f"Synced {bm25_count} BM25 document(s) ({total_tokens} lexical tokens)",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            metadata={
                "algorithm": "BM25Okapi (k1=1.5, b=0.75)",
                "tokenizer": "Hyphen-Code & Subtoken Regex ([a-z0-9]+(?:-[a-z0-9]+)*)",
                "bm25_doc_count": bm25_count,
                "total_tokens": total_tokens,
                "avg_tokens_per_chunk": round(total_tokens / bm25_count, 1) if bm25_count else 0.0,
                "corpus_path": str(corpus_path),
            },
        )

        total_ms = (time.perf_counter() - run_t0) * 1000.0
        manager.finish_run(
            run_id,
            status="completed",
            total_duration_ms=total_ms,
            result={
                "files_processed": succeeded_files,
                "chunks_per_file": chunks_per_file,
                "total_chunks_indexed": len(all_chunks),
                "chroma_count": final_chroma_count,
                "bm25_doc_count": bm25_count,
                "embedding_dim": embedding_dim,
                "failed_files": failed_files,
            },
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - run_t0) * 1000.0
        manager.finish_run(
            run_id,
            status="failed",
            total_duration_ms=total_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


def execute_traced_query(
    run_id: str,
    manager: RunManager,
    query: str,
    model: str = DEFAULT_GROQ_MODEL,
    top_n: int = RERANK_TOP_N,
    dense_top_k: int = DENSE_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    rrf_k: int = RRF_K,
    chroma_dir: Path = CHROMA_DIR,
    corpus_path: Path = BM25_CORPUS_PATH,
) -> None:
    """Execute the actual hybrid RAG + Groq pipeline while emitting live stage telemetry events."""
    run_t0 = time.perf_counter()
    current_stage = "query_input"
    try:
        # 1. QUERY INPUT & VALIDATION
        t0 = time.perf_counter()
        manager.emit(run_id, "query_input", "running", "Validating query and checking local index...")
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query string cannot be empty.")

        client = get_chroma_client(chroma_dir)
        collection = get_or_create_collection(client)
        total_docs = collection.count()

        manager.emit(
            run_id,
            "query_input",
            "completed",
            f"Accepted query ({len(cleaned_query)} chars) against {total_docs} indexed chunks",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            metadata={
                "raw_query": query,
                "cleaned_query": cleaned_query,
                "char_length": len(cleaned_query),
                "word_count": len(cleaned_query.split()),
                "indexed_chunks_available": total_docs,
                "config": {
                    "groq_model": model,
                    "top_n": top_n,
                    "dense_top_k": dense_top_k,
                    "bm25_top_k": bm25_top_k,
                    "rrf_k": rrf_k,
                },
            },
        )

        if total_docs == 0:
            empty_resp = RAGResponse(
                answer="This information is not found in the provided company documents (the knowledge base index is currently empty).",
                citations=[],
                is_supported=False,
                confidence="low",
            )
            manager.finish_run(
                run_id,
                status="completed",
                total_duration_ms=(time.perf_counter() - run_t0) * 1000.0,
                result={"response": empty_resp.model_dump(), "chunks": [], "tokens": None},
            )
            return

        # 2. QUERY EMBEDDING
        current_stage = "query_embedding"
        t0 = time.perf_counter()
        manager.emit(run_id, "query_embedding", "running", f"Encoding query with {EMBEDDING_MODEL_NAME}...")
        embedder = get_embedding_model()
        prefixed_query = BGE_QUERY_PREFIX + cleaned_query
        q_vec_np = embedder.encode(
            prefixed_query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        q_vec_list = q_vec_np.tolist()
        embed_ms = (time.perf_counter() - t0) * 1000.0

        manager.emit(
            run_id,
            "query_embedding",
            "completed",
            f"Encoded 384-d normalized query vector in {embed_ms:.1f} ms",
            duration_ms=embed_ms,
            metadata={
                "model": EMBEDDING_MODEL_NAME,
                "query_prefix": BGE_QUERY_PREFIX,
                "dimensions": len(q_vec_list),
                "vector_preview_first_8": [round(float(x), 5) for x in q_vec_list[:8]],
            },
        )

        # 3. CHROMADB DENSE RETRIEVAL
        current_stage = "dense_retrieval"
        t0 = time.perf_counter()
        manager.emit(run_id, "dense_retrieval", "running", f"Querying ChromaDB HNSW (top {dense_top_k})...")
        n_dense = min(dense_top_k, total_docs)
        chroma_res = collection.query(
            query_embeddings=[q_vec_list],
            n_results=n_dense,
            include=["documents", "metadatas", "distances"],
        )
        dense_hits: list[dict[str, Any]] = []
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
                        "dense_score": round(1.0 - float(dist), 4),
                    }
                )
        dense_ms = (time.perf_counter() - t0) * 1000.0
        manager.emit(
            run_id,
            "dense_retrieval",
            "completed",
            f"Retrieved {len(dense_hits)} dense candidate(s) (top cos={dense_hits[0]['dense_score'] if dense_hits else 0:.4f})",
            duration_ms=dense_ms,
            metadata={
                "technique": "HNSW Cosine Similarity",
                "requested_k": dense_top_k,
                "retrieved_count": len(dense_hits),
                "hits": [
                    {
                        "rank": idx,
                        "chunk_id": h["chunk_id"],
                        "source_file": h["source_file"],
                        "page_or_section": h["page_or_section"],
                        "cosine_score": h["dense_score"],
                        "preview": h["text"][:220],
                    }
                    for idx, h in enumerate(dense_hits, start=1)
                ],
            },
        )

        # 4. BM25 SPARSE KEYWORD RETRIEVAL
        current_stage = "bm25_retrieval"
        t0 = time.perf_counter()
        manager.emit(run_id, "bm25_retrieval", "running", f"Scoring BM25Okapi lexical corpus (top {bm25_top_k})...")
        bm25_hits: list[dict[str, Any]] = []
        query_tokens = tokenize_for_bm25(cleaned_query)
        bm25_index, corpus_records = load_bm25_index(corpus_path)

        if bm25_index is not None and corpus_records and query_tokens:
            scores = bm25_index.get_scores(query_tokens)
            scored_indices = sorted(
                [(idx, float(sc)) for idx, sc in enumerate(scores) if float(sc) > 0.0],
                key=lambda pair: pair[1],
                reverse=True,
            )
            for idx, sc in scored_indices[:bm25_top_k]:
                rec = corpus_records[idx]
                bm25_hits.append(
                    {
                        "chunk_id": rec["chunk_id"],
                        "text": rec["text"],
                        "source_file": rec["source_file"],
                        "page_or_section": rec["page_or_section"],
                        "bm25_score": round(sc, 4),
                    }
                )
        bm25_ms = (time.perf_counter() - t0) * 1000.0
        if not bm25_hits:
            manager.emit(
                run_id,
                "bm25_retrieval",
                "skipped",
                "Zero lexical token overlap in BM25 corpus — relying on dense semantic candidates",
                duration_ms=bm25_ms,
                metadata={"query_tokens": query_tokens, "retrieved_count": 0, "hits": []},
            )
        else:
            manager.emit(
                run_id,
                "bm25_retrieval",
                "completed",
                f"Matched {len(bm25_hits)} keyword candidate(s) (top bm25={bm25_hits[0]['bm25_score']:.4f})",
                duration_ms=bm25_ms,
                metadata={
                    "technique": "BM25Okapi + Hyphen-Code Tokenizer",
                    "query_tokens": query_tokens,
                    "requested_k": bm25_top_k,
                    "retrieved_count": len(bm25_hits),
                    "hits": [
                        {
                            "rank": idx,
                            "chunk_id": h["chunk_id"],
                            "source_file": h["source_file"],
                            "page_or_section": h["page_or_section"],
                            "bm25_score": h["bm25_score"],
                            "preview": h["text"][:220],
                        }
                        for idx, h in enumerate(bm25_hits, start=1)
                    ],
                },
            )

        # 5. RECIPROCAL RANK FUSION (RRF)
        current_stage = "rrf_fusion"
        t0 = time.perf_counter()
        manager.emit(run_id, "rrf_fusion", "running", f"Fusing rankings via RRF (k={rrf_k})...")
        fused_candidates = reciprocal_rank_fusion(dense_hits, bm25_hits, rrf_k=rrf_k)
        both_matched = sum(1 for c in fused_candidates if c.dense_rank is not None and c.bm25_rank is not None)
        rrf_ms = (time.perf_counter() - t0) * 1000.0

        rrf_rank_by_id = {c.chunk_id: idx for idx, c in enumerate(fused_candidates, start=1)}
        manager.emit(
            run_id,
            "rrf_fusion",
            "completed",
            f"Merged into {len(fused_candidates)} unique candidate(s) ({both_matched} dual-hit)",
            duration_ms=rrf_ms,
            metadata={
                "formula": f"sum(1 / ({rrf_k} + rank))",
                "rrf_k": rrf_k,
                "dense_input_count": len(dense_hits),
                "bm25_input_count": len(bm25_hits),
                "unique_fused_count": len(fused_candidates),
                "dual_retriever_overlap_count": both_matched,
                "candidates": [
                    {
                        "rrf_rank": idx,
                        "chunk_id": c.chunk_id,
                        "source_file": c.source_file,
                        "page_or_section": c.page_or_section,
                        "rrf_score": c.rrf_score,
                        "dense_rank": c.dense_rank,
                        "bm25_rank": c.bm25_rank,
                    }
                    for idx, c in enumerate(fused_candidates, start=1)
                ],
            },
        )

        # 6. FLASHRANK CROSS-ENCODER RERANKING
        current_stage = "cross_encoder_rerank"
        t0 = time.perf_counter()
        manager.emit(
            run_id,
            "cross_encoder_rerank",
            "running",
            f"Reranking {len(fused_candidates)} candidate(s) with {FLASHRANK_MODEL_NAME}...",
        )
        ranker = get_reranker()
        cand_by_id = {c.chunk_id: c for c in fused_candidates}
        passages = [{"id": c.chunk_id, "text": c.text} for c in fused_candidates]
        reranked_all = ranker.rerank(RerankRequest(query=cleaned_query, passages=passages))

        final_chunks: list[RetrievedChunk] = []
        rank_comparisons: list[dict[str, Any]] = []
        for final_rank, item in enumerate(reranked_all, start=1):
            ch = cand_by_id[item["id"]]
            score_val = round(float(item["score"]), 4)
            ch.rerank_score = score_val
            prev_rrf_rank = rrf_rank_by_id[ch.chunk_id]
            rank_comparisons.append(
                {
                    "final_rank": final_rank,
                    "rrf_rank_before": prev_rrf_rank,
                    "rank_delta": prev_rrf_rank - final_rank,
                    "kept_in_top_n": final_rank <= top_n,
                    "chunk_id": ch.chunk_id,
                    "source_file": ch.source_file,
                    "page_or_section": ch.page_or_section,
                    "rerank_score": score_val,
                    "rrf_score": ch.rrf_score,
                    "dense_rank": ch.dense_rank,
                    "dense_score": ch.dense_score,
                    "bm25_rank": ch.bm25_rank,
                    "bm25_score": ch.bm25_score,
                    "text": ch.text,
                }
            )
            if final_rank <= top_n:
                final_chunks.append(ch)

        rerank_ms = (time.perf_counter() - t0) * 1000.0
        manager.emit(
            run_id,
            "cross_encoder_rerank",
            "completed",
            f"Selected Top {len(final_chunks)} / {len(fused_candidates)} chunk(s) (top score={final_chunks[0].rerank_score if final_chunks else 0:.4f})",
            duration_ms=rerank_ms,
            metadata={
                "model": FLASHRANK_MODEL_NAME,
                "runtime": "ONNXRuntime CPU (max_length=512)",
                "input_count": len(fused_candidates),
                "output_count": len(final_chunks),
                "comparisons": rank_comparisons,
            },
        )

        # 7. CONTEXT ASSEMBLY
        current_stage = "context_assembly"
        t0 = time.perf_counter()
        manager.emit(run_id, "context_assembly", "running", "Assembling XML-escaped prompt context...")
        context_xml = format_chunks_for_prompt(final_chunks)
        user_msg_content = (
            f"{context_xml}\n\n"
            f"User Question: {cleaned_query}\n\n"
            "Respond strictly in JSON matching the RAGResponse schema."
        )
        ctx_ms = (time.perf_counter() - t0) * 1000.0
        manager.emit(
            run_id,
            "context_assembly",
            "completed",
            f"Assembled {len(final_chunks)} chunks into {len(context_xml):,} XML characters",
            duration_ms=ctx_ms,
            metadata={
                "chunks_injected": len(final_chunks),
                "xml_context_chars": len(context_xml),
                "system_prompt_chars": len(SYSTEM_PROMPT),
                "xml_escaping_enabled": True,
                "system_prompt": SYSTEM_PROMPT,
                "assembled_xml": context_xml,
            },
        )

        # 8. GROQ LLM GENERATION
        current_stage = "llm_generation"
        t0 = time.perf_counter()
        manager.emit(run_id, "llm_generation", "running", f"Calling Groq ({model}) with strict JSON Schema...")
        groq_client = get_groq_client()
        schema_dict = RAGResponse.model_json_schema()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg_content},
        ]

        raw_llm_json = ""
        used_mode = "json_schema (strict)"
        attempts = 0
        usage_meta: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

        parsed_raw_obj: RAGResponse | None = None
        for attempt_idx in range(1, 3):
            attempts = attempt_idx
            try:
                comp = groq_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_completion_tokens=4096,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "rag_response",
                            "strict": True,
                            "schema": schema_dict,
                        },
                    },
                )
                raw_llm_json = comp.choices[0].message.content or ""
                if comp.usage:
                    usage_meta = {
                        "prompt_tokens": getattr(comp.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(comp.usage, "completion_tokens", None),
                        "total_tokens": getattr(comp.usage, "total_tokens", None),
                    }
                parsed_raw_obj = RAGResponse.model_validate_json(raw_llm_json)
                break
            except (BadRequestError, ValidationError) as exc:
                err_s = str(exc)
                recoverable = (
                    isinstance(exc, ValidationError)
                    or "json_validate_failed" in err_s
                    or "json_schema" in err_s
                    or "response_format" in err_s
                )
                if not recoverable:
                    raise
                if "json_schema" in err_s or "response_format" in err_s:
                    break

        if parsed_raw_obj is None:
            used_mode = "json_object (fallback)"
            attempts += 1
            schema_prompt = SYSTEM_PROMPT + "\n\nJSON SCHEMA TO FOLLOW EXACTLY:\n" + json.dumps(schema_dict, indent=2)
            comp = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": schema_prompt}, messages[1]],
                temperature=0.0,
                max_completion_tokens=4096,
                response_format={"type": "json_object"},
            )
            raw_llm_json = comp.choices[0].message.content or ""
            if comp.usage:
                usage_meta = {
                    "prompt_tokens": getattr(comp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(comp.usage, "completion_tokens", None),
                    "total_tokens": getattr(comp.usage, "total_tokens", None),
                }
            parsed_raw_obj = RAGResponse.model_validate_json(raw_llm_json)

        llm_ms = (time.perf_counter() - t0) * 1000.0
        tok_label = f"{usage_meta['total_tokens']} tokens" if usage_meta["total_tokens"] else "completed"
        manager.emit(
            run_id,
            "llm_generation",
            "completed",
            f"Groq {model} responded in {llm_ms:.0f} ms ({tok_label})",
            duration_ms=llm_ms,
            metadata={
                "provider": "Groq Cloud LPU",
                "model": model,
                "temperature": 0.0,
                "max_completion_tokens": 4096,
                "response_format_mode": used_mode,
                "attempts": attempts,
                "token_usage": usage_meta,
                "raw_json_output": raw_llm_json,
            },
        )

        # 9. PYDANTIC & QUOTE CONTAINMENT VERIFICATION
        current_stage = "citation_verification"
        t0 = time.perf_counter()
        manager.emit(run_id, "citation_verification", "running", "Verifying source filenames & quote substrings...")

        texts_by_source: dict[str, list[str]] = {}
        for ch in final_chunks:
            texts_by_source.setdefault(ch.source_file, []).append(_normalize_for_quote_match(ch.text))

        verified_citations: list[Citation] = []
        stripped_citations: list[dict[str, str]] = []
        for cit in parsed_raw_obj.citations:
            norm_q = _normalize_for_quote_match(cit.quote)
            if not norm_q or cit.source_file not in texts_by_source:
                stripped_citations.append(
                    {"source_file": cit.source_file, "quote": cit.quote, "reason": "Source file not in retrieved set"}
                )
                continue
            if any(norm_q in blob for blob in texts_by_source[cit.source_file]):
                verified_citations.append(cit)
            else:
                stripped_citations.append(
                    {
                        "source_file": cit.source_file,
                        "quote": cit.quote,
                        "reason": "Quote substring not found in source chunk",
                    }
                )

        if not parsed_raw_obj.is_supported or not verified_citations:
            ans_txt = parsed_raw_obj.answer
            if parsed_raw_obj.is_supported and not verified_citations:
                ans_txt = (
                    "This information could not be verified with an exact citation from the provided company documents."
                )
            final_response = RAGResponse(
                answer=ans_txt,
                citations=[],
                is_supported=False,
                confidence="low",
            )
        else:
            final_response = RAGResponse(
                answer=parsed_raw_obj.answer,
                citations=verified_citations,
                is_supported=True,
                confidence=parsed_raw_obj.confidence,
            )

        ver_ms = (time.perf_counter() - t0) * 1000.0
        verdict_msg = (
            f"Verified {len(final_response.citations)} citation(s) · Confidence: {final_response.confidence.upper()}"
            if final_response.is_supported
            else "Strict Abstention enforced (Not supported by company documents)"
        )
        manager.emit(
            run_id,
            "citation_verification",
            "completed",
            verdict_msg,
            duration_ms=ver_ms,
            metadata={
                "is_supported": final_response.is_supported,
                "confidence": final_response.confidence,
                "raw_citations_count": len(parsed_raw_obj.citations),
                "verified_citations_count": len(final_response.citations),
                "stripped_citations": stripped_citations,
                "final_response": final_response.model_dump(),
            },
        )

        total_ms = (time.perf_counter() - run_t0) * 1000.0
        retrieve_total_ms = round(embed_ms + dense_ms + bm25_ms + rrf_ms + rerank_ms, 2)
        manager.finish_run(
            run_id,
            status="completed",
            total_duration_ms=total_ms,
            result={
                "query": cleaned_query,
                "model": model,
                "response": final_response.model_dump(),
                "chunks": [c.model_dump() for c in final_chunks],
                "timings": {
                    "retrieve_ms": retrieve_total_ms,
                    "llm_ms": round(llm_ms, 2),
                    "total_ms": round(total_ms, 2),
                },
                "tokens": usage_meta,
            },
        )
    except Exception as exc:
        err_text = f"{type(exc).__name__}: {exc}"
        manager.emit(run_id, current_stage, "failed", err_text)
        manager.finish_run(
            run_id,
            status="failed",
            total_duration_ms=(time.perf_counter() - run_t0) * 1000.0,
            error=err_text,
        )
