"""Local ingestion pipeline: chunking -> BGE embeddings -> ChromaDB + synced BM25 corpus."""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    BM25_CORPUS_PATH,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    DEFAULT_EMBEDDING_PROFILE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROFILES,
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
)
from src.ingestion.chunker import make_chunk_id, recursive_split_text
from src.ingestion.parsers import parse_file
from src.schemas import ChunkMetadata, DocumentChunk

def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or unicodedata.category(ch).startswith("M")


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text preserving hyphenated IDs and all Unicode scripts (including combining marks)."""
    lowered = unicodedata.normalize("NFC", text).lower()
    raw_tokens: list[str] = []
    buf: list[str] = []
    for ch in lowered:
        if _is_word_char(ch) or (ch == "-" and buf):
            buf.append(ch)
        else:
            if buf:
                tok = "".join(buf).rstrip("-")
                if tok:
                    raw_tokens.append(tok)
                buf.clear()
    if buf:
        tok = "".join(buf).rstrip("-")
        if tok:
            raw_tokens.append(tok)

    tokens: list[str] = []
    for tok in raw_tokens:
        tokens.append(tok)
        if "-" in tok:
            tokens.extend(part for part in tok.split("-") if part)
    return tokens


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load and cache a local CPU SentenceTransformer model."""
    return SentenceTransformer(model_name, device="cpu")


def get_chroma_client(chroma_dir: Path = CHROMA_DIR) -> chromadb.ClientAPI:
    """Return a local PersistentClient backed by ./chroma_db."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Get or create the cosine-similarity company knowledge collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_file(
    file_path: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Parse a single file and split its sections into overlapping DocumentChunks."""
    sections = parse_file(file_path)
    file_type = "image" if file_path.suffix.lower() in IMAGE_EXTENSIONS else "document"
    chunks: list[DocumentChunk] = []
    chunk_counter = 0

    for section in sections:
        split_texts = recursive_split_text(
            section.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for piece in split_texts:
            chunk_id = make_chunk_id(section.source_file, chunk_counter, piece)
            meta = ChunkMetadata(
                source_file=section.source_file,
                source=section.source_file,
                file_type=file_type,
                page_or_section=section.page_or_section,
                chunk_id=chunk_id,
            )
            chunks.append(DocumentChunk(chunk_id=chunk_id, text=piece, metadata=meta))
            chunk_counter += 1

    return chunks


@dataclass
class IngestionReport:
    """Summary metrics returned after indexing company files."""

    files_processed: list[str]
    chunks_per_file: dict[str, int]
    total_chunks_indexed: int
    chroma_count: int
    bm25_doc_count: int
    embedding_dim: int
    failed_files: dict[str, str] = field(default_factory=dict)


# In-memory BM25 cache keyed by corpus path -> (file signature, index, records).
# Avoids re-reading JSON and rebuilding BM25Okapi on every query.
_BM25_CACHE: dict[str, tuple[tuple[int, int], BM25Okapi | None, list[dict]]] = {}
_BM25_CACHE_LOCK = threading.Lock()


def _cache_key(corpus_path: Path) -> str:
    return str(Path(corpus_path).resolve())


def invalidate_bm25_cache(corpus_path: Path | None = None) -> None:
    """Drop cached BM25 index for one corpus path (or all paths when None)."""
    with _BM25_CACHE_LOCK:
        if corpus_path is None:
            _BM25_CACHE.clear()
        else:
            _BM25_CACHE.pop(_cache_key(corpus_path), None)


def save_bm25_corpus(records: list[dict], corpus_path: Path = BM25_CORPUS_PATH) -> None:
    """Atomically persist BM25 records (temp file + os.replace) and invalidate the cache."""
    corpus_path = Path(corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = corpus_path.with_name(corpus_path.name + ".tmp")
    tmp_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    os.replace(tmp_path, corpus_path)
    invalidate_bm25_cache(corpus_path)


def load_bm25_index(
    corpus_path: Path = BM25_CORPUS_PATH,
) -> tuple[BM25Okapi | None, list[dict]]:
    """Load persisted BM25 corpus records and BM25Okapi, cached until the file changes."""
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        return None, []
    stat = corpus_path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    key = _cache_key(corpus_path)

    with _BM25_CACHE_LOCK:
        cached = _BM25_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]

    records: list[dict] = json.loads(corpus_path.read_text(encoding="utf-8"))
    index: BM25Okapi | None = None
    if records:
        index = BM25Okapi([rec["tokens"] for rec in records])
    else:
        records = []

    with _BM25_CACHE_LOCK:
        _BM25_CACHE[key] = (signature, index, records)
    return index, records


def ingest_directory(
    data_dir: Path = DATA_DIR,
    chroma_dir: Path = CHROMA_DIR,
    corpus_path: Path = BM25_CORPUS_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    reset_index: bool = True,
    embedding_profile: str = DEFAULT_EMBEDDING_PROFILE,
) -> IngestionReport:
    """Parse, chunk, embed, and persist all supported company files into ChromaDB + BM25."""
    data_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    client = get_chroma_client(chroma_dir)
    if reset_index:
        existing_names = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing_names:
            client.delete_collection(COLLECTION_NAME)
    collection = get_or_create_collection(client)

    all_chunks: list[DocumentChunk] = []
    chunks_per_file: dict[str, int] = {}
    succeeded_files: list[str] = []
    failed_files: dict[str, str] = {}

    for file_path in files:
        try:
            file_chunks = chunk_file(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception as exc:
            failed_files[file_path.name] = f"{type(exc).__name__}: {exc}"
            continue
        succeeded_files.append(file_path.name)
        chunks_per_file[file_path.name] = len(file_chunks)
        all_chunks.extend(file_chunks)

    # When updating incrementally (reset_index=False), purge prior chunks of updated files in ChromaDB
    if not reset_index and succeeded_files:
        for fname in succeeded_files:
            collection.delete(where={"source_file": fname})

    embedding_dim = 0
    if all_chunks:
        prof = EMBEDDING_PROFILES.get(embedding_profile, EMBEDDING_PROFILES[DEFAULT_EMBEDDING_PROFILE])
        embedder = get_embedding_model(prof["model_name"])
        texts = [c.text for c in all_chunks]
        p_prefix = prof.get("passage_prefix", "")
        encode_inputs = [p_prefix + t for t in texts] if p_prefix else texts
        embeddings_np = embedder.encode(
            encode_inputs,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedding_dim = int(embeddings_np.shape[1])

        collection.upsert(
            ids=[c.chunk_id for c in all_chunks],
            documents=texts,
            embeddings=embeddings_np.tolist(),
            metadatas=[c.metadata.model_dump() for c in all_chunks],
        )

    # Persist BM25 corpus synchronized 1:1 with ChromaDB
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
        # Merge with existing records if not resetting
        _, existing_records = load_bm25_index(corpus_path)
        updated_files = set(succeeded_files)
        kept_records = [r for r in existing_records if r["source_file"] not in updated_files]
        new_records = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source_file": c.metadata.source_file,
                "page_or_section": c.metadata.page_or_section,
                "tokens": tokenize_for_bm25(c.text),
            }
            for c in all_chunks
        ]
        bm25_records = kept_records + new_records

    save_bm25_corpus(bm25_records, corpus_path)

    bm25_index, loaded_records = load_bm25_index(corpus_path)
    bm25_count = len(loaded_records) if bm25_index is not None else 0

    return IngestionReport(
        files_processed=succeeded_files,
        chunks_per_file=chunks_per_file,
        total_chunks_indexed=len(all_chunks),
        chroma_count=collection.count(),
        bm25_doc_count=bm25_count,
        embedding_dim=embedding_dim,
        failed_files=failed_files,
    )
