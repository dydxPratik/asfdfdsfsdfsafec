"""Local ingestion pipeline: chunking -> BGE embeddings -> ChromaDB + synced BM25 corpus."""

from __future__ import annotations

import json
import re
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
    EMBEDDING_MODEL_NAME,
    SUPPORTED_EXTENSIONS,
)
from src.ingestion.chunker import make_chunk_id, recursive_split_text
from src.ingestion.parsers import parse_file
from src.schemas import ChunkMetadata, DocumentChunk

# Keep hyphenated IDs (e.g. pol-exp-402, nv-emp-1042) AND individual alphanumeric sub-tokens
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text preserving exact hyphenated company codes/IDs plus sub-parts."""
    lowered = text.lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(lowered):
        tok = match.group(0)
        tokens.append(tok)
        if "-" in tok:
            tokens.extend(part for part in tok.split("-") if part)
    return tokens


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load and cache the local CPU SentenceTransformer model."""
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


def load_bm25_index(
    corpus_path: Path = BM25_CORPUS_PATH,
) -> tuple[BM25Okapi | None, list[dict]]:
    """Load persisted BM25 corpus records and instantiate BM25Okapi."""
    if not corpus_path.exists():
        return None, []
    records: list[dict] = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not records:
        return None, []
    tokenized_corpus = [rec["tokens"] for rec in records]
    return BM25Okapi(tokenized_corpus), records


def ingest_directory(
    data_dir: Path = DATA_DIR,
    chroma_dir: Path = CHROMA_DIR,
    corpus_path: Path = BM25_CORPUS_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    reset_index: bool = True,
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
        embedder = get_embedding_model()
        texts = [c.text for c in all_chunks]
        embeddings_np = embedder.encode(
            texts,
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

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(json.dumps(bm25_records, indent=2), encoding="utf-8")

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
