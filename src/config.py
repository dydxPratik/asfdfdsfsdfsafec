"""Central configuration for paths, models, chunking, and retrieval hyperparameters."""

import os
from pathlib import Path

# Suppress noisy Windows non-admin symlink warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
BM25_CORPUS_PATH: Path = CHROMA_DIR / "bm25_corpus.json"
FLASHRANK_CACHE_DIR: Path = CHROMA_DIR / "flashrank_cache"

COLLECTION_NAME: str = "company_knowledge"

# Local CPU Models
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "
FLASHRANK_MODEL_NAME: str = "ms-marco-MiniLM-L-12-v2"

# Chunking (tuned for 512-token context window of bge-small-en-v1.5 & MiniLM reranker)
CHUNK_SIZE: int = 700
CHUNK_OVERLAP: int = 120

# Hybrid Retrieval & Reranking
DENSE_TOP_K: int = 12
BM25_TOP_K: int = 12
RRF_K: int = 60
RERANK_TOP_N: int = 6

# Supported file extensions
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".md", ".txt", ".csv", ".xlsx", ".xlsm")

# Active Groq LLM models (verified live via Groq models.list API)
DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-120b"
AVAILABLE_GROQ_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
)
