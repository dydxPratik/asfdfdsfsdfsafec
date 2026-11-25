# RAGQuest — Local Hybrid Enterprise RAG

Production-grade, zero-hallucination Retrieval-Augmented Generation (RAG) system running **100% locally** for storage, embeddings, keyword search, and cross-encoder reranking, paired with **Groq Structured Outputs + Pydantic** for strict source citations.

## Tech Stack
- **Package Manager:** `uv` (Python 3.12)
- **Vector Store:** Local persistent `chromadb` (`./chroma_db`) — zero cloud database accounts
- **Dense Embeddings:** Local `sentence-transformers` (`BAAI/bge-small-en-v1.5`, CPU)
- **Hybrid Search:** Dense Cosine + `rank-bm25` (`BM25Okapi`) fused via **Reciprocal Rank Fusion (RRF)**
- **Local Reranker:** `flashrank` (`ms-marco-MiniLM-L-12-v2` quantized ONNX on CPU)
- **Grounded Generation:** `groq` SDK + strict `pydantic` schema (`answer`, `citations`, `is_supported`, `confidence`)
- **Document Parsers:** `pymupdf` (`.pdf`), Python stdlib (`.md`, `.txt`, `.csv`)
- **UI:** `streamlit` (`app.py`) with drag-and-drop upload, one-click re-indexing, and **Sources & Retrieval Debug** inspector

## Quick Start (PowerShell)

1. Ensure `.env` contains your Groq key:
   ```env
   GROQ_API_KEY=gsk_...
   ```
2. Launch the Streamlit app:
   ```powershell
   uv run streamlit run app.py
   ```
3. *(Optional)* Run individual phase verification suites:
   ```powershell
   uv run python -u scripts/test_phase1_ingest.py
   uv run python -u scripts/test_phase2_retrieve.py
   uv run python -u scripts/test_phase3_generate.py
   uv run python -u scripts/test_phase4_ui.py
   ```
