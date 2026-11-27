# RAGQuest Studio — Real-Time AI Observability & Local Hybrid RAG Platform

Production-grade, zero-hallucination Retrieval-Augmented Generation (RAG) & Real-Time Observability Platform running **100% locally** for storage, embeddings, keyword search, OCR, and cross-encoder reranking, paired with **Groq Structured Outputs + Pydantic** and an interactive **React + TypeScript + React Flow (`@xyflow/react`)** live DAG visualizer streamed over **FastAPI Server-Sent Events (SSE)**.

## Tech Stack
- **Frontend Observability UI:** `React 18` + `TypeScript` + `Vite` + `Tailwind CSS` + `@xyflow/react` (live animated DAG) + `Framer Motion` + `Lucide React`
- **Real-Time API & Telemetry:** `FastAPI` + `Uvicorn` + thread-safe multi-run **Server-Sent Events (SSE)** (`src/api/`)
- **Package Manager:** `uv` (Python 3.12)
- **Vector Store:** Local persistent `chromadb` (`./chroma_db`) — zero cloud database accounts
- **Dense Embeddings:** Local `sentence-transformers` (`BAAI/bge-small-en-v1.5`, 384-d CPU)
- **Hybrid Search & Reranker:** Dense Cosine + `rank-bm25` (`BM25Okapi`) → **Reciprocal Rank Fusion (RRF)** → `flashrank` (`ms-marco-MiniLM-L-12-v2` quantized ONNX on CPU)
- **Grounded Generation:** `groq` SDK + strict `pydantic` schema (`answer`, `citations`, `is_supported`, `confidence`) + quote containment verification
- **Multimodal & Document Parsers:** `pymupdf` (`.pdf`), `openpyxl` (`.xlsx`, `.xlsm`), `Pillow` + `rapidocr-onnxruntime` (`.png`, `.jpg`, `.jpeg`, `.webp`), Python stdlib (`.md`, `.txt`, `.csv`)

## Quick Start (PowerShell)

1. Ensure `.env` contains your Groq key:
   ```env
   GROQ_API_KEY=gsk_...
   ```
2. Build the React frontend (first time or after UI edits) and launch the full platform on `http://localhost:8000`:
   ```powershell
   cd frontend; npm run build; cd ..
   uv run python run_server.py
   ```
3. *(Optional — Hot-Reload Frontend Dev Mode on `:5173` + FastAPI on `:8000`):*
   ```powershell
   # Terminal 1:
   uv run python run_server.py
   # Terminal 2:
   cd frontend; npm run dev
   ```
4. *(Optional)* Run the full automated test suite:
   ```powershell
   uv run python -m unittest discover -s tests -p "test_*.py" -v
   ```
