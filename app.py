"""RAGQuest — Local Hybrid RAG + FlashRank Reranking + Grounded Groq Streamlit App."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from src.config import (
    AVAILABLE_GROQ_MODELS,
    BM25_CORPUS_PATH,
    CHROMA_DIR,
    DATA_DIR,
    DEFAULT_GROQ_MODEL,
    RERANK_TOP_N,
    SUPPORTED_EXTENSIONS,
)
from src.generation.generator import generate_grounded_answer
from src.ingestion.indexer import (
    get_chroma_client,
    get_embedding_model,
    get_or_create_collection,
    ingest_directory,
)
from src.retrieval.hybrid_retriever import get_reranker, hybrid_retrieve_and_rerank
from src.schemas import RAGResponse, RetrievedChunk

st.set_page_config(
    page_title="RAGQuest — Enterprise Local Hybrid RAG",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _escape_dollars(text: str) -> str:
    """Prevent Streamlit Markdown KaTeX from treating currency '$' amounts as math spans."""
    return text.replace("$", "\\$")


@st.cache_resource(show_spinner="Loading local CPU embedding & reranker models...")
def warm_local_models() -> bool:
    """Cache BAAI/bge-small-en-v1.5 and FlashRank ONNX session across Streamlit reruns."""
    get_embedding_model()
    get_reranker()
    return True


def get_index_stats() -> tuple[list[Path], int]:
    """Return sorted source files in ./data and current chunk count in local ChromaDB."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    client = get_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)
    return files, collection.count()


def ensure_initial_index() -> None:
    """Auto-index ./data on cold start if ChromaDB or BM25 index is missing."""
    files, count = get_index_stats()
    if files and (count == 0 or not BM25_CORPUS_PATH.exists()):
        ingest_directory(
            data_dir=DATA_DIR,
            chroma_dir=CHROMA_DIR,
            corpus_path=BM25_CORPUS_PATH,
            reset_index=True,
        )


def render_assistant_turn(
    response: RAGResponse,
    chunks: list[RetrievedChunk],
    show_debug: bool,
    timings: dict[str, float] | None = None,
) -> None:
    """Render grounded answer, status badge, exact citations, and debug retrieval panel."""
    if response.is_supported and response.confidence == "high":
        st.caption("✅ **Grounded in Company Documents** · Confidence: **HIGH**")
    elif response.is_supported:
        st.caption("⚠️ **Partially Grounded in Company Documents** · Confidence: **MEDIUM**")
    else:
        st.caption("🛑 **Not Found in Company Documents (Strict Abstention)** · Confidence: **LOW**")

    st.markdown(_escape_dollars(response.answer))

    if response.citations:
        st.markdown("**Verified Citations:**")
        for idx, cit in enumerate(response.citations, start=1):
            st.markdown(
                f"{idx}. **`{cit.source_file}`** — *\"{_escape_dollars(cit.quote)}\"*"
            )

    if show_debug:
        timing_str = ""
        if timings:
            timing_str = (
                f" · Retrieve+Rerank: {timings.get('retrieve_ms', 0):.0f} ms"
                f" | Groq LLM: {timings.get('llm_ms', 0):.0f} ms"
            )
        with st.expander(
            f"🔍 Sources & Retrieval Debug ({len(chunks)} reranked chunks{timing_str})",
            expanded=False,
        ):
            if not chunks:
                st.info("No chunks were retrieved from the local index.")
            else:
                for rank, ch in enumerate(chunks, start=1):
                    st.markdown(
                        f"**Rank #{rank} — `{ch.source_file}`** (`{ch.page_or_section}`)  \n"
                        f"`chunk_id`: `{ch.chunk_id}`  \n"
                        f"**FlashRank Score:** `{ch.rerank_score:.4f}` | "
                        f"**RRF Score:** `{ch.rrf_score:.6f}` | "
                        f"**Dense Rank:** `{ch.dense_rank}` (`cos={ch.dense_score}`) | "
                        f"**BM25 Rank:** `{ch.bm25_rank}` (`bm25={ch.bm25_score}`)"
                    )
                    st.code(ch.text, language="text")
                    st.divider()

            st.markdown("**Validated Pydantic `RAGResponse` JSON:**")
            st.code(response.model_dump_json(indent=2), language="json")


def run_rag_turn(query: str, model_name: str, top_n: int) -> None:
    """Append user message, execute hybrid RAG + Groq generation, and store turn in session_state."""
    st.session_state.messages.append({"role": "user", "content": query})

    t0 = time.perf_counter()
    chunks = hybrid_retrieve_and_rerank(query=query, top_n=top_n)
    t1 = time.perf_counter()
    response = generate_grounded_answer(query=query, retrieved_chunks=chunks, model=model_name)
    t2 = time.perf_counter()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "response": response.model_dump(),
            "chunks": [c.model_dump() for c in chunks],
            "timings": {
                "retrieve_ms": (t1 - t0) * 1000.0,
                "llm_ms": (t2 - t1) * 1000.0,
            },
        }
    )


def main() -> None:
    warm_local_models()
    ensure_initial_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("📂 Company Knowledge Base")
        files, chunk_count = get_index_stats()
        col_a, col_b = st.columns(2)
        col_a.metric("Source Files", len(files))
        col_b.metric("Indexed Chunks", chunk_count)

        uploaded_files = st.file_uploader(
            "Upload Company Files (.pdf, .md, .txt, .csv, .xlsx, .png, .jpg, .webp)",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            help="Files are saved locally to ./data and indexed locally into ChromaDB + BM25.",
        )

        if st.button("🔄 Save & Re-Index Knowledge Base", type="primary"):
            saved_names: list[str] = []
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            if uploaded_files:
                for uf in uploaded_files:
                    safe_name = Path(uf.name).name
                    target_path = DATA_DIR / safe_name
                    target_path.write_bytes(uf.getvalue())
                    saved_names.append(safe_name)

            with st.spinner("Chunking, embedding (BGE-small), and updating ChromaDB + BM25..."):
                report = ingest_directory(
                    data_dir=DATA_DIR,
                    chroma_dir=CHROMA_DIR,
                    corpus_path=BM25_CORPUS_PATH,
                    reset_index=True,
                )
            if report.failed_files:
                failed_summary = "; ".join(f"{k} ({v})" for k, v in report.failed_files.items())
                st.warning(f"Skipped unreadable file(s): {failed_summary}")
            if saved_names:
                st.success(
                    f"Saved {len(saved_names)} file(s) and indexed {report.total_chunks_indexed} chunks!"
                )
            else:
                st.success(
                    f"Re-indexed {len(report.files_processed)} file(s) ({report.total_chunks_indexed} chunks)."
                )
            st.rerun()

        with st.expander(f"📄 Indexed Files in ./data ({len(files)})", expanded=False):
            if not files:
                st.caption("No files in `./data` yet.")
            else:
                for fpath in files:
                    kb = fpath.stat().st_size / 1024.0
                    st.markdown(f"- `{fpath.name}` ({kb:.1f} KB)")

        st.divider()
        st.subheader("⚙️ Retrieval & Generation")
        selected_model = st.selectbox(
            "Groq Model",
            options=list(AVAILABLE_GROQ_MODELS),
            index=list(AVAILABLE_GROQ_MODELS).index(DEFAULT_GROQ_MODEL),
        )
        top_n = st.slider(
            "Top-N Reranked Chunks",
            min_value=3,
            max_value=10,
            value=RERANK_TOP_N,
            help="Chunks passed to Groq after Hybrid RRF + FlashRank CPU reranking.",
        )
        show_debug = st.toggle(
            "Sources & Retrieval Debug",
            value=True,
            help="Show exact retrieved chunks, filenames, RRF scores, and FlashRank scores under answers.",
        )

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # ---------------- MAIN CHAT PANE ----------------
    st.title("🧭 RAGQuest — Local Hybrid Enterprise RAG")
    st.caption(
        "100% Local ChromaDB + `BAAI/bge-small-en-v1.5` + `rank-bm25` (RRF) + `FlashRank` CPU Reranker "
        "→ Strictly Grounded `Groq` + `Pydantic` Citations"
    )

    if not st.session_state.messages:
        st.markdown("##### Try a sample company query:")
        sample_queries = [
            "What is the lodging cap in Zurich under POL-EXP-402 and when must VaultSpend receipts be submitted?",
            "Who is employee NV-EMP-1001, what project do they lead, and how long is the Helios-9 warranty?",
            "What are the SEV-1 incident notification SLAs in SEC-RUN-880?",
            "What is NexaVault's cryptocurrency 401(k) match and stock ticker?",
        ]
        cols = st.columns(2)
        for idx, sq in enumerate(sample_queries):
            if cols[idx % 2].button(sq, key=f"sample_q_{idx}"):
                with st.spinner("Running Hybrid RRF + FlashRank + Grounded Groq generation..."):
                    try:
                        run_rag_turn(sq, model_name=selected_model, top_n=top_n)
                    except Exception as exc:
                        st.error(f"Error while processing query: {exc}")
                st.rerun()

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(_escape_dollars(msg["content"]))
            else:
                resp_obj = RAGResponse.model_validate(msg["response"])
                chunk_objs = [RetrievedChunk.model_validate(c) for c in msg["chunks"]]
                render_assistant_turn(
                    response=resp_obj,
                    chunks=chunk_objs,
                    show_debug=show_debug,
                    timings=msg.get("timings"),
                )

    # Chat input bar
    user_prompt = st.chat_input("Ask a question about company policies, warranties, security SLAs, or employees...")
    if user_prompt:
        with st.chat_message("user"):
            st.markdown(_escape_dollars(user_prompt))
        with st.chat_message("assistant"):
            with st.spinner("Retrieving (ChromaDB + BM25 → RRF → FlashRank) & verifying answer via Groq..."):
                try:
                    t0 = time.perf_counter()
                    chunks = hybrid_retrieve_and_rerank(query=user_prompt, top_n=top_n)
                    t1 = time.perf_counter()
                    response = generate_grounded_answer(
                        query=user_prompt,
                        retrieved_chunks=chunks,
                        model=selected_model,
                    )
                    t2 = time.perf_counter()
                    timings = {
                        "retrieve_ms": (t1 - t0) * 1000.0,
                        "llm_ms": (t2 - t1) * 1000.0,
                    }
                    render_assistant_turn(
                        response=response,
                        chunks=chunks,
                        show_debug=show_debug,
                        timings=timings,
                    )
                    st.session_state.messages.append({"role": "user", "content": user_prompt})
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "response": response.model_dump(),
                            "chunks": [c.model_dump() for c in chunks],
                            "timings": timings,
                        }
                    )
                except Exception as exc:
                    st.error(f"RAG pipeline error: {exc}")


if __name__ == "__main__":
    main()
