"""Phase 1 Verification Script: Generates a sample company PDF and runs full local ingestion."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf

from src.config import BM25_CORPUS_PATH, CHROMA_DIR, DATA_DIR
from src.ingestion.indexer import get_chroma_client, get_or_create_collection, ingest_directory, load_bm25_index


def create_sample_company_pdf(pdf_path: Path) -> None:
    """Create a realistic 2-page NexaVault Robotics hardware & warranty policy PDF."""
    doc = pymupdf.open()

    page1_text = (
        "NEXAVAULT ROBOTICS INC. — ENTERPRISE FLEET WARRANTY & SLA SCHEDULE\n"
        "Document ID: POL-WAR-910 | Effective: February 1, 2026\n\n"
        "1. Autonomous Mobile Robot (AMR) Hardware Coverage\n"
        "All Helios-9 and Titan-4 warehouse robotics units include a standard 36-month bumper-to-bumper "
        "hardware warranty covering LiDAR arrays, brushless drive actuators, and solid-state battery packs.\n\n"
        "Battery Health Guarantee (Clause BAT-80): NexaVault guarantees that LFP battery modules will retain "
        "at least 80% of rated nameplate capacity (4.8 kWh) for 3,500 charge cycles or 36 months, whichever "
        "occurs first. Modules dropping below 80% threshold qualify for free field replacement within 5 business days.\n\n"
        "2. Exclusions\n"
        "Damage caused by unauthorized firmware flashing, operating on floor gradients steeper than 12 degrees, "
        "or immersion in standing water deeper than 40 millimeters voids the standard actuator warranty."
    )

    page2_text = (
        "NEXAVAULT ROBOTICS INC. — ENTERPRISE SUPPORT TIERS (POL-WAR-910 PAGE 2)\n\n"
        "3. Mission-Critical Support Tiers\n"
        "- Gold Fleet Care ($14,500 USD / site / year): 8x5 remote telemetry diagnostics, next-business-day "
        "spare parts dispatch, and quarterly preventative calibration.\n"
        "- Platinum 24/7 Autonomous Care ($29,000 USD / site / year): 24/7/365 live robotics SRE support, "
        "4-hour on-site field engineer arrival in metro zones, and dedicated spare Helios-9 hot-swap unit on site.\n\n"
        "4. RMA Return Procedure (Code RMA-NV-77)\n"
        "Customers must submit telemetry crash bundles via portal rma.nexavault.internal and display the "
        "approved RMA-NV-77 barcode on the outer antistatic crate before carrier pickup."
    )

    p1 = doc.new_page(width=612, height=792)
    p1.insert_textbox(pymupdf.Rect(50, 50, 562, 742), page1_text, fontsize=11, fontname="helv")

    p2 = doc.new_page(width=612, height=792)
    p2.insert_textbox(pymupdf.Rect(50, 50, 562, 742), page2_text, fontsize=11, fontname="helv")

    doc.save(pdf_path)
    doc.close()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = DATA_DIR / "nexavault_hardware_warranty.pdf"
    create_sample_company_pdf(pdf_path)

    print("=" * 72)
    print("PHASE 1 TEST: LOCAL DOCUMENT INGESTION (ChromaDB + BM25)")
    print("=" * 72)

    report = ingest_directory(
        data_dir=DATA_DIR,
        chroma_dir=CHROMA_DIR,
        corpus_path=BM25_CORPUS_PATH,
        reset_index=True,
    )

    print(f"Files Processed ({len(report.files_processed)}): {report.files_processed}")
    print("Chunks per File:")
    for fname, count in report.chunks_per_file.items():
        print(f"  - {fname}: {count} chunks")
    print(f"Total Chunks Indexed : {report.total_chunks_indexed}")
    print(f"ChromaDB Count       : {report.chroma_count} (persisted at {CHROMA_DIR})")
    print(f"BM25 Corpus Count    : {report.bm25_doc_count} (persisted at {BM25_CORPUS_PATH})")
    print(f"Embedding Dimension  : {report.embedding_dim} (BAAI/bge-small-en-v1.5)")

    assert report.chroma_count == report.total_chunks_indexed > 0, "ChromaDB count mismatch!"
    assert report.bm25_doc_count == report.total_chunks_indexed, "BM25 count mismatch!"
    assert report.embedding_dim == 384, f"Expected 384-d BGE embeddings, got {report.embedding_dim}"

    # Inspect stored metadata in ChromaDB and verify BM25 scoring sanity check
    client = get_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)
    peek = collection.get(limit=4, include=["documents", "metadatas"])

    print("\nSample Stored Chunks & Metadata from ChromaDB:")
    for doc_text, meta in zip(peek["documents"], peek["metadatas"]):
        preview = doc_text.replace("\n", " ")[:110]
        print(
            f"  [{meta['chunk_id']}] "
            f"file={meta['source_file']} | loc={meta['page_or_section']} | "
            f"len={len(doc_text)} chars\n    -> \"{preview}...\""
        )

    bm25, records = load_bm25_index(BM25_CORPUS_PATH)
    assert bm25 is not None
    print("\nPhase 1 Status: PASSED (ChromaDB & BM25 indexes built and verified locally).")


if __name__ == "__main__":
    main()
