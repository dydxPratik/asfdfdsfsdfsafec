"""Shared pytest configuration, offline HuggingFace cache detection, and live-LLM marker gating."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _has_local_bge_cache() -> bool:
    """Return True if BAAI/bge-small-en-v1.5 is already cached locally so offline runs skip Hub pings."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub"))
    model_cache = hub_dir / "models--BAAI--bge-small-en-v1.5" / "snapshots"
    return model_cache.exists() and any(model_cache.iterdir())


if _has_local_bge_cache():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live-llm",
        action="store_true",
        default=False,
        help="Execute tests that make live network calls to the Groq LLM API.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_live = config.getoption("--run-live-llm") or os.environ.get("RAGQUEST_LIVE_LLM") == "1"
    if run_live:
        return

    skip_live = pytest.mark.skip(
        reason="Skipped live Groq API test during offline/fast pytest run (pass --run-live-llm to enable)."
    )
    for item in items:
        if "live_llm" in item.keywords or item.name == "test_e2e_image_text_table_textfree_unanswerable_and_regression":
            item.add_marker(skip_live)
