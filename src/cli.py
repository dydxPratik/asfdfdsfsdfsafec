"""Installed CLI entrypoint (`uv run ragquest`) for querying RAGQuest from the terminal."""

from __future__ import annotations

import argparse

from src.generation.generator import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGQuest Local Hybrid RAG CLI")
    parser.add_argument(
        "query",
        nargs="?",
        default="What are the core collaboration hours and on-site days in POL-WRK-101?",
        help="Company policy question to ask",
    )
    args = parser.parse_args()
    response, _ = answer_question(args.query)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
