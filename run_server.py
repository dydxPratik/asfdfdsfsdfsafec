"""Single-command launcher for the RAGQuest Real-Time AI Observability Platform."""

from __future__ import annotations

import uvicorn


def main() -> None:
    print("=" * 72)
    print("RAGQuest Studio — Real-Time AI Observability Platform")
    print("Open in Browser: http://localhost:8000")
    print("=" * 72)
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
