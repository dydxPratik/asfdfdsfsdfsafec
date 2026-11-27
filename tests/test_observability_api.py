"""Integration tests for the FastAPI + SSE Real-Time Observability Backend."""

from __future__ import annotations

import json
import time
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.server import app, run_manager
from src.schemas import Citation, RAGResponse


class TestObservabilityAPIEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_overview_and_documents_endpoints(self) -> None:
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)
        h_json = h.json()
        self.assertEqual(h_json["status"], "ok")
        self.assertGreaterEqual(h_json["chroma_chunks"], 1)

        ov = self.client.get("/api/overview")
        self.assertEqual(ov.status_code, 200)
        self.assertIn("telemetry", ov.json())

        docs = self.client.get("/api/documents")
        self.assertEqual(docs.status_code, 200)
        self.assertGreaterEqual(len(docs.json()["documents"]), 4)

    def test_retrieval_explorer_side_by_side(self) -> None:
        res = self.client.post(
            "/api/retrieval/explore",
            json={"query": "employee NV-EMP-1042 hub", "top_n": 3},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertGreaterEqual(len(body["dense"]), 1)
        self.assertGreaterEqual(len(body["bm25"]), 1)
        self.assertEqual(len(body["reranked"]), 3)
        self.assertIn("rank_delta", body["reranked"][0])

    def test_traced_query_run_and_sse_snapshot_recovery(self) -> None:
        fake_llm = RAGResponse(
            answer="The lodging cap in Zurich is $340 USD per night.",
            citations=[
                Citation(
                    source_file="nexavault_hr_expense_policy.md",
                    quote="Lodging cap: **$340 USD / night**",
                )
            ],
            is_supported=True,
            confidence="high",
        ).model_dump_json()

        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=fake_llm))],
            usage=SimpleNamespace(prompt_tokens=1500, completion_tokens=180, total_tokens=1680),
        )

        with patch("src.api.traced_pipeline.get_groq_client", return_value=mock_groq):
            start_res = self.client.post(
                "/api/query/start",
                json={"query": "What is the lodging cap in Zurich?", "top_n": 4},
            )
            self.assertEqual(start_res.status_code, 200)
            run_id = start_res.json()["run_id"]

            # Poll until background thread finishes (typically <1.5s locally with mocked Groq)
            deadline = time.time() + 15.0
            final_run = None
            while time.time() < deadline:
                r = self.client.get(f"/api/runs/{run_id}").json()
                if r["status"] in ("completed", "failed"):
                    final_run = r
                    break
                time.sleep(0.1)

            self.assertIsNotNone(final_run)
            self.assertEqual(final_run["status"], "completed", msg=final_run.get("error"))
            self.assertEqual(final_run["stages"]["cross_encoder_rerank"]["status"], "completed")
            self.assertEqual(final_run["stages"]["llm_generation"]["status"], "completed")
            self.assertEqual(
                final_run["stages"]["llm_generation"]["metadata"]["token_usage"]["total_tokens"],
                1680,
            )
            self.assertEqual(final_run["stages"]["citation_verification"]["status"], "completed")
            self.assertTrue(final_run["result"]["response"]["is_supported"])

            # Verify SSE stream endpoint replays authoritative snapshot cleanly
            with self.client.stream("GET", f"/api/runs/{run_id}/stream") as sse_resp:
                self.assertEqual(sse_resp.status_code, 200)
                lines = [ln for ln in sse_resp.iter_lines() if ln.startswith("data: ")]
                self.assertGreaterEqual(len(lines), 1)
                first_msg = json.loads(lines[0].replace("data: ", "", 1))
                self.assertEqual(first_msg["type"], "snapshot")
                self.assertEqual(first_msg["run"]["run_id"], run_id)


if __name__ == "__main__":
    unittest.main()
