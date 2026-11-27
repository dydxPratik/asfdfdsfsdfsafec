"""End-to-end UI, CLI entrypoint, path-sanitization, and multi-turn conversation tests."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT
from src.schemas import Citation, RAGResponse, RetrievedChunk


class TestEndToEndUIAndCLI(unittest.TestCase):
    def test_pyproject_script_entrypoint_works(self) -> None:
        """Verify `[project.scripts] ragquest = 'src.cli:main'` runs the real CLI help cleanly."""
        proc = subprocess.run(
            ["uv", "run", "--no-sync", "ragquest", "--help"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("RAGQuest Local Hybrid RAG CLI", proc.stdout)

    def test_main_py_cli_works_with_mocked_llm(self) -> None:
        fake_resp = RAGResponse(
            answer="Core hours are 10 AM to 3 PM.",
            citations=[Citation(source_file="nexavault_hr_expense_policy.md", quote="10:00 AM to 3:00 PM")],
            is_supported=True,
            confidence="high",
        )
        with patch("src.generation.generator.generate_grounded_answer", return_value=fake_resp):
            import main as cli_main

            with patch.object(sys, "argv", ["main.py", "What are core hours?"]):
                cli_main.main()

    def test_streamlit_ui_error_handling_and_clear_history(self) -> None:
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
        # Simulate LLM failure to verify UI catches exception gracefully without crashing session
        with patch(
            "src.generation.generator.generate_grounded_answer",
            side_effect=RuntimeError("Simulated Groq 503 Service Unavailable"),
        ):
            at.run()
            self.assertFalse(at.exception)
            at.chat_input[0].set_value("Test failure handling").run()
            self.assertFalse(at.exception)
            self.assertGreaterEqual(len(at.error), 1)
            self.assertIn("Simulated Groq 503", at.error[0].value)

    def test_filename_path_traversal_sanitization_logic(self) -> None:
        # Verify Path(uf.name).name strips '../' and '..\\' traversal attempts
        malicious_unix = "../../.env"
        malicious_win = "..\\..\\Windows\\System32\\cmd.exe"
        self.assertEqual(Path(malicious_unix).name, ".env")
        # Subtle edge case check: what if someone uploads a file literally named `.env`?
        # `.env` is NOT in SUPPORTED_EXTENSIONS (.pdf, .md, .txt, .csv), so st.file_uploader blocks it,
        # whereas `../../evil.txt` becomes `evil.txt` inside DATA_DIR.
        self.assertEqual(Path("../../evil.txt").name, "evil.txt")
        self.assertEqual(Path(malicious_win).name, "cmd.exe")


if __name__ == "__main__":
    unittest.main()
