"""Phase 4 Verification Script: End-to-end headless Streamlit AppTest of app.py."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest


def main() -> None:
    print("=" * 78)
    print("PHASE 4 TEST: STREAMLIT UI HEADLESS APPTEST (app.py)")
    print("=" * 78)

    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()

    assert not at.exception, f"Streamlit initial boot raised exception: {at.exception}"
    print(f"Initial UI Boot      : OK (zero exceptions)")
    print(f"Sidebar Metrics      : {[ (m.label, m.value) for m in at.sidebar.metric ]}")
    print(f"Sidebar Toggles      : {[ (t.label, t.value) for t in at.sidebar.toggle ]}")

    # Submit a real user query through st.chat_input
    test_query = "What are the SEV-1 incident notification SLAs in SEC-RUN-880?"
    print(f"\nSubmitting Chat Query: {test_query}")
    at.chat_input[0].set_value(test_query).run()

    assert not at.exception, f"Chat submission raised exception: {at.exception}"
    assert len(at.chat_message) == 2, f"Expected 2 chat messages (user + assistant), got {len(at.chat_message)}"

    # Check assistant markdown & expanders ("Sources & Retrieval Debug")
    assistant_msg = at.chat_message[1]
    md_blocks = [m.value for m in assistant_msg.markdown]
    expander_labels = [e.label for e in at.expander]

    print("\nAssistant Rendered Markdown Blocks:")
    for idx, block in enumerate(md_blocks[:4], start=1):
        print(f"  [{idx}] {block[:160]}")

    print("\nRendered Expanders:")
    for lbl in expander_labels:
        print(f"  - {lbl}")

    debug_expanders = [lbl for lbl in expander_labels if "Sources & Retrieval Debug" in lbl]
    assert len(debug_expanders) >= 1, "Expected 'Sources & Retrieval Debug' expander to be rendered!"

    # Toggle off debug view in sidebar and verify expander hides cleanly
    at.sidebar.toggle[0].set_value(False).run()
    assert not at.exception, f"Toggling debug panel raised exception: {at.exception}"
    after_toggle_debug = [e.label for e in at.expander if "Sources & Retrieval Debug" in e.label]
    assert len(after_toggle_debug) == 0, "Expected debug expander to hide when toggle=False!"
    print("Debug Toggle Check   : OK (Expander visible when ON, hidden when OFF)")

    print("\nPhase 4 Status: PASSED (Streamlit UI, Chat RAG, Citations & Debug Panel verified).")


if __name__ == "__main__":
    main()
