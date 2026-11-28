"""
tests/test_streamlit_app.py
---------------------------
Programmatic UI & End-to-End Streamlit AppTest suite for app/streamlit_app.py.

Verifies:
  1. All 5 pages render without exceptions.
  2. Overview Dashboard starts at zero (no fabricated historical stats).
  3. Empty/incomplete Analyze Post inputs show user-friendly validation errors
     without inventing missing metadata or crashing.
  4. Preset fill + Analyze Post runs live SocialThreatPipeline across Agents 1, 2, 3.
  5. REVIEW decisions enter Human Review Queue and preserve both agent outputs.
  6. Human reviewer resolution calls resolve() and preserves ai_decision.
  7. Overview Dashboard counters and Analysis Results reflect session state.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "streamlit_app.py"))


def _new_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def _navigate(at: AppTest, page_name: str) -> None:
    nav = next(r for r in at.sidebar.radio if r.key == "active_page")
    nav.set_value(page_name).run()
    assert not at.exception


class TestStreamlitDashboard:

    def test_all_pages_load_with_zero_initial_stats(self):
        at = _new_app()
        # Page 1: Overview Dashboard starts at zero
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Total Analyzed"] == "0"
        assert metrics["Approved"] == "0"
        assert metrics["Review Required"] == "0"
        assert metrics["Restricted"] == "0"
        assert metrics["Bot Detections"] == "0"
        assert metrics["Fake News"] == "0"

        for page in (
            "Analyze a Post",
            "Analysis Results",
            "Human Review Queue",
            "System Information",
            "Overview Dashboard",
        ):
            _navigate(at, page)
            assert not at.exception

    def test_analyze_post_rejects_incomplete_metadata_safely(self):
        at = _new_app()
        _navigate(at, "Analyze a Post")

        submit_btn = next(b for b in at.button if b.label == "Analyze Post")
        submit_btn.click().run()

        assert not at.exception
        assert len(at.error) >= 1
        assert len(at.session_state["analysis_history"]) == 0

    def test_full_e2e_analysis_review_queue_resolution_and_session_state(self):
        at = _new_app()
        _navigate(at, "Analyze a Post")

        # 1) Submit Human + Real News preset -> APPROVE (not queued)
        btn_hr = next(b for b in at.button if b.label == "Human + Real News")
        btn_hr.click().run()
        submit_btn = next(b for b in at.button if b.label == "Analyze Post")
        submit_btn.click().run()
        assert not at.exception
        assert len(at.session_state["analysis_history"]) == 1
        assert at.session_state["latest_result"]["moderation"]["decision"] == "APPROVE"
        assert at.session_state["latest_result"]["review_queue"]["queued"] is False

        # 2) Submit Bot + Real News preset -> REVIEW (queued)
        btn_br = next(b for b in at.button if b.label == "Bot + Real News")
        btn_br.click().run()
        submit_btn = next(b for b in at.button if b.label == "Analyze Post")
        submit_btn.click().run()
        assert not at.exception
        assert len(at.session_state["analysis_history"]) == 2
        latest_br = at.session_state["latest_result"]
        assert latest_br["fake_news"]["prediction"] == "real"
        assert latest_br["bot_detection"]["prediction"] == "bot"
        assert latest_br["moderation"]["decision"] == "REVIEW"
        assert latest_br["review_queue"]["queued"] is True

        queue = at.session_state["review_queue"]
        assert len(queue.get_pending()) == 1
        pending_case = queue.get_pending()[0]
        assert pending_case.ai_decision == "REVIEW"
        assert pending_case.input_predictions["fake_news"]["prediction"] == "real"
        assert pending_case.input_predictions["bot_detection"]["prediction"] == "bot"

        # 3) Navigate to Analysis Results page -> cards render cleanly
        _navigate(at, "Analysis Results")
        assert not at.exception
        assert len(at.metric) >= 6

        # 4) Navigate to Human Review Queue and resolve the pending REVIEW case
        _navigate(at, "Human Review Queue")
        assert not at.exception

        # Fill reviewer ID & notes and submit human decision in one form submission
        rev_input = next(ti for ti in at.text_input if "Reviewer ID" in ti.label)
        rev_input.input("moderator_alice")
        notes_input = next(ta for ta in at.text_area if "Reviewer Notes" in ta.label)
        notes_input.input("Verified factual news posted by automated news feed.")
        decision_box = next(sb for sb in at.selectbox if "Human Reviewer Decision" in sb.label)
        decision_box.select("APPROVE")

        resolve_btn = next(b for b in at.button if b.label == "Submit Human Decision")
        resolve_btn.click().run()
        assert not at.exception
        assert len(at.success) >= 1

        resolved_queue = at.session_state["review_queue"]
        assert len(resolved_queue.get_pending()) == 0
        assert len(resolved_queue.get_resolved()) == 1
        resolved_case = resolved_queue.get_resolved()[0]
        assert resolved_case.human_decision == "APPROVE"
        assert resolved_case.ai_decision == "REVIEW"  # Original AI decision preserved!
        assert resolved_case.reviewer_id == "moderator_alice"

        # 5) Overview Dashboard reflects exact updated session counts
        _navigate(at, "Overview Dashboard")
        assert not at.exception
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Total Analyzed"] == "2"
        assert metrics["Approved"] == "1"
        assert metrics["Review Required"] == "1"
        assert metrics["Restricted"] == "0"
        assert metrics["Bot Detections"] == "1"
        assert metrics["Fake News"] == "0"

        # 6) System Information renders cleanly with synthetic disclaimer
        _navigate(at, "System Information")
        assert not at.exception
        assert len(at.dataframe) >= 1
