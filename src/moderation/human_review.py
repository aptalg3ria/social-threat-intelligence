"""
src/moderation/human_review.py
------------------------------
Lightweight human-in-the-loop review queue for Supervisor decisions.

Preserves the original AI decision separately from the authorized human
reviewer's final disposition (APPROVE, REJECT, or RESTRICT).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


VALID_HUMAN_DECISIONS = ("APPROVE", "REJECT", "RESTRICT")


@dataclass
class ReviewCase:
    """Represents a moderation case routed to human review."""

    case_id: str
    created_at: str
    ai_decision: str
    risk_level: str
    risk_score: float
    review_reasons: list[str]
    input_predictions: dict
    status: str = "PENDING"
    human_decision: str | None = None
    reviewer_id: str | None = None
    reviewer_notes: str = ""
    reviewed_at: str | None = None

    def to_dict(self) -> dict:
        """Serialize the review case into a plain dictionary."""
        return {
            "case_id": self.case_id,
            "created_at": self.created_at,
            "status": self.status,
            "ai_decision": self.ai_decision,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "review_reasons": list(self.review_reasons),
            "human_decision": self.human_decision,
            "reviewer_id": self.reviewer_id,
            "reviewer_notes": self.reviewer_notes,
            "reviewed_at": self.reviewed_at,
            "input_predictions": copy.deepcopy(self.input_predictions),
        }


class ReviewQueue:
    """
    In-memory human review queue that captures cases needing review and
    records authorized reviewer decisions without overwriting AI outputs.
    """

    def __init__(self) -> None:
        self._cases: dict[str, ReviewCase] = {}

    def submit(self, supervisor_output: dict, case_id: str | None = None) -> ReviewCase | None:
        """
        Inspect a Supervisor output and enqueue it if human review is required.

        Returns the created ReviewCase, or None if human_review_required is False.
        """
        if not isinstance(supervisor_output, dict):
            raise TypeError(
                f"supervisor_output must be a dict, got {type(supervisor_output).__name__}."
            )
        if "human_review_required" not in supervisor_output:
            raise ValueError("supervisor_output is missing 'human_review_required'.")

        if not supervisor_output["human_review_required"]:
            return None

        cid = case_id or f"case-{uuid.uuid4().hex[:12]}"
        if cid in self._cases:
            raise ValueError(f"Review case '{cid}' already exists.")

        case = ReviewCase(
            case_id=cid,
            created_at=datetime.now(timezone.utc).isoformat(),
            ai_decision=str(supervisor_output["decision"]),
            risk_level=str(supervisor_output["risk_level"]),
            risk_score=float(supervisor_output["risk_score"]),
            review_reasons=list(supervisor_output.get("reasons", [])),
            input_predictions=copy.deepcopy(supervisor_output.get("input_predictions", {})),
        )
        self._cases[cid] = case
        return case

    def resolve(
        self,
        case_id: str,
        human_decision: str,
        reviewer_id: str,
        notes: str = "",
    ) -> ReviewCase:
        """
        Record an authorized human reviewer's final decision for a pending case.

        Allowed human_decision values: APPROVE, REJECT, RESTRICT.
        """
        if case_id not in self._cases:
            raise KeyError(f"Review case '{case_id}' not found.")

        case = self._cases[case_id]
        if case.status == "RESOLVED":
            raise ValueError(f"Review case '{case_id}' is already resolved.")

        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ValueError("Authorized 'reviewer_id' must be a non-empty string.")

        if not isinstance(human_decision, str):
            raise ValueError("human_decision must be a string.")
        normalized_decision = human_decision.strip().upper()
        if normalized_decision not in VALID_HUMAN_DECISIONS:
            raise ValueError(
                f"Invalid human_decision {human_decision!r}. "
                f"Expected one of {VALID_HUMAN_DECISIONS}."
            )

        case.human_decision = normalized_decision
        case.reviewer_id = reviewer_id.strip()
        case.reviewer_notes = notes.strip() if isinstance(notes, str) else ""
        case.reviewed_at = datetime.now(timezone.utc).isoformat()
        case.status = "RESOLVED"
        return case

    def get_case(self, case_id: str) -> ReviewCase:
        """Retrieve a review case by ID."""
        if case_id not in self._cases:
            raise KeyError(f"Review case '{case_id}' not found.")
        return self._cases[case_id]

    def get_pending(self) -> list[ReviewCase]:
        """Return all cases awaiting human review."""
        return [c for c in self._cases.values() if c.status == "PENDING"]

    def get_resolved(self) -> list[ReviewCase]:
        """Return all resolved cases."""
        return [c for c in self._cases.values() if c.status == "RESOLVED"]

    def list_cases(self) -> list[ReviewCase]:
        """Return all cases in submission order."""
        return list(self._cases.values())
