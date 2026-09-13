"""RecoverOps Part B — intent routing.

Deterministic rule table — no LLM. Evaluated top-down; the FIRST rule whose
condition holds wins. ``reason()`` documents the routing inputs without
deciding, so the router stays auditable as pure logic.

ROUTING RULES (evaluated in order)
----------------------------------
R1  manual_review — conflicting evidence
    Any known field has multiple detections whose parsed values disagree.
    WHY FIRST: a contradiction cannot be auto-resolved, so no downstream rule
    may act on the document (spec: "Low confidence / conflicting evidence →
    manual_review").

R2  missing_financial_evidence — amount unusable
    amount_due has no usable parsed value (absent, region-only detection,
    unparseable text, or confidence below the trust floor).
    WHY BEFORE R3: without an amount there is no receivable to characterise,
    so the due-date question is moot.

R3  missing_due_date — amount usable, date_due unusable
    Split from R2 so vendors can be asked for the one missing artifact.

R4  manual_review — low confidence
    Both required values are usable, but a required field's best confidence
    is below SAFE_CONFIDENCE (still at/above the trust floor, otherwise R2
    applies). Usable but not trusted enough to act on automatically.

R5  overdue_payment
    date_due < as_of − OVERDUE_GRACE_DAYS and amount usable.
    The obligation has matured: recovery action applies.

R6  upcoming_due
    date_due >= as_of − OVERDUE_GRACE_DAYS and amount usable.

R7  manual_review — fallback
    Currently unreachable (R2–R6 cover all evidence shapes); kept as an
    explicit safety net for future rule changes.

Intent describes the DOCUMENT's situation; the guardrail state (SAFE /
REVIEW / BLOCKED) describes how much the evidence can be TRUSTED. The two are
combined in recovery.decide(): a non-SAFE state always forces human review
even when routing itself was unambiguous.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional, Tuple

from src.reasoning import config
from src.reasoning.schemas import Intent, NormalizedEvidence


def _overdue_cutoff(as_of: date) -> date:
    """First date a due date is considered overdue (grace included)."""
    return as_of - timedelta(days=config.OVERDUE_GRACE_DAYS)


def _usable_amount(ev: NormalizedEvidence) -> Optional[float]:
    """amount_due value only when present, non-conflicting, and parseable."""
    rf = ev.field("amount_due")
    if rf is None or rf.conflicting or rf.resolved_value is None:
        return None
    return float(rf.resolved_value)


def _usable_due_date(ev: NormalizedEvidence) -> Optional[date]:
    rf = ev.field("date_due")
    if rf is None or rf.conflicting or rf.resolved_value is None:
        return None
    return date.fromisoformat(str(rf.resolved_value))


def _conflicting_fields(ev: NormalizedEvidence) -> List[str]:
    return sorted(name for name, rf in ev.fields.items() if rf.conflicting)


def _low_confidence_fields(ev: NormalizedEvidence) -> List[str]:
    return sorted(
        name
        for name in config.REQUIRED_FIELDS
        if ev.field(name) is not None
        and ev.field(name).best_confidence < config.SAFE_CONFIDENCE
    )


def route_intent(ev: NormalizedEvidence) -> Tuple[Intent, List[str]]:
    """Apply the rule table and return ``(intent, rules_fired)``."""
    conflicts = _conflicting_fields(ev)
    if conflicts:
        return Intent.MANUAL_REVIEW, ["R1-conflicting-evidence"]

    amount = _usable_amount(ev)
    if amount is None:
        return Intent.MISSING_FINANCIAL_EVIDENCE, ["R2-amount-unusable"]

    due = _usable_due_date(ev)
    if due is None:
        return Intent.MISSING_DUE_DATE, ["R3-due-date-unusable"]

    low = _low_confidence_fields(ev)
    if low:
        return Intent.MANUAL_REVIEW, ["R4-low-confidence"]

    if due < _overdue_cutoff(ev.as_of_date):
        return Intent.OVERDUE_PAYMENT, ["R5-past-due"]
    return Intent.UPCOMING_DUE, ["R6-not-yet-due"]


def reason(ev: NormalizedEvidence) -> str:
    """Human-readable documentation of the routing inputs (audit only)."""
    amount = _usable_amount(ev)
    due = _usable_due_date(ev)
    parts = []
    parts.append(
        f"amount_due usable={amount is not None}"
        + (f" (value={amount:.2f})" if amount is not None else "")
    )
    parts.append(
        f"date_due usable={due is not None}"
        + (f" (value={due.isoformat()})" if due is not None else "")
    )
    parts.append(f"as_of={ev.as_of_date.isoformat()}")
    return "; ".join(parts)


__all__ = [
    "route_intent",
    "reason",
    "_overdue_cutoff",
    "_usable_amount",
    "_usable_due_date",
]
