"""RecoverOps Part B — confidence guardrail.

Computes an evidence confidence STATE from auditable checks. The state is a
property of the *evidence as a whole* (detection confidence, field
availability, conflicts, parse failures, date consistency) — never a single
raw model confidence.

STATES
------
SAFE     All required evidence available and trusted; no contradictions.
         → automated (but still non-binding) recommendation is allowed.

REVIEW   Some evidence exists and is usable, but confidence is insufficient
         for a fully automated path.
         → recommendation produced, flagged for human confirmation.

BLOCKED  Critical evidence missing, untrusted, or contradictory. No decision
         may be trusted.
         → generic escalate-to-human action only.

CHECKS (G1–G6)
--------------
G1 required_fields_present   : required fields have at least a parsed value
                               or an extraction-backed (non region-only)
                               detection.
G2 required_fields_trusted   : every required field's best confidence ≥
                               REVIEW_CONFIDENCE.
G3 required_fields_confident : every required field's best confidence ≥
                               SAFE_CONFIDENCE (soft: failure → REVIEW).
G4 no_conflicting_evidence   : no field has disagreeing parsed values.
G5 no_unparseable_values     : extracted text parsed cleanly (soft).
G6 date_consistency          : when both dates are parsed, issue ≤ due
                               (soft).

Any hard check failing (G1, G2, G4) ⇒ BLOCKED. All hard checks passing with
at least one soft check failing ⇒ REVIEW. Everything passing ⇒ SAFE.

Intent vs. state: the router (intent_router.py) decides what the document's
situation is — it may route low-confidence but usable evidence to
manual_review directly. The guardrail independently grades how much the
evidence can be TRUSTED; recovery.decide() combines both so that a non-SAFE
state always forces human review.
"""
from __future__ import annotations

from typing import List

from src.reasoning import config
from src.reasoning.schemas import (
    ConfidenceState,
    GuardrailCheck,
    GuardrailReport,
    NormalizedEvidence,
)


def _date(iso: str):
    from datetime import date as _date_type

    return _date_type.fromisoformat(iso)


def evaluate_guardrail(ev: NormalizedEvidence) -> GuardrailReport:
    """Run G1–G6 over normalized evidence and derive the confidence state."""
    checks: List[GuardrailCheck] = []
    blocking: List[str] = []
    review: List[str] = []

    # G1 — required field availability ------------------------------------
    # A field counts as available when it has a parsed value OR an
    # extraction-backed detection (text was found even if it did not parse —
    # that failure is graded by the soft check G5, not here). Region-only
    # detections (no text at all) do NOT count: a bbox is not a value.
    missing: List[str] = []
    for name in config.REQUIRED_FIELDS:
        rf = ev.field(name)
        if rf is None:
            missing.append(name)
            continue
        has_value = rf.resolved_value is not None
        has_extraction = rf.present and not rf.region_only
        if not (has_value or has_extraction):
            missing.append(name)
    g1 = GuardrailCheck(
        check_id="G1-required-fields-present",
        passed=not missing,
        detail=(
            "all required fields present"
            if not missing
            else f"missing usable evidence for: {', '.join(missing)}"
        ),
    )
    checks.append(g1)
    if not g1.passed:
        blocking.append(g1.detail)

    # G2 — trust floor on required fields ----------------------------------
    untrusted: List[str] = []
    for name in config.REQUIRED_FIELDS:
        rf = ev.field(name)
        if rf is None or rf.best_confidence < config.REVIEW_CONFIDENCE:
            untrusted.append(name)
    g2 = GuardrailCheck(
        check_id="G2-required-fields-trusted",
        passed=not untrusted,
        detail=(
            "required fields meet the trust floor"
            if not untrusted
            else f"confidence below trust floor for: {', '.join(untrusted)}"
        ),
    )
    checks.append(g2)
    if not g2.passed:
        blocking.append(g2.detail)

    # G4 — conflicts --------------------------------------------------------
    conflicting: List[str] = [n for n, rf in ev.fields.items() if rf.conflicting]
    g4 = GuardrailCheck(
        check_id="G4-no-conflicting-evidence",
        passed=not conflicting,
        detail=(
            "no contradictory detections"
            if not conflicting
            else f"conflicting values for: {', '.join(conflicting)}"
        ),
    )
    checks.append(g4)
    if not g4.passed:
        blocking.append(g4.detail)

    # G3 — safe-confidence bar (soft) --------------------------------------
    not_confident: List[str] = []
    for name in config.REQUIRED_FIELDS:
        rf = ev.field(name)
        if rf is None or rf.best_confidence < config.SAFE_CONFIDENCE:
            not_confident.append(name)
    g3 = GuardrailCheck(
        check_id="G3-required-fields-confident",
        passed=not not_confident,
        detail=(
            "required fields meet the safe-confidence bar"
            if not not_confident
            else f"confidence below safe bar for: {', '.join(not_confident)}"
        ),
    )
    checks.append(g3)
    if not g3.passed:
        review.append(g3.detail)

    # G5 — parse failures (soft) -------------------------------------------
    parse_problems: List[str] = [
        f"{n}: {v}" for n, rf in ev.fields.items() for v in rf.parse_failures
    ]
    g5 = GuardrailCheck(
        check_id="G5-no-unparseable-values",
        passed=not parse_problems,
        detail=(
            "all extracted values parsed cleanly"
            if not parse_problems
            else f"unparseable values: {'; '.join(parse_problems)}"
        ),
    )
    checks.append(g5)
    if not g5.passed:
        review.append(g5.detail)

    # G6 — date consistency (soft) -----------------------------------------
    issue_rf = ev.field("date_issue")
    due_rf = ev.field("date_due")
    if (
        issue_rf is not None
        and due_rf is not None
        and issue_rf.resolved_value is not None
        and due_rf.resolved_value is not None
    ):
        issue = _date(str(issue_rf.resolved_value))
        due = _date(str(due_rf.resolved_value))
        consistent = issue <= due
    else:
        consistent = True  # not checkable without both values — not a failure
    g6 = GuardrailCheck(
        check_id="G6-date-consistency",
        passed=consistent,
        detail=(
            "issue date on or before due date"
            if consistent
            else "date_issue is after date_due"
        ),
    )
    checks.append(g6)
    if not g6.passed:
        review.append(g6.detail)

    if blocking:
        state = ConfidenceState.BLOCKED
    elif review:
        state = ConfidenceState.REVIEW
    else:
        state = ConfidenceState.SAFE

    return GuardrailReport(
        state=state, checks=checks, blocking_reasons=blocking, review_reasons=review
    )
