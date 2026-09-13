"""RecoverOps Part B — recovery decision (bounded recommendations only).

``decide()`` composes the reasoning layer:

    EvidenceRequest → normalize_evidence → evaluate_guardrail → route_intent
    → bounded recommendation → auditable Decision

FINANCIAL SAFETY
----------------
RecoverOps NEVER executes financial transactions. It does not initiate
payments, modify bank information, transfer funds, send legally binding
communications, or make irreversible financial decisions. It emits bounded
recommendations only; uncertain outcomes are escalated to human review.

RECOMMENDATION MAP (intent → action)
------------------------------------
overdue_payment            → payment_reminder
upcoming_due               → pre_due_reminder
missing_due_date           → invoice_verification_request
missing_financial_evidence → document_ocr_review_request
manual_review              → escalate_to_human

SEPARATION OF CONCERNS
----------------------
The intent (what the document's situation is) comes from the router. The
confidence state (how much the evidence can be trusted) comes from the
guardrail. They are combined only at the end:

* ``requires_human_review`` is True whenever the state is not SAFE *or* the
  router fired a manual_review rule — an unsafe or self-declared-uncertain
  outcome can never yield an un-flagged recommendation.
* Intent is never overridden by the guardrail: intent describes the
  document, the state describes the trust. A REVIEW-state overdue invoice is
  still an overdue invoice — but its reminder recommendation is explicitly
  marked for human confirmation.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.reasoning.evidence import normalize_evidence
from src.reasoning.guardrail import evaluate_guardrail
from src.reasoning.intent_router import reason as route_reason
from src.reasoning.intent_router import route_intent
from src.reasoning.schemas import (
    ConfidenceState,
    Decision,
    EvidenceRequest,
    Intent,
    NormalizedEvidence,
)

RECOMMENDED_ACTIONS: dict = {
    Intent.OVERDUE_PAYMENT: "payment_reminder",
    Intent.UPCOMING_DUE: "pre_due_reminder",
    Intent.MISSING_DUE_DATE: "invoice_verification_request",
    Intent.MISSING_FINANCIAL_EVIDENCE: "document_ocr_review_request",
    Intent.MANUAL_REVIEW: "escalate_to_human",
}


def _build_reason(ev, intent: Intent, guardrail, rules) -> str:
    """Deterministic explanation assembled from the actual evidence."""
    if intent == Intent.MANUAL_REVIEW:
        if "R1-conflicting-evidence" in rules:
            return (
                "Conflicting evidence: multiple detections disagree on the same "
                "field value, so no automated action is possible; escalated to "
                "human review."
            )
        if "R4-low-confidence" in rules:
            return (
                "Evidence values are usable but detection confidence is below "
                "the safe threshold; escalated to human review."
            )
        return "Routed to manual review by rule fallback."

    if intent == Intent.OVERDUE_PAYMENT:
        text = (
            f"Due date has passed (as_of {ev.as_of_date.isoformat()}) and amount due "
            "evidence is available."
        )
    elif intent == Intent.UPCOMING_DUE:
        text = (
            f"Due date has not passed (as_of {ev.as_of_date.isoformat()}) and amount due "
            "evidence is available."
        )
    elif intent == Intent.MISSING_DUE_DATE:
        text = (
            "Amount due evidence is available but the due date has no usable value; "
            "invoice verification is required before any recovery action."
        )
    else:  # MISSING_FINANCIAL_EVIDENCE
        text = (
            "Required financial evidence (amount and/or due date) is missing, "
            "unparseable, or untrusted; document/OCR review is required."
        )

    if guardrail.state == ConfidenceState.REVIEW:
        text += f" Review flagged: {'; '.join(guardrail.review_reasons)}."
    elif guardrail.state == ConfidenceState.BLOCKED:
        text += f" Blocked: {'; '.join(guardrail.blocking_reasons)}."

    return text


def decide(request: EvidenceRequest) -> Decision:
    """Produce an auditable, bounded recovery decision for one document."""
    ev = normalize_evidence(request)
    guardrail = evaluate_guardrail(ev)
    intent, rules_fired = route_intent(ev)

    action = RECOMMENDED_ACTIONS[intent]
    reason = _build_reason(ev, intent, guardrail, rules_fired)

    return Decision(
        timestamp=datetime.now(timezone.utc).isoformat(),
        document_id=ev.document_id,
        intent=intent,
        confidence_state=guardrail.state,
        recommended_action=action,
        reason=reason,
        requires_human_review=(
            guardrail.state != ConfidenceState.SAFE
            or intent == Intent.MANUAL_REVIEW
        ),
        rules_fired=rules_fired,
        guardrail_checks=guardrail.checks,
        evidence=list(ev.fields.values()),
    )
