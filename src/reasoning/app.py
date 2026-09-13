"""RecoverOps Part B — minimal FastAPI contract for the reasoning layer.

Endpoints:
    GET  /reason/health — liveness/metadata (no auth, no data).
    POST /reason        — structured evidence in, auditable decision out.

DEC-B1..B4 are the financial-safety binding decisions on this surface:
the API only ever returns bounded recommendations; it exposes no payment,
fund-transfer, bank-detail, or communications endpoints, and /reason never
mutates state.

DECOUPLING: this app imports nothing from the RT-DETR /detect
implementation. /reason accepts plain structured JSON evidence, so it can be
tested independently with mock evidence (see tests/fixtures/reasoning/).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.reasoning.recovery import decide
from src.reasoning.schemas import Decision, EvidenceRequest

app = FastAPI(
    title="RecoverOps Part B — Reasoning Layer",
    version="0.1.0",
    description=(
        "Deterministic, auditable recovery reasoning over structured invoice "
        "evidence. Produces bounded recommendations only — never executes "
        "financial transactions."
    ),
)

_DECISIONS_RECORDED = 0  # in-process audit counter (no persistence by design)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@app.get("/reason/health", response_model=HealthResponse)
def reason_health() -> HealthResponse:
    """Liveness probe for the reasoning layer."""
    return HealthResponse(status="ok", service="reasoning", version=app.version)


@app.post("/reason", response_model=Decision)
def reason(request: EvidenceRequest) -> Decision:
    """Normalize evidence → guardrail → route intent → bounded recommendation.

    Always returns 200 with a full decision object. Evidence-quality problems
    are encoded in the decision itself (confidence_state BLOCKED, intent
    manual_review, requires_human_review true) rather than as HTTP errors,
    because 'this document cannot be trusted' IS the decision.
    """
    global _DECISIONS_RECORDED
    decision = decide(request)
    _DECISIONS_RECORDED += 1
    return decision


def decisions_recorded() -> int:
    """Audit counter (in-process). Kept trivial by design; no secrets/PII stored."""
    return _DECISIONS_RECORDED
