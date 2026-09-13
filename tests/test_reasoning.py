"""RecoverOps Part B — reasoning layer unit tests.

Every test states its expected result explicitly. Run with: pytest
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.reasoning.config import SAFE_CONFIDENCE
from src.reasoning.evidence import (
    amounts_agree,
    normalize_evidence,
    parse_amount,
    parse_date,
)
from src.reasoning.guardrail import evaluate_guardrail
from src.reasoning.intent_router import route_intent
from src.reasoning.recovery import decide
from src.reasoning.schemas import (
    ConfidenceState,
    Decision,
    EvidenceField,
    EvidenceRequest,
    FieldName,
    Intent,
    SourceType,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reasoning"
FIXTURE_FILES = sorted(FIXTURES.glob("*.json"))

AS_OF = date(2026, 9, 1)
PAST = "2026-08-15"
FUTURE = "2026-09-30"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def field(
    name: str,
    value: str | None = None,
    confidence: float = 0.9,
    source: str = "rtdetr",
    bbox: list | None = None,
) -> EvidenceField:
    return EvidenceField(
        field=FieldName(name),
        value=value,
        source=SourceType(source),
        confidence=confidence,
        bbox=bbox,
    )


def ev(fields, document_id="doc", as_of=AS_OF) -> EvidenceRequest:
    return EvidenceRequest(
        document_id=document_id,
        fields=list(fields),
        as_of_date=as_of,
    )


def load_fixture(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def fixture_request(path: Path) -> EvidenceRequest:
    return EvidenceRequest(**load_fixture(path)["request"])


# ---------------------------------------------------------------------------
# Fixtures (static data) sanity
# ---------------------------------------------------------------------------
def test_all_six_scenarios_present():
    assert len(FIXTURE_FILES) == 6


@pytest.mark.parametrize("path", FIXTURE_FILES)
def test_fixture_is_valid_and_labeled_synthetic(path: Path):
    data = load_fixture(path)
    assert "TEST FIXTURE" in data["_label"]
    assert "not a real invoice" in data["_label"]
    req = EvidenceRequest(**data["request"])  # must validate strictly
    assert req.document_id


# ---------------------------------------------------------------------------
# Evidence parsing / normalization
# ---------------------------------------------------------------------------
def test_parse_amount_handles_common_invoice_formats():
    assert parse_amount("Rs. 55,000.00") == 55000.00
    assert parse_amount("INR 1,23,456.78") == 123456.78
    assert parse_amount("$ 2,500") == 2500.0
    assert parse_amount("55000.00") == 55000.00
    assert parse_amount("2.500,00") == 2500.0  # European decimal comma honoured


def test_parse_amount_rejects_garbage():
    assert parse_amount("") is None
    assert parse_amount("N/A") is None
    assert parse_amount("abc123") is None
    assert parse_amount(".99") is None  # ambiguous cents-only string not guessed


def test_parse_date_formats_and_rejects_garbage():
    assert parse_date("2026-08-15") == "2026-08-15"
    assert parse_date("15/08/2026") == "2026-08-15"  # day-first
    assert parse_date("15.08.2026") == "2026-08-15"
    assert parse_date("2026-13-99") is None
    assert parse_date("soon") is None


def test_region_only_detection_is_present_but_valueless():
    req = ev([field("amount_due", None, 0.95, bbox=[1, 2, 3, 4])])
    state = normalize_evidence(req)
    rf = state.field("amount_due")
    assert rf.present is True
    assert rf.region_only is True
    assert rf.resolved_value is None


def test_conflicting_amounts_marked_and_value_withheld():
    req = ev(
        [
            field("amount_due", "54000.00", 0.91),
            field("amount_due", "55000.00", 0.88),
            field("date_due", PAST, 0.9),
        ]
    )
    state = normalize_evidence(req)
    rf = state.field("amount_due")
    assert rf.conflicting is True
    assert rf.resolved_value is None  # never silently pick a side


def test_amounts_agree_uses_configured_tolerance():
    assert amounts_agree(100.0, 100.0) is True
    assert amounts_agree(54000.0, 55000.0) is False  # tolerance is 0.0 by default


# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------
def test_route_overdue_when_due_date_past_and_amount_present():
    req = ev([field("amount_due", "55000.00"), field("date_due", PAST)])
    intent, rules = route_intent(normalize_evidence(req))
    assert intent == Intent.OVERDUE_PAYMENT
    assert "R5-past-due" in rules


def test_route_upcoming_when_due_date_future_and_amount_present():
    req = ev([field("amount_due", "18750.50"), field("date_due", FUTURE)])
    intent, _ = route_intent(normalize_evidence(req))
    assert intent == Intent.UPCOMING_DUE


def test_route_missing_due_date_when_amount_present_without_due():
    req = ev([field("amount_due", "22000"), field("date_issue", "2026-08-20")])
    intent, rules = route_intent(normalize_evidence(req))
    assert intent == Intent.MISSING_DUE_DATE
    assert "R3-due-date-unusable" in rules


def test_route_missing_financial_evidence_when_amount_missing():
    req = ev([field("date_due", FUTURE), field("vendor_name", "Acme Supplies Ltd")])
    intent, _ = route_intent(normalize_evidence(req))
    assert intent == Intent.MISSING_FINANCIAL_EVIDENCE


def test_route_missing_financial_evidence_when_amount_region_only():
    # RT-DETR found the region but no OCR text: value cannot be asserted.
    req = ev([field("amount_due", None, 0.95), field("date_due", PAST, 0.9)])
    intent, _ = route_intent(normalize_evidence(req))
    assert intent == Intent.MISSING_FINANCIAL_EVIDENCE


def test_due_date_equal_to_as_of_is_upcoming_not_overdue():
    req = ev([field("amount_due", "100"), field("date_due", "2026-09-01")])
    intent, _ = route_intent(normalize_evidence(req))
    assert intent == Intent.UPCOMING_DUE  # grace days = 0 → overdue strictly before as_of


def test_conflicting_evidence_routes_to_manual_review():
    req = ev(
        [
            field("amount_due", "54000.00", 0.91),
            field("amount_due", "55000.00", 0.88),
            field("date_due", PAST, 0.9),
        ]
    )
    intent, rules = route_intent(normalize_evidence(req))
    assert intent == Intent.MANUAL_REVIEW
    assert "R1-conflicting-evidence" in rules


def test_overdue_respects_configured_grace_days(monkeypatch):
    from src.reasoning import config, intent_router

    monkeypatch.setattr(config, "OVERDUE_GRACE_DAYS", 3)
    req = ev([field("amount_due", "100"), field("date_due", "2026-08-30")])
    intent, _ = intent_router.route_intent(normalize_evidence(req))
    assert intent == Intent.UPCOMING_DUE  # within 3-day grace on 2026-09-01


def test_low_confidence_routes_to_manual_review():
    # Usable values but below the SAFE bar → manual_review (spec rule 5).
    req = ev([field("amount_due", "9500", 0.64), field("date_due", FUTURE, 0.62)])
    intent, rules = route_intent(normalize_evidence(req))
    assert intent == Intent.MANUAL_REVIEW
    assert "R4-low-confidence" in rules


def test_unknown_fields_are_ignored():
    req = ev([field("amount_due", "55000.00"), field("date_due", PAST)])
    req.fields.append(
        EvidenceField(field=FieldName.DOCUMENT_ID, value="INV-001", confidence=0.9)
    )
    state = normalize_evidence(req)
    assert set(state.fields) == set(
        ("amount_due", "date_due", "document_id", "date_issue", "vendor_name")
    )


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------
def test_guardrail_safe_when_all_required_evidence_trusted():
    req = ev([field("amount_due", "55000.00", 0.94), field("date_due", PAST, 0.92)])
    report = evaluate_guardrail(normalize_evidence(req))
    assert report.state == ConfidenceState.SAFE
    assert report.blocking_reasons == []
    assert report.review_reasons == []


def test_guardrail_review_when_below_safe_threshold():
    # Above the REVIEW floor, below the SAFE bar → REVIEW, not BLOCKED.
    assert SAFE_CONFIDENCE == 0.80
    req = ev([field("amount_due", "9500", 0.64), field("date_due", FUTURE, 0.62)])
    report = evaluate_guardrail(normalize_evidence(req))
    assert report.state == ConfidenceState.REVIEW
    assert any(c.check_id.startswith("G3") for c in report.checks)


def test_guardrail_blocked_when_required_field_missing():
    req = ev([field("amount_due", "55000.00", 0.94)])  # no date_due at all
    report = evaluate_guardrail(normalize_evidence(req))
    assert report.state == ConfidenceState.BLOCKED
    assert any("date_due" in r for r in report.blocking_reasons)


def test_guardrail_blocked_when_required_field_untrusted():
    req = ev([field("amount_due", "55000.00", 0.55), field("date_due", PAST, 0.9)])
    report = evaluate_guardrail(normalize_evidence(req))
    assert report.state == ConfidenceState.BLOCKED  # below the 0.60 trust floor


def test_guardrail_blocked_on_conflicting_amounts():
    req = ev(
        [
            field("amount_due", "54000.00", 0.91),
            field("amount_due", "55000.00", 0.88),
            field("date_due", PAST, 0.9),
        ]
    )
    report = evaluate_guardrail(normalize_evidence(req))
    assert report.state == ConfidenceState.BLOCKED
    assert any("amount_due" in r for r in report.blocking_reasons)


def test_guardrail_review_on_unparseable_extracted_text():
    req = ev([field("amount_due", "SEE ATTACHED", 0.9), field("date_due", PAST, 0.9)])
    report = evaluate_guardrail(normalize_evidence(req))
    assert report.state == ConfidenceState.REVIEW  # G5 soft failure


def test_guardrail_review_on_date_inconsistency():
    req = ev(
        [
            field("amount_due", "55000.00", 0.94),
            field("date_due", PAST, 0.92),
            field("date_issue", "2026-09-01", 0.9),  # issued after its own due date
        ]
    )
    report = evaluate_guardrail(normalize_evidence(req))
    assert report.state == ConfidenceState.REVIEW
    assert any(c.check_id.startswith("G6") for c in report.checks)


def test_guardrail_empty_evidence_is_blocked():
    report = evaluate_guardrail(normalize_evidence(ev([])))
    assert report.state == ConfidenceState.BLOCKED


# ---------------------------------------------------------------------------
# Full decide() — recommendations, escalation, auditability
# ---------------------------------------------------------------------------
def test_decide_overdue_recommends_payment_reminder_without_escalation():
    d = decide(
        ev([field("amount_due", "55000.00", 0.94), field("date_due", PAST, 0.92)], "test-001")
    )
    assert isinstance(d, Decision)
    assert d.intent == Intent.OVERDUE_PAYMENT
    assert d.confidence_state == ConfidenceState.SAFE
    assert d.recommended_action == "payment_reminder"
    assert d.requires_human_review is False
    assert d.reason  # deterministic, non-empty explanation


def test_decide_upcoming_recommends_pre_due_reminder():
    d = decide(ev([field("amount_due", "18750.50", 0.9), field("date_due", FUTURE, 0.91)]))
    assert d.intent == Intent.UPCOMING_DUE
    assert d.recommended_action == "pre_due_reminder"
    assert d.requires_human_review is False


def test_decide_missing_due_date_requests_invoice_verification_and_escalates():
    d = decide(ev([field("amount_due", "22000", 0.93), field("date_issue", "2026-08-20", 0.88)]))
    assert d.intent == Intent.MISSING_DUE_DATE
    assert d.confidence_state == ConfidenceState.BLOCKED
    assert d.recommended_action == "invoice_verification_request"  # spec action map
    assert d.requires_human_review is True  # BLOCKED state always escalates


def test_decide_missing_amount_requests_document_review_and_escalates():
    d = decide(ev([field("date_due", FUTURE, 0.9), field("vendor_name", "Acme Supplies Ltd", 0.87)]))
    assert d.intent == Intent.MISSING_FINANCIAL_EVIDENCE
    assert d.confidence_state == ConfidenceState.BLOCKED
    assert d.recommended_action == "document_ocr_review_request"  # spec action map
    assert d.requires_human_review is True  # BLOCKED state always escalates


def test_decide_low_confidence_escalates_to_manual_review():
    d = decide(ev([field("amount_due", "9500", 0.64), field("date_due", FUTURE, 0.62)]))
    assert d.intent == Intent.MANUAL_REVIEW  # usable but below SAFE bar → human decides
    assert d.confidence_state == ConfidenceState.REVIEW
    assert d.recommended_action == "escalate_to_human"
    assert d.requires_human_review is True  # never automated without trust


def test_decide_conflicting_amounts_escalate_to_human():
    d = decide(
        ev(
            [
                field("amount_due", "54000.00", 0.91),
                field("amount_due", "55000.00", 0.88),
                field("date_due", PAST, 0.9),
            ]
        )
    )
    assert d.intent == Intent.MANUAL_REVIEW  # conflicting → human resolves
    assert d.confidence_state == ConfidenceState.BLOCKED
    assert d.recommended_action == "escalate_to_human"
    assert d.requires_human_review is True


def test_decide_blocked_never_attaches_specific_recovery_action():
    # Even a blatantly overdue conflicting document must not yield a reminder.
    d = decide(
        ev(
            [
                field("amount_due", "54000.00", 0.91),
                field("amount_due", "55000.00", 0.88),
                field("date_due", "2020-01-01", 0.9),
            ]
        )
    )
    assert d.recommended_action == "escalate_to_human"
    assert d.recommended_action != "payment_reminder"


def test_decision_is_auditable():
    d = decide(ev([field("amount_due", "55000.00", 0.94), field("date_due", PAST, 0.92)], "audit-doc"))
    assert d.timestamp  # ISO timestamp present
    assert d.document_id == "audit-doc"
    assert d.rules_fired, "routing rule ids must be recorded"
    assert any(c.check_id.startswith("G") for c in d.guardrail_checks)
    assert {rf.field for rf in d.evidence} >= {"amount_due", "date_due"}


def test_decide_is_deterministic_except_timestamp():
    req = ev([field("amount_due", "55000.00", 0.94), field("date_due", PAST, 0.92)], "doc-42")
    d1, d2 = decide(req), decide(req)
    assert d1.intent == d2.intent
    assert d1.confidence_state == d2.confidence_state
    assert d1.recommended_action == d2.recommended_action
    assert d1.reason == d2.reason
    assert d1.rules_fired == d2.rules_fired


# ---------------------------------------------------------------------------
# Fixture-driven end-to-end (all 6 scenarios through the same entry point)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", FIXTURE_FILES)
def test_fixture_end_to_end(path: Path):
    d = decide(fixture_request(path))
    name = path.name
    if name.startswith("01_"):
        assert (d.intent, d.confidence_state, d.recommended_action, d.requires_human_review) == (
            Intent.OVERDUE_PAYMENT, ConfidenceState.SAFE, "payment_reminder", False)
    elif name.startswith("02_"):
        assert (d.intent, d.confidence_state, d.recommended_action, d.requires_human_review) == (
            Intent.UPCOMING_DUE, ConfidenceState.SAFE, "pre_due_reminder", False)
    elif name.startswith("03_"):
        assert (d.intent, d.confidence_state, d.recommended_action, d.requires_human_review) == (
            Intent.MISSING_DUE_DATE, ConfidenceState.BLOCKED, "invoice_verification_request", True)
    elif name.startswith("04_"):
        assert (d.intent, d.confidence_state, d.recommended_action, d.requires_human_review) == (
            Intent.MISSING_FINANCIAL_EVIDENCE, ConfidenceState.BLOCKED, "document_ocr_review_request", True)
    elif name.startswith("05_"):
        assert (d.intent, d.confidence_state, d.recommended_action, d.requires_human_review) == (
            Intent.MANUAL_REVIEW, ConfidenceState.REVIEW, "escalate_to_human", True)
    elif name.startswith("06_"):
        assert (d.intent, d.confidence_state, d.recommended_action, d.requires_human_review) == (
            Intent.MANUAL_REVIEW, ConfidenceState.BLOCKED, "escalate_to_human", True)
    else:
        pytest.fail(f"unexpected fixture {name}")


# ---------------------------------------------------------------------------
# API contract — POST /reason
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from src.reasoning.app import app

    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/reason/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_reason_endpoint_returns_decision_json(client):
    r = client.post(
        "/reason",
        json={
            "document_id": "test-001",
            "as_of_date": "2026-09-01",
            "fields": [
                {
                    "field": "amount_due",
                    "value": "55000.00",
                    "value_normalized": 55000.0,
                    "source": "rtdetr",
                    "confidence": 0.94,
                    "bbox": [412.0, 96.0, 560.0, 118.0],
                },
                {
                    "field": "date_due",
                    "value": "2026-08-15",
                    "value_normalized": "2026-08-15",
                    "source": "rtdetr",
                    "confidence": 0.92,
                    "bbox": [380.0, 150.0, 500.0, 170.0],
                },
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["document_id"] == "test-001"
    assert body["intent"] == "overdue_payment"
    assert body["confidence_state"] == "SAFE"
    assert body["recommended_action"] == "payment_reminder"
    assert body["requires_human_review"] is False
    assert "Due date has passed" in body["reason"]
    assert body["timestamp"]
    assert isinstance(body["rules_fired"], list)
    assert isinstance(body["guardrail_checks"], list)


def test_reason_endpoint_low_confidence_marks_human_review(client):
    r = client.post(
        "/reason",
        json={
            "document_id": "test-005",
            "as_of_date": "2026-09-01",
            "fields": [
                {"field": "amount_due", "value": "9500", "confidence": 0.64},
                {"field": "date_due", "value": "2026-10-05", "confidence": 0.62},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["confidence_state"] == "REVIEW"
    assert body["intent"] == "manual_review"
    assert body["requires_human_review"] is True


def test_reason_endpoint_empty_evidence_blocked(client):
    r = client.post("/reason", json={"document_id": "empty-case", "fields": []})
    assert r.status_code == 200
    body = r.json()
    assert body["confidence_state"] == "BLOCKED"
    assert body["intent"] == "missing_financial_evidence"
    assert body["recommended_action"] == "document_ocr_review_request"
    assert body["requires_human_review"] is True


def test_reason_endpoint_rejects_unknown_fields(client):
    r = client.post(
        "/reason",
        json={"document_id": "x", "fields": [], "totally_unknown": 1},
    )
    assert r.status_code == 422


def test_reason_endpoint_rejects_invalid_bbox(client):
    r = client.post(
        "/reason",
        json={
            "document_id": "x",
            "fields": [
                {"field": "amount_due", "value": "1", "confidence": 0.9, "bbox": [3, 2, 1, 0]}
            ],
        },
    )
    assert r.status_code == 422


@pytest.mark.parametrize("path", FIXTURE_FILES)
def test_reason_endpoint_accepts_every_fixture(client, path: Path):
    r = client.post("/reason", json=load_fixture(path)["request"])
    assert r.status_code == 200
    assert "confidence_state" in r.json()
