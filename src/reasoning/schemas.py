"""RecoverOps Part B — strict structured schemas for evidence and decisions.

DESIGN NOTE (RT-DETR vs. text extraction)
-----------------------------------------
RT-DETR provides *spatial detection of field regions*: class + bbox + confidence.
It does NOT read the text inside the region. Therefore:

* ``EvidenceField.bbox``             — where the field is on the page (RT-DETR).
* ``EvidenceField.confidence``       — detection confidence (or OCR confidence
                                       when ``source`` is OCR).
* ``EvidenceField.value``            — raw text, populated by an OCR /
                                       text-extraction step connected later.
* ``EvidenceField.value_normalized`` — parsed value (float for amounts, ISO
                                       date string for dates), produced by
                                       ``evidence.normalize_evidence``.

A region-only detection carries ``value=None``. Values are NEVER fabricated
from bounding boxes. Strict types (``StrictStr``) prevent numeric strings from
silently masquerading as parsed values.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class FieldName(str, Enum):
    """Invoice field classes — must match Part A ``src/data/config.py``."""

    AMOUNT_DUE = "amount_due"
    DATE_DUE = "date_due"
    DOCUMENT_ID = "document_id"
    DATE_ISSUE = "date_issue"
    VENDOR_NAME = "vendor_name"


class SourceType(str, Enum):
    """Where an evidence entry came from. RT-DETR alone yields regions."""

    RTDETR = "rtdetr"
    OCR = "ocr"
    MANUAL = "manual"
    EXTERNAL = "external"


class ConfidenceState(str, Enum):
    """Guardrail states (see guardrail.py for the state machine)."""

    SAFE = "SAFE"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"


class Intent(str, Enum):
    """Supported routing intents (see intent_router.py for the rules)."""

    OVERDUE_PAYMENT = "overdue_payment"
    UPCOMING_DUE = "upcoming_due"
    MISSING_DUE_DATE = "missing_due_date"
    MISSING_FINANCIAL_EVIDENCE = "missing_financial_evidence"
    MANUAL_REVIEW = "manual_review"


class EvidenceField(BaseModel):
    """One detected/extracted occurrence of an invoice field.

    Multiple entries with the same ``field`` are allowed: disagreeing parsed
    values are a contradiction handled by the router/guardrail, never
    silently resolved.
    """

    model_config = ConfigDict(extra="forbid")

    field: FieldName
    value: Optional[StrictStr] = None
    value_normalized: Optional[Union[StrictStr, float]] = None
    source: SourceType = SourceType.RTDETR
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: Optional[List[float]] = None

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (float(c) for c in v)
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox must satisfy x2 >= x1 and y2 >= y1")
        return [x1, y1, x2, y2]


class EvidenceRequest(BaseModel):
    """Input contract for POST /reason (and for the pure ``decide()`` API).

    ``as_of_date`` pins the reference "today" so decisions are reproducible;
    when omitted, the current UTC date is used at runtime.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=128)
    fields: List[EvidenceField] = Field(default_factory=list)
    as_of_date: Optional[date] = None


class ResolvedField(BaseModel):
    """Normalized per-field view computed by ``evidence.normalize_evidence``."""

    model_config = ConfigDict(extra="forbid")

    field: FieldName
    present: bool = False
    conflicting: bool = False
    resolved_value: Optional[Union[StrictStr, float]] = None
    best_confidence: float = 0.0
    region_only: bool = False
    entries: List[EvidenceField] = Field(default_factory=list)
    parse_failures: List[str] = Field(default_factory=list)


class NormalizedEvidence(BaseModel):
    """Invoice state after evidence normalization."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    as_of_date: date
    fields: Dict[str, ResolvedField] = Field(default_factory=dict)

    def field(self, name: str) -> Optional[ResolvedField]:
        return self.fields.get(name)


class GuardrailCheck(BaseModel):
    """One auditable guardrail evaluation (check id + outcome + detail)."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    passed: bool
    detail: str = ""


class GuardrailReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: ConfidenceState
    checks: List[GuardrailCheck] = Field(default_factory=list)
    blocking_reasons: List[str] = Field(default_factory=list)
    review_reasons: List[str] = Field(default_factory=list)


class Decision(BaseModel):
    """Auditable decision object returned by ``decide()`` and POST /reason.

    Contains no secrets and no PII beyond the invoice fields the caller
    already supplied as evidence.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    document_id: str
    intent: Intent
    confidence_state: ConfidenceState
    recommended_action: str
    reason: str
    requires_human_review: bool
    rules_fired: List[str] = Field(default_factory=list)
    guardrail_checks: List[GuardrailCheck] = Field(default_factory=list)
    evidence: List[ResolvedField] = Field(default_factory=list)
