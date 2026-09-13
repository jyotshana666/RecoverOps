"""RecoverOps Part B — evidence normalization.

Converts a raw list of ``EvidenceField`` detections into an
``NormalizedEvidence`` invoice state. This module is the ONLY place where raw
text values are parsed into normalized values (float amounts / ISO dates).

Parsing is deliberately conservative:

* A value that cannot be parsed is recorded as a ``parse_failure`` and is
  treated as *absent* for routing — never guessed from the bbox.
* Regions with no OCR text yet (``value=None``) are ``present`` but flagged
  ``region_only``; they cannot satisfy the value requirement of any intent.
* Multiple entries for the same field are preserved. Disagreeing parsed
  values are marked ``conflicting``; the guardrail handles them. Nothing is
  silently resolved.
"""
from __future__ import annotations

from datetime import date, timezone, datetime
from typing import Dict, List, Optional

from src.reasoning import config
from src.reasoning.schemas import (
    EvidenceField,
    EvidenceRequest,
    FieldName,
    NormalizedEvidence,
    ResolvedField,
)


def _norm_text(raw: str) -> str:
    """Whitespace-normalize a raw OCR/extraction string."""
    return " ".join(raw.split())


def parse_amount(raw: str) -> Optional[float]:
    """Parse an amount string into a float.

    Accepts common invoice formats and thousand separators; currency symbols
    and letters (e.g. ``Rs.``/``INR``/``USD``) are stripped from the
    boundaries. If anything but digits/separators remains, parsing fails.
    """
    cleaned = _norm_text(raw).upper()
    for symbol in ("INR", "USD", "RS.", "RS", "$", "€", "£", "₹"):
        cleaned = cleaned.replace(symbol, "")
    cleaned = cleaned.strip()

    if not cleaned or cleaned.startswith("."):
        return None
    if not all(ch.isdigit() or ch in ",. " for ch in cleaned):
        return None

    # Normalise separators: spaces are always grouping; if both '.' and ','
    # appear, the last one is the decimal point (Indian/US convention).
    cleaned = cleaned.replace(" ", "")
    if "." in cleaned and "," in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        head, _, tail = cleaned.partition(",")
        if len(tail) in (1, 2) and head:  # 1,00 / 1,000 → thousands, 1,00 → decimal
            cleaned = cleaned.replace(",", "") if len(tail) == 3 else cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")

    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y")


def parse_date(raw: str) -> Optional[str]:
    """Parse a date string and return an ISO ``YYYY-MM-DD`` string.

    NOTE: ``DD/MM/YYYY`` and ``DD.MM.YY`` are parsed day-first (common for
    Indian invoices); ``YYYY-MM-DD`` is unambiguous. Unparseable input
    returns ``None`` — never a fabricated date.
    """
    text = _norm_text(raw)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Entry-level helpers (exported for the guardrail + tests)
# ---------------------------------------------------------------------------
def is_region_only(entry: EvidenceField) -> bool:
    """True when a detection has no extracted text (RT-DETR region only)."""
    return entry.value is None


def entry_normalized(entry: EvidenceField) -> Optional[float]:
    """Normalized float for amount entries (None when absent/unparseable)."""
    if entry.value is None:
        return None
    if isinstance(entry.value_normalized, (int, float)) and not isinstance(
        entry.value_normalized, bool
    ):
        return float(entry.value_normalized)
    return parse_amount(entry.value)


def entry_date(entry: EvidenceField) -> Optional[str]:
    """Normalized ISO date string for date entries (None when absent/unparseable)."""
    if entry.value is None:
        return None
    if isinstance(entry.value_normalized, str) and entry.value_normalized:
        return entry.value_normalized
    return parse_date(entry.value)


def amounts_agree(a: float, b: float) -> bool:
    """True when two amount values agree within the configured tolerance."""
    tol = config.AMOUNT_CONFLICT_RELATIVE_TOLERANCE
    if tol <= 0.0:
        return a == b
    base = max(abs(a), abs(b))
    return abs(a - b) <= tol * base


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _resolve_field(
    name: FieldName, entries: List[EvidenceField]
) -> ResolvedField:
    rf = ResolvedField(field=name, entries=entries, best_confidence=0.0)

    if not entries:
        return rf

    rf.present = True
    rf.best_confidence = max(e.confidence for e in entries)

    if name == FieldName.AMOUNT_DUE:
        values: List[float] = []
        for e in entries:
            parsed = entry_normalized(e)
            if parsed is None:
                if is_region_only(e):
                    rf.region_only = True
                else:
                    rf.parse_failures.append(str(e.value))
            else:
                values.append(parsed)
        rf.resolved_value = values[0] if values else None
        rf.conflicting = len(
            {v for v in values if not amounts_agree(v, values[0])}
        ) > 0 or len({v for v in values}) > 1 and not all(
            amounts_agree(values[0], v) for v in values
        )
        if rf.conflicting:
            rf.resolved_value = None

    elif name in (FieldName.DATE_DUE, FieldName.DATE_ISSUE):
        values_s: List[str] = []
        for e in entries:
            parsed = entry_date(e)
            if parsed is None:
                if is_region_only(e):
                    rf.region_only = True
                else:
                    rf.parse_failures.append(str(e.value))
            else:
                values_s.append(parsed)
        rf.resolved_value = values_s[0] if values_s else None
        rf.conflicting = len(set(values_s)) > 1
        if rf.conflicting:
            rf.resolved_value = None

    else:  # document_id, vendor_name — free text, no numeric parsing
        texts = {_norm_text(e.value) for e in entries if e.value is not None}
        rf.resolved_value = next(iter(texts)) if len(texts) == 1 else None
        rf.conflicting = len(texts) > 1

    return rf


def normalize_evidence(
    request: EvidenceRequest, as_of: Optional[date] = None
) -> NormalizedEvidence:
    """Group raw evidence per field and compute the invoice state.

    ``as_of`` pins the reference date for reproducibility; defaults to the
    current UTC date.
    """
    grouped: Dict[str, List[EvidenceField]] = {}
    for e in request.fields:
        if e.field.value in config.KNOWN_FIELDS:
            grouped.setdefault(e.field.value, []).append(e)

    fields: Dict[str, ResolvedField] = {}
    for name in config.KNOWN_FIELDS:
        fields[name] = _resolve_field(FieldName(name), grouped.get(name, []))

    reference = as_of or request.as_of_date
    if reference is None:
        reference = datetime.now(timezone.utc).date()

    return NormalizedEvidence(
        document_id=request.document_id, as_of_date=reference, fields=fields
    )
