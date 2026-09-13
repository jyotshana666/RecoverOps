"""RecoverOps Part B — single source of truth for reasoning thresholds and rules.

WHY A SINGLE CONFIG MODULE
--------------------------
The Part B specification forbids scattering magic numbers through the
reasoning code. Every tunable value below is imported by name from
``evidence.py`` / ``guardrail.py`` / ``intent_router.py`` / ``recovery.py``.

IMPORTANT — THESE ARE INITIAL ENGINEERING THRESHOLDS
----------------------------------------------------
None of these values are empirically validated optima. They are deliberate,
conservative starting points chosen so the system fails towards human review
rather than towards automated financial action. They must be calibrated
against labelled validation data (e.g. the DOCILE validation split used in
Part A) before any production use.
"""

# ---------------------------------------------------------------------------
# Field contract (must stay aligned with Part A: src/data/config.py)
# ---------------------------------------------------------------------------
# Part A's RT-DETR model is trained on exactly these five region classes.
KNOWN_FIELDS: frozenset = frozenset(
    {
        "amount_due",
        "date_due",
        "document_id",
        "date_issue",
        "vendor_name",
    }
)

# Required for any financial decision. ``amount_due`` is the recoverable
# quantity; ``date_due`` is what makes it actionable (overdue vs upcoming).
# Without both, RecoverOps cannot characterise the receivable at all, so
# missing either must BLOCK the decision.
REQUIRED_FIELDS: tuple = ("amount_due", "date_due")

# vendor_name / date_issue / document_id detections are useful context but are
# not needed to route an intent, so their absence must not block the pipeline.
OPTIONAL_FIELDS: tuple = ("vendor_name", "date_issue", "document_id")

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------
# Per-field confidence at or above which evidence is considered trustworthy
# enough for an automated (but still non-binding) recommendation. Kept high on
# purpose: RT-DETR confidence measures *region localisation*, not value
# correctness, so the bar for trusting a value must be stricter than the bar
# for trusting a box.
SAFE_CONFIDENCE: float = 0.80

# Hard floor. Below this, a required field's evidence is treated as
# unreliable noise: critical evidence that cannot be trusted is handled like
# missing/contradictory evidence and BLOCKS the decision instead of merely
# flagging review. Exists so "confidently wrong" detections cannot reach a
# SAFE state through the normal review path.
REVIEW_CONFIDENCE: float = 0.60

# ---------------------------------------------------------------------------
# Date rules
# ---------------------------------------------------------------------------
# Grace days subtracted from the reference date before the overdue comparison.
# 0 = a due date strictly before the reference date is overdue. Configurable
# so a payment-grace-period policy can be adopted without code changes.
OVERDUE_GRACE_DAYS: int = 0

# ---------------------------------------------------------------------------
# Conflict rules
# ---------------------------------------------------------------------------
# Two amount_due candidates conflict when their relative difference exceeds
# this tolerance. 0.0 means amounts must match exactly after normalisation:
# for money, approximate equality is not acceptable until OCR provenance is
# good enough to justify fuzzier matching.
AMOUNT_CONFLICT_RELATIVE_TOLERANCE: float = 0.0
