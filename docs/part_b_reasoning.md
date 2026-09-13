# RecoverOps — Part B: Minimal Reasoning Layer

Part B is a deterministic, auditable reasoning layer that consumes structured
financial evidence (produced by Part A's RT-DETR detection pipeline) and emits
**bounded recovery recommendations**. It never executes financial transactions.

```
Part A:  image → RT-DETR → detections (class + bbox + confidence)
Part B:  detections → evidence normalization → invoice state → intent routing
         → confidence guardrail → recovery recommendation
         → human escalation when required
```

**No LLM is used.** Every decision comes from an explicit, documented rule
table, making the output reproducible and auditable.

## Module map

| Module | Responsibility |
|---|---|
| `src/reasoning/schemas.py` | Strict pydantic models: evidence, invoice state, guardrail report, decision |
| `src/reasoning/config.py` | Single source of truth for thresholds/required fields (documented) |
| `src/reasoning/evidence.py` | Normalization: grouping, amount/date parsing, conflict marking |
| `src/reasoning/intent_router.py` | Deterministic rule table R1–R7 → one of five intents |
| `src/reasoning/guardrail.py` | Auditable checks G1–G6 → `SAFE` / `REVIEW` / `BLOCKED` |
| `src/reasoning/recovery.py` | `decide()`: composes the above into an auditable `Decision` |
| `src/reasoning/app.py` | Minimal FastAPI surface: `POST /reason`, `GET /reason/health` |

## 1. Evidence model

RT-DETR detects **field regions** (class + bbox + confidence). It does **not**
read text. The schema is therefore built so an OCR/text-extraction step can be
connected later:

```json
{
  "field": "amount_due",
  "value": "55000.00",          // raw text; null = region-only detection
  "value_normalized": 55000.0,  // parsed float / ISO date (set by evidence.py)
  "confidence": 0.94,           // detection (or OCR) confidence
  "bbox": [412, 96, 560, 118],  // RT-DETR region, [x1, y1, x2, y2]
  "source": "rtdetr"            // rtdetr | ocr | manual | external
}
```

* A region-only detection (`value: null`) is recorded as `present` with
  `region_only: true` — it can never satisfy a value requirement, and values
  are **never fabricated from bounding boxes**.
* Multiple entries per field are preserved. Disagreeing parsed values are
  marked `conflicting` and are never silently resolved.
* Field names must match Part A's class mapping (`src/data/config.py`):
  `amount_due`, `date_due`, `document_id`, `date_issue`, `vendor_name`.
* Unparseable text (e.g. `"SEE ATTACHED"`) is recorded as a `parse_failure`
  and treated as absent for routing — never guessed.

## 2. Intent routing (`intent_router.py`)

Rules are evaluated top-down; the first match wins. All rules are documented
in the module docstring and fired rule IDs are recorded in every decision
(`rules_fired`).

| Rule | Condition | Intent |
|---|---|---|
| R1 | Any field has conflicting parsed values | `manual_review` |
| R2 | `amount_due` has no usable value (absent / region-only / unparseable / below trust floor) | `missing_financial_evidence` |
| R3 | `amount_due` usable, `date_due` unusable | `missing_due_date` |
| R4 | Both usable, but a required field is below `SAFE_CONFIDENCE` | `manual_review` |
| R5 | `date_due < as_of − OVERDUE_GRACE_DAYS` and amount usable | `overdue_payment` |
| R6 | `date_due ≥ as_of − OVERDUE_GRACE_DAYS` and amount usable | `upcoming_due` |
| R7 | Fallback (defensive; unreachable today) | `manual_review` |

**Intent vs. confidence state:** intent describes the document's situation;
the guardrail state describes how much the evidence can be trusted. They are
combined only at the end (see §4).

## 3. Confidence guardrail (`guardrail.py`)

The state is computed from auditable checks — never from a single raw model
confidence. Every check outcome is returned in the decision
(`guardrail_checks`).

| Check | Kind | Meaning |
|---|---|---|
| G1 required-fields-present | hard | Required fields have a parsed value or extraction-backed detection (region-only does not count) |
| G2 required-fields-trusted | hard | Best confidence per required field ≥ `REVIEW_CONFIDENCE` (0.60) |
| G3 required-fields-confident | soft | Best confidence per required field ≥ `SAFE_CONFIDENCE` (0.80) |
| G4 no-conflicting-evidence | hard | No field has disagreeing parsed values |
| G5 no-unparseable-values | soft | Extracted text parsed cleanly |
| G6 date-consistency | soft | `date_issue ≤ date_due` when both are available |

**State machine:**

* `SAFE` — all hard checks pass, all soft checks pass → automated (non-binding) recommendation.
* `REVIEW` — hard checks pass, at least one soft check fails → recommendation + human confirmation.
* `BLOCKED` — any hard check fails → no specific recovery action may be trusted.

## 4. Recovery recommendation (`recovery.py`)

`decide(request)` composes: `normalize_evidence → evaluate_guardrail →
route_intent → recommendation → Decision`.

| Intent | Recommended action |
|---|---|
| `overdue_payment` | `payment_reminder` |
| `upcoming_due` | `pre_due_reminder` |
| `missing_due_date` | `invoice_verification_request` |
| `missing_financial_evidence` | `document_ocr_review_request` |
| `manual_review` | `escalate_to_human` |

`requires_human_review` is **true** whenever the state is not `SAFE` **or**
the router fired a manual-review rule. So:

* a `REVIEW`-state overdue invoice still gets `payment_reminder`, but it is explicitly flagged for human confirmation;
* a `BLOCKED` state never blocks the *diagnosis* (the intent is still reported) but always escalates.

## 5. Human escalation

Escalation is forced by construction, not by exception handling:

* intent `manual_review` (conflicting evidence R1, low confidence R4, fallback R7);
* guardrail state `REVIEW` or `BLOCKED` (`requires_human_review = true`).

## 6. API contract (`app.py`)

`POST /reason` — structured evidence in, decision out. Always HTTP 200 with a
full decision: evidence-quality problems are encoded *in the decision*
(`BLOCKED` / `manual_review`) because "this document cannot be trusted" **is**
the decision. Malformed requests (unknown keys, invalid bbox, bad enums) are
rejected with 422. `GET /reason/health` for liveness.

```jsonc
// POST /reason
{
  "document_id": "test-001",
  "as_of_date": "2026-09-01",   // optional; pins "today" for reproducibility
  "fields": [ { "field": "amount_due", "value": "55000.00",
                "confidence": 0.94, "bbox": [412,96,560,118] } ]
}

// 200 response
{
  "timestamp": "2026-09-13T12:34:56.789+00:00",
  "document_id": "test-001",
  "intent": "overdue_payment",
  "confidence_state": "SAFE",
  "recommended_action": "payment_reminder",
  "requires_human_review": false,
  "reason": "Due date has passed (as_of 2026-09-01) and amount due evidence is available.",
  "rules_fired": ["R5-past-due"],
  "guardrail_checks": [ { "check_id": "G1-required-fields-present", "passed": true, "detail": "..." } ],
  "evidence": [ ... resolved per-field view ... ]
}
```

**Decoupling:** `app.py` imports nothing from the RT-DETR/`/detect`
implementation. `/reason` accepts plain structured JSON and can be tested
entirely with mock evidence (`tests/fixtures/reasoning/`). No state is
mutated; no secrets or extra PII are stored — the in-process decision counter
is the only runtime bookkeeping.

## 7. Configuration (`config.py`)

All thresholds live in one place, each with a rationale:

| Setting | Value | Why |
|---|---|---|
| `SAFE_CONFIDENCE` | 0.80 | Trust bar for automated (non-binding) recommendations. Deliberately stricter than typical detection operating points because detection confidence measures *localisation*, not value correctness. |
| `REVIEW_CONFIDENCE` | 0.60 | Hard trust floor; below it evidence is treated as noise and BLOCKS the decision, so "confidently wrong" regions cannot reach SAFE. |
| `REQUIRED_FIELDS` | `amount_due`, `date_due` | Without a recoverable amount and a due date, no financial decision can be characterised. |
| `OVERDUE_GRACE_DAYS` | 0 | Configurable grace period; 0 = strictly past-due. |
| `AMOUNT_CONFLICT_RELATIVE_TOLERANCE` | 0.0 | Money must match exactly while OCR provenance is unproven. |

**These are initial engineering thresholds, not empirically validated
optima.** They are chosen to fail towards human review. Calibrate them
against labelled validation data (e.g. the DOCILE validation split) before
any production use.

## 8. Test scenarios and execution

Six deterministic fixtures in `tests/fixtures/reasoning/` (**synthetic test
data — NOT real invoices**), all pinned to `as_of_date` for reproducibility:

| # | Scenario | Intent | State | Action | Human review |
|---|---|---|---|---|---|
| 01 | Clear overdue invoice | `overdue_payment` | `SAFE` | `payment_reminder` | no |
| 02 | Upcoming payment | `upcoming_due` | `SAFE` | `pre_due_reminder` | no |
| 03 | Missing due date | `missing_due_date` | `BLOCKED` | `invoice_verification_request` | yes |
| 04 | Missing amount | `missing_financial_evidence` | `BLOCKED` | `document_ocr_review_request` | yes |
| 05 | Low-confidence detection | `manual_review` | `REVIEW` | `escalate_to_human` | yes |
| 06 | Conflicting evidence (amounts) | `manual_review` | `BLOCKED` | `escalate_to_human` | yes |

Run:

```bash
pytest
```

`tests/test_reasoning.py` covers: intent routing (all five intents, ordered
rules, grace days, due-today boundary), missing/region-only/unparseable
evidence, amount & date parsing, conflict marking, guardrail states (SAFE /
REVIEW / BLOCKED per check), human escalation, recommendation generation,
decision auditability, determinism, fixture-driven end-to-end, and the HTTP
contract via `TestClient` (including 422 rejection paths). **Current status:
58 passed.**

## 9. Limitations

1. **RT-DETR does not extract values.** It detects field regions. Until an
   OCR/text-extraction step is connected, `value` is `null` and every
   document routes to `missing_financial_evidence`. This is by design: the
   reasoning layer never fabricates values from bounding boxes.
2. **No persistence/queueing.** `POST /reason` is stateless; storing
   decisions and driving outbound workflows is out of scope for the Part B MVP.
3. **Unvalidated thresholds.** `SAFE_CONFIDENCE`/`REVIEW_CONFIDENCE` are
   engineering starting points; calibrate against labelled data.
4. **Date parsing is format-list based** (ISO, DD/MM/YYYY, DD.MM.YYYY —
   day-first by convention). Other locale formats fail safe (→ parse failure,
   REVIEW), they are not guessed.
5. **English/currency symbol set only** in amount parsing; anything else
   fails safe.
6. **No authentication/rate limiting** on `/reason` in this MVP; it performs
   no sensitive operations, but an API layer should be added before exposure.
7. **Single-currency assumption** in normalization; currency symbols are
   stripped and not compared across fields.

## Financial safety

RecoverOps does **not** initiate payments, modify bank information, transfer
funds, send legally binding communications, or make irreversible financial
decisions. The reasoning layer emits bounded recommendations only; every
uncertain outcome is routed to human review. There are no endpoints and no
code paths that move money.
