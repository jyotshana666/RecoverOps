# Reasoning test fixtures — SYNTHETIC TEST DATA

**These are NOT real invoices.** Every file in this directory contains
hand-written synthetic values created solely to test the Part B reasoning
layer deterministically. No personal data, no customer documents, no real
amounts, no secrets.

## Files

| Fixture | Scenario                        | Expected intent               | Expected state |
|---------|---------------------------------|-------------------------------|----------------|
| `01_clear_overdue.json`        | Clear overdue invoice          | `overdue_payment`            | `SAFE`    |
| `02_upcoming_payment.json`     | Upcoming payment               | `upcoming_due`               | `SAFE`    |
| `03_missing_due_date.json`     | Missing due date               | `missing_due_date`           | `BLOCKED` |
| `04_missing_amount.json`       | Missing amount                 | `missing_financial_evidence` | `BLOCKED` |
| `05_low_confidence.json`       | Low-confidence detection       | `manual_review`              | `REVIEW`  |
| `06_conflicting_evidence.json` | Conflicting evidence (amounts) | `manual_review`              | `BLOCKED` |

Each file has the shape `{"_label", "description", "request"}` where `request`
is exactly the POST `/reason` body (`EvidenceRequest`). The `request` object is
what tests submit; `_label`/`description` are human documentation only.

All fixtures pin `as_of_date` so routing outcomes are deterministic forever.
