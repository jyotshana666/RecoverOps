# Local Integration & End-to-End Execution Guide

This document details the local integration between Part A (RT-DETR-L field detector) and Part B (deterministic financial recovery reasoning).

---

## 1. Pipeline Architecture

```
                       [ Input Invoice (Image / PDF) ]
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    InvoiceDetector (Part A)   │
                      │       (RT-DETR-L Model)       │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │        DetectionResult        │
                      │  - class_id, class_name       │
                      │  - confidence, bounding_box   │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │   Detection→Evidence Adapter  │
                      │     (src/pipeline/adapter)    │
                      │  - Maps to EvidenceField      │
                      │  - Sets value=None (no OCR)   │
                      │  - Sets source=RTDETR         │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │        EvidenceRequest        │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │      decide() (Part B)        │
                      │  - Evidence Normalization     │
                      │  - Confidence Guardrails      │
                      │  - Intent Routing             │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      [ Final ProcessResponse / Decision ]
```

---

## 2. Model Weight Management & Environment Variables

| Variable | Default Value | Description |
|---|---|---|
| `MODEL_PATH` | `models/best.pt` | Path to the trained RT-DETR-L PyTorch checkpoint (`.pt` file). |

### Where to Place `best.pt`:
- After completing training on Kaggle, download `best.pt` and place it at:
  `models/best.pt`
- Alternatively, specify any custom path via:
  ```bash
  export MODEL_PATH="/path/to/your/rtdetr_l_best.pt"
  ```

---

## 3. Running and Testing the Backend

### Run Test Suite:
```bash
pytest
```
Expected output: 83 passed tests covering reasoning, detection, pipeline adaptation, and API endpoints.

### Start Local Server:
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```

### Endpoints:
- `GET  /health` — Unified health probe showing model loading state.
- `GET  /reason/health` — Part B reasoning health probe.
- `POST /reason` — Pure structured JSON reasoning endpoint (mockable without model weights).
- `POST /detect` — Multipart form upload for image/PDF field region detection.
- `POST /process` — Multipart form upload for end-to-end detection → reasoning pipeline.

---

## 4. OCR / Text Extraction Limitation Note

- **Current State**: Part A extracts spatial bounding boxes and class probabilities (`amount_due`, `date_due`, `document_id`, `date_issue`, `vendor_name`).
- **Honest Evidence Contract**: Because an optical character recognition (OCR) text-transcription module is not yet connected, the adapter strictly sets `value=None` for detected bounding boxes.
- **Guardrail Behavior**: When `value=None`, Part B's confidence guardrail identifies the fields as `region_only=True`. In the absence of parsed amounts and dates, the guardrail triggers `requires_human_review=True` with state `BLOCKED` or `REVIEW`. This satisfies strict financial safety requirements by preventing ungrounded automated recovery actions.
