# RecoverOps: Autonomous Invoice Detection & Financial Recovery Reasoning

RecoverOps provides an end-to-end, auditable architecture for invoice field localization (Part A) and deterministic financial recovery reasoning (Part B).

---

## 1. Architecture Overview

```
Invoice Image / PDF
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Part A: RT-DETR-L Field Detector                      │
│  - Classes: amount_due, date_due, document_id,         │
│             date_issue, vendor_name                    │
│  - Bounding box extraction & confidence scoring        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Detection → Evidence Adapter                          │
│  - Maps detection regions to EvidenceField schema      │
│  - Strictly honest: region-only (value=None) when OCR  │
│    is not present; never fabricates text values        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Part B: Deterministic Reasoning Layer                 │
│  - Financial safety guardrails (G1–G6)                 │
│  - Intent routing & bounded recommendations            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
Unified API Decision Response (JSON)
```

---

## 2. Local Workflow & Deployment

### Step 1: Train RT-DETR-L on Kaggle
Follow [docs/kaggle_training.md](file:///c:/Work/Projects/RecoverOps/docs/kaggle_training.md) using the Kaggle notebook:
`kaggle/rtdetr_baseline_training.ipynb`.

### Step 2: Download Trained Checkpoint
Download `best.pt` from Kaggle:
`/kaggle/working/RecoverOps/experiments/baseline/runs/rtdetr_l_baseline/weights/best.pt`

### Step 3: Place Weights Locally
Place the weights file at the default model location or configure `MODEL_PATH`:
```bash
# Default location:
cp /path/to/downloaded/best.pt models/best.pt

# Or export environment variable:
export MODEL_PATH="c:/Work/Projects/RecoverOps/models/best.pt"
```

### Step 4: Start the Unified FastAPI Backend
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 3. Testing API Endpoints

### 1. Health Probe
```bash
curl http://localhost:8000/health
```

### 2. Pure Reasoning Endpoint (`POST /reason`)
```bash
curl -X POST http://localhost:8000/reason \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "INV-2026-001",
    "as_of_date": "2026-09-01",
    "fields": [
      {
        "field": "amount_due",
        "value": "1500.00",
        "source": "ocr",
        "confidence": 0.95,
        "bbox": [10.0, 10.0, 100.0, 50.0]
      },
      {
        "field": "date_due",
        "value": "2026-08-15",
        "source": "ocr",
        "confidence": 0.92,
        "bbox": [10.0, 60.0, 100.0, 100.0]
      }
    ]
  }'
```

### 3. Field Detection Endpoint (`POST /detect`)
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@sample_invoice.png" \
  -F "document_id=INV_SAMPLE_01"
```

### 4. End-to-End Processing Endpoint (`POST /process`)
```bash
curl -X POST http://localhost:8000/process \
  -F "file=@sample_invoice.png" \
  -F "document_id=INV_SAMPLE_01" \
  -F "as_of_date=2026-09-15"
```

---

## 4. OCR / Text Extraction Status & Constraints
* **Current State**: RT-DETR-L identifies the *spatial regions* of invoice fields.
* **Integrity Guardrail**: Detection bounding boxes alone carry `value=None`. The system strictly avoids fabricating invoice numbers or amounts from coordinates.
* When OCR is not connected, region-only detections are flagged `region_only=True` by the reasoning layer, triggering safety guardrails (`BLOCKED` / `REVIEW` requiring human confirmation) rather than initiating automated financial actions.
