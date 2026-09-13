# RecoverOps: Autonomous Invoice Detection & Financial Recovery Reasoning

RecoverOps is an end-to-end, auditable platform integrating **RT-DETR-L field region detection** (Part A) with **deterministic financial recovery reasoning and guardrails** (Part B).

---

## 1. Production Architecture

```
                      [ Client / Browser ]
                               │
                               ▼ HTTPS
                 ┌───────────────────────────┐
                 │   Vercel React Frontend   │
                 │    (Vite / HTML5 / CSS)   │
                 └─────────────┬─────────────┘
                               │
                               ▼ HTTPS (REST API)
                 ┌───────────────────────────┐
                 │   Render FastAPI Backend  │
                 │   (Docker / Python 3.12)  │
                 └─────────────┬─────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌──────────────────────────┐            ┌──────────────────────────┐
│  Part A: RT-DETR-L       │            │  Part B: Deterministic   │
│  Field Region Detector   │            │  Financial Reasoning     │
│  - amount_due (0)        │            │  - Safety Guardrails     │
│  - date_due (1)          │            │  - Intent Routing        │
│  - document_id (2)       │            │  - Bounded Decisions     │
│  - date_issue (3)        │            └────────────▲─────────────┘
│  - vendor_name (4)       │                         │
└───────────┬──────────────┘                         │
            │                                        │
            ▼                                        │
┌──────────────────────────┐                         │
│ Detection→Evidence       ├─────────────────────────┘
│ Adapter (src/pipeline)   │
│ - Strict honesty         │
│ - value=None (no OCR)    │
└──────────────────────────┘
```

---

## 2. API Endpoints

| Method | Endpoint | Payload / Format | Description |
|---|---|---|---|
| `GET` | `/health` | None | Unified system liveness and model loading status (fast, non-blocking). |
| `GET` | `/reason/health` | None | Liveness probe for the reasoning layer (Part B backward compatible). |
| `POST` | `/reason` | `EvidenceRequest` JSON | Pure deterministic financial reasoning evaluation (fully mockable). |
| `POST` | `/detect` | Multipart Form (`file`, `document_id`) | RT-DETR-L field region detection returning bounding boxes and scores. |
| `POST` | `/process` | Multipart Form (`file`, `document_id`, `as_of_date`) | Full pipeline: Image → Detection → Adapter → Reasoning Decision. |

---

## 3. Local Development & Execution

### Option A: Local Native Execution
1. **Start Backend**:
   ```bash
   uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
   ```
2. **Start Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser.

### Option B: Local Docker Container
1. **Build Container**:
   ```bash
   docker build -t recoverops-backend:latest .
   ```
2. **Run Container**:
   ```bash
   docker run -p 8000:8000 \
     -e PORT=8000 \
     -e MODEL_PATH=/app/models/best.pt \
     -e FRONTEND_URL=http://localhost:5173 \
     recoverops-backend:latest
   ```
3. **Verify Container Health**:
   ```bash
   curl http://localhost:8000/health
   ```

---

## 4. Render Deployment (Backend)

The FastAPI backend is configured for automated Docker deployment on Render using [render.yaml](file:///c:/Work/Projects/RecoverOps/render.yaml).

### Required Environment Variables on Render:
- `PORT`: Supplied dynamically by Render (e.g. `10000` or `8000`).
- `MODEL_PATH`: Location of the trained RT-DETR weights (e.g. `/app/models/best.pt` or persistent disk `/data/models/best.pt`).
- `FRONTEND_URL`: URL of your deployed Vercel frontend (e.g. `https://recoverops.vercel.app`), enabling CORS.

### Health Check:
- Configure Render health check path to `/health`.

---

## 5. Vercel Deployment (Frontend)

The frontend is located in `frontend/` and configured for Vercel deployment with [frontend/vercel.json](file:///c:/Work/Projects/RecoverOps/frontend/vercel.json).

### Vercel Project Settings:
- **Root Directory**: `frontend`
- **Framework Preset**: `Vite`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

### Required Environment Variables on Vercel:
- `VITE_API_URL`: Public HTTPS URL of your Render backend:
  ```text
  VITE_API_URL=https://recoverops-backend.onrender.com
  ```

---

## 6. Model Weight Strategy (`MODEL_PATH`)

1. Train RT-DETR-L using the Kaggle notebook: `kaggle/recoverops.ipynb`.
2. Download `best.pt` from Kaggle output:
   `/kaggle/working/RecoverOps/experiments/baseline/runs/rtdetr_l_baseline/weights/best.pt`
3. Place `best.pt` at `models/best.pt` (or set `MODEL_PATH=/path/to/best.pt`).
4. **Graceful Standby**: If weights are not mounted or not found at startup, the backend starts in standby mode:
   - `GET /health` reports `model_loaded: false`.
   - `POST /reason` remains 100% functional.
   - `POST /detect` and `POST /process` return clear HTTP 503 explaining how to supply weights.

---

## 7. Important Constraint: Field Localization vs. OCR

- **Current State**: Part A RT-DETR-L extracts spatial field *regions* (bounding boxes and class probabilities).
- **Zero Fabrication**: In accordance with the Part B specification, detection bounding boxes strictly carry `value=None` rather than inventing text values from coordinates.
- **Safety Guardrail Behavior**: Missing textual evidence causes Part B guardrails to safely flag `requires_human_review=True` with confidence state `BLOCKED` or `REVIEW`. Automated financial actions are never executed without verified text transcription.
