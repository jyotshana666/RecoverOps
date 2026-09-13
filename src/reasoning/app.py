"""RecoverOps Unified FastAPI Backend.

Provides endpoints for:
1. Pure Reasoning API (Part B contract):
   - GET  /reason/health — liveness probe for reasoning layer
   - POST /reason        — deterministic recovery decision from structured EvidenceRequest
2. Detection API (Part A contract):
   - POST /detect        — image/PDF in, structured RT-DETR detections out
3. End-to-End Pipeline:
   - POST /process       — image/PDF in, RT-DETR detection → evidence adapter → reasoning decision out
4. System Health:
   - GET  /health         — overall system liveness and model availability

DEC-B1..B4 are the financial-safety binding decisions on this surface:
the API only ever returns bounded recommendations; it exposes no payment,
fund-transfer, bank-detail, or communications endpoints, and never mutates state.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.detection.config import get_model_path
from src.detection.detector import InvoiceDetector
from src.detection.schemas import Detection, DetectionResult
from src.pipeline.adapter import (
    detection_to_evidence_request,
    process_invoice_document,
)
from src.reasoning.recovery import decide
from src.reasoning.schemas import Decision, EvidenceRequest

app = FastAPI(
    title="RecoverOps — Unified Invoice Detection & Recovery Reasoning API",
    version="0.2.0",
    description=(
        "Deterministic, auditable invoice field detection (RT-DETR-L) and "
        "recovery reasoning layer. Produces bounded recommendations only — "
        "never executes financial transactions."
    ),
)

# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

frontend_env = os.getenv("FRONTEND_URL", "").strip()
if frontend_env:
    if frontend_env == "*":
        allowed_origins = ["*"]
    else:
        configured_origins = [orig.strip() for orig in frontend_env.split(",") if orig.strip()]
        allowed_origins = list(set(default_origins + configured_origins))
else:
    allowed_origins = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_DECISIONS_RECORDED = 0  # in-process audit counter (no persistence by design)

# Singleton lazy-loaded detector instance
_detector_instance: Optional[InvoiceDetector] = None

ALLOWED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
    ".webp",
    ".pdf",
}


def get_detector() -> InvoiceDetector:
    """Retrieve or initialize the InvoiceDetector singleton (lazy loaded)."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = InvoiceDetector(lazy_load=True)
    return _detector_instance


def set_detector(detector: Optional[InvoiceDetector]) -> None:
    """Setter for dependency injection and unit testing."""
    global _detector_instance
    _detector_instance = detector


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    version: str
    model_path: Optional[str] = None
    model_loaded: bool = False


class ProcessResponse(BaseModel):
    """Complete end-to-end response containing detections, evidence, and decision."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    decision: Decision
    detections: List[Detection] = Field(default_factory=list)
    evidence_request: EvidenceRequest


# ---------------------------------------------------------------------------
# Health Probes
# ---------------------------------------------------------------------------
@app.get("/reason/health", response_model=HealthResponse)
def reason_health() -> HealthResponse:
    """Liveness probe for the reasoning layer (Part B backward compatibility)."""
    return HealthResponse(status="ok", service="reasoning", version=app.version)


@app.get("/health", response_model=HealthResponse)
def unified_health() -> HealthResponse:
    """Liveness probe for the unified RecoverOps service."""
    model_path = get_model_path()
    model_exists = model_path.exists()
    return HealthResponse(
        status="ok",
        service="recoverops-unified",
        version=app.version,
        model_path=str(model_path),
        model_loaded=model_exists,
    )


# ---------------------------------------------------------------------------
# Part B: Direct Reasoning Endpoint (Preserved Exactly)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Part A: Field Detection Endpoint
# ---------------------------------------------------------------------------
@app.post("/detect", response_model=DetectionResult)
async def detect(
    file: UploadFile = File(..., description="Invoice image (.png, .jpg, etc.) or PDF file"),
    document_id: Optional[str] = Form(None, description="Optional document identifier"),
    conf_threshold: Optional[float] = Form(None, description="Optional detection confidence threshold override"),
) -> DetectionResult:
    """Run RT-DETR-L field region detection on an uploaded invoice file."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed types: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    doc_id = document_id if document_id else Path(file.filename).stem

    detector = get_detector()
    try:
        detection_result = detector.predict(
            image=file_bytes,
            document_id=doc_id,
            conf_threshold=conf_threshold,
        )
        return detection_result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RT-DETR detector weights not available: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process image for detection: {e}",
        ) from e


# ---------------------------------------------------------------------------
# Unified Pipeline Endpoint: Image -> Detection -> Evidence -> Decision
# ---------------------------------------------------------------------------
@app.post("/process", response_model=ProcessResponse)
async def process(
    file: UploadFile = File(..., description="Invoice image or PDF document"),
    document_id: Optional[str] = Form(None, description="Optional document identifier"),
    as_of_date: Optional[date] = Form(None, description="Optional reference date (YYYY-MM-DD) for decision consistency"),
    conf_threshold: Optional[float] = Form(None, description="Optional detection confidence threshold override"),
) -> ProcessResponse:
    """Execute complete end-to-end invoice detection, evidence adaptation, and recovery reasoning."""
    global _DECISIONS_RECORDED

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed types: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    doc_id = document_id if document_id else Path(file.filename).stem

    detector = get_detector()
    try:
        detection_result = detector.predict(
            image=file_bytes,
            document_id=doc_id,
            conf_threshold=conf_threshold,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RT-DETR detector weights not available: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process image for detection: {e}",
        ) from e

    evidence_request = detection_to_evidence_request(
        detection_result=detection_result,
        as_of_date=as_of_date,
    )

    decision = decide(evidence_request)
    _DECISIONS_RECORDED += 1

    return ProcessResponse(
        document_id=doc_id,
        decision=decision,
        detections=detection_result.detections,
        evidence_request=evidence_request,
    )


def decisions_recorded() -> int:
    """Audit counter (in-process). Kept trivial by design; no secrets/PII stored."""
    return _DECISIONS_RECORDED
