"""Detection-to-Evidence Adapter for RecoverOps.

Bridges Part A (RT-DETR spatial detection) and Part B (deterministic reasoning).
Converts raw DetectionResult objects into the strict Part B EvidenceRequest schema
without fabricating text values from bounding boxes.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
import numpy as np

from src.detection.detector import InvoiceDetector
from src.detection.schemas import Detection, DetectionResult
from src.reasoning.recovery import decide
from src.reasoning.schemas import (
    Decision,
    EvidenceField,
    EvidenceRequest,
    FieldName,
    SourceType,
)


def detection_to_evidence_request(
    detection_result: DetectionResult,
    as_of_date: Optional[date] = None,
    ocr_extracted_values: Optional[Dict[int, str]] = None,
) -> EvidenceRequest:
    """Convert a Part A DetectionResult into a Part B EvidenceRequest.

    Design constraints:
    * Spatial bounding boxes represent detected field regions (source=RTDETR).
    * If OCR text is not supplied, `value` remains `None` (region-only detection).
    * Bounding boxes are preserved as `[x1, y1, x2, y2]`.
    * Unknown or unrecognized classes are skipped safely.

    Args:
        detection_result: Output from RT-DETR InvoiceDetector.
        as_of_date: Optional reference date for reproducible date reasoning.
        ocr_extracted_values: Optional dictionary mapping detection index to OCR raw string.

    Returns:
        A strictly validated EvidenceRequest instance conforming to Part B schemas.
    """
    evidence_fields: List[EvidenceField] = []

    for idx, det in enumerate(detection_result.detections):
        # Validate that class_name is a recognized Part B FieldName
        try:
            field_name = FieldName(det.class_name)
        except ValueError:
            # Skip any unmapped/out-of-domain detection
            continue

        raw_value: Optional[str] = None
        if ocr_extracted_values and idx in ocr_extracted_values:
            raw_value = str(ocr_extracted_values[idx])

        evidence_field = EvidenceField(
            field=field_name,
            value=raw_value,
            value_normalized=None,  # Normalization happens inside Part B's normalize_evidence
            source=SourceType.RTDETR,
            confidence=float(det.confidence),
            bbox=det.bounding_box,
        )
        evidence_fields.append(evidence_field)

    # Ensure document_id is non-empty and adheres to schema limits (min 1, max 128)
    doc_id = detection_result.document_id.strip() if detection_result.document_id else "unknown_document"
    if len(doc_id) > 128:
        doc_id = doc_id[:128]
    if not doc_id:
        doc_id = "unknown_document"

    return EvidenceRequest(
        document_id=doc_id,
        fields=evidence_fields,
        as_of_date=as_of_date,
    )


def process_invoice_document(
    detector: InvoiceDetector,
    image: Union[str, Path, bytes, Image.Image, np.ndarray],
    document_id: Optional[str] = None,
    as_of_date: Optional[date] = None,
) -> Tuple[DetectionResult, EvidenceRequest, Decision]:
    """Execute the complete end-to-end invoice analysis pipeline.

    Steps:
        1. Image -> RT-DETR Detection -> DetectionResult
        2. DetectionResult -> Adapter -> EvidenceRequest
        3. EvidenceRequest -> Deterministic Reasoning (decide) -> Decision

    Returns:
        Tuple of (DetectionResult, EvidenceRequest, Decision)
    """
    detection_result = detector.predict(image, document_id=document_id)
    evidence_request = detection_to_evidence_request(detection_result, as_of_date=as_of_date)
    decision = decide(evidence_request)
    return detection_result, evidence_request, decision
