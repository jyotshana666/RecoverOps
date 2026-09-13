"""Pipeline package bridging Part A detection and Part B reasoning."""

from src.pipeline.adapter import (
    detection_to_evidence_request,
    process_invoice_document,
)

__all__ = [
    "detection_to_evidence_request",
    "process_invoice_document",
]
