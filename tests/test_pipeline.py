"""Unit tests for detection-to-evidence adapter and end-to-end pipeline."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.detection.schemas import Detection, DetectionResult
from src.pipeline.adapter import (
    detection_to_evidence_request,
    process_invoice_document,
)
from src.reasoning.schemas import (
    ConfidenceState,
    Decision,
    EvidenceField,
    EvidenceRequest,
    FieldName,
    Intent,
    SourceType,
)


class TestDetectionToEvidenceAdapter:
    def test_empty_detections_adapter(self):
        detection_res = DetectionResult(
            document_id="empty_doc",
            image_width=640,
            image_height=640,
            detections=[],
        )
        evidence_req = detection_to_evidence_request(detection_res)

        assert isinstance(evidence_req, EvidenceRequest)
        assert evidence_req.document_id == "empty_doc"
        assert len(evidence_req.fields) == 0
        assert evidence_req.as_of_date is None

    def test_valid_detections_mapping(self):
        detection_res = DetectionResult(
            document_id="inv_2026_001",
            image_width=1000,
            image_height=1400,
            detections=[
                Detection(
                    class_id=0,
                    class_name="amount_due",
                    confidence=0.95,
                    bounding_box=[100.0, 500.0, 300.0, 550.0],
                ),
                Detection(
                    class_id=1,
                    class_name="date_due",
                    confidence=0.91,
                    bounding_box=[100.0, 600.0, 250.0, 640.0],
                ),
                Detection(
                    class_id=4,
                    class_name="vendor_name",
                    confidence=0.88,
                    bounding_box=[50.0, 50.0, 400.0, 100.0],
                ),
            ],
        )

        as_of = date(2026, 9, 15)
        evidence_req = detection_to_evidence_request(detection_res, as_of_date=as_of)

        assert evidence_req.document_id == "inv_2026_001"
        assert evidence_req.as_of_date == as_of
        assert len(evidence_req.fields) == 3

        f0 = evidence_req.fields[0]
        assert f0.field == FieldName.AMOUNT_DUE
        assert f0.source == SourceType.RTDETR
        assert f0.confidence == 0.95
        assert f0.bbox == [100.0, 500.0, 300.0, 550.0]
        # Crucial: value must be None (no OCR fabrication)
        assert f0.value is None
        assert f0.value_normalized is None

        f1 = evidence_req.fields[1]
        assert f1.field == FieldName.DATE_DUE
        assert f1.confidence == 0.91
        assert f1.value is None

        f2 = evidence_req.fields[2]
        assert f2.field == FieldName.VENDOR_NAME
        assert f2.confidence == 0.88

    def test_honest_values_no_fabrication(self):
        """Verify that detector bounding boxes alone never produce fabricated text values."""
        detection_res = DetectionResult(
            document_id="doc_integrity",
            image_width=500,
            image_height=500,
            detections=[
                Detection(
                    class_id=0,
                    class_name="amount_due",
                    confidence=0.99,
                    bounding_box=[10.0, 10.0, 100.0, 50.0],
                )
            ],
        )
        evidence_req = detection_to_evidence_request(detection_res)
        for ef in evidence_req.fields:
            assert ef.value is None, "Detection alone MUST NOT fabricate string text values"
            assert ef.value_normalized is None

    def test_optional_ocr_values_propagation(self):
        """When OCR values are explicitly provided, they populate the value field."""
        detection_res = DetectionResult(
            document_id="doc_with_ocr",
            image_width=500,
            image_height=500,
            detections=[
                Detection(
                    class_id=0,
                    class_name="amount_due",
                    confidence=0.92,
                    bounding_box=[10.0, 10.0, 100.0, 50.0],
                ),
                Detection(
                    class_id=1,
                    class_name="date_due",
                    confidence=0.87,
                    bounding_box=[10.0, 60.0, 100.0, 100.0],
                ),
            ],
        )
        ocr_map = {0: "$1,250.00", 1: "2026-10-01"}
        evidence_req = detection_to_evidence_request(detection_res, ocr_extracted_values=ocr_map)

        assert evidence_req.fields[0].value == "$1,250.00"
        assert evidence_req.fields[1].value == "2026-10-01"

    def test_skip_unrecognized_class(self):
        detection_res = DetectionResult(
            document_id="doc_unknown_cls",
            image_width=500,
            image_height=500,
            detections=[
                Detection(
                    class_id=0,
                    class_name="amount_due",
                    confidence=0.9,
                    bounding_box=[10.0, 10.0, 50.0, 50.0],
                ),
                Detection(
                    class_id=99,
                    class_name="some_unrecognized_label",
                    confidence=0.7,
                    bounding_box=[20.0, 20.0, 60.0, 60.0],
                ),
            ],
        )
        evidence_req = detection_to_evidence_request(detection_res)
        assert len(evidence_req.fields) == 1
        assert evidence_req.fields[0].field == FieldName.AMOUNT_DUE


class TestProcessInvoicePipeline:
    def test_process_invoice_document_end_to_end_mock(self):
        mock_detector = MagicMock()
        mock_detector.predict.return_value = DetectionResult(
            document_id="pipeline_test_doc",
            image_width=800,
            image_height=1000,
            detections=[
                Detection(
                    class_id=0,
                    class_name="amount_due",
                    confidence=0.95,
                    bounding_box=[100.0, 200.0, 300.0, 250.0],
                ),
                Detection(
                    class_id=1,
                    class_name="date_due",
                    confidence=0.92,
                    bounding_box=[100.0, 300.0, 250.0, 340.0],
                ),
            ],
        )

        det_res, evid_req, decision = process_invoice_document(
            detector=mock_detector,
            image=b"dummy_image_bytes",
            document_id="pipeline_test_doc",
            as_of_date=date(2026, 9, 15),
        )

        assert isinstance(det_res, DetectionResult)
        assert isinstance(evid_req, EvidenceRequest)
        assert isinstance(decision, Decision)
        assert decision.document_id == "pipeline_test_doc"
        # Since region-only detections have value=None (no OCR), Part B guardrails correctly evaluate this
        assert decision.confidence_state in {ConfidenceState.BLOCKED, ConfidenceState.REVIEW}
        assert decision.requires_human_review is True
