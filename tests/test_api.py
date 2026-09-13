"""Unit tests for the unified FastAPI application endpoints."""

from __future__ import annotations

import io
from datetime import date
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.detection.schemas import Detection, DetectionResult
from src.reasoning.app import app, set_detector


@pytest.fixture(autouse=True)
def reset_detector_singleton():
    """Ensure clean detector state before and after each test."""
    set_detector(None)
    yield
    set_detector(None)


def make_png_bytes(width: int = 50, height: int = 50) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoints:
    def test_reason_health(self):
        client = TestClient(app)
        resp = client.get("/reason/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "reasoning"

    def test_unified_health(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "recoverops-unified"

    def test_cors_headers(self):
        client = TestClient(app)
        headers = {"Origin": "http://localhost:5173"}
        resp = client.get("/health", headers=headers)
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


class TestReasonEndpointRegression:
    def test_reason_valid_request(self):
        client = TestClient(app)
        payload = {
            "document_id": "INV-2026-001",
            "as_of_date": "2026-09-01",
            "fields": [
                {
                    "field": "amount_due",
                    "value": "1500.00",
                    "source": "ocr",
                    "confidence": 0.95,
                    "bbox": [10.0, 10.0, 100.0, 50.0],
                },
                {
                    "field": "date_due",
                    "value": "2026-08-15",
                    "source": "ocr",
                    "confidence": 0.92,
                    "bbox": [10.0, 60.0, 100.0, 100.0],
                },
            ],
        }
        resp = client.post("/reason", json=payload)
        assert resp.status_code == 200
        decision = resp.json()
        assert decision["document_id"] == "INV-2026-001"
        assert decision["intent"] == "overdue_payment"
        assert decision["confidence_state"] == "SAFE"
        assert decision["requires_human_review"] is False


class TestDetectEndpoint:
    def test_detect_missing_weights_returns_503(self):
        # When default detector is not found, /detect returns 503
        client = TestClient(app)
        png_bytes = make_png_bytes()
        files = {"file": ("invoice.png", png_bytes, "image/png")}
        resp = client.post("/detect", files=files)
        assert resp.status_code == 503
        assert "RT-DETR detector weights not available" in resp.json()["detail"]

    def test_detect_unsupported_file_extension(self):
        client = TestClient(app)
        files = {"file": ("script.sh", b"echo hello", "text/plain")}
        resp = client.post("/detect", files=files)
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_detect_empty_file(self):
        client = TestClient(app)
        files = {"file": ("empty.png", b"", "image/png")}
        resp = client.post("/detect", files=files)
        assert resp.status_code == 400
        assert "Uploaded file is empty" in resp.json()["detail"]

    def test_detect_success_with_mock_detector(self):
        mock_det = MagicMock()
        mock_det.predict.return_value = DetectionResult(
            document_id="mock_doc_99",
            image_width=600,
            image_height=800,
            detections=[
                Detection(
                    class_id=0,
                    class_name="amount_due",
                    confidence=0.96,
                    bounding_box=[50.0, 100.0, 200.0, 150.0],
                )
            ],
        )
        set_detector(mock_det)

        client = TestClient(app)
        png_bytes = make_png_bytes(600, 800)
        files = {"file": ("invoice_sample.png", png_bytes, "image/png")}
        data = {"document_id": "mock_doc_99", "conf_threshold": "0.3"}

        resp = client.post("/detect", files=files, data=data)
        assert resp.status_code == 200
        res = resp.json()
        assert res["document_id"] == "mock_doc_99"
        assert res["image_width"] == 600
        assert len(res["detections"]) == 1
        assert res["detections"][0]["class_name"] == "amount_due"
        assert res["detections"][0]["confidence"] == 0.96


class TestProcessEndpoint:
    def test_process_missing_weights_returns_503(self):
        client = TestClient(app)
        png_bytes = make_png_bytes()
        files = {"file": ("invoice.png", png_bytes, "image/png")}
        resp = client.post("/process", files=files)
        assert resp.status_code == 503

    def test_process_unsupported_file(self):
        client = TestClient(app)
        files = {"file": ("invoice.exe", b"binary", "application/octet-stream")}
        resp = client.post("/process", files=files)
        assert resp.status_code == 400

    def test_process_success_with_mock_detector(self):
        mock_det = MagicMock()
        mock_det.predict.return_value = DetectionResult(
            document_id="end_to_end_doc",
            image_width=700,
            image_height=900,
            detections=[
                Detection(
                    class_id=0,
                    class_name="amount_due",
                    confidence=0.91,
                    bounding_box=[10.0, 20.0, 150.0, 60.0],
                ),
                Detection(
                    class_id=1,
                    class_name="date_due",
                    confidence=0.89,
                    bounding_box=[10.0, 70.0, 150.0, 110.0],
                ),
            ],
        )
        set_detector(mock_det)

        client = TestClient(app)
        png_bytes = make_png_bytes(700, 900)
        files = {"file": ("invoice.png", png_bytes, "image/png")}
        data = {"document_id": "end_to_end_doc", "as_of_date": "2026-09-01"}

        resp = client.post("/process", files=files, data=data)
        assert resp.status_code == 200
        body = resp.json()

        assert body["document_id"] == "end_to_end_doc"
        assert len(body["detections"]) == 2
        assert body["evidence_request"]["document_id"] == "end_to_end_doc"
        assert len(body["evidence_request"]["fields"]) == 2

        # Verify decision structure
        decision = body["decision"]
        assert decision["document_id"] == "end_to_end_doc"
        assert "requires_human_review" in decision
        assert "confidence_state" in decision
