"""Unit tests for InvoiceDetector and detection schemas."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from src.detection.config import CLASS_MAPPING, REVERSE_CLASS_MAPPING
from src.detection.detector import InvoiceDetector
from src.detection.schemas import Detection, DetectionResult


def create_dummy_png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Helper to generate valid in-memory PNG bytes."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestDetectionSchemas:
    def test_valid_detection(self):
        det = Detection(
            class_id=0,
            class_name="amount_due",
            confidence=0.95,
            bounding_box=[10.0, 20.0, 100.0, 50.0],
        )
        assert det.class_id == 0
        assert det.class_name == "amount_due"
        assert det.confidence == 0.95
        assert det.bounding_box == [10.0, 20.0, 100.0, 50.0]

    def test_invalid_bbox_length(self):
        with pytest.raises(ValidationError):
            Detection(
                class_id=0,
                class_name="amount_due",
                confidence=0.9,
                bounding_box=[10.0, 20.0, 100.0],
            )

    def test_invalid_bbox_coordinates(self):
        # x2 < x1
        with pytest.raises(ValidationError):
            Detection(
                class_id=0,
                class_name="amount_due",
                confidence=0.9,
                bounding_box=[100.0, 20.0, 50.0, 80.0],
            )

    def test_detection_result_schema(self):
        result = DetectionResult(
            document_id="doc_123",
            image_width=800,
            image_height=600,
            detections=[
                Detection(
                    class_id=0,
                    class_name="amount_due",
                    confidence=0.92,
                    bounding_box=[10.0, 10.0, 100.0, 50.0],
                )
            ],
        )
        assert result.document_id == "doc_123"
        assert len(result.detections) == 1
        assert result.image_width == 800


class TestInvoiceDetector:
    def test_missing_weights_raises_filenotfound(self, tmp_path):
        nonexistent = tmp_path / "nonexistent_model.pt"
        with pytest.raises(FileNotFoundError) as exc_info:
            InvoiceDetector(model_path=nonexistent, lazy_load=False)
        assert "RT-DETR model checkpoint not found" in str(exc_info.value)

    def test_lazy_load_missing_weights(self, tmp_path):
        nonexistent = tmp_path / "nonexistent_model.pt"
        detector = InvoiceDetector(model_path=nonexistent, lazy_load=True)
        assert detector._model is None
        with pytest.raises(FileNotFoundError):
            _ = detector.model

    def test_predict_with_mock_rtdetr(self):
        detector = InvoiceDetector(model_path="dummy.pt", lazy_load=True)

        # Mock RTDETR inference output
        mock_model = MagicMock()
        mock_box_1 = MagicMock()
        mock_box_1.cls.item.return_value = 0
        mock_box_1.conf.item.return_value = 0.94
        mock_box_1.xyxy.cpu().numpy().tolist.return_value = [50.0, 100.0, 200.0, 140.0]

        mock_box_2 = MagicMock()
        mock_box_2.cls.item.return_value = 1
        mock_box_2.conf.item.return_value = 0.88
        mock_box_2.xyxy.cpu().numpy().tolist.return_value = [50.0, 150.0, 180.0, 190.0]

        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 2
        mock_boxes.__getitem__.side_effect = [mock_box_1, mock_box_2]
        mock_boxes.cls = [mock_box_1.cls, mock_box_2.cls]
        mock_boxes.conf = [mock_box_1.conf, mock_box_2.conf]
        mock_boxes.xyxy = [mock_box_1.xyxy, mock_box_2.xyxy]

        mock_res = MagicMock()
        mock_res.boxes = mock_boxes
        mock_model.predict.return_value = [mock_res]

        detector._model = mock_model

        png_bytes = create_dummy_png_bytes(400, 300)
        res = detector.predict(image=png_bytes, document_id="invoice_test_01")

        assert isinstance(res, DetectionResult)
        assert res.document_id == "invoice_test_01"
        assert res.image_width == 400
        assert res.image_height == 300
        assert len(res.detections) == 2

        d0 = res.detections[0]
        assert d0.class_id == 0
        assert d0.class_name == "amount_due"
        assert d0.confidence == 0.94
        assert d0.bounding_box == [50.0, 100.0, 200.0, 140.0]

        d1 = res.detections[1]
        assert d1.class_id == 1
        assert d1.class_name == "date_due"
        assert d1.confidence == 0.88

    def test_preprocess_input_formats(self):
        detector = InvoiceDetector(model_path="dummy.pt", lazy_load=True)

        # 1. PIL Image
        pil_img = Image.new("RGB", (50, 50), color=(0, 0, 0))
        out, w, h = detector._preprocess_input(pil_img)
        assert w == 50 and h == 50

        # 2. Numpy array
        np_arr = np.zeros((60, 80, 3), dtype=np.uint8)
        out, w, h = detector._preprocess_input(np_arr)
        assert w == 80 and h == 60

        # 3. Valid bytes
        png_bytes = create_dummy_png_bytes(120, 90)
        out, w, h = detector._preprocess_input(png_bytes)
        assert w == 120 and h == 90

        # 4. Invalid bytes
        with pytest.raises(ValueError):
            detector._preprocess_input(b"not an image")

        # 5. Invalid type
        with pytest.raises(TypeError):
            detector._preprocess_input(12345)
