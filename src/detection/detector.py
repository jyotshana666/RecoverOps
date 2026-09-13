"""Invoice field detector wrapping Ultralytics RT-DETR-L.

Provides a clean object-oriented abstraction for running field detection
on invoice documents (images or PDFs), returning strictly typed
DetectionResult objects without leaking framework internals.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.detection.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMG_SIZE,
    REVERSE_CLASS_MAPPING,
    get_model_path,
)
from src.detection.schemas import Detection, DetectionResult


class InvoiceDetector:
    """RT-DETR-L detector for invoice field region extraction."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        img_size: int = DEFAULT_IMG_SIZE,
        lazy_load: bool = False,
    ):
        self.model_path = Path(model_path).resolve() if model_path else get_model_path()
        self.device = device
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self._model = None

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load RT-DETR model checkpoint from configured path."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"RT-DETR model checkpoint not found at: {self.model_path}. "
                "Please place the trained 'best.pt' in the models/ directory "
                "or set the MODEL_PATH environment variable."
            )

        from ultralytics import RTDETR

        try:
            self._model = RTDETR(str(self.model_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize RT-DETR model from '{self.model_path}': {e}"
            ) from e

    @property
    def model(self):
        """Lazy model accessor."""
        if self._model is None:
            self._load_model()
        return self._model

    def _preprocess_input(
        self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]
    ) -> Tuple[Image.Image, int, int]:
        """Normalize various input formats into a PIL Image and extract dimensions."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Input image file not found: {path}")
            if path.suffix.lower() == ".pdf":
                img = self._render_pdf_page(path.read_bytes(), page_index=0)
            else:
                img = Image.open(path).convert("RGB")
        elif isinstance(image_input, bytes):
            # Check for PDF magic bytes (%PDF)
            if image_input.startswith(b"%PDF"):
                img = self._render_pdf_page(image_input, page_index=0)
            else:
                try:
                    img = Image.open(io.BytesIO(image_input)).convert("RGB")
                except Exception as e:
                    raise ValueError(f"Could not decode image bytes: {e}") from e
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}. "
                "Expected str, Path, bytes, PIL.Image, or np.ndarray."
            )

        width, height = img.size
        return img, width, height

    def _render_pdf_page(self, pdf_bytes: bytes, page_index: int = 0) -> Image.Image:
        """Render a single PDF page to a PIL Image at 150 DPI."""
        try:
            import pymupdf
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            if page_index < 0 or page_index >= len(doc):
                raise ValueError(f"Page index {page_index} out of range for PDF with {len(doc)} pages.")
            page = doc[page_index]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        except ImportError:
            raise RuntimeError(
                "pymupdf is required to process PDF invoice files. "
                "Install with: pip install pymupdf"
            )

    def predict(
        self,
        image: Union[str, Path, bytes, Image.Image, np.ndarray],
        document_id: Optional[str] = None,
        conf_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Run RT-DETR inference on an invoice image or document.

        Args:
            image: Image path, raw bytes, PIL Image, or numpy array.
            document_id: Optional document identifier; defaults to filename or 'doc_unknown'.
            conf_threshold: Optional override for detection confidence threshold.

        Returns:
            DetectionResult with extracted bounding boxes and class labels.
        """
        pil_img, width, height = self._preprocess_input(image)

        if document_id is None:
            if isinstance(image, (str, Path)):
                document_id = Path(image).stem
            else:
                document_id = "doc_unknown"

        effective_conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        predict_kwargs = {
            "source": pil_img,
            "conf": effective_conf,
            "imgsz": self.img_size,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)

        detections: List[Detection] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()

                # Map class ID to authoritative class name
                class_name = REVERSE_CLASS_MAPPING.get(cls_id, f"unknown_class_{cls_id}")

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        bounding_box=xyxy,
                    )
                )

        return DetectionResult(
            document_id=document_id,
            image_width=width,
            image_height=height,
            detections=detections,
        )
