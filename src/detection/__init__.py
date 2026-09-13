"""Detection package for RecoverOps Part A (RT-DETR-L)."""

from src.detection.config import (
    CLASS_MAPPING,
    DEFAULT_MODEL_PATH,
    REVERSE_CLASS_MAPPING,
    get_model_path,
)
from src.detection.detector import InvoiceDetector
from src.detection.schemas import Detection, DetectionResult

__all__ = [
    "InvoiceDetector",
    "Detection",
    "DetectionResult",
    "CLASS_MAPPING",
    "REVERSE_CLASS_MAPPING",
    "DEFAULT_MODEL_PATH",
    "get_model_path",
]
