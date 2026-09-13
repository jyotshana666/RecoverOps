"""Detection configuration and model path resolution.

Provides single-source resolution of model checkpoint paths via
environment variable ``MODEL_PATH`` with fallback to local ``models/best.pt``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Authoritative class definitions matching Part A and Part B
CLASS_MAPPING = {
    "amount_due": 0,
    "date_due": 1,
    "document_id": 2,
    "date_issue": 3,
    "vendor_name": 4,
}

REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# Default local checkpoint location
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "best.pt"

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
DEFAULT_IOU_THRESHOLD: float = 0.5
DEFAULT_IMG_SIZE: int = 640


def get_model_path() -> Path:
    """Resolve production RT-DETR model weights path.

    Checks environment variable ``MODEL_PATH`` first. If not set, defaults
    to ``models/best.pt``.
    """
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path).resolve()
    return DEFAULT_MODEL_PATH.resolve()
