"""Detection schemas for RT-DETR invoice field extraction.

Defines structured detection objects returned by the RT-DETR-L detector,
ensuring strict types, validated bounding boxes, and alignment with
the authoritative 5 invoice field classes.
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Detection(BaseModel):
    """A single detected invoice field region on a page."""

    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(ge=0, description="Class ID matching detector taxonomy")
    class_name: str = Field(description="Class name (e.g. amount_due, date_due)")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    bounding_box: List[float] = Field(description="Bounding box [x1, y1, x2, y2]")

    @field_validator("bounding_box")
    @classmethod
    def _validate_bounding_box(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("bounding_box must be exactly 4 coordinates [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (float(coord) for coord in v)
        if x2 < x1 or y2 < y1:
            raise ValueError(f"bounding_box coordinates must satisfy x2 >= x1 and y2 >= y1, got [{x1}, {y1}, {x2}, {y2}]")
        return [x1, y1, x2, y2]


class DetectionResult(BaseModel):
    """Complete detection result for an invoice document/image."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="Identifier of the processed document")
    image_width: int = Field(gt=0, description="Input image width in pixels")
    image_height: int = Field(gt=0, description="Input image height in pixels")
    detections: List[Detection] = Field(default_factory=list, description="List of detected field regions")
