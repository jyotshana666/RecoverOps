"""RecoverOps Part B — minimal deterministic reasoning layer.

Pipeline:
    detections → evidence normalization → invoice state → intent routing
    → confidence guardrail → recovery recommendation → human escalation

Design constraints (per the Part B specification):
    * Deterministic and auditable — explicit rules only, no LLM.
    * RT-DETR detects FIELD REGIONS, not values; text extraction (OCR) is
      connected later and values are never fabricated from bounding boxes.
    * Bounded output: recommendations only. RecoverOps never initiates
      payments, moves funds, or sends legally binding communications.
    * All thresholds/rules live in ``src/reasoning/config.py``.
"""
from src.reasoning.recovery import decide

__all__ = ["decide"]
