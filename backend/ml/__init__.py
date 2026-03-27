"""
ML module for ASD Screening — MLHC 2026.

Public API:
  ml_inference(modalities, ...) -> dict[str, float]
    Drop-in replacement for _mock_confidence() in backend/routers/submit.py.
    Uses CNN-BiLSTM model with MC Dropout if checkpoint exists,
    otherwise falls back to random mock scores.
"""

from .inference import ml_inference

__all__ = ["ml_inference"]
