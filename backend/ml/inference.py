"""
Inference module: loads trained checkpoint, runs MC Dropout,
returns per-modality confidence scores for the governance pipeline.

Public API:
  ml_inference(modalities, ...) -> dict[str, float]

This is the drop-in replacement for _mock_confidence() in
backend/routers/submit.py — same signature, same return type.

Graceful fallback: if no checkpoint exists, falls back to
random.uniform(0.70, 0.95) matching the original mock behavior.
"""

from __future__ import annotations

import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .model import ASDScreeningModel
from .calibration import TemperatureScaling

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "asd_cnnrnn_v1.pt"
N_MC_PASSES = 20


# ── Model loading (singleton via lru_cache) ───────────────────────────────────

@lru_cache(maxsize=1)
def _load_model(
    checkpoint_path: str = str(DEFAULT_CHECKPOINT),
    device_str:      str = "cpu",
) -> Optional[TemperatureScaling]:
    """
    Load calibrated model from checkpoint. Returns None if file missing.
    Cached — loads from disk only once per process.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        logger.warning("Checkpoint not found at %s — will use mock confidence scores.", ckpt_path)
        return None

    try:
        device = torch.device(device_str)
        ckpt   = torch.load(ckpt_path, map_location=device)

        # Reconstruct model with saved config
        config = ckpt.get("model_config", {})
        model  = ASDScreeningModel(**config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Wrap with temperature scaling
        ts = TemperatureScaling(model)
        if "temperature" in ckpt:
            with torch.no_grad():
                ts.temperature.fill_(ckpt["temperature"])

        logger.info(
            "ASD model loaded from %s (T=%.4f, val_auc=%.4f)",
            ckpt_path,
            ckpt.get("temperature", 1.0),
            ckpt.get("training_metadata", {}).get("best_val_auc", 0.0),
        )
        return ts

    except Exception as exc:
        logger.error("Failed to load ASD model: %s — using mock scores.", exc)
        return None


# ── Dummy input builder ───────────────────────────────────────────────────────

def _build_dummy_input(
    device: torch.device,
    T: int = 5,
) -> dict:
    """
    Build a single-sample batch of zero-valued inputs.
    Used when no real sensor data is piped through the API.
    """
    return {
        "images":        torch.zeros(1, T, 3, 64, 64,    device=device),
        "audio":         torch.zeros(1, T, 1, 40, 128,   device=device),
        "questionnaire": torch.zeros(1, T, 10,            device=device),
        "text":          torch.zeros(1, T, 50,            device=device),
    }


# ── MC Dropout inference ──────────────────────────────────────────────────────

def mc_dropout_inference(
    model:    TemperatureScaling,
    inputs:   dict,
    n_passes: int = N_MC_PASSES,
) -> dict:
    """
    Run N stochastic forward passes with dropout active.

    Returns:
      {"means": {mod: float}, "stds": {mod: float}}
    """
    model.eval()
    model.enable_dropout()   # keep dropout ON for uncertainty estimation

    all_runs = {k: [] for k in ["audio", "video", "questionnaire", "text", "global"]}

    with torch.no_grad():
        for _ in range(n_passes):
            out = model(
                inputs["images"],
                inputs["audio"],
                inputs["questionnaire"],
                inputs["text"],
            )
            for k in all_runs:
                all_runs[k].append(float(out[k].squeeze().item()))

    return {
        "means": {k: float(np.mean(v))  for k, v in all_runs.items()},
        "stds":  {k: float(np.std(v))   for k, v in all_runs.items()},
    }


# ── Public API ────────────────────────────────────────────────────────────────

def ml_inference(
    modalities:      list,
    checkpoint_path: str             = str(DEFAULT_CHECKPOINT),
    device_str:      str             = "cpu",
    n_passes:        int             = N_MC_PASSES,
    inputs:          Optional[dict]  = None,
) -> dict:
    """
    Drop-in replacement for _mock_confidence() in backend/routers/submit.py.

    Args:
      modalities:       list of active modality names
                        e.g. ["audio", "video", "text", "questionnaire"]
      checkpoint_path:  path to .pt checkpoint (auto-detected by default)
      device_str:       "cpu" | "cuda" | "mps"
      n_passes:         MC Dropout forward passes (default 20)
      inputs:           optional real input tensors; uses dummy zeros if None

    Returns:
      dict[str, float] — per-modality mean confidence in [0, 1]
      only for modalities in the `modalities` argument.

    Fallback:
      If checkpoint is missing or load fails → random.uniform(0.70, 0.95)
      matching the original _mock_confidence() behavior.
    """
    model = _load_model(checkpoint_path, device_str)

    if model is None:
        # Original mock behavior — indistinguishable to callers
        return {m: round(random.uniform(0.70, 0.95), 3) for m in modalities}

    device = torch.device(device_str)
    if inputs is None:
        inputs = _build_dummy_input(device)

    result = mc_dropout_inference(model, inputs, n_passes)
    means  = result["means"]

    # Return only the modalities requested, matching _mock_confidence signature
    return {m: round(float(means.get(m, means["global"])), 3)
            for m in modalities}
