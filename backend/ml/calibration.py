"""
Post-hoc temperature scaling calibration for ASDScreeningModel.

Temperature scaling is a single-parameter method that learns a scalar T
to minimize NLL on the validation set:
  calibrated_prob = sigmoid(raw_logit / T)

T > 1: model becomes less confident (useful when overconfident)
T < 1: model becomes more confident
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import ASDScreeningModel


class TemperatureScaling(nn.Module):
    """
    Wraps a trained ASDScreeningModel and adds a learnable temperature
    parameter applied to all output logits.

    Only the temperature is trainable during calibration — base model is frozen.
    """

    def __init__(self, model: ASDScreeningModel):
        super().__init__()
        self.model = model
        # Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Single learnable temperature (initialised to 1.0 = no change)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def _scale_logit(self, prob: torch.Tensor) -> torch.Tensor:
        """
        Convert sigmoid output back to logit, scale by 1/T, re-apply sigmoid.
        Clamp prob away from 0/1 to avoid log(0).
        """
        prob_clamped = prob.clamp(1e-6, 1 - 1e-6)
        logit = torch.log(prob_clamped / (1 - prob_clamped))  # inverse sigmoid
        return torch.sigmoid(logit / self.temperature)

    def forward(
        self,
        images:        torch.Tensor,
        audio:         torch.Tensor,
        questionnaire: torch.Tensor,
        text:          torch.Tensor,
    ) -> dict:
        """
        Run base model then apply temperature scaling to all heads.
        Returns same dict structure as ASDScreeningModel.forward().
        """
        raw = self.model(images, audio, questionnaire, text)
        return {k: self._scale_logit(v) for k, v in raw.items()}

    def calibrate(
        self,
        val_loader: DataLoader,
        device:     torch.device,
        lr:         float = 0.01,
        max_iter:   int   = 50,
    ) -> float:
        """
        Fit temperature on validation set using LBFGS.

        Returns optimal temperature value.
        """
        self.model.eval()
        self.to(device)

        # Collect all val logits and labels first
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["images"].to(device)
                aud  = batch["audio"].to(device)
                qst  = batch["questionnaire"].to(device)
                txt  = batch["text"].to(device)
                lbl  = batch["label"].to(device).float()

                raw = self.model(imgs, aud, qst, txt)
                prob = raw["global"].squeeze(1)
                prob_c = prob.clamp(1e-6, 1 - 1e-6)
                logit = torch.log(prob_c / (1 - prob_c))
                all_logits.append(logit)
                all_labels.append(lbl)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Optimise temperature with LBFGS
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        def eval_closure():
            optimizer.zero_grad()
            scaled = all_logits / self.temperature
            loss = criterion(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)

        # Clamp temperature to reasonable range
        with torch.no_grad():
            self.temperature.clamp_(0.1, 10.0)

        return float(self.temperature.item())

    def save(self, path: str) -> None:
        """Save temperature value as JSON."""
        Path(path).write_text(
            json.dumps({"temperature": float(self.temperature.item())}, indent=2)
        )

    @classmethod
    def load(cls, model: ASDScreeningModel, path: str) -> "TemperatureScaling":
        """Load saved temperature JSON and return wrapped calibrated model."""
        ts = cls(model)
        data = json.loads(Path(path).read_text())
        with torch.no_grad():
            ts.temperature.fill_(data["temperature"])
        return ts

    def enable_dropout(self) -> None:
        """Enable dropout in base model for MC Dropout inference."""
        self.model.enable_dropout()
