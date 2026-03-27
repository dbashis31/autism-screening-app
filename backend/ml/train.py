"""
Training script for CNN-BiLSTM ASD screening model.

Usage:
  cd autism-screening-app
  python -m backend.ml.train [OPTIONS]

Options:
  --data-source   synthetic | kaggle  (default: synthetic)
  --data-root     path to data/raw/   (required if --data-source kaggle)
  --epochs        int (default: 50)
  --batch-size    int (default: 32)
  --lr            float (default: 3e-4)
  --patience      int (default: 10)
  --device        cpu | cuda | mps   (default: auto-detect)
  --seed          int (default: 42)
  --no-calibrate  flag: skip temperature scaling step
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .calibration import TemperatureScaling
from .dataset import get_dataloaders
from .model import ASDScreeningModel


# ── Loss ─────────────────────────────────────────────────────────────────────

def multimodal_bce_loss(
    outputs:          dict,
    labels:           torch.Tensor,   # (B,) float32
    pos_weight:       torch.Tensor,
    global_weight:    float = 0.7,
    modality_weight:  float = 0.3,
) -> torch.Tensor:
    """
    Combined loss:
      global_weight * BCE(global, y)
      + modality_weight * mean(BCE(modality_m, y))
    All losses use pos_weight for class imbalance.
    """
    pw = pos_weight.to(labels.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    def _logit(prob):
        p = prob.squeeze(1).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    global_loss = criterion(_logit(outputs["global"]), labels)
    modal_keys  = ["audio", "video", "questionnaire", "text"]
    modal_loss  = torch.stack([
        criterion(_logit(outputs[m]), labels) for m in modal_keys
    ]).mean()

    return global_weight * global_loss + modality_weight * modal_loss


# ── Metric helpers ────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_aucs(
    model:  ASDScreeningModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Return AUC-ROC per modality + global on loader."""
    model.eval()
    all_labels = []
    all_probs  = {k: [] for k in ["audio", "video", "questionnaire", "text", "global"]}

    for batch in loader:
        imgs = batch["images"].to(device)
        aud  = batch["audio"].to(device)
        qst  = batch["questionnaire"].to(device)
        txt  = batch["text"].to(device)
        lbl  = batch["label"].cpu().numpy()

        out  = model(imgs, aud, qst, txt)
        all_labels.extend(lbl)
        for k in all_probs:
            all_probs[k].extend(out[k].squeeze(1).cpu().numpy())

    y = np.array(all_labels)
    return {
        k: float(roc_auc_score(y, np.array(v)))
        for k, v in all_probs.items()
    }


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, checkpoint_path: Path = None):
        self.patience         = patience
        self.checkpoint_path  = checkpoint_path
        self.best_val_auc     = -1.0
        self.epochs_no_improve = 0

    def __call__(self, val_auc: float, model: ASDScreeningModel) -> bool:
        """Returns True if training should stop."""
        if val_auc > self.best_val_auc:
            self.best_val_auc      = val_auc
            self.epochs_no_improve = 0
            if self.checkpoint_path is not None:
                torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.epochs_no_improve += 1
        return self.epochs_no_improve >= self.patience


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      ASDScreeningModel,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    device:     torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs = batch["images"].to(device)
        aud  = batch["audio"].to(device)
        qst  = batch["questionnaire"].to(device)
        txt  = batch["text"].to(device)
        lbl  = batch["label"].to(device).float()

        optimizer.zero_grad()
        out  = model(imgs, aud, qst, txt)
        loss = multimodal_bce_loss(out, lbl, pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(
    model:      ASDScreeningModel,
    loader:     DataLoader,
    pos_weight: torch.Tensor,
    device:     torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels, all_global = [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            aud  = batch["audio"].to(device)
            qst  = batch["questionnaire"].to(device)
            txt  = batch["text"].to(device)
            lbl  = batch["label"].to(device).float()

            out  = model(imgs, aud, qst, txt)
            loss = multimodal_bce_loss(out, lbl, pos_weight)
            total_loss  += loss.item()
            all_labels.extend(lbl.cpu().numpy())
            all_global.extend(out["global"].squeeze(1).cpu().numpy())

    y   = np.array(all_labels)
    auc = float(roc_auc_score(y, np.array(all_global)))
    return {"loss": total_loss / max(len(loader), 1), "auc_global": auc}


# ── Main train function ───────────────────────────────────────────────────────

def train(
    data_source:    str           = "synthetic",
    data_root:      Optional[Path] = None,
    epochs:         int           = 50,
    batch_size:     int           = 32,
    lr:             float         = 3e-4,
    patience:       int           = 10,
    checkpoint_dir: Path          = Path("backend/ml/checkpoints"),
    device_str:     Optional[str] = None,
    seed:           int           = 42,
    calibrate:      bool          = True,
    progress_cb:    Optional[callable] = None,  # (epoch, total_epochs, val_auc) -> None
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if device_str is None:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    print(f"Training on device: {device}")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt    = checkpoint_dir / "best_model.pt"
    final_ckpt   = checkpoint_dir / "asd_cnnrnn_v1.pt"
    curves_path  = checkpoint_dir / "training_curves.json"

    # Data
    print(f"Loading data (source={data_source}) ...")
    train_loader, val_loader, test_loader, pos_weight = get_dataloaders(
        use_synthetic=(data_source == "synthetic"),
        data_root=data_root,
        batch_size=batch_size,
        seed=seed,
    )
    pos_weight = pos_weight.to(device)
    print(f"  train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, "
          f"test={len(test_loader.dataset)}, "
          f"pos_weight={pos_weight.item():.2f}")

    # Model
    model = ASDScreeningModel().to(device)
    print(f"Model parameters: {model.count_parameters():,} trainable")

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    early_stop = EarlyStopping(patience=patience, checkpoint_path=best_ckpt)
    history    = {"train_loss": [], "val_loss": [], "val_auc": []}

    print(f"\nStarting training for up to {epochs} epochs (patience={patience}) ...\n")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, pos_weight, device)
        val_metrics = evaluate(model, val_loader, pos_weight, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc_global"])

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_auc={val_metrics['auc_global']:.4f} | "
              f"elapsed={elapsed:.0f}s")

        if progress_cb is not None:
            try:
                progress_cb(epoch, epochs, val_metrics["auc_global"])
            except Exception:  # noqa: BLE001
                pass  # never let a bad callback kill training

        if early_stop(val_metrics["auc_global"], model):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best val AUC={early_stop.best_val_auc:.4f})")
            break

    # Load best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"\nBest val AUC: {early_stop.best_val_auc:.4f}")

    # Test set evaluation
    print("Evaluating on test set ...")
    test_aucs = compute_aucs(model, test_loader, device)
    print(f"Test AUCs: { {k: f'{v:.4f}' for k, v in test_aucs.items()} }")

    # Temperature scaling calibration
    temperature = 1.0
    if calibrate:
        print("Calibrating temperature scaling on val set ...")
        ts = TemperatureScaling(model)
        temperature = ts.calibrate(val_loader, device)
        print(f"Optimal temperature: {temperature:.4f}")
    else:
        ts = TemperatureScaling(model)

    # Save final checkpoint (includes model weights + metadata)
    metadata = {
        "model_state_dict":  model.state_dict(),
        "model_config":      model.hparams,
        "temperature":       temperature,
        "training_metadata": {
            "best_val_auc":    early_stop.best_val_auc,
            "test_aucs":       test_aucs,
            "epochs_trained":  len(history["train_loss"]),
            "data_source":     data_source,
            "seed":            seed,
        },
    }
    torch.save(metadata, final_ckpt)
    curves_path.write_text(json.dumps(history, indent=2))
    print(f"\nCheckpoint saved → {final_ckpt}")

    return {
        "best_val_auc":    early_stop.best_val_auc,
        "test_metrics":    test_aucs,
        "temperature":     temperature,
        "checkpoint_path": str(final_ckpt),
        "epochs_trained":  len(history["train_loss"]),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train CNN-BiLSTM ASD screening model")
    p.add_argument("--data-source",   default="synthetic", choices=["synthetic", "kaggle"])
    p.add_argument("--data-root",     default=None,  type=Path)
    p.add_argument("--epochs",        default=50,    type=int)
    p.add_argument("--batch-size",    default=32,    type=int)
    p.add_argument("--lr",            default=3e-4,  type=float)
    p.add_argument("--patience",      default=10,    type=int)
    p.add_argument("--checkpoint-dir",default="backend/ml/checkpoints", type=Path)
    p.add_argument("--device",        default=None)
    p.add_argument("--seed",          default=42,    type=int)
    p.add_argument("--no-calibrate",  action="store_true")
    args = p.parse_args()

    result = train(
        data_source    = args.data_source,
        data_root      = args.data_root,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        patience       = args.patience,
        checkpoint_dir = args.checkpoint_dir,
        device_str     = args.device,
        seed           = args.seed,
        calibrate      = not args.no_calibrate,
    )
    print("\n" + "="*60)
    print("Training complete!")
    print(f"  Best val AUC : {result['best_val_auc']:.4f}")
    print(f"  Test AUC     : {result['test_metrics']['global']:.4f}")
    print(f"  Temperature  : {result['temperature']:.4f}")
    print(f"  Checkpoint   : {result['checkpoint_path']}")
    print("="*60)


if __name__ == "__main__":
    main()
