"""
Autism Detection CNN - Transfer Learning with ResNet50
Dataset: Autistic Children Facial Image Dataset
Classes: autistic vs non_autistic
"""

import os
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR    = Path("/Users/dpatra/Documents/final-paper/autism-screening-app")
DATA_DIR    = BASE_DIR / "data/raw/archive/Autistic Children Facial Image Dataset"
OUTPUT_DIR  = BASE_DIR / "data/model_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR     = OUTPUT_DIR / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
WEIGHT_DECAY= 1e-4
NUM_WORKERS = 0          # set 0 for macOS stability
DEVICE      = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")
print(f"🖥️  Device: {DEVICE}")
print(f"📁 Outputs → {RUN_DIR}\n")

# ──────────────────────────────────────────────
# DATA TRANSFORMS
# ──────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ──────────────────────────────────────────────
# DATASETS & LOADERS
# ──────────────────────────────────────────────
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
val_ds   = datasets.ImageFolder(DATA_DIR / "valid", transform=val_transforms)
test_ds  = datasets.ImageFolder(DATA_DIR / "test",  transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

CLASS_NAMES = train_ds.classes          # ['autistic', 'non_autistic']
print(f"📊 Class mapping : {train_ds.class_to_idx}")
print(f"   Train samples : {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\n")

# ──────────────────────────────────────────────
# MODEL  (ResNet-50 + custom head)
# ──────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last two residual blocks for fine-tuning
    for layer in [model.layer3, model.layer4]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),          # binary: autistic vs non_autistic
    )
    return model

model = build_model().to(DEVICE)
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🏗️  Model      : ResNet-50 (fine-tuned)")
print(f"   Parameters : {total_params:,} total | {trainable_params:,} trainable\n")

# ──────────────────────────────────────────────
# LOSS / OPTIMIZER / SCHEDULER
# ──────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
def run_epoch(loader, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            probs  = torch.softmax(outputs, dim=1)[:, 1]
            preds  = outputs.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return avg_loss, accuracy, auc

history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","train_auc","val_auc"]}
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
patience, patience_ctr = 5, 0

print("=" * 60)
print(f"{'Ep':>3} {'T-Loss':>8} {'V-Loss':>8} {'T-Acc':>7} {'V-Acc':>7} {'T-AUC':>7} {'V-AUC':>7}  LR")
print("=" * 60)

start = time.time()
for epoch in range(1, EPOCHS + 1):
    t_loss, t_acc, t_auc = run_epoch(train_loader, "train")
    v_loss, v_acc, v_auc = run_epoch(val_loader,   "val")
    scheduler.step()

    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)
    history["train_auc"].append(t_auc)
    history["val_auc"].append(v_auc)

    lr_now = optimizer.param_groups[0]["lr"]
    flag   = " ✅" if v_acc > best_val_acc else ""
    print(f"{epoch:>3} {t_loss:>8.4f} {v_loss:>8.4f} {t_acc:>6.2%} {v_acc:>6.2%} "
          f"{t_auc:>6.4f} {v_auc:>6.4f}  {lr_now:.2e}{flag}")

    if v_acc > best_val_acc:
        best_val_acc   = v_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_ctr   = 0
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break

elapsed = time.time() - start
print(f"\n⏱️  Training time: {elapsed/60:.1f} min")
print(f"🏆 Best Val Accuracy: {best_val_acc:.4f}")

# ──────────────────────────────────────────────
# SAVE BEST MODEL
# ──────────────────────────────────────────────
model.load_state_dict(best_model_wts)
model_path = RUN_DIR / "best_model.pth"
torch.save({
    "epoch":           EPOCHS,
    "model_state_dict": best_model_wts,
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_acc":    best_val_acc,
    "class_names":     CLASS_NAMES,
    "config": {
        "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
    }
}, model_path)
print(f"\n💾 Model saved → {model_path}")

# ──────────────────────────────────────────────
# TEST EVALUATION
# ──────────────────────────────────────────────
print("\n📋 Evaluating on TEST set …")
model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        out  = model(imgs)
        prob = torch.softmax(out, dim=1)[:, 1]
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(prob.cpu().numpy())

test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
test_auc = roc_auc_score(all_labels, all_probs)
cm       = confusion_matrix(all_labels, all_preds)
report   = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
report_str = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)

print(f"\n  Test Accuracy : {test_acc:.4f}")
print(f"  Test AUC      : {test_auc:.4f}")
print("\nClassification Report:")
print(report_str)

# ──────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Autism Detection CNN – Training Summary", fontsize=14, fontweight="bold")

# 1 – Accuracy
ax = axes[0]
ax.plot(history["train_acc"], label="Train")
ax.plot(history["val_acc"],   label="Val")
ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(alpha=0.3)

# 2 – Loss
ax = axes[1]
ax.plot(history["train_loss"], label="Train")
ax.plot(history["val_loss"],   label="Val")
ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

# 3 – Confusion Matrix
ax = axes[2]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(CLASS_NAMES, rotation=15)
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Test)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plot_path = RUN_DIR / "training_summary.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📊 Plot saved → {plot_path}")

# ROC Curve
fig, ax = plt.subplots(figsize=(6, 6))
fpr, tpr, _ = roc_curve(all_labels, all_probs)
ax.plot(fpr, tpr, lw=2, label=f"AUC = {test_auc:.4f}")
ax.plot([0,1],[0,1],"--", color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve – Autism Detection"); ax.legend(); ax.grid(alpha=0.3)
roc_path = RUN_DIR / "roc_curve.png"
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"📈 ROC curve saved → {roc_path}")

# ──────────────────────────────────────────────
# SAVE RESULTS JSON
# ──────────────────────────────────────────────
results = {
    "run_id":         RUN_ID,
    "model":          "ResNet50-FineTuned",
    "device":         str(DEVICE),
    "config":         {"img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
                       "epochs_run": len(history["train_acc"]),
                       "lr": LR, "weight_decay": WEIGHT_DECAY},
    "dataset":        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds),
                       "classes": CLASS_NAMES},
    "best_val_acc":   round(best_val_acc, 6),
    "test_accuracy":  round(test_acc, 6),
    "test_auc":       round(test_auc, 6),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "history":        {k: [round(v, 6) for v in vals] for k, vals in history.items()},
    "training_time_sec": round(elapsed, 1),
    "outputs": {
        "model":   str(model_path),
        "plot":    str(plot_path),
        "roc":     str(roc_path),
    }
}

results_path = RUN_DIR / "results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"📄 Results JSON → {results_path}")

print("\n" + "="*60)
print(f"✅  DONE  |  Test Acc: {test_acc:.4f}  |  AUC: {test_auc:.4f}")
print(f"📁  All outputs in: {RUN_DIR}")
print("="*60)
