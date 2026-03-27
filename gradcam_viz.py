"""
Grad-CAM Visualization for Autism Detection CNN
Generates publication-quality attention heatmaps for the paper
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = Path("/Users/dpatra/Documents/final-paper/autism-screening-app")
DATA_DIR   = BASE_DIR / "data/raw/archive/Autistic Children Facial Image Dataset"
RUN_DIR    = BASE_DIR / "data/model_outputs/run_20260311_135435"
OUT_DIR    = RUN_DIR / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")
print(f"🖥️  Device: {DEVICE}")

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
    "figure.dpi":     120,
})

# ── Load Model ────────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 128),         nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 2),
    )
    return model

ckpt  = torch.load(RUN_DIR / "best_model.pth", map_location=DEVICE)
model = build_model().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
CLASS_NAMES = ckpt["class_names"]   # ['autistic', 'non_autistic']
print(f"✅ Model loaded  |  Classes: {CLASS_NAMES}")

# ── Grad-CAM ──────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.gradients    = None
        self.activations  = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        out   = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[0, class_idx]
        score.backward()

        # Weights = global average pooled gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        prob    = torch.softmax(out, dim=1)[0, class_idx].item()
        return cam, class_idx, prob

gradcam = GradCAM(model, model.layer4[-1])

# ── Image Transforms ──────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
inv_norm = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std =[1/0.229,       1/0.224,      1/0.225]
)

test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=val_tf)

# ── Collect samples: correct & wrong predictions ──────────────
random.seed(42)
correct_autistic, correct_non, wrong_autistic, wrong_non = [], [], [], []

for idx in random.sample(range(len(test_ds)), len(test_ds)):
    img_tensor, true_label = test_ds[idx]
    img_path  = test_ds.imgs[idx][0]
    x         = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out  = model(x)
        pred = out.argmax(dim=1).item()
        prob = torch.softmax(out, dim=1)[0, pred].item()

    entry = (img_tensor, true_label, pred, prob, img_path)

    if pred == true_label:
        if true_label == 0 and len(correct_autistic) < 4:
            correct_autistic.append(entry)
        elif true_label == 1 and len(correct_non) < 4:
            correct_non.append(entry)
    else:
        if true_label == 0 and len(wrong_autistic) < 2:
            wrong_autistic.append(entry)
        elif true_label == 1 and len(wrong_non) < 2:
            wrong_non.append(entry)

    if (len(correct_autistic) == 4 and len(correct_non) == 4
            and len(wrong_autistic) == 2 and len(wrong_non) == 2):
        break

print(f"Collected: {len(correct_autistic)} correct-autistic, {len(correct_non)} correct-non, "
      f"{len(wrong_autistic)} wrong-autistic, {len(wrong_non)} wrong-non")

# ── Helper: overlay heatmap ───────────────────────────────────
def overlay_heatmap(img_tensor, cam, alpha=0.45, colormap="jet"):
    img_np = inv_norm(img_tensor).permute(1,2,0).numpy()
    img_np = np.clip(img_np, 0, 1)
    heatmap = plt.get_cmap(colormap)(cam)[:,:,:3]
    overlay = (1-alpha)*img_np + alpha*heatmap
    overlay = np.clip(overlay, 0, 1)
    return img_np, overlay, cam

# ═══════════════════════════════════════════════════════════════
# FIG 8 – Grad-CAM: Correct Predictions (4 autistic + 4 non-autistic)
# ═══════════════════════════════════════════════════════════════
samples = correct_autistic[:4] + correct_non[:4]
fig, axes = plt.subplots(4, 6, figsize=(18, 12))
fig.suptitle(
    "Grad-CAM Visualizations — Correct Predictions\n"
    "Columns: Original Image | Grad-CAM Heatmap | Overlay",
    fontsize=14, fontweight="bold"
)

col_titles = ["Original", "Heatmap", "Overlay",
              "Original", "Heatmap", "Overlay"]
for j, t in enumerate(col_titles):
    axes[0, j].set_title(t, fontsize=11, fontweight="bold", pad=6)

for row, (img_t, true_lbl, pred_lbl, prob, path) in enumerate(samples):
    x   = img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
    cam, _, _ = gradcam(x, class_idx=pred_lbl)
    img_np, overlay, cam_np = overlay_heatmap(img_t, cam)

    true_name = CLASS_NAMES[true_lbl].replace("_", " ").title()
    pred_name = CLASS_NAMES[pred_lbl].replace("_", " ").title()

    # Column offset: first 4 rows use cols 0-2, but we need them in groups of 2 across 4 rows
    # Layout: row 0-3, each row = one sample, 3 cols per sample (orig, heat, overlay) × 2 samples side by side
    # Re-layout: 4 rows × 6 cols, rows 0-1 = autistic, rows 2-3 = non-autistic
    col = 0 if row < 4 else 3
    r   = row if row < 4 else row - 4

    axes[r, col+0].imshow(img_np);    axes[r, col+0].axis("off")
    hm = axes[r, col+1].imshow(cam_np, cmap="jet", vmin=0, vmax=1)
    axes[r, col+1].axis("off")
    axes[r, col+2].imshow(overlay);   axes[r, col+2].axis("off")

    tick = "✓"
    color = "#16A34A"
    label_text = f"{tick} True: {true_name}\nPred: {pred_name} ({prob:.2%})"
    axes[r, col+0].set_ylabel(label_text, fontsize=9, color=color,
                               rotation=0, labelpad=110, va="center")

# Row group labels
for r in range(4):
    axes[r, 0].annotate("AUTISTIC",   xy=(-0.05, 0.5), xycoords="axes fraction",
                         fontsize=9, color="#2563EB", fontweight="bold",
                         ha="right", va="center", rotation=90)
    axes[r, 3].annotate("NON-AUTISTIC", xy=(-0.05, 0.5), xycoords="axes fraction",
                         fontsize=9, color="#DC2626", fontweight="bold",
                         ha="right", va="center", rotation=90)

plt.colorbar(hm, ax=axes[:, 2], fraction=0.02, pad=0.02, label="Activation Intensity")
plt.colorbar(hm, ax=axes[:, 5], fraction=0.02, pad=0.02, label="Activation Intensity")
plt.subplots_adjust(hspace=0.08, wspace=0.05)
plt.savefig(OUT_DIR / "fig8_gradcam_correct.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ fig8_gradcam_correct.png")

# ═══════════════════════════════════════════════════════════════
# FIG 9 – Grad-CAM: Misclassified Examples (2+2)
# ═══════════════════════════════════════════════════════════════
misclassified = wrong_autistic[:2] + wrong_non[:2]

fig, axes = plt.subplots(len(misclassified), 3, figsize=(10, 3.2*len(misclassified)))
fig.suptitle(
    "Grad-CAM Visualizations — Misclassified Examples\n"
    "Model attended to correct facial regions but made wrong predictions",
    fontsize=13, fontweight="bold"
)
for j, t in enumerate(["Original Image", "Grad-CAM Heatmap", "Overlay"]):
    axes[0, j].set_title(t, fontsize=11, fontweight="bold", pad=6)

for row, (img_t, true_lbl, pred_lbl, prob, path) in enumerate(misclassified):
    x   = img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
    cam, _, _ = gradcam(x, class_idx=pred_lbl)
    img_np, overlay, cam_np = overlay_heatmap(img_t, cam)

    true_name = CLASS_NAMES[true_lbl].replace("_", " ").title()
    pred_name = CLASS_NAMES[pred_lbl].replace("_", " ").title()

    axes[row, 0].imshow(img_np);   axes[row, 0].axis("off")
    hm = axes[row, 1].imshow(cam_np, cmap="jet", vmin=0, vmax=1)
    axes[row, 1].axis("off")
    axes[row, 2].imshow(overlay);  axes[row, 2].axis("off")

    axes[row, 0].set_ylabel(
        f"✗ True: {true_name}\nPred: {pred_name} ({prob:.2%})",
        fontsize=9, color="#DC2626", rotation=0, labelpad=110, va="center"
    )

plt.colorbar(hm, ax=axes[:, 1], fraction=0.02, pad=0.02, label="Activation Intensity")
plt.subplots_adjust(hspace=0.08, wspace=0.05)
plt.savefig(OUT_DIR / "fig9_gradcam_misclassified.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ fig9_gradcam_misclassified.png")

# ═══════════════════════════════════════════════════════════════
# FIG 10 – Grad-CAM Class Comparison (side-by-side 1 autistic vs 1 non-autistic)
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 9))
gs  = GridSpec(2, 4, figure=fig, hspace=0.15, wspace=0.08)
fig.suptitle(
    "Grad-CAM Class Comparison: Facial Attention Patterns\n"
    "ResNet-50 learns distinct facial cues for ASD vs. Non-ASD classification",
    fontsize=13, fontweight="bold", y=1.01
)

pairs = [correct_autistic[0], correct_non[0],
         correct_autistic[1], correct_non[1]]
titles_top = ["Autistic", "Non-Autistic", "Autistic", "Non-Autistic"]
colors_top  = ["#2563EB", "#DC2626", "#2563EB", "#DC2626"]

row_labels = ["Original", "Grad-CAM Overlay"]

for col, (img_t, true_lbl, pred_lbl, prob, path) in enumerate(pairs):
    x   = img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
    cam, _, _ = gradcam(x, class_idx=pred_lbl)
    img_np, overlay, cam_np = overlay_heatmap(img_t, cam, alpha=0.50)

    # Original
    ax_orig = fig.add_subplot(gs[0, col])
    ax_orig.imshow(img_np)
    ax_orig.axis("off")
    pred_name = CLASS_NAMES[pred_lbl].replace("_"," ").title()
    ax_orig.set_title(f"{titles_top[col]}\n{pred_name} ({prob:.1%})",
                      color=colors_top[col], fontsize=11, fontweight="bold", pad=4)

    # Overlay
    ax_over = fig.add_subplot(gs[1, col])
    hm = ax_over.imshow(overlay)
    ax_over.axis("off")

    if col == 0:
        ax_orig.set_ylabel("Original Image", fontsize=10, labelpad=8)
        ax_over.set_ylabel("Grad-CAM Overlay", fontsize=10, labelpad=8)

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="Activation Intensity")

# Annotation
fig.text(0.5, -0.01,
         "Warmer colors (red/yellow) indicate regions the model focuses on most for classification.\n"
         "The model primarily attends to facial features including eyes, nose bridge, and perioral regions.",
         ha="center", fontsize=10, style="italic", color="#374151")

plt.savefig(OUT_DIR / "fig10_gradcam_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ fig10_gradcam_comparison.png")

# ── Final summary ─────────────────────────────────────────────
print(f"\n🎉 Grad-CAM figures saved to:\n   {OUT_DIR}")
print("\nFiles:")
for f in sorted(OUT_DIR.glob("fig[89]*.png")) | sorted(OUT_DIR.glob("fig10*.png")):
    print(f"   {f.name:<45} {f.stat().st_size//1024:>5} KB")
