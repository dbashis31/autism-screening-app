"""
t-SNE & UMAP Feature Embedding Visualizations
Extracts deep features from ResNet-50 and plots class separation
for the Autism Detection paper
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path("/Users/dpatra/Documents/final-paper/autism-screening-app")
DATA_DIR = BASE_DIR / "data/raw/archive/Autistic Children Facial Image Dataset"
RUN_DIR  = BASE_DIR / "data/model_outputs/run_20260311_135435"
OUT_DIR  = RUN_DIR / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")
print(f"🖥️  Device : {DEVICE}")

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     12,
    "axes.titlesize":13,
    "axes.labelsize":12,
    "axes.linewidth":1.2,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "savefig.dpi":   300,
    "savefig.bbox":  "tight",
    "figure.dpi":    120,
})

BLUE      = "#2563EB"
RED       = "#DC2626"
GREEN     = "#16A34A"
PURPLE    = "#7C3AED"
GRAY      = "#6B7280"
COLORS    = [BLUE, RED]
CLASS_LABELS = ["Autistic", "Non-Autistic"]

# ── Load Model ────────────────────────────────────────────────
def build_model():
    m = models.resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Linear(in_f, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 2),
    )
    return m

ckpt  = torch.load(RUN_DIR / "best_model.pth", map_location=DEVICE)
model = build_model().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
CLASS_NAMES = ckpt["class_names"]
print(f"✅ Model loaded  |  Classes: {CLASS_NAMES}")

# ── Feature Extractor (penultimate 128-d layer) ───────────────
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Extract everything up to the last Linear (128→2)
        self.features = nn.Sequential(*list(backbone.fc.children())[:-1])
        self.backbone.fc = nn.Identity()   # remove final classifier

    def forward(self, x):
        x = self.backbone(x)   # ResNet body → (B, 2048)
        # Run through first 6 sub-layers of original fc (up to ReLU before Dropout[1])
        return x

# Simpler: hook approach
embeddings_128 = {}
def hook_fn(module, input, output):
    embeddings_128["feat"] = output.detach().cpu()

# Hook the last BatchNorm1d(128) → captures 128-d representation
hook = model.fc[7].register_forward_hook(hook_fn)   # BN after second linear

# ── Data Loaders ──────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

all_feats, all_labels, all_splits, all_preds = [], [], [], []

for split_name, split_path in [("Train", DATA_DIR/"train"),
                                ("Val",   DATA_DIR/"valid"),
                                ("Test",  DATA_DIR/"test")]:
    ds = datasets.ImageFolder(split_path, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out  = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            feats = embeddings_128["feat"].numpy()

            all_feats.append(feats)
            all_labels.extend(labels.numpy())
            all_splits.extend([split_name] * len(labels))
            all_preds.extend(preds)

    print(f"   {split_name}: {len(ds)} samples extracted")

hook.remove()

X      = np.vstack(all_feats)           # (N, 128)
y      = np.array(all_labels)           # (N,)  0=autistic, 1=non_autistic
splits = np.array(all_splits)
preds  = np.array(all_preds)
correct = (preds == y).astype(int)

print(f"\n✅ Feature matrix: {X.shape}  |  Classes: {np.bincount(y)}")

# ── Dimensionality Reduction ───────────────────────────────────
print("\n⏳ Running PCA (pre-reduce to 50-d for speed) …")
pca50 = PCA(n_components=50, random_state=42)
X50   = pca50.fit_transform(X)
var_explained = pca50.explained_variance_ratio_.sum()
print(f"   PCA variance retained: {var_explained:.1%}")

print("⏳ Running t-SNE …")
tsne  = TSNE(n_components=2, perplexity=40, max_iter=1200,
             learning_rate="auto", init="pca", random_state=42)
X_tsne = tsne.fit_transform(X50)

print("⏳ Running UMAP …")
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.15,
                    metric="cosine", random_state=42)
X_umap  = reducer.fit_transform(X50)

# ── Silhouette Scores ─────────────────────────────────────────
sil_tsne = silhouette_score(X_tsne, y)
sil_umap = silhouette_score(X_umap, y)
print(f"\n📊 Silhouette Score — t-SNE: {sil_tsne:.4f}  |  UMAP: {sil_umap:.4f}")

# ═══════════════════════════════════════════════════════════════
# FIG 11 — t-SNE by class  (clean, paper-ready)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

for cls_idx, (cls_name, color) in enumerate(zip(CLASS_LABELS, COLORS)):
    mask = y == cls_idx
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=color, label=f"{cls_name} (n={mask.sum()})",
               alpha=0.65, s=18, edgecolors="none", linewidths=0)

ax.set_title(f"t-SNE Feature Embedding\n(Silhouette Score = {sil_tsne:.4f})", pad=10)
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.legend(markerscale=2, framealpha=0.9)
ax.text(0.01, 0.01,
        f"Features: 128-d  |  Perplexity: 40  |  PCA pre-reduction: {var_explained:.0%} var retained",
        transform=ax.transAxes, fontsize=8, color=GRAY, va="bottom")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig11_tsne_by_class.png", dpi=300)
plt.close()
print("✅ fig11_tsne_by_class.png")

# ═══════════════════════════════════════════════════════════════
# FIG 12 — UMAP by class
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

for cls_idx, (cls_name, color) in enumerate(zip(CLASS_LABELS, COLORS)):
    mask = y == cls_idx
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
               c=color, label=f"{cls_name} (n={mask.sum()})",
               alpha=0.65, s=18, edgecolors="none")

ax.set_title(f"UMAP Feature Embedding\n(Silhouette Score = {sil_umap:.4f})", pad=10)
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.legend(markerscale=2, framealpha=0.9)
ax.text(0.01, 0.01,
        "Features: 128-d  |  n_neighbors: 30  |  min_dist: 0.15  |  metric: cosine",
        transform=ax.transAxes, fontsize=8, color=GRAY, va="bottom")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig12_umap_by_class.png", dpi=300)
plt.close()
print("✅ fig12_umap_by_class.png")

# ═══════════════════════════════════════════════════════════════
# FIG 13 — t-SNE: correct vs wrong predictions
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("t-SNE Embedding — Prediction Correctness", fontsize=13, fontweight="bold")

# Left: by class
ax = axes[0]
for cls_idx, (cls_name, color) in enumerate(zip(CLASS_LABELS, COLORS)):
    mask = y == cls_idx
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=color, alpha=0.55, s=16, edgecolors="none", label=cls_name)
ax.set_title("(a) Coloured by True Class")
ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
ax.legend(markerscale=2)

# Right: by correct/wrong
ax = axes[1]
corr_colors  = {1: GREEN, 0: RED}
corr_labels  = {1: f"Correct (n={correct.sum()})", 0: f"Misclassified (n={(~correct.astype(bool)).sum()})"}
corr_markers = {1: "o", 0: "X"}
for c_val in [1, 0]:
    mask = correct == c_val
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=corr_colors[c_val], alpha=0.6 if c_val == 1 else 0.9,
               s=16 if c_val == 1 else 55,
               marker=corr_markers[c_val],
               edgecolors="white" if c_val == 0 else "none",
               linewidths=0.5,
               label=corr_labels[c_val], zorder=3 if c_val == 0 else 2)
ax.set_title("(b) Coloured by Prediction Outcome")
ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
ax.legend(markerscale=1.5)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig13_tsne_correctness.png", dpi=300)
plt.close()
print("✅ fig13_tsne_correctness.png")

# ═══════════════════════════════════════════════════════════════
# FIG 14 — Combined t-SNE + UMAP + Train/Val/Test split
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 11))
gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)
fig.suptitle(
    "Deep Feature Space Visualization — ResNet-50 Penultimate Layer (128-d)\n"
    f"Total samples: {len(y)} | t-SNE Silhouette: {sil_tsne:.4f} | UMAP Silhouette: {sil_umap:.4f}",
    fontsize=13, fontweight="bold", y=1.01
)

SPLIT_COLORS = {"Train": BLUE, "Val": ORANGE if False else "#F59E0B", "Test": PURPLE}

# (a) t-SNE by class
ax = fig.add_subplot(gs[0, 0])
for cls_idx, (cls_name, color) in enumerate(zip(CLASS_LABELS, COLORS)):
    mask = y == cls_idx
    ax.scatter(X_tsne[mask,0], X_tsne[mask,1], c=color, alpha=0.55, s=12,
               edgecolors="none", label=cls_name)
ax.set_title(f"(a) t-SNE — by Class\nSilhouette={sil_tsne:.4f}")
ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
ax.legend(fontsize=9, markerscale=2)

# (b) UMAP by class
ax = fig.add_subplot(gs[0, 1])
for cls_idx, (cls_name, color) in enumerate(zip(CLASS_LABELS, COLORS)):
    mask = y == cls_idx
    ax.scatter(X_umap[mask,0], X_umap[mask,1], c=color, alpha=0.55, s=12,
               edgecolors="none", label=cls_name)
ax.set_title(f"(b) UMAP — by Class\nSilhouette={sil_umap:.4f}")
ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
ax.legend(fontsize=9, markerscale=2)

# (c) t-SNE — misclassified highlighted
ax = fig.add_subplot(gs[0, 2])
correct_mask = correct == 1
wrong_mask   = correct == 0
ax.scatter(X_tsne[correct_mask,0], X_tsne[correct_mask,1],
           c=GREEN, alpha=0.45, s=12, edgecolors="none", label=f"Correct ({correct_mask.sum()})", zorder=2)
ax.scatter(X_tsne[wrong_mask,0], X_tsne[wrong_mask,1],
           c=RED, alpha=0.9, s=55, marker="X", edgecolors="white", linewidths=0.5,
           label=f"Misclassified ({wrong_mask.sum()})", zorder=3)
ax.set_title("(c) t-SNE — Prediction Outcome")
ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
ax.legend(fontsize=9, markerscale=1.2)

# (d) t-SNE by train/val/test split
ax = fig.add_subplot(gs[1, 0])
for sp_name, sp_color in [("Train","#2563EB"), ("Val","#F59E0B"), ("Test","#7C3AED")]:
    mask = splits == sp_name
    ax.scatter(X_tsne[mask,0], X_tsne[mask,1], c=sp_color, alpha=0.5, s=12,
               edgecolors="none", label=f"{sp_name} ({mask.sum()})")
ax.set_title("(d) t-SNE — by Data Split")
ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
ax.legend(fontsize=9, markerscale=2)

# (e) UMAP by train/val/test split
ax = fig.add_subplot(gs[1, 1])
for sp_name, sp_color in [("Train","#2563EB"), ("Val","#F59E0B"), ("Test","#7C3AED")]:
    mask = splits == sp_name
    ax.scatter(X_umap[mask,0], X_umap[mask,1], c=sp_color, alpha=0.5, s=12,
               edgecolors="none", label=f"{sp_name} ({mask.sum()})")
ax.set_title("(e) UMAP — by Data Split")
ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
ax.legend(fontsize=9, markerscale=2)

# (f) PCA variance explained
ax = fig.add_subplot(gs[1, 2])
var_ratio  = pca50.explained_variance_ratio_
cumulative = np.cumsum(var_ratio)
ax.bar(range(1, 21), var_ratio[:20]*100, color=BLUE, alpha=0.7, label="Per-component")
ax.plot(range(1, 21), cumulative[:20]*100, color=RED, marker="o",
        markersize=4, linewidth=2, label="Cumulative")
ax.axhline(var_explained*100, color=GRAY, linestyle="--", linewidth=1,
           label=f"50-PC total: {var_explained:.1%}")
ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance Explained (%)")
ax.set_title("(f) PCA Variance Explained\n(Top 20 of 50 components)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_xticks(range(1, 21, 2))

plt.savefig(OUT_DIR / "fig14_embedding_combined.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ fig14_embedding_combined.png")

# ── Final listing ─────────────────────────────────────────────
print(f"\n🎉 All embedding figures saved to:\n   {OUT_DIR}\n")
for f in sorted(OUT_DIR.glob("fig1[1-4]*.png")):
    print(f"   {f.name:<45}  {f.stat().st_size//1024:>5} KB")

print(f"\n📊 Silhouette Score Summary")
print(f"   t-SNE : {sil_tsne:.4f}  {'(good separation)' if sil_tsne > 0.3 else '(moderate separation)'}")
print(f"   UMAP  : {sil_umap:.4f}  {'(good separation)' if sil_umap > 0.3 else '(moderate separation)'}")
