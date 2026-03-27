import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

FIG_DIR = Path("/Users/dpatra/Documents/final-paper/autism-screening-app/data/model_outputs/run_20260311_135435/paper_figures")
OUT     = FIG_DIR / "grid_3x3.png"

FIGS = [
    ("fig1_accuracy_curve.png",     "(a) Accuracy Curve"),
    ("fig2_loss_curve.png",         "(b) Loss Curve"),
    ("fig3_auc_curve.png",          "(c) AUC Curve"),
    ("fig4_roc_curve.png",          "(d) ROC Curve"),
    ("fig5_confusion_matrix.png",   "(e) Confusion Matrix"),
    ("fig6_classification_metrics.png", "(f) Classification Metrics"),
    ("fig10_gradcam_comparison.png","(g) Grad-CAM Attention"),
    ("fig11_tsne_by_class.png",     "(h) t-SNE Embedding"),
    ("fig12_umap_by_class.png",     "(i) UMAP Embedding"),
]

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.patch.set_facecolor("white")
fig.suptitle(
    "Autism Detection CNN — ResNet-50 Transfer Learning\n"
    "Test Accuracy: 89.0%  |  AUC: 0.9585  |  t-SNE Silhouette: 0.61  |  UMAP Silhouette: 0.86",
    fontsize=15, fontweight="bold", y=1.01, color="#111827"
)

for ax, (fname, title) in zip(axes.flat, FIGS):
    img = mpimg.imread(FIG_DIR / fname)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1E3A8A", pad=6)

plt.tight_layout(h_pad=1.5, w_pad=0.8)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✅ Saved → {OUT}  ({OUT.stat().st_size//1024} KB)")
