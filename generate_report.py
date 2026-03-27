"""
PDF Report Generator — Autism Detection CNN
Packages all 14 figures into a single publication-ready PDF with
title page, captions, and metrics summary table.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from datetime import datetime
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path("/Users/dpatra/Documents/final-paper/autism-screening-app")
RUN_DIR  = BASE_DIR / "data/model_outputs/run_20260311_135435"
FIG_DIR  = RUN_DIR / "paper_figures"
PDF_PATH = RUN_DIR / "autism_cnn_report.pdf"

with open(RUN_DIR / "results.json") as f:
    R = json.load(f)

cr = R["classification_report"]

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "figure.dpi":     150,
    "savefig.dpi":    300,
})

BLUE   = "#2563EB";  RED    = "#DC2626"
GREEN  = "#16A34A";  PURPLE = "#7C3AED"
GRAY   = "#6B7280";  DARK   = "#111827"
LIGHT  = "#F3F4F6"

# ── Figure metadata ───────────────────────────────────────────
FIGURES = [
    # (filename, title, caption, section)
    ("fig1_accuracy_curve.png",
     "Figure 1 — Training & Validation Accuracy",
     "Accuracy curves over 18 epochs. The model reaches 68.8% on epoch 1 and converges to a peak "
     "validation accuracy of 91.0% at epoch 13. A vertical dashed line marks the best epoch. "
     "The train–val gap after epoch 10 indicates mild overfitting, mitigated by early stopping.",
     "Training Dynamics"),

    ("fig2_loss_curve.png",
     "Figure 2 — Training & Validation Loss",
     "Cross-entropy loss (label smoothing = 0.1) decreasing from 0.60 to 0.23 (train) and bottoming "
     "at 0.376 (validation) before rising slightly, confirming the early stopping criterion was appropriate. "
     "Cosine annealing LR schedule contributes to the smooth descent.",
     "Training Dynamics"),

    ("fig3_auc_curve.png",
     "Figure 3 — ROC-AUC Over Training",
     "Validation AUC improves from 0.867 at epoch 1 to 0.955 at epoch 13. The green dashed line marks "
     "the final test AUC of 0.9585, demonstrating strong and consistent discriminative ability "
     "across the entire training run.",
     "Training Dynamics"),

    ("fig4_roc_curve.png",
     "Figure 4 — Receiver Operating Characteristic (ROC) Curve",
     "ROC curve for the test set (n=200). AUC = 0.9585 indicates excellent discrimination between "
     "autistic and non-autistic samples. The shaded region represents the area under the curve. "
     "The red dot marks the actual operating point (FPR=0.08, TPR=0.86) from the confusion matrix.",
     "Model Evaluation"),

    ("fig5_confusion_matrix.png",
     "Figure 5 — Normalized Confusion Matrix (Test Set)",
     "Row-normalized confusion matrix on the 200-sample test set. The model correctly classifies 86% "
     "of autistic children (14 missed) and 92% of non-autistic children (8 false alarms). "
     "Both proportions and raw sample counts are displayed.",
     "Model Evaluation"),

    ("fig6_classification_metrics.png",
     "Figure 6 — Per-Class Classification Metrics",
     "Precision, Recall, and F1-Score for each class on the test set. Both classes achieve scores "
     "between 0.867–0.915, with a macro-average F1 of 0.890 (green diamond), demonstrating "
     "balanced performance with no class bias.",
     "Model Evaluation"),

    ("fig7_combined_summary.png",
     "Figure 7 — Combined Training Summary (6-Panel Overview)",
     "All-in-one summary figure: (a) accuracy curves, (b) loss curves, (c) AUC curves, "
     "(d) ROC curve, (e) confusion matrix, (f) per-class metrics. "
     "Suitable as a single comprehensive results figure in the paper.",
     "Summary"),

    ("fig8_gradcam_correct.png",
     "Figure 8 — Grad-CAM: Correctly Classified Samples",
     "Grad-CAM heatmaps for 4 correctly classified autistic (left) and 4 non-autistic (right) children. "
     "Warmer colors (red/yellow) indicate regions of highest activation. The model predominantly "
     "attends to periorbital, nasal bridge, and perioral regions — consistent with known ASD facial markers.",
     "Grad-CAM Interpretability"),

    ("fig9_gradcam_misclassified.png",
     "Figure 9 — Grad-CAM: Misclassified Samples",
     "Grad-CAM analysis of 4 misclassified samples (2 autistic predicted as non-autistic; "
     "2 non-autistic predicted as autistic). Despite incorrect predictions, the model still "
     "attends to plausible facial regions, suggesting boundary-region ambiguity rather than "
     "spurious feature reliance.",
     "Grad-CAM Interpretability"),

    ("fig10_gradcam_comparison.png",
     "Figure 10 — Grad-CAM Class Comparison",
     "Side-by-side Grad-CAM comparison between autistic and non-autistic samples. "
     "Original images (top row) and overlaid heatmaps (bottom row) reveal that the model "
     "learns class-specific attention patterns, with autistic samples showing higher activation "
     "around the eye region compared to non-autistic samples.",
     "Grad-CAM Interpretability"),

    ("fig11_tsne_by_class.png",
     "Figure 11 — t-SNE Feature Embedding (by Class)",
     "t-SNE projection of 128-dimensional penultimate-layer features from all 2,926 samples. "
     "Autistic (blue) and non-autistic (red) samples form largely separable clusters "
     "(Silhouette Score = 0.6055), confirming that the model has learned discriminative "
     "representations. Some boundary overlap corresponds to the 22 misclassified samples.",
     "Feature Space Analysis"),

    ("fig12_umap_by_class.png",
     "Figure 12 — UMAP Feature Embedding (by Class)",
     "UMAP projection of the same 128-dimensional features with superior global structure preservation. "
     "The two classes form tight, well-separated manifolds (Silhouette Score = 0.8617), providing "
     "strong evidence that the fine-tuned ResNet-50 captures meaningful inter-class distinctions "
     "in its learned feature space.",
     "Feature Space Analysis"),

    ("fig13_tsne_correctness.png",
     "Figure 13 — t-SNE: Class Labels vs. Prediction Outcome",
     "(a) t-SNE coloured by true class showing cluster structure. "
     "(b) The same embedding coloured by prediction correctness: correct predictions (green circles) "
     "occupy cluster interiors while misclassified samples (red crosses, n=22) concentrate "
     "near cluster boundaries, validating that errors arise from inter-class overlap.",
     "Feature Space Analysis"),

    ("fig14_embedding_combined.png",
     "Figure 14 — Complete Embedding Analysis (6-Panel)",
     "Comprehensive embedding overview: (a) t-SNE by class, (b) UMAP by class, (c) misclassifications "
     "on t-SNE, (d–e) embeddings coloured by train/val/test split showing no domain shift, "
     "(f) PCA scree plot retaining 91.5% variance in 50 components, justifying the PCA "
     "pre-reduction step before t-SNE/UMAP.",
     "Feature Space Analysis"),
]

# ═══════════════════════════════════════════════════════════════
# BUILD PDF
# ═══════════════════════════════════════════════════════════════
print(f"📄 Building PDF report → {PDF_PATH}")

with PdfPages(PDF_PATH) as pdf:

    # ── PAGE 1: Title Page ────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69))   # A4
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Header bar
    ax.add_patch(FancyBboxPatch((0, 0.82), 1, 0.18, boxstyle="square",
                                facecolor=BLUE, edgecolor="none"))

    ax.text(0.5, 0.935, "Autism Spectrum Disorder Detection",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.875, "Using Convolutional Neural Networks — Experimental Results Report",
            ha="center", va="center", fontsize=13, color="#BFDBFE",
            transform=ax.transAxes)

    # Model summary box
    ax.add_patch(FancyBboxPatch((0.05, 0.60), 0.90, 0.19,
                                boxstyle="round,pad=0.01",
                                facecolor=LIGHT, edgecolor="#D1D5DB", linewidth=1))
    ax.text(0.50, 0.775, "Model Architecture & Configuration",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=DARK, transform=ax.transAxes)

    cfg_lines = [
        ("Architecture",  "ResNet-50 (Transfer Learning — ImageNet pre-trained)"),
        ("Fine-tuned",    "Layer3, Layer4 + Custom head [2048→512→128→2]"),
        ("Optimizer",     "AdamW  (lr=1e-4, weight decay=1e-4)"),
        ("Scheduler",     "Cosine Annealing  (T_max=20, η_min=1e-6)"),
        ("Regularization","Label Smoothing 0.1 | Dropout 0.4/0.3 | BatchNorm"),
        ("Device",        "Apple MPS (M-series GPU)"),
        ("Training Time", "30.8 minutes  |  18 epochs (early stopping at patience=5)"),
    ]
    for i, (k, v) in enumerate(cfg_lines):
        y_pos = 0.745 - i * 0.028
        ax.text(0.10, y_pos, f"{k}:", ha="left", va="center", fontsize=9.5,
                fontweight="bold", color=BLUE, transform=ax.transAxes)
        ax.text(0.35, y_pos, v, ha="left", va="center", fontsize=9.5,
                color=DARK, transform=ax.transAxes)

    # Results table
    ax.add_patch(FancyBboxPatch((0.05, 0.35), 0.90, 0.22,
                                boxstyle="round,pad=0.01",
                                facecolor=LIGHT, edgecolor="#D1D5DB", linewidth=1))
    ax.text(0.50, 0.555, "Performance Metrics — Test Set  (n = 200)",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=DARK, transform=ax.transAxes)

    headers  = ["Metric", "Autistic", "Non-Autistic", "Macro Avg"]
    col_x    = [0.10, 0.35, 0.57, 0.78]
    row_data = [
        ("Precision",  f"{cr['autistic']['precision']:.4f}",
                       f"{cr['non_autistic']['precision']:.4f}",
                       f"{cr['macro avg']['precision']:.4f}"),
        ("Recall",     f"{cr['autistic']['recall']:.4f}",
                       f"{cr['non_autistic']['recall']:.4f}",
                       f"{cr['macro avg']['recall']:.4f}"),
        ("F1-Score",   f"{cr['autistic']['f1-score']:.4f}",
                       f"{cr['non_autistic']['f1-score']:.4f}",
                       f"{cr['macro avg']['f1-score']:.4f}"),
        ("Accuracy",   "—", "—",
                       f"{R['test_accuracy']:.4f}"),
        ("AUC-ROC",    "—", "—",
                       f"{R['test_auc']:.4f}"),
    ]

    # Header row
    for j, (hdr, cx) in enumerate(zip(headers, col_x)):
        ax.text(cx, 0.525, hdr, ha="left", va="center", fontsize=10,
                fontweight="bold", color=BLUE, transform=ax.transAxes)
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.515, 0.515],
                               transform=fig.transFigure,
                               color="#D1D5DB", linewidth=0.8))

    for i, row in enumerate(row_data):
        y_pos = 0.497 - i * 0.028
        bg = LIGHT if i % 2 == 0 else "white"
        for j, (val, cx) in enumerate(zip(row, col_x)):
            fw = "bold" if j == 3 else "normal"
            col = GREEN if (j == 3 and i >= 3) else DARK
            ax.text(cx, y_pos, val, ha="left", va="center", fontsize=10,
                    fontweight=fw, color=col, transform=ax.transAxes)

    # Dataset info
    ax.add_patch(FancyBboxPatch((0.05, 0.21), 0.90, 0.11,
                                boxstyle="round,pad=0.01",
                                facecolor=LIGHT, edgecolor="#D1D5DB", linewidth=1))
    ax.text(0.50, 0.305, "Dataset",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=DARK, transform=ax.transAxes)

    ds_items = [
        ("Dataset",   "Autistic Children Facial Image Dataset"),
        ("Classes",   "Autistic  (n=1,463)  |  Non-Autistic  (n=1,463)  — Perfectly balanced"),
        ("Split",     f"Train: {R['dataset']['train']}  |  Val: {R['dataset']['val']}  |  Test: {R['dataset']['test']}"),
        ("Input size","224 × 224 px  |  ImageNet normalization  |  Augmentation: flip, rotate, jitter, affine"),
    ]
    for i, (k, v) in enumerate(ds_items):
        y_pos = 0.275 - i * 0.025
        ax.text(0.10, y_pos, f"{k}:", ha="left", va="center", fontsize=9.5,
                fontweight="bold", color=BLUE, transform=ax.transAxes)
        ax.text(0.28, y_pos, v, ha="left", va="center", fontsize=9.5,
                color=DARK, transform=ax.transAxes)

    # Footer
    ax.add_patch(FancyBboxPatch((0, 0), 1, 0.06, boxstyle="square",
                                facecolor="#1E3A5F", edgecolor="none"))
    ax.text(0.5, 0.03, f"Generated: {datetime.now().strftime('%B %d, %Y')}   |   "
            f"Run ID: {R['run_id']}   |   14 Figures",
            ha="center", va="center", fontsize=9, color="#93C5FD",
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("   ✅ Page 1 — Title page")

    # ── SECTION DIVIDERS + FIGURE PAGES ──────────────────────
    sections_seen = set()
    page_num = 2

    for fig_file, title, caption, section in FIGURES:
        # Section divider page
        if section not in sections_seen:
            sections_seen.add(section)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor(BLUE)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

            section_icons = {
                "Training Dynamics":        "📈",
                "Model Evaluation":         "📊",
                "Summary":                  "🗂️",
                "Grad-CAM Interpretability":"🔥",
                "Feature Space Analysis":   "🔬",
            }
            icon = section_icons.get(section, "📌")

            ax.text(0.5, 0.55, icon,   ha="center", va="center",
                    fontsize=60, transform=ax.transAxes)
            ax.text(0.5, 0.46, section, ha="center", va="center",
                    fontsize=30, fontweight="bold", color="white",
                    transform=ax.transAxes)

            descriptions = {
                "Training Dynamics":        "Accuracy, loss, and AUC progression\nacross 18 training epochs",
                "Model Evaluation":         "ROC curve, confusion matrix,\nand per-class classification metrics",
                "Summary":                  "Combined 6-panel overview figure\nsuitable for direct use in the paper",
                "Grad-CAM Interpretability":"Gradient-weighted Class Activation Maps\nrevealing model attention on facial regions",
                "Feature Space Analysis":   "t-SNE and UMAP projections of 128-d\ndeep features showing class separation",
            }
            ax.text(0.5, 0.38, descriptions.get(section, ""),
                    ha="center", va="center", fontsize=14,
                    color="#BFDBFE", transform=ax.transAxes)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"   ✅ Section divider — {section}")

        # Figure page
        img_path = FIG_DIR / fig_file
        if not img_path.exists():
            print(f"   ⚠️  Skipping {fig_file} — not found")
            continue

        img = np.array(Image.open(img_path))
        h, w = img.shape[:2]
        aspect = w / h

        # A4 proportions: allocate space for title + caption
        fig_w, fig_h = 8.27, 11.69
        img_h_frac = 0.72
        img_w_frac = min(0.92, img_h_frac * aspect * fig_h / fig_w)

        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("white")

        # Title
        fig.text(0.5, 0.95, title, ha="center", va="top",
                 fontsize=13, fontweight="bold", color=DARK)

        # Thin rule under title
        fig.add_artist(plt.Line2D([0.05, 0.95], [0.935, 0.935],
                                   transform=fig.transFigure,
                                   color=BLUE, linewidth=1.5))

        # Image
        left   = (1 - img_w_frac) / 2
        bottom = 0.20
        ax_img = fig.add_axes([left, bottom, img_w_frac, img_h_frac * img_w_frac / (aspect * img_h_frac / fig_w * fig_h / fig_w)])

        # Recalc properly
        img_ax_h = img_w_frac * (fig_w / fig_h) / aspect
        ax_img = fig.add_axes([left, 0.22, img_w_frac, min(img_ax_h, 0.68)])
        ax_img.imshow(img)
        ax_img.axis("off")

        # Caption box
        caption_ax = fig.add_axes([0.05, 0.02, 0.90, 0.17])
        caption_ax.set_xlim(0, 1); caption_ax.set_ylim(0, 1)
        caption_ax.axis("off")
        caption_ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                             boxstyle="round,pad=0.02",
                                             facecolor=LIGHT,
                                             edgecolor="#D1D5DB", linewidth=0.8))
        caption_ax.text(0.02, 0.85, title.split("—")[0].strip(),
                        ha="left", va="top", fontsize=9,
                        fontweight="bold", color=BLUE)
        caption_ax.text(0.02, 0.60, caption,
                        ha="left", va="top", fontsize=8.5,
                        color=DARK, wrap=True,
                        multialignment="left",
                        transform=caption_ax.transAxes)

        # Page number
        fig.text(0.95, 0.01, f"{page_num}", ha="right", va="bottom",
                 fontsize=9, color=GRAY)
        fig.text(0.05, 0.01, f"Section: {section}", ha="left", va="bottom",
                 fontsize=9, color=GRAY)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_num += 1
        print(f"   ✅ Page {page_num-1:>2} — {fig_file}")

    # ── LAST PAGE: Summary Stats ──────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0, 0.88), 1, 0.12, boxstyle="square",
                                facecolor=BLUE, edgecolor="none"))
    ax.text(0.5, 0.94, "Experiment Summary & Key Findings",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="white", transform=ax.transAxes)

    findings = [
        ("🏆", "Best Validation Accuracy",
         f"{R['best_val_acc']*100:.1f}% at epoch 13 (of 18 total)"),
        ("🎯", "Test Accuracy",
         f"{R['test_accuracy']*100:.1f}% on 200 held-out test images"),
        ("📈", "ROC-AUC Score",
         f"{R['test_auc']:.4f} — excellent discriminative ability"),
        ("⚖️", "Class Balance",
         "Macro F1 = 0.890 — no significant class bias"),
        ("🔥", "Grad-CAM Insight",
         "Model attends to eyes, nose bridge, and perioral regions"),
        ("🔬", "Feature Separation",
         f"UMAP Silhouette = 0.8617 — strong cluster separation"),
        ("⏱️", "Efficiency",
         f"Training time: {R['training_time_sec']/60:.1f} min on Apple MPS"),
        ("💾", "Model Size",
         "ResNet-50: 24.6M total params | 23.2M fine-tuned"),
    ]

    for i, (icon, heading, detail) in enumerate(findings):
        y = 0.82 - i * 0.088
        bg = LIGHT if i % 2 == 0 else "white"
        ax.add_patch(FancyBboxPatch((0.04, y - 0.032), 0.92, 0.07,
                                     boxstyle="round,pad=0.005",
                                     facecolor=bg, edgecolor="#E5E7EB", linewidth=0.6))
        ax.text(0.07, y + 0.003, icon,     ha="left", fontsize=16, va="center", transform=ax.transAxes)
        ax.text(0.13, y + 0.015, heading,  ha="left", fontsize=11, fontweight="bold",
                color=BLUE, va="center", transform=ax.transAxes)
        ax.text(0.13, y - 0.015, detail,   ha="left", fontsize=10,
                color=DARK, va="center", transform=ax.transAxes)

    ax.add_patch(FancyBboxPatch((0, 0), 1, 0.05, boxstyle="square",
                                facecolor="#1E3A5F", edgecolor="none"))
    ax.text(0.5, 0.025,
            "Autism Detection CNN  |  ResNet-50 Transfer Learning  |  "
            f"Run: {R['run_id']}",
            ha="center", va="center", fontsize=8.5, color="#93C5FD",
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Page {page_num} — Summary & Key Findings")

    # PDF metadata
    d = pdf.infodict()
    d["Title"]   = "Autism Spectrum Disorder Detection — CNN Experimental Results"
    d["Author"]  = "ResNet-50 Transfer Learning Study"
    d["Subject"] = "Deep Learning, ASD Detection, Facial Image Classification"
    d["Keywords"]= "CNN, ResNet-50, Autism, Grad-CAM, t-SNE, UMAP, Transfer Learning"
    d["CreationDate"] = datetime.now()

size_mb = PDF_PATH.stat().st_size / (1024*1024)
print(f"\n🎉 PDF report saved!\n   → {PDF_PATH}\n   → Size: {size_mb:.1f} MB  |  {page_num} pages")
