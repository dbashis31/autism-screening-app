"""
POST /predict/image
Simple image upload endpoint — runs the trained ResNet-50 and returns
autistic / non-autistic prediction with confidence + Grad-CAM heatmap.
"""

import io
import base64
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parents[2] / \
    "data/model_outputs/run_20260311_135435/best_model.pth"
DEVICE     = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")
CLASS_NAMES = ["autistic", "non_autistic"]
IMG_SIZE    = 224

# ── Transform ─────────────────────────────────────────────────────────────────
_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

_inv_norm = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std =[1/0.229,       1/0.224,      1/0.225]
)

# ── Model builder ─────────────────────────────────────────────────────────────
def _build_model() -> nn.Module:
    m = models.resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Linear(in_f, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 2),
    )
    return m

@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        logger.warning("Model not found at %s", MODEL_PATH)
        return None
    ckpt  = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model = _build_model().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("ResNet-50 loaded from %s on %s", MODEL_PATH, DEVICE)
    return model

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class _GradCAM:
    def __init__(self, model):
        self.grads = None
        self.acts  = None
        model.layer4[-1].register_forward_hook(
            lambda m, i, o: setattr(self, "acts", o))
        model.layer4[-1].register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, model, x, cls_idx):
        model.zero_grad()
        out   = model(x)
        out[0, cls_idx].backward()
        w   = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    prediction:        str           # "autistic" | "non_autistic"
    label:             str           # human-readable
    confidence:        float         # 0-1
    autistic_prob:     float
    non_autistic_prob: float
    gradcam_image:     str           # base64 PNG overlay
    model_path:        str
    device:            str

# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.post("/image", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    # Validate
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload must be an image file.")

    # Read & preprocess
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image — ensure it is a valid JPG/PNG.")

    # Resize for display
    display_img = pil_img.resize((IMG_SIZE, IMG_SIZE))

    model = _load_model()
    if model is None:
        raise HTTPException(503, f"Model checkpoint not found at {MODEL_PATH}")

    tensor = _tf(pil_img).unsqueeze(0).to(DEVICE).requires_grad_(True)

    # Grad-CAM + prediction
    gcam  = _GradCAM(model)

    out   = model(tensor)
    probs = torch.softmax(out, dim=1)[0]
    pred_idx  = int(probs.argmax())
    pred_name = CLASS_NAMES[pred_idx]

    # Compute Grad-CAM
    cam = gcam(model, tensor, pred_idx)

    # Build overlay image
    import matplotlib.cm as cm_module
    img_np  = np.array(display_img).astype(float) / 255.0
    heatmap = cm_module.jet(cam)[:, :, :3]
    overlay = np.clip(0.55 * img_np + 0.45 * heatmap, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    # Encode overlay as base64
    buf = io.BytesIO()
    Image.fromarray(overlay_uint8).save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode()

    return PredictionResult(
        prediction        = pred_name,
        label             = "Autistic" if pred_idx == 0 else "Non-Autistic",
        confidence        = round(float(probs[pred_idx]), 4),
        autistic_prob     = round(float(probs[0]), 4),
        non_autistic_prob = round(float(probs[1]), 4),
        gradcam_image     = overlay_b64,
        model_path        = str(MODEL_PATH),
        device            = str(DEVICE),
    )
