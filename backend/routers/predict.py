"""
POST /predict/image
Image upload endpoint — runs the trained CNN-BiLSTM multimodal model's
visual pathway and returns ASD risk prediction with Grad-CAM heatmap.
"""

import io
import base64
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])

MODEL_PATH = Path(__file__).resolve().parents[1] / "ml" / "checkpoints" / "asd_cnnrnn_v1.pt"
DEVICE     = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")
IMG_SIZE    = 64
DISPLAY_SIZE = 224

_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=1)
def _load_model():
    from ml.model import ASDScreeningModel

    if not MODEL_PATH.exists():
        logger.warning("Model not found at %s", MODEL_PATH)
        return None

    ckpt   = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)
    config = ckpt.get("model_config", {})

    model = ASDScreeningModel(**config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("ASDScreeningModel loaded from %s on %s", MODEL_PATH, DEVICE)
    return model


class _GradCAM:
    def __init__(self, model):
        self.grads = None
        self.acts  = None
        target = model.visual_encoder.features[7]
        target.register_forward_hook(
            lambda m, i, o: setattr(self, "acts", o))
        target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, scalar_output, display_size):
        scalar_output.backward(retain_graph=True)
        w   = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, (display_size, display_size),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class PredictionResult(BaseModel):
    prediction:        str
    label:             str
    confidence:        float
    autistic_prob:     float
    non_autistic_prob: float
    gradcam_image:     str
    model_path:        str
    device:            str


@router.post("/image", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload must be an image file.")

    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image — ensure it is a valid JPG/PNG.")

    display_img = pil_img.resize((DISPLAY_SIZE, DISPLAY_SIZE))

    model = _load_model()
    if model is None:
        raise HTTPException(503, f"Model checkpoint not found at {MODEL_PATH}")

    img_tensor = _tf(pil_img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    dummy_audio = torch.zeros(1, 1, 1, 40, 128, device=DEVICE)
    dummy_quest = torch.zeros(1, 1, 10, device=DEVICE)
    dummy_text  = torch.zeros(1, 1, 50, device=DEVICE)

    img_tensor.requires_grad_(True)
    gcam = _GradCAM(model)

    model.zero_grad()
    outputs = model(img_tensor, dummy_audio, dummy_quest, dummy_text)

    asd_prob     = outputs["video"].squeeze().item()
    non_asd_prob = 1.0 - asd_prob
    is_autistic  = asd_prob >= 0.5
    pred_name    = "autistic" if is_autistic else "non_autistic"
    confidence   = asd_prob if is_autistic else non_asd_prob

    cam = gcam(outputs["video"].squeeze(), DISPLAY_SIZE)

    import matplotlib.cm as cm_module
    img_np  = np.array(display_img).astype(float) / 255.0
    heatmap = cm_module.jet(cam)[:, :, :3]
    overlay = np.clip(0.55 * img_np + 0.45 * heatmap, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay_uint8).save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode()

    return PredictionResult(
        prediction        = pred_name,
        label             = "Autistic" if is_autistic else "Non-Autistic",
        confidence        = round(confidence, 4),
        autistic_prob     = round(asd_prob, 4),
        non_autistic_prob = round(non_asd_prob, 4),
        gradcam_image     = overlay_b64,
        model_path        = str(MODEL_PATH),
        device            = str(DEVICE),
    )
