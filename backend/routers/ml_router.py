"""
ML management endpoints (admin-only).

POST /ml/train   — kick off a background training run
GET  /ml/status  — checkpoint info, training metrics, model config
GET  /ml/logs    — last N training log lines
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from ..dependencies import require_role

router = APIRouter(prefix="/ml", tags=["ml"])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
_CKPT_PATH = _BASE_DIR / "ml" / "checkpoints" / "asd_cnnrnn_v1.pt"
_FIGURES_DIR = _BASE_DIR / "ml" / "figures"
_METRICS_JSON = _FIGURES_DIR / "metrics_summary.json"

# ---------------------------------------------------------------------------
# In-process training state (singleton, safe for single-worker uvicorn)
# ---------------------------------------------------------------------------
_training_state: dict = {
    "status": "idle",       # idle | running | done | failed
    "started_at": None,
    "finished_at": None,
    "error": None,
    "epochs_done": 0,
    "log_lines": [],
}
_training_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    data_source: str = "synthetic"   # "synthetic" | "kaggle"
    data_root: Optional[str] = None  # only needed for kaggle
    epochs: int = 50
    batch_size: int = 32
    lr: float = 3e-4
    n_mc_passes: int = 20
    generate_figures: bool = True


class TrainResponse(BaseModel):
    status: str
    message: str
    started_at: Optional[str] = None


class MLStatus(BaseModel):
    checkpoint_exists: bool
    checkpoint_path: str
    checkpoint_size_mb: Optional[float] = None
    temperature: Optional[float] = None
    training_metadata: Optional[dict] = None
    ml_model_config: Optional[dict] = None
    training_state: dict
    figures: list[str]
    paper_metrics: Optional[dict] = None


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------
def _training_worker(req: TrainRequest) -> None:
    """Runs in a daemon thread; updates _training_state throughout."""
    with _training_lock:
        _training_state.update(
            status="running",
            started_at=datetime.utcnow().isoformat(),
            finished_at=None,
            error=None,
            epochs_done=0,
            log_lines=[],
        )

    try:
        # Import lazily so the router loads even without torch installed
        from ..ml.train import train as run_train  # noqa: PLC0415

        def _progress_cb(epoch: int, total: int, val_auc: float) -> None:
            with _training_lock:
                _training_state["epochs_done"] = epoch
                line = (
                    f"[{datetime.utcnow().strftime('%H:%M:%S')}] "
                    f"Epoch {epoch}/{total}  val_auc={val_auc:.4f}"
                )
                _training_state["log_lines"].append(line)
                # Keep last 200 lines to avoid unbounded memory growth
                _training_state["log_lines"] = _training_state["log_lines"][-200:]

        run_train(
            data_source=req.data_source,
            data_root=req.data_root,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            progress_cb=_progress_cb,
        )

        if req.generate_figures:
            from ..ml.paper_metrics import generate_all_figures  # noqa: PLC0415

            _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            generate_all_figures(
                checkpoint_path=str(_CKPT_PATH),
                output_dir=str(_FIGURES_DIR),
                n_mc_passes=req.n_mc_passes,
                threshold=0.65,
            )

        with _training_lock:
            _training_state.update(
                status="done",
                finished_at=datetime.utcnow().isoformat(),
            )

    except Exception as exc:  # noqa: BLE001
        with _training_lock:
            _training_state.update(
                status="failed",
                finished_at=datetime.utcnow().isoformat(),
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/train", response_model=TrainResponse)
def trigger_training(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
    _role: str = Depends(require_role("admin")),
):
    """
    Kick off model training in a background thread (admin-only).
    Returns immediately; poll GET /ml/status for progress.
    """
    with _training_lock:
        if _training_state["status"] == "running":
            raise HTTPException(
                409, "Training already in progress. Poll GET /ml/status."
            )

    background_tasks.add_task(_training_worker, req)

    return TrainResponse(
        status="accepted",
        message=(
            f"Training started: data_source={req.data_source}, "
            f"epochs={req.epochs}. Poll GET /ml/status for progress."
        ),
        started_at=datetime.utcnow().isoformat(),
    )


@router.get("/status", response_model=MLStatus)
def get_ml_status(
    _role: str = Depends(require_role("admin")),
):
    """
    Return ML system status: checkpoint existence, temperature, training
    metadata, generated figures, and paper metrics. Admin-only.
    """
    # ── checkpoint info ──────────────────────────────────────────────────
    ckpt_exists = _CKPT_PATH.exists()
    ckpt_size_mb: Optional[float] = None
    temperature: Optional[float] = None
    training_metadata: Optional[dict] = None
    model_cfg: Optional[dict] = None

    if ckpt_exists:
        ckpt_size_mb = round(_CKPT_PATH.stat().st_size / 1e6, 2)
        try:
            checkpoint = torch.load(str(_CKPT_PATH), map_location="cpu",
                                    weights_only=False)
            temperature = float(checkpoint.get("temperature", 1.0))
            training_metadata = checkpoint.get("training_metadata")
            model_cfg = checkpoint.get("model_config")
        except Exception:  # noqa: BLE001
            pass  # checkpoint may be mid-write; skip gracefully

    # ── generated figures ────────────────────────────────────────────────
    figures: list[str] = []
    if _FIGURES_DIR.exists():
        figures = sorted(
            p.name
            for p in _FIGURES_DIR.iterdir()
            if p.suffix in {".pdf", ".png"}
        )

    # ── paper metrics summary JSON ───────────────────────────────────────
    paper_metrics: Optional[dict] = None
    if _METRICS_JSON.exists():
        try:
            paper_metrics = json.loads(_METRICS_JSON.read_text())
        except Exception:  # noqa: BLE001
            pass

    with _training_lock:
        state_copy = dict(_training_state)

    return MLStatus(
        checkpoint_exists=ckpt_exists,
        checkpoint_path=str(_CKPT_PATH),
        checkpoint_size_mb=ckpt_size_mb,
        temperature=temperature,
        training_metadata=training_metadata,
        ml_model_config=model_cfg,
        training_state=state_copy,
        figures=figures,
        paper_metrics=paper_metrics,
    )


@router.get("/logs")
def get_training_logs(
    last_n: int = 50,
    _role: str = Depends(require_role("admin")),
):
    """
    Return the last N training log lines (admin-only).
    Useful for polling progress without WebSockets.
    """
    with _training_lock:
        lines = list(_training_state.get("log_lines", []))
        status = _training_state["status"]
        epochs_done = _training_state["epochs_done"]

    return {
        "status": status,
        "epochs_done": epochs_done,
        "lines": lines[-last_n:],
    }
