"""POST /sessions/{id}/submit — assembles scenario, runs pipeline, persists result."""

import random
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session as DBSession
from ..models import Session, ConsentRecord
from ..schemas import SubmitPayload, CaregiverResult, ClinicianResult
from ..dependencies import get_db
from ..pipeline import run_pipeline

# Use real CNN-BiLSTM inference when a checkpoint is available;
# falls back to random mock scores automatically if the checkpoint is missing.
try:
    from ..ml import ml_inference as _get_confidence
except Exception:  # noqa: BLE001 — torch not installed yet
    import random as _random
    def _get_confidence(modalities, **_kw):  # type: ignore[misc]
        return {m: round(_random.uniform(0.70, 0.95), 3) for m in modalities}

router = APIRouter(prefix="/sessions", tags=["submit"])


@router.post("/{session_id}/submit")
def submit_session(
    session_id: str,
    payload: SubmitPayload,
    x_role: str = Header(default="caregiver"),
    db: DBSession = Depends(get_db),
):
    # Load session
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(404, "Session not found")

    # Load latest consent record
    consent_row = (
        db.query(ConsentRecord)
        .filter(ConsentRecord.session_id == session_id)
        .order_by(ConsentRecord.id.desc())
        .first()
    )
    consent_record = (
        {"permitted_ops": consent_row.permitted_ops,
         "expiry_date": consent_row.expiry_date}
        if consent_row else None
    )

    # Build confidence scores — real ML model, or mock fallback if no checkpoint
    confidence_scores = payload.confidence_scores or _get_confidence(payload.modalities)

    # Build scenario dict matching agent expectations
    scenario = {
        "session_id": session_id,
        "child_id": s.child_id,
        "role": x_role,
        "model_id": s.model_id,
        "requested_operation": "inference",
        "consent_record": consent_record,
        "modalities": payload.modalities,
        "audio_snr_db": payload.audio_snr_db,
        "child_age_months": payload.child_age_months,
        "report_type": payload.report_type,
        "cross_modal_conflict": payload.cross_modal_conflict,
        "force_abstain": payload.force_abstain,
        "confidence_scores": confidence_scores,
        "consent_scope_change": payload.consent_scope_change,
    }

    # Run the five-agent pipeline
    result = run_pipeline(scenario, db)

    # Persist result to session row
    s.pipeline_status = result["pipeline_status"]
    s.pipeline_result = result
    s.modalities = payload.modalities
    s.audio_snr_db = payload.audio_snr_db
    s.child_age_months = payload.child_age_months
    s.report_type = payload.report_type
    s.cross_modal_conflict = payload.cross_modal_conflict
    s.confidence_scores = confidence_scores
    db.commit()

    # Return role-gated response
    if x_role == "clinician":
        return ClinicianResult(
            session_id=session_id,
            child_id=s.child_id,
            pipeline_status=result["pipeline_status"],
            caregiver_report=result.get("caregiver_report"),
            clinician_report=result.get("clinician_report"),
            enabled_modalities=result.get("enabled_modalities", []),
            applicability_warnings=result.get("applicability_warnings", []),
            abstention_reason=result.get("abstention_reason"),
            confidence_scores=result.get("confidence_scores", {}),
            block_reason=result.get("block_reason"),
        )

    return CaregiverResult(
        session_id=session_id,
        child_id=s.child_id,
        pipeline_status=result["pipeline_status"],
        caregiver_report=result.get("caregiver_report"),
    )
