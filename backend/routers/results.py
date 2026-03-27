from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session as DBSession
from ..models import Session
from ..schemas import CaregiverResult, ClinicianResult
from ..dependencies import get_db

router = APIRouter(prefix="/sessions", tags=["results"])


@router.get("/{session_id}/results")
def get_results(
    session_id: str,
    x_role: str = Header(default="caregiver"),
    db: DBSession = Depends(get_db),
):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(404, "Session not found")
    if s.pipeline_status == "pending":
        raise HTTPException(400, "Session has not been submitted yet")

    r = s.pipeline_result or {}

    if x_role == "clinician":
        return ClinicianResult(
            session_id=session_id,
            child_id=s.child_id,
            pipeline_status=s.pipeline_status,
            caregiver_report=r.get("caregiver_report"),
            clinician_report=r.get("clinician_report"),
            enabled_modalities=r.get("enabled_modalities", []),
            applicability_warnings=r.get("applicability_warnings", []),
            abstention_reason=r.get("abstention_reason"),
            confidence_scores=r.get("confidence_scores", {}),
            block_reason=r.get("block_reason"),
        )

    return CaregiverResult(
        session_id=session_id,
        child_id=s.child_id,
        pipeline_status=s.pipeline_status,
        caregiver_report=r.get("caregiver_report"),
    )
