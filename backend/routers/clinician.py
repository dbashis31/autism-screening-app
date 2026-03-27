from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func
from ..models import Session, AbstentionHistory
from ..schemas import EscalationItem, ClinicianResult
from ..dependencies import get_db, require_role

router = APIRouter(prefix="/clinician", tags=["clinician"])


@router.get("/queue", dependencies=[Depends(require_role("clinician", "admin"))])
def get_escalation_queue(db: DBSession = Depends(get_db)):
    """Children with 2+ consecutive abstentions requiring clinician review."""
    rows = (
        db.query(
            AbstentionHistory.child_id,
            func.count(AbstentionHistory.id).label("count"),
            func.max(AbstentionHistory.session_id).label("last_session"),
            func.max(AbstentionHistory.reason).label("last_reason"),
            func.max(AbstentionHistory.created_at).label("last_created"),
        )
        .group_by(AbstentionHistory.child_id)
        .having(func.count(AbstentionHistory.id) >= 2)
        .all()
    )
    return [
        EscalationItem(
            child_id=r.child_id,
            abstention_count=r.count,
            last_session_id=r.last_session,
            last_reason=r.last_reason,
            last_abstention=r.last_created.isoformat(),
        )
        for r in rows
    ]


@router.get("/sessions", dependencies=[Depends(require_role("clinician", "admin"))])
def list_clinician_sessions(skip: int = 0, limit: int = 20,
                             db: DBSession = Depends(get_db)):
    sessions = (
        db.query(Session)
        .filter(Session.pipeline_status.in_(["complete", "blocked", "abstained"]))
        .order_by(Session.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    results = []
    for s in sessions:
        r = s.pipeline_result or {}
        results.append(ClinicianResult(
            id=s.id,
            session_id=s.id,
            child_id=s.child_id,
            pipeline_status=s.pipeline_status,
            created_at=s.created_at.isoformat() if s.created_at else "",
            modalities=s.modalities or [],
            caregiver_report=r.get("caregiver_report"),
            clinician_report=r.get("clinician_report"),
            enabled_modalities=r.get("enabled_modalities", s.modalities or []),
            applicability_warnings=r.get("applicability_warnings", []),
            abstention_reason=r.get("abstention_reason"),
            confidence_scores=s.confidence_scores or r.get("confidence_scores", {}),
            block_reason=r.get("block_reason"),
            pipeline_result=r,
        ))
    return results


@router.get("/sessions/{session_id}/abstentions",
            dependencies=[Depends(require_role("clinician", "admin"))])
def get_abstentions(session_id: str, db: DBSession = Depends(get_db)):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(404, "Session not found")
    rows = (
        db.query(AbstentionHistory)
        .filter(AbstentionHistory.child_id == s.child_id)
        .order_by(AbstentionHistory.created_at)
        .all()
    )
    return {
        "child_id": s.child_id,
        "abstentions": [
            {"session_id": r.session_id, "reason": r.reason,
             "created_at": r.created_at.isoformat()}
            for r in rows
        ],
    }
