from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session as DBSession
from ..dependencies import get_db, require_role
from ..audit import get_audit_log
from ..metrics import compute_metrics
from ..schemas import AuditLogResponse, AuditEntry

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/audit-log", response_model=AuditLogResponse,
            dependencies=[Depends(require_role("admin"))])
def list_audit_log(
    agent: str | None = Query(default=None),
    decision: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, le=200),
    db: DBSession = Depends(get_db),
):
    total, entries = get_audit_log(db, agent=agent, decision=decision,
                                   session_id=session_id, skip=skip, limit=limit)
    return AuditLogResponse(total=total,
                            entries=[AuditEntry(**e) for e in entries])


@router.get("/metrics", dependencies=[Depends(require_role("admin"))])
def get_metrics(db: DBSession = Depends(get_db)):
    return compute_metrics(db)
