from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from ..models import Session, ConsentRecord
from ..schemas import ConsentCreate
from ..dependencies import get_db

router = APIRouter(prefix="/sessions", tags=["consent"])


@router.post("/{session_id}/consent")
def add_consent(session_id: str, payload: ConsentCreate,
                db: DBSession = Depends(get_db)):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(404, "Session not found")

    record = ConsentRecord(
        session_id=session_id,
        permitted_ops=payload.permitted_ops,
        expiry_date=payload.expiry_date,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"session_id": session_id, "consent_id": record.id, "status": "saved"}
