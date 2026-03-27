import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from ..database import SessionLocal
from ..models import Session
from ..schemas import SessionCreate
from ..dependencies import get_db

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("")
def create_session(payload: SessionCreate, db: DBSession = Depends(get_db)):
    sid = str(uuid.uuid4())
    session = Session(
        id=sid,
        child_id=payload.child_id,
        role=payload.role,
        model_id=payload.model_id,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"session_id": sid, "child_id": payload.child_id,
            "role": payload.role, "created_at": session.created_at.isoformat()}


@router.get("/{session_id}")
def get_session(session_id: str, db: DBSession = Depends(get_db)):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": s.id, "child_id": s.child_id, "role": s.role,
        "model_id": s.model_id, "pipeline_status": s.pipeline_status,
        "created_at": s.created_at.isoformat(),
    }
