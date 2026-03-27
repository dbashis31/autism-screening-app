"""DB-backed audit logging and abstention history helpers."""

import datetime
from typing import Callable
from sqlalchemy.orm import Session as DBSession
from .models import AuditLog, AbstentionHistory


def make_log_fn(db: DBSession, _session_id: str) -> Callable:
    """Returns a log_fn closure that persists entries to the audit_log table."""
    def log_fn(agent: str, session_id: str, decision: str, reason: str,
               details: dict | None = None) -> None:
        entry = AuditLog(
            timestamp=datetime.datetime.utcnow(),
            agent=agent,
            session_id=session_id,
            decision=decision,
            reason=reason,
            details=details or {},
        )
        db.add(entry)
        db.commit()
    return log_fn


def write_abstention(db: DBSession, child_id: str, session_id: str,
                     reason: str) -> None:
    row = AbstentionHistory(
        child_id=child_id,
        session_id=session_id,
        reason=reason,
        created_at=datetime.datetime.utcnow(),
    )
    db.add(row)
    db.commit()


def get_abstention_history(db: DBSession, child_id: str) -> list[dict]:
    rows = (
        db.query(AbstentionHistory)
        .filter(AbstentionHistory.child_id == child_id)
        .order_by(AbstentionHistory.created_at)
        .all()
    )
    return [{"session_id": r.session_id, "reason": r.reason,
             "created_at": r.created_at.isoformat()} for r in rows]


def get_audit_log(db: DBSession, agent: str | None = None,
                  decision: str | None = None, session_id: str | None = None,
                  skip: int = 0, limit: int = 50) -> tuple[int, list[dict]]:
    q = db.query(AuditLog)
    if agent:
        q = q.filter(AuditLog.agent == agent)
    if decision:
        q = q.filter(AuditLog.decision == decision)
    if session_id:
        q = q.filter(AuditLog.session_id == session_id)
    total = q.count()
    rows = q.order_by(AuditLog.timestamp.desc()).offset(skip).limit(limit).all()
    return total, [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "agent": r.agent,
            "session_id": r.session_id,
            "decision": r.decision,
            "reason": r.reason,
            "details": r.details or {},
        }
        for r in rows
    ]
