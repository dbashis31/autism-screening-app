"""FastAPI shared dependencies."""

from fastapi import Header, HTTPException
from sqlalchemy.orm import Session as DBSession
from .database import SessionLocal


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_role(*allowed: str):
    """Returns a dependency that enforces the X-Role header."""
    def _check(x_role: str = Header(default="caregiver")):
        if x_role not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{x_role}' is not permitted for this endpoint. "
                       f"Required: {list(allowed)}",
            )
        return x_role
    return _check
