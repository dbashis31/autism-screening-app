"""SQLAlchemy ORM table definitions."""

import datetime
from sqlalchemy import String, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    child_id: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(20), default="caregiver")
    model_id: Mapped[str] = mapped_column(String(64), default="model-v2.1-signed")
    modalities: Mapped[list] = mapped_column(JSON, default=list)
    audio_snr_db: Mapped[float | None] = mapped_column(Float, nullable=True)
    child_age_months: Mapped[int | None] = mapped_column(Integer, nullable=True)
    report_type: Mapped[str] = mapped_column(String(32), default="standard")
    requested_operation: Mapped[str] = mapped_column(String(32), default="inference")
    cross_modal_conflict: Mapped[bool] = mapped_column(Boolean, default=False)
    force_abstain: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence_scores: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    consent_scope_change: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    pipeline_status: Mapped[str] = mapped_column(String(20), default="pending")
    pipeline_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )


class ConsentRecord(Base):
    __tablename__ = "consent_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"))
    permitted_ops: Mapped[list] = mapped_column(JSON, default=list)
    expiry_date: Mapped[str] = mapped_column(String(10))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    agent: Mapped[str] = mapped_column(String(50))
    session_id: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(50))
    reason: Mapped[str] = mapped_column(String(500))
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class AbstentionHistory(Base):
    __tablename__ = "abstention_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    child_id: Mapped[str] = mapped_column(String(64))
    session_id: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
