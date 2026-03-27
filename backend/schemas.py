"""Pydantic request / response schemas."""

from pydantic import BaseModel, Field
from typing import Any


class SessionCreate(BaseModel):
    child_id: str
    role: str = "caregiver"
    model_id: str = "model-v2.1-signed"


class ConsentCreate(BaseModel):
    permitted_ops: list[str] = ["inference"]
    expiry_date: str  # "YYYY-MM-DD"


class SubmitPayload(BaseModel):
    modalities: list[str] = ["audio", "video", "text", "questionnaire"]
    audio_snr_db: float | None = None
    child_age_months: int | None = None
    report_type: str = "standard"
    cross_modal_conflict: bool = False
    force_abstain: bool = False
    confidence_scores: dict[str, float] | None = None
    consent_scope_change: dict | None = None


class CaregiverResult(BaseModel):
    session_id: str
    child_id: str
    pipeline_status: str
    caregiver_report: str | None


class ClinicianResult(BaseModel):
    id: str = ""                    # session UUID (alias for session_id in list views)
    session_id: str
    child_id: str
    pipeline_status: str
    created_at: str = ""
    modalities: list[str] = []
    caregiver_report: str | None
    clinician_report: dict | None
    enabled_modalities: list[str]
    applicability_warnings: list[str]
    abstention_reason: str | None
    confidence_scores: dict[str, float]
    block_reason: str | None
    pipeline_result: dict = {}


class AuditEntry(BaseModel):
    id: int
    timestamp: str
    agent: str
    session_id: str
    decision: str
    reason: str
    details: dict


class AuditLogResponse(BaseModel):
    total: int
    entries: list[AuditEntry]


class EscalationItem(BaseModel):
    child_id: str
    abstention_count: int
    last_session_id: str
    last_reason: str
    last_abstention: str      # ISO timestamp of most recent abstention
