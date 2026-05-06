"""Shared constants for all governance agents."""

# ── Model registry (now keyed by modality support) ────────────────────────────

APPROVED_MODEL_REGISTRY: dict[str, dict] = {
    "model-v2.1-signed": {
        "modalities": {"audio", "video", "text", "questionnaire"},
        "version": "2.1",
        "signed": True,
        "status": "deployed",
    },
    "model-v2.0-signed": {
        "modalities": {"audio", "video", "text", "questionnaire"},
        "version": "2.0",
        "signed": True,
        "status": "deployed",
    },
    "asd-cnnrnn-v1": {
        "modalities": {"audio", "video"},
        "version": "1.0",
        "signed": True,
        "status": "deployed",
    },
}

# ── Caregiver vocabulary (governance-approved, non-diagnostic) ────────────────

APPROVED_CAREGIVER_VOCAB: list[str] = [
    "Your child's developmental screen is complete. A clinician will review and follow up with you.",
    "Additional information is needed before results can be shared. A clinician will contact you.",
    "Your child's screening session has been logged. No further action is needed at this time.",
]

# ── Role-based access control ─────────────────────────────────────────────────

VALID_ROLES: set[str] = {"clinician", "caregiver", "researcher", "admin"}

ROLE_DATA_ACCESS: dict[str, set[str]] = {
    "clinician":  {"screening_results", "confidence_scores", "modality_data", "audit_log", "demographic_data"},
    "caregiver":  {"screening_results"},
    "researcher": {"screening_results", "confidence_scores", "demographic_data"},
    "admin":      {"screening_results", "confidence_scores", "modality_data", "audit_log", "demographic_data"},
}

# ── Thresholds ────────────────────────────────────────────────────────────────

BIAS_RISK_THRESHOLD: float = 0.7          # τ_bias — ABSTAIN if bias score exceeds this
CONFIDENCE_THRESHOLD: float = 0.65        # τ_conf — ABSTAIN if confidence below this
INCONSISTENCY_THRESHOLD: float = 0.3      # τ_var  — ABSTAIN if cross-modal variance exceeds this
AUDIO_SNR_THRESHOLD_DB: float = 15.0
AGE_RANGE_MONTHS: tuple[int, int] = (18, 72)
MIN_MODALITY_COUNT: int = 2

# ── Diagnostic language filter ────────────────────────────────────────────────

DIAGNOSTIC_TERMS: set[str] = {
    "autism", "asd", "autistic", "diagnosis", "diagnosed", "diagnostic",
    "disorder", "spectrum", "clinical finding", "pathology", "deficit",
    "impairment", "abnormal", "atypical", "symptomatic", "comorbid",
    "prognosis", "etiology",
}
