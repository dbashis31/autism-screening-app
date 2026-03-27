"""Dev utility endpoints — expose the 10 research scenarios for UI demo pre-fill."""

from fastapi import APIRouter
from ..agents.constants import APPROVED_CAREGIVER_VOCAB

router = APIRouter(prefix="/dev", tags=["dev"])

_VALID_CONSENT = {"permitted_ops": ["inference"], "expiry_date": "2027-12-31"}
_HIGH_CONF = {"audio": 0.88, "video": 0.91, "text": 0.77, "questionnaire": 0.81}

SCENARIOS = {
    "S-01": {
        "label": "S-01: No consent",
        "consent": None,
        "audio_snr_db": None, "child_age_months": None,
        "cross_modal_conflict": False, "force_abstain": False,
        "report_type": "standard",
        "confidence_scores": None,
        "consent_scope_change": None,
    },
    "S-02": {
        "label": "S-02: Low-SNR audio (10 dB)",
        "consent": _VALID_CONSENT,
        "audio_snr_db": 10, "child_age_months": None,
        "cross_modal_conflict": False, "force_abstain": False,
        "report_type": "standard",
        "confidence_scores": {"video": 0.84, "text": 0.76, "questionnaire": 0.80},
        "consent_scope_change": None,
    },
    "S-03": {
        "label": "S-03: High confidence — full report",
        "consent": _VALID_CONSENT,
        "audio_snr_db": None, "child_age_months": None,
        "cross_modal_conflict": False, "force_abstain": False,
        "report_type": "standard",
        "confidence_scores": _HIGH_CONF,
        "consent_scope_change": None,
    },
    "S-04": {
        "label": "S-04: Inter-modal conflict → abstention",
        "consent": _VALID_CONSENT,
        "audio_snr_db": None, "child_age_months": None,
        "cross_modal_conflict": True, "force_abstain": False,
        "report_type": "standard",
        "confidence_scores": {"audio": 0.88, "video": 0.72, "text": 0.55, "questionnaire": 0.50},
        "consent_scope_change": None,
    },
    "S-05": {
        "label": "S-05: Caregiver requests clinician report (blocked)",
        "consent": _VALID_CONSENT,
        "audio_snr_db": None, "child_age_months": None,
        "cross_modal_conflict": False, "force_abstain": False,
        "report_type": "clinician_report",
        "confidence_scores": _HIGH_CONF,
        "consent_scope_change": None,
    },
    "S-06": {
        "label": "S-06: Force abstention (for escalation testing)",
        "consent": _VALID_CONSENT,
        "audio_snr_db": None, "child_age_months": None,
        "cross_modal_conflict": False, "force_abstain": True,
        "report_type": "standard",
        "confidence_scores": None,
        "consent_scope_change": None,
    },
    "S-07": {
        "label": "S-07: Unsigned model",
        "consent": _VALID_CONSENT,
        "audio_snr_db": None, "child_age_months": None,
        "cross_modal_conflict": False, "force_abstain": False,
        "report_type": "standard",
        "confidence_scores": None,
        "consent_scope_change": None,
        "_model_id_override": "model-v3.0-unsigned",
    },
    "S-08": {
        "label": "S-08: Standard — vocabulary check",
        "consent": _VALID_CONSENT,
        "audio_snr_db": None, "child_age_months": None,
        "cross_modal_conflict": False, "force_abstain": False,
        "report_type": "standard",
        "confidence_scores": _HIGH_CONF,
        "consent_scope_change": None,
    },
    "S-09": {
        "label": "S-09: Age boundary (18 months)",
        "consent": _VALID_CONSENT,
        "audio_snr_db": None, "child_age_months": 18,
        "cross_modal_conflict": False, "force_abstain": False,
        "report_type": "standard",
        "confidence_scores": _HIGH_CONF,
        "consent_scope_change": None,
    },
    "S-10": {
        "label": "S-10: Mid-session consent scope change",
        "consent": _VALID_CONSENT,
        "audio_snr_db": None, "child_age_months": None,
        "cross_modal_conflict": False, "force_abstain": False,
        "report_type": "standard",
        "confidence_scores": {"video": 0.84, "text": 0.76, "questionnaire": 0.80},
        "consent_scope_change": {"removed_modalities": ["audio"]},
    },
}


@router.get("/scenarios")
def get_scenarios():
    return SCENARIOS


@router.get("/approved-vocab")
def get_approved_vocab():
    return APPROVED_CAREGIVER_VOCAB
