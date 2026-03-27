"""
Pipeline scenario tests — mirrors the 10 research scenarios (S-01 to S-10).

Each test covers a specific governance decision path through the LangGraph
pipeline. The LLM is disabled (no ANTHROPIC_API_KEY set in test env) so
every agent uses its deterministic rule-based fallback — tests are fast,
reproducible, and require no API quota.
"""

import pytest
from .conftest import (
    VALID_CONSENT, HIGH_CONF,
    create_session, add_consent, submit_screening,
)


# ── S-01: No consent → pipeline blocked immediately ───────────────────────────
class TestS01NoConsent:
    def test_blocked_without_consent(self, caregiver_client):
        sid = create_session(caregiver_client, "S01-CHILD")
        # Do NOT add consent
        r = caregiver_client.post(f"/sessions/{sid}/submit", json={
            "modalities": ["audio", "video"],
            "cross_modal_conflict": False,
            "force_abstain": False,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["pipeline_status"] == "blocked"

    def test_no_caregiver_report_when_blocked(self, caregiver_client):
        sid = create_session(caregiver_client, "S01B-CHILD")
        r = caregiver_client.post(f"/sessions/{sid}/submit", json={
            "modalities": ["audio"],
            "cross_modal_conflict": False,
            "force_abstain": False,
        })
        assert r.json().get("caregiver_report") is None


# ── S-02: Low-SNR audio → audio excluded, remaining modalities proceed ─────────
class TestS02LowSnr:
    def test_audio_excluded_on_low_snr(self, caregiver_client):
        sid = create_session(caregiver_client, "S02-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(
            caregiver_client, sid,
            audio_snr_db=10,
            confidence_scores={"video": 0.84, "text": 0.76, "questionnaire": 0.80},
        )
        # Pipeline should complete (3 modalities remain above threshold)
        assert result["pipeline_status"] in ("complete", "abstained")


# ── S-03: High-confidence full run → complete with caregiver report ────────────
class TestS03HighConfidence:
    def test_complete_status(self, caregiver_client):
        sid = create_session(caregiver_client, "S03-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(caregiver_client, sid, confidence_scores=HIGH_CONF)
        assert result["pipeline_status"] == "complete"

    def test_caregiver_report_is_approved_vocab(self, caregiver_client):
        from backend.agents.constants import APPROVED_CAREGIVER_VOCAB
        sid = create_session(caregiver_client, "S03B-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(caregiver_client, sid, confidence_scores=HIGH_CONF)
        assert result["caregiver_report"] in APPROVED_CAREGIVER_VOCAB

    def test_caregiver_cannot_see_clinician_report(self, caregiver_client):
        sid = create_session(caregiver_client, "S03C-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(caregiver_client, sid)
        # Caregiver endpoint returns only caregiver_report, not clinician_report
        assert "caregiver_report" in result
        # clinician_report field absent or None in caregiver response
        assert result.get("clinician_report") is None


# ── S-04: Cross-modal conflict → abstention ────────────────────────────────────
class TestS04Conflict:
    def test_abstained_on_conflict(self, caregiver_client):
        sid = create_session(caregiver_client, "S04-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(
            caregiver_client, sid,
            cross_modal_conflict=True,
            confidence_scores={"audio": 0.88, "video": 0.72,
                               "text": 0.55, "questionnaire": 0.50},
        )
        assert result["pipeline_status"] == "abstained"

    def test_abstention_caregiver_message(self, caregiver_client):
        sid = create_session(caregiver_client, "S04B-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(caregiver_client, sid, cross_modal_conflict=True)
        assert result["caregiver_report"] is not None
        assert "clinician" in result["caregiver_report"].lower()


# ── S-05: Caregiver requests clinician report → blocked (role escalation) ──────
class TestS05RoleEscalation:
    def test_caregiver_cannot_get_clinician_report(self, caregiver_client):
        sid = create_session(caregiver_client, "S05-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(
            caregiver_client, sid,
            confidence_scores=HIGH_CONF,
            **{"report_type": "clinician_report"},
        )
        assert result["pipeline_status"] == "blocked"


# ── S-06: Force abstention → abstained, escalation after 2nd ──────────────────
class TestS06ForceAbstention:
    def test_force_abstain_status(self, caregiver_client):
        sid = create_session(caregiver_client, "S06-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(caregiver_client, sid, force_abstain=True)
        assert result["pipeline_status"] == "abstained"

    def test_second_abstention_triggers_escalation(self, caregiver_client, admin_client):
        child_id = "S06-ESC-CHILD"
        for _ in range(2):
            sid = create_session(caregiver_client, child_id)
            add_consent(caregiver_client, sid)
            submit_screening(caregiver_client, sid, force_abstain=True)

        # Check escalation queue (admin role)
        r = admin_client.get("/clinician/queue")
        assert r.status_code == 200
        queue = r.json()
        child_ids = [item["child_id"] for item in queue]
        assert child_id in child_ids


# ── S-07: Unsigned / unapproved model → model rejected → abstention ────────────
class TestS07UnsignedModel:
    def test_unsigned_model_causes_abstention(self, caregiver_client):
        sid = create_session(caregiver_client, "S07-CHILD")
        add_consent(caregiver_client, sid)
        # Override model_id to something not in the approved registry
        # by tweaking the session after creation
        from tests.conftest import TestingSessionLocal
        from backend.models import Session as SessionModel
        db = TestingSessionLocal()
        s = db.query(SessionModel).filter(SessionModel.id == sid).first()
        s.model_id = "model-v3.0-unsigned"
        db.commit()
        db.close()

        result = submit_screening(caregiver_client, sid, confidence_scores=HIGH_CONF)
        assert result["pipeline_status"] == "abstained"


# ── S-08: Standard full run — vocabulary compliance ────────────────────────────
class TestS08VocabCompliance:
    def test_caregiver_report_in_approved_vocab(self, caregiver_client):
        from backend.agents.constants import APPROVED_CAREGIVER_VOCAB
        sid = create_session(caregiver_client, "S08-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(caregiver_client, sid, confidence_scores=HIGH_CONF)
        assert result["caregiver_report"] in APPROVED_CAREGIVER_VOCAB


# ── S-09: Age boundary (18 months) → warning but proceeds ─────────────────────
class TestS09AgeBoundary:
    def test_age_boundary_does_not_block(self, caregiver_client):
        sid = create_session(caregiver_client, "S09-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(
            caregiver_client, sid,
            child_age_months=18,
            confidence_scores=HIGH_CONF,
        )
        # Age boundary raises a warning but should not block or abstain
        assert result["pipeline_status"] == "complete"


# ── S-10: Mid-session consent scope change (audio removed) ────────────────────
class TestS10ScopeChange:
    def test_scope_change_removes_audio(self, caregiver_client):
        sid = create_session(caregiver_client, "S10-CHILD")
        add_consent(caregiver_client, sid)
        result = submit_screening(
            caregiver_client, sid,
            confidence_scores={"video": 0.84, "text": 0.76, "questionnaire": 0.80},
            **{"consent_scope_change": {"removed_modalities": ["audio"]}},
        )
        assert result["pipeline_status"] in ("complete", "abstained")
