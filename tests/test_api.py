"""
API integration tests — exercises every HTTP endpoint through the FastAPI
TestClient with a real (in-memory) SQLite database.
"""

import pytest
from .conftest import (
    VALID_CONSENT, HIGH_CONF,
    create_session, add_consent, submit_screening,
)


# ── Session endpoints ──────────────────────────────────────────────────────────

class TestSessionEndpoints:
    def test_create_session_returns_session_id(self, client):
        r = client.post("/sessions", json={"child_id": "API-CHILD-01", "role": "caregiver"})
        assert r.status_code == 200
        body = r.json()
        assert "session_id" in body
        assert body["child_id"] == "API-CHILD-01"

    def test_get_session(self, client):
        sid = create_session(client, "API-CHILD-02")
        r = client.get(f"/sessions/{sid}")
        assert r.status_code == 200
        assert r.json()["session_id"] == sid

    def test_get_nonexistent_session_returns_404(self, client):
        r = client.get("/sessions/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404


# ── Consent endpoints ──────────────────────────────────────────────────────────

class TestConsentEndpoints:
    def test_add_consent(self, client):
        sid = create_session(client, "CONSENT-CHILD-01")
        r = client.post(f"/sessions/{sid}/consent", json=VALID_CONSENT)
        assert r.status_code == 200

    def test_consent_on_missing_session_404(self, client):
        r = client.post(
            "/sessions/00000000-0000-0000-0000-000000000000/consent",
            json=VALID_CONSENT,
        )
        assert r.status_code == 404


# ── Submit endpoint ────────────────────────────────────────────────────────────

class TestSubmitEndpoints:
    def test_submit_returns_pipeline_status(self, caregiver_client):
        sid = create_session(caregiver_client, "SUBMIT-CHILD-01")
        add_consent(caregiver_client, sid)
        result = submit_screening(caregiver_client, sid)
        assert "pipeline_status" in result

    def test_submit_without_consent_blocked(self, caregiver_client):
        sid = create_session(caregiver_client, "SUBMIT-CHILD-NOCONSENT")
        r = caregiver_client.post(f"/sessions/{sid}/submit", json={
            "modalities": ["audio", "video"],
            "cross_modal_conflict": False,
            "force_abstain": False,
        })
        assert r.status_code == 200
        assert r.json()["pipeline_status"] == "blocked"

    def test_submit_missing_session_404(self, caregiver_client):
        r = caregiver_client.post(
            "/sessions/00000000-0000-0000-0000-000000000000/submit",
            json={"modalities": ["audio"], "cross_modal_conflict": False, "force_abstain": False},
        )
        assert r.status_code == 404


# ── Clinician endpoints ────────────────────────────────────────────────────────

class TestClinicianEndpoints:
    def test_clinician_sessions_requires_clinician_role(self, caregiver_client):
        r = caregiver_client.get("/clinician/sessions")
        assert r.status_code == 403

    def test_clinician_can_list_sessions(self, clinician_client, caregiver_client):
        # Create and complete a session as caregiver
        sid = create_session(caregiver_client, "CLIN-CHILD-01")
        add_consent(caregiver_client, sid)
        submit_screening(caregiver_client, sid, confidence_scores=HIGH_CONF)

        r = clinician_client.get("/clinician/sessions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_escalation_queue_requires_clinician_role(self, caregiver_client):
        r = caregiver_client.get("/clinician/queue")
        assert r.status_code == 403

    def test_escalation_queue_accessible_to_clinician(self, clinician_client):
        r = clinician_client.get("/clinician/queue")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_clinician_sessions_include_abstained(self, clinician_client, caregiver_client):
        sid = create_session(caregiver_client, "CLIN-ABS-CHILD-01")
        add_consent(caregiver_client, sid)
        submit_screening(caregiver_client, sid, force_abstain=True)

        r = clinician_client.get("/clinician/sessions")
        assert r.status_code == 200
        statuses = [s["pipeline_status"] for s in r.json()]
        assert "abstained" in statuses


# ── Admin endpoints ────────────────────────────────────────────────────────────

class TestAdminEndpoints:
    def test_audit_log_requires_admin_role(self, caregiver_client):
        r = caregiver_client.get("/admin/audit-log")
        assert r.status_code == 403

    def test_clinician_cannot_access_audit_log(self, clinician_client):
        r = clinician_client.get("/admin/audit-log")
        assert r.status_code == 403

    def test_admin_can_get_audit_log(self, admin_client):
        r = admin_client.get("/admin/audit-log")
        assert r.status_code == 200
        body = r.json()
        assert "total" in body
        assert "entries" in body

    def test_audit_log_pagination(self, admin_client):
        r = admin_client.get("/admin/audit-log?limit=5&skip=0")
        assert r.status_code == 200
        body = r.json()
        assert len(body["entries"]) <= 5

    def test_audit_log_filter_by_decision(self, admin_client):
        r = admin_client.get("/admin/audit-log?decision=ALLOW")
        assert r.status_code == 200
        for entry in r.json()["entries"]:
            assert entry["decision"] == "ALLOW"

    def test_admin_can_get_metrics(self, admin_client):
        r = admin_client.get("/admin/metrics")
        assert r.status_code == 200
        metrics = r.json()
        # All 7 defined metric keys must be present
        expected_keys = {
            "Policy Gate Accuracy (PGA)",
            "Abstention Precision (AP)",
            "Vocabulary Compliance Rate (VCR)",
            "Consent Gate Latency (CGL)",
            "Audit Log Completeness (ALC)",
            "Inter-Rater Agreement (IRA)",
            "Role Isolation Rate (RIR)",
        }
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_metrics_have_required_fields(self, admin_client):
        r = admin_client.get("/admin/metrics")
        metrics = r.json()
        for key, val in metrics.items():
            if key == "_meta":
                continue
            assert "value" in val, f"{key} missing 'value'"
            assert "threshold" in val, f"{key} missing 'threshold'"
            assert "pass" in val, f"{key} missing 'pass'"

    def test_metrics_requires_admin(self, caregiver_client):
        r = caregiver_client.get("/admin/metrics")
        assert r.status_code == 403


# ── Dev / mock data endpoints ──────────────────────────────────────────────────

class TestDevEndpoints:
    def test_get_scenarios(self, client):
        r = client.get("/dev/scenarios")
        assert r.status_code == 200
        scenarios = r.json()
        assert len(scenarios) >= 10
        assert "S-01" in scenarios

    def test_scenario_has_required_fields(self, client):
        r = client.get("/dev/scenarios")
        for key, sc in r.json().items():
            assert "label" in sc, f"Scenario {key} missing label"

    def test_get_approved_vocab(self, client):
        r = client.get("/dev/approved-vocab")
        assert r.status_code == 200
        vocab = r.json()
        assert isinstance(vocab, list)
        assert len(vocab) == 3


# ── Role isolation ─────────────────────────────────────────────────────────────

class TestRoleIsolation:
    """Verify that the X-Role header actually gates access."""

    @pytest.mark.parametrize("role,endpoint,expected_code", [
        ("caregiver", "/clinician/sessions", 403),
        ("caregiver", "/clinician/queue",    403),
        ("caregiver", "/admin/audit-log",    403),
        ("caregiver", "/admin/metrics",      403),
        ("clinician", "/admin/audit-log",    403),
        ("clinician", "/admin/metrics",      403),
        ("admin",     "/clinician/sessions", 200),
        ("admin",     "/clinician/queue",    200),
        ("admin",     "/admin/audit-log",    200),
        ("admin",     "/admin/metrics",      200),
    ])
    def test_role_access(self, client, role, endpoint, expected_code):
        r = client.get(endpoint, headers={"X-Role": role})
        assert r.status_code == expected_code, (
            f"Role '{role}' on {endpoint}: expected {expected_code}, got {r.status_code}"
        )
