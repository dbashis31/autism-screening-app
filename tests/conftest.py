"""
Shared pytest fixtures for the ASD Governance Screening API test suite.

Uses an in-memory SQLite database so tests are completely isolated — the
production screening.db is never touched.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.dependencies import get_db
from backend.main import app

# ── In-memory SQLite (isolated per test session) ───────────────────────────────
TEST_DB_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with in-memory DB wired in."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def caregiver_client(client):
    """Client that sends X-Role: caregiver on every request."""
    client.headers.update({"X-Role": "caregiver"})
    return client


@pytest.fixture
def clinician_client(client):
    """Client that sends X-Role: clinician on every request."""
    client.headers.update({"X-Role": "clinician"})
    return client


@pytest.fixture
def admin_client(client):
    """Client that sends X-Role: admin on every request."""
    client.headers.update({"X-Role": "admin"})
    return client


# ── Common helpers ────────────────────────────────────────────────────────────

VALID_CONSENT = {"permitted_ops": ["inference"], "expiry_date": "2028-12-31"}
HIGH_CONF     = {"audio": 0.88, "video": 0.91, "text": 0.77, "questionnaire": 0.81}


def create_session(client, child_id="CHILD-TEST") -> str:
    """Create a session and return its session_id."""
    r = client.post("/sessions", json={"child_id": child_id, "role": "caregiver"})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def add_consent(client, session_id: str, consent: dict = None) -> None:
    """Attach a consent record to a session."""
    payload = consent or VALID_CONSENT
    r = client.post(f"/sessions/{session_id}/consent", json=payload)
    assert r.status_code == 200, r.text


def submit_screening(client, session_id: str, **overrides) -> dict:
    """Submit a screening session and return the response JSON."""
    payload = {
        "modalities": ["audio", "video", "text", "questionnaire"],
        "audio_snr_db": 20,
        "child_age_months": 36,
        "cross_modal_conflict": False,
        "force_abstain": False,
        "confidence_scores": HIGH_CONF,
        **overrides,
    }
    r = client.post(f"/sessions/{session_id}/submit", json=payload)
    assert r.status_code == 200, r.text
    return r.json()
