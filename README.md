# ASD Screening MVP — MLHC 2026

A governance-compliant web application for longitudinal autism screening support.

## Quick Start

### Backend (Python 3.11 + FastAPI)

```bash
cd autism-screening-app
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Frontend (React + Vite)

```bash
cd autism-screening-app/frontend
npm install && npm run dev
# App: http://localhost:5173
```

---

## Three User Views

| URL | Role | Purpose |
|---|---|---|
| `/caregiver` | Parent/Guardian | 3-step consent → submit → result wizard |
| `/clinician` | Clinician | Escalation queue, abstention history, full reports |
| `/admin` | Admin | Live governance metrics + paginated audit log |

---

## Governance Architecture (5 Agents)

1. **Ethics & Consent** — blocks if no valid consent or expired
2. **Bias & Applicability** — disables audio if SNR < 15 dB; warns on age edge cases
3. **Model Selection** — rejects models not in approved registry
4. **Confidence & Abstention** — abstains on low confidence, cross-modal conflict, or insufficient modalities
5. **Explanation & Reporting** — enforces vocabulary compliance; blocks role escalation

---

## 7 Governance Metrics (Admin Dashboard)

| Metric | Threshold |
|---|---|
| Policy Gate Accuracy (PGA) | ≥ 95% |
| Abstention Precision (AP) | ≥ 90% |
| Vocabulary Compliance Rate (VCR) | 100% |
| Consent Gate Latency (CGL) | < 500 ms |
| Audit Log Completeness (ALC) | 100% |
| Inter-Rater Agreement (IRA) | κ ≥ 0.70 (requires expert scorer) |
| Role Isolation Rate (RIR) | 100% |

---

## Demo Scenarios

Hit `GET /dev/scenarios` to see all 10 pre-built test fixtures (S-01 through S-10).
Load them in the caregiver UI via the "Load research scenario" dropdown.

Key scenarios:
- **S-01** — No consent → BLOCKED
- **S-03** — Full modalities, SNR=22 → COMPLETE ✅
- **S-05** — Force abstain → amber "additional information" message
- **S-06** → repeat 3× to trigger escalation queue
