# ASD Governance Screening Platform

A governance-compliant web application for longitudinal autism screening support,
built for the MLHC 2026 paper. The platform implements a 5-agent LangGraph
pipeline with LLM-augmented reasoning (Claude) and deterministic rule-based
fallbacks for every governance decision.

## Quick Start

### Backend (Python 3.11+ / FastAPI)

```bash
cd autism-screening-app
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Frontend (React 19 + Vite)

```bash
cd autism-screening-app/frontend
npm install && npm run dev
# App: http://localhost:5173
```

---

## Three User Views

| URL | Role | Purpose |
|---|---|---|
| `/caregiver` | Parent/Guardian | 3-step consent -> submit -> result wizard |
| `/clinician` | Clinician | Escalation queue, abstention history, full reports |
| `/admin` | Admin | Live governance metrics + paginated audit log |

---

## Governance Architecture (5-Agent LangGraph Pipeline)

```
ethics_consent
  |  BLOCKED -> END
  +- ALLOWED -> bias_applicability
                  |  ABSTAIN -> explanation_reporting -> END
                  +- PASS -> model_selection
                              |  REJECTED -> END
                              +- APPROVED -> confidence_abstention
                                              +- explanation_reporting -> END
```

1. **Ethics & Consent (Algorithm 1)** -- Verifies user identity and role, checks consent validity/expiry, validates requested operation scope, enforces role-based data access, applies mid-session scope narrowing, LLM compliance review
2. **Bias & Applicability (Algorithm 2)** -- Checks modality data availability, assesses signal quality (audio SNR), validates demographic applicability (age boundaries), LLM estimates combined bias risk score, ABSTAINs if data missing or bias > threshold
3. **Model Selection & Execution (Algorithm 3)** -- Identifies required modalities, queries registry for approved models, selects highest-version candidate, verifies signature/version/deployment status, LLM security provenance review, executes inference
4. **Confidence & Abstention (Algorithm 4)** -- Checks model rejection, force-abstain flag, minimum modality count, per-modality confidence vs threshold, computes cross-modal inconsistency (variance) vs threshold, detects repeated uncertainty with escalation, LLM interprets confidence patterns
5. **Explanation & Reporting (Algorithm 5)** -- Role escalation block, role-gated report generation (caregiver vs clinician), explicit diagnostic language removal filter, confidence/abstention status inclusion, audit metadata attachment, LLM generates structured clinician reports

---

## 7 Governance Metrics (Admin Dashboard)

| Metric | Threshold |
|---|---|
| Policy Gate Accuracy (PGA) | >= 95% |
| Abstention Precision (AP) | >= 90% |
| Vocabulary Compliance Rate (VCR) | 100% |
| Consent Gate Latency (CGL) | < 500 ms |
| Audit Log Completeness (ALC) | 100% |
| Inter-Rater Agreement (IRA) | kappa >= 0.70 (requires expert scorer) |
| Role Isolation Rate (RIR) | 100% |

---

## 10 Synthetic Research Scenarios

Hit `GET /dev/scenarios` to see all 10 pre-built test fixtures (S-01 through S-10).
Load them in the caregiver UI via the "Load research scenario" dropdown.

| ID | Scenario | Governance Path | Expected Outcome |
|---|---|---|---|
| **S-01** | No consent record | Ethics agent blocks immediately -- no downstream agents execute | `blocked` (consent_absent_or_expired) |
| **S-02** | Low-SNR audio (10 dB) | Bias agent disables audio modality (SNR < 15 dB threshold); remaining 3 modalities proceed through model selection, confidence, and reporting | `complete` or `abstained` |
| **S-03** | High-confidence full run | All 4 modalities with high confidence scores (0.77-0.91); full pipeline executes; caregiver receives approved vocabulary report | `complete` |
| **S-04** | Cross-modal conflict | Conflicting scores across modalities trigger abstention in confidence agent; clinician notified; caregiver gets "additional information needed" message | `abstained` (inter_modal_conflict) |
| **S-05** | Caregiver requests clinician report | Role escalation blocked -- caregiver role cannot access clinician-grade diagnostic report | `blocked` (role_not_authorized) |
| **S-06** | Force abstention | Force-abstain flag triggers immediate abstention; 2nd consecutive abstention for same child triggers escalation to clinician queue | `abstained` (insufficient_confidence_data) |
| **S-07** | Unsigned/unapproved model | Model ID not in approved registry -> model rejected -> confidence agent forces abstention | `abstained` (model_not_approved) |
| **S-08** | Standard full run (vocab check) | Identical to S-03 but specifically validates caregiver report string matches approved vocabulary list exactly | `complete` |
| **S-09** | Age boundary (18 months) | Child age at lower boundary of validated range (18-72 months) raises applicability warning but does NOT block or abstain | `complete` |
| **S-10** | Mid-session consent scope change | Caregiver revokes audio consent mid-session; ethics agent disables audio modality; remaining modalities proceed | `complete` or `abstained` |

---

## Test Suite

### Running Tests

```bash
cd autism-screening-app
pytest tests/ -v
```

The test suite runs with no `ANTHROPIC_API_KEY` set, so all agents use their
deterministic rule-based fallback outputs. Tests are fast, reproducible, and
require no API quota.

---

### Test File Structure

| File | Scope | Tests |
|---|---|---|
| `tests/test_agents.py` | Unit tests for each governance agent in isolation | 30 |
| `tests/test_llm_pipeline.py` | LangGraph graph structure, LLM fallback, end-to-end flow | 15 |
| `tests/test_pipeline_scenarios.py` | 10 research scenarios (S-01 to S-10) through full API | 14 |
| `tests/test_api.py` | HTTP API integration tests (sessions, consent, submit, clinician, admin, roles) | 22 |

**Total: 81 test events**

---

### Agent Unit Tests (`test_agents.py`)

#### Agent 1: Ethics & Consent (11 tests)

| # | Test | What It Verifies |
|---|---|---|
| 1 | `test_allows_valid_consent` | Valid consent record with future expiry and matching operation -> ALLOWED |
| 2 | `test_blocks_when_no_consent` | Missing consent record (None) -> BLOCKED with reason `consent_absent_or_expired` |
| 3 | `test_blocks_expired_consent` | Consent with past expiry date (2020-01-01) -> BLOCKED |
| 4 | `test_blocks_malformed_expiry` | Consent with non-date expiry string -> treated as expired -> BLOCKED |
| 5 | `test_blocks_operation_out_of_scope` | Requested operation `longitudinal_tracking` not in `permitted_ops` -> BLOCKED with reason `operation_out_of_scope` |
| 6 | `test_scope_change_disables_modalities` | Mid-session consent change removes audio from enabled modalities without blocking |
| 7 | `test_consent_latency_recorded` | Consent gate latency (ms) is recorded and > 0 |
| 8 | `test_blocks_missing_user_id` | No `user_id` in scenario -> BLOCKED with reason `missing_user_id` |
| 9 | `test_blocks_invalid_role` | Role not in VALID_ROLES set -> BLOCKED with reason `invalid_or_missing_role` |
| 10 | `test_blocks_unauthorized_data_access` | Caregiver requesting `audit_log` data type -> BLOCKED with reason `role_data_access_denied` |
| 11 | `test_allows_authorized_data_access` | Caregiver requesting `screening_results` (permitted) -> ALLOWED |

#### Agent 2: Bias & Applicability (10 tests)

| # | Test | What It Verifies |
|---|---|---|
| 12 | `test_all_modalities_pass_by_default` | All 4 modalities enabled when SNR and age are within range |
| 13 | `test_audio_excluded_when_snr_below_threshold` | SNR=10 dB (< 15 dB threshold) -> audio removed from enabled modalities; warning logged |
| 14 | `test_audio_kept_when_snr_at_threshold` | SNR=15 dB (rule is `< 15`) -> audio kept |
| 15 | `test_age_boundary_warning_low` | Age 18 months (at lower boundary) -> applicability warning |
| 16 | `test_age_boundary_warning_high` | Age 72 months (at upper boundary) -> applicability warning |
| 17 | `test_no_age_warning_in_range` | Age 36 months (within range) -> no warning |
| 18 | `test_unknown_age_no_warning` | Age=None (not provided) -> no warning |
| 19 | `test_abstains_on_missing_modality_data` | All modality data missing from `submitted_data` -> insufficient modalities -> ABSTAIN |
| 20 | `test_no_abstain_with_sufficient_data` | 4 modalities with data present -> COMPLETE (no ABSTAIN) |
| 21 | `test_inherits_disabled_modalities_from_ethics` | Modalities disabled by ethics consent scope change are excluded from bias check |

#### Agent 3: Model Selection & Execution (8 tests)

| # | Test | What It Verifies |
|---|---|---|
| 22 | `test_approved_models_accepted` | Every model in APPROVED_MODEL_REGISTRY passes whitelist check |
| 23 | `test_unknown_model_rejected` | Model ID `model-v99.0-unsigned` -> REJECTED (not in registry) |
| 24 | `test_selects_highest_version` | When multiple models support required modalities, highest version is selected |
| 25 | `test_rejects_model_not_supporting_modalities` | Model that doesn't support all required modalities -> REJECTED |
| 26 | `test_model_verification_checks_signed` | Model with `signed=False` -> REJECTED by verification gate |
| 27 | `test_model_verification_checks_status` | Model with `status != "deployed"` -> REJECTED |
| 28 | `test_inference_result_populates_scores` | After model approval, confidence_scores are populated from inference result |
| 29 | `test_approved_model_has_output_fields` | Approved model output contains `model_id`, `model_version`, `inference_result` |

#### Agent 4: Confidence & Abstention (12 tests)

| # | Test | What It Verifies |
|---|---|---|
| 30 | `test_proceeds_on_high_confidence` | All modality scores >= 0.65 with low variance -> REPORT |
| 31 | `test_abstains_on_low_confidence` | One modality score (0.40) below 0.65 threshold -> ABSTAIN with reason `low_confidence` |
| 32 | `test_abstains_when_force_abstain` | `force_abstain=True` flag -> ABSTAIN with reason `insufficient_confidence_data` |
| 33 | `test_abstains_when_model_rejected` | `model_rejected=True` -> ABSTAIN with reason `model_not_approved` |
| 34 | `test_abstains_with_insufficient_modalities` | Only 1 active modality (below min 2) -> ABSTAIN with reason `insufficient_modalities` |
| 35 | `test_abstains_on_high_inconsistency` | Cross-modal variance > 0.3 (tau_var threshold) -> ABSTAIN with reason `inter_modal_inconsistency` |
| 36 | `test_passes_on_low_inconsistency` | Cross-modal variance < 0.3 -> REPORT (no ABSTAIN) |
| 37 | `test_abstains_on_repeated_uncertainty` | 2+ prior abstentions for same child_id -> ABSTAIN with escalation |
| 38 | `test_escalation_logged_on_repeated_uncertainty` | Repeated uncertainty triggers `human_in_the_loop` ESCALATION_QUEUED log entry |
| 39 | `test_confidence_scores_stored_in_state` | Active modality scores propagated to pipeline state |
| 40 | `test_abstention_written_to_db` | Abstention calls `write_abstention` via `db_ops` |
| 41 | `test_no_abstention_written_on_pass` | When all thresholds pass, no abstention written to DB |

#### Agent 5: Explanation & Reporting (10 tests)

| # | Test | What It Verifies |
|---|---|---|
| 42 | `test_complete_path_status` | Full successful run -> `pipeline_status == "complete"` |
| 43 | `test_caregiver_report_in_approved_vocab` | Caregiver report string is exactly one of the 3 approved vocabulary entries |
| 44 | `test_clinician_report_is_structured` | Clinician report is a dict with `type == "full_diagnostic_support"` |
| 45 | `test_abstention_report_type` | Abstention path produces clinician report with `type == "abstention"` |
| 46 | `test_role_escalation_blocked` | Caregiver requesting `clinician_report` type -> BLOCKED |
| 47 | `test_clinician_report_none_when_blocked` | Blocked role escalation -> `clinician_report is None` |
| 48 | `test_diagnostic_language_removed` | Caregiver report does not contain any term from DIAGNOSTIC_TERMS set |
| 49 | `test_clinician_report_has_audit_metadata` | Clinician report dict contains `audit_metadata` with session_id and agent_chain |
| 50 | `test_abstention_caregiver_gets_waiting_message` | Abstention path gives caregiver the "additional information needed" vocab entry |
| 51 | `test_clinician_role_gets_full_report` | Clinician role receives structured diagnostic support report (not None) |

---

### LangGraph Pipeline Tests (`test_llm_pipeline.py`)

#### Graph Structure (2 tests)

| # | Test | What It Verifies |
|---|---|---|
| 52 | `test_all_nodes_present` | Graph contains all 5 agent nodes plus `__start__` and `__end__` |
| 53 | `test_graph_is_compiled` | Graph is a CompiledGraph with `invoke()` method |

#### LLM Fallback (2 tests)

| # | Test | What It Verifies |
|---|---|---|
| 54 | `test_call_llm_returns_none_without_key` | `call_llm()` returns None when ANTHROPIC_API_KEY is unset |
| 55 | `test_call_llm_json_returns_none_without_key` | `call_llm_json()` returns None when ANTHROPIC_API_KEY is unset |

#### End-to-End Graph Flow (11 tests)

| # | Test | What It Verifies |
|---|---|---|
| 56 | `test_happy_path_complete` | Full pipeline -> `pipeline_status == "complete"` with both reports present |
| 57 | `test_all_agents_ran_on_happy_path` | All 5 agent keys present in `agent_outputs` dict |
| 58 | `test_no_consent_stops_at_ethics` | No consent -> only `ethics_consent` in agent_outputs (short-circuit) |
| 59 | `test_force_abstain_path` | Force abstain -> `abstained` status with correct reason |
| 60 | `test_cross_modal_conflict_abstains` | Cross-modal conflict flag -> abstention |
| 61 | `test_low_snr_excludes_audio` | SNR=8 dB -> audio not in enabled_modalities |
| 62 | `test_clinician_report_is_dict` | Clinician report output is a dictionary (not string/None) |
| 63 | `test_llm_reasoning_keys_present` | `llm_reasoning` dict exists even without API key (values may be empty) |
| 64 | `test_consent_latency_greater_than_zero` | Consent gate timing is recorded |
| 65 | `test_role_escalation_blocked` | Caregiver requesting clinician_report -> blocked at explanation agent |
| 66 | `test_bias_abstain_skips_to_reporting` | When bias agent ABSTAINs, model_selection and confidence_abstention are skipped |

---

### Research Scenario Tests (`test_pipeline_scenarios.py`)

These mirror the 10 research scenarios through the full FastAPI HTTP stack with
an in-memory SQLite database.

| # | Test | Scenario | What It Verifies |
|---|---|---|---|
| 67 | `test_blocked_without_consent` | S-01 | Submit without consent -> `pipeline_status == "blocked"` |
| 68 | `test_no_caregiver_report_when_blocked` | S-01 | Blocked session returns no caregiver report |
| 69 | `test_audio_excluded_on_low_snr` | S-02 | SNR=10 dB -> pipeline completes with audio excluded |
| 70 | `test_complete_status` | S-03 | High confidence full run -> `complete` |
| 71 | `test_caregiver_report_is_approved_vocab` | S-03 | Caregiver report matches approved vocabulary |
| 72 | `test_caregiver_cannot_see_clinician_report` | S-03 | Caregiver endpoint returns `clinician_report: null` |
| 73 | `test_abstained_on_conflict` | S-04 | Cross-modal conflict -> `abstained` |
| 74 | `test_abstention_caregiver_message` | S-04 | Abstention caregiver message mentions "clinician" |
| 75 | `test_caregiver_cannot_get_clinician_report` | S-05 | Role escalation -> `blocked` |
| 76 | `test_force_abstain_status` | S-06 | Force abstain -> `abstained` |
| 77 | `test_second_abstention_triggers_escalation` | S-06 | 2 consecutive abstentions -> child appears in escalation queue |
| 78 | `test_unsigned_model_causes_abstention` | S-07 | Unsigned model -> `abstained` |
| 79 | `test_caregiver_report_in_approved_vocab` | S-08 | Vocabulary compliance check |
| 80 | `test_age_boundary_does_not_block` | S-09 | Age=18 months -> warning but still `complete` |
| 81 | `test_scope_change_removes_audio` | S-10 | Mid-session scope change -> pipeline proceeds without audio |

---

### API Integration Tests (`test_api.py`)

#### Session Endpoints (3 tests)

| # | Test | What It Verifies |
|---|---|---|
| 82 | `test_create_session_returns_session_id` | POST /sessions returns session_id and child_id |
| 83 | `test_get_session` | GET /sessions/{id} returns correct session data |
| 84 | `test_get_nonexistent_session_returns_404` | GET /sessions/{invalid_id} -> 404 |

#### Consent Endpoints (2 tests)

| # | Test | What It Verifies |
|---|---|---|
| 85 | `test_add_consent` | POST /sessions/{id}/consent -> 200 |
| 86 | `test_consent_on_missing_session_404` | Consent on non-existent session -> 404 |

#### Submit Endpoints (3 tests)

| # | Test | What It Verifies |
|---|---|---|
| 87 | `test_submit_returns_pipeline_status` | POST /sessions/{id}/submit returns pipeline_status field |
| 88 | `test_submit_without_consent_blocked` | Submit without prior consent -> blocked |
| 89 | `test_submit_missing_session_404` | Submit to non-existent session -> 404 |

#### Clinician Endpoints (4 tests)

| # | Test | What It Verifies |
|---|---|---|
| 90 | `test_clinician_sessions_requires_clinician_role` | Caregiver accessing /clinician/sessions -> 403 |
| 91 | `test_clinician_can_list_sessions` | Clinician can GET /clinician/sessions -> 200 with list |
| 92 | `test_escalation_queue_requires_clinician_role` | Caregiver accessing /clinician/queue -> 403 |
| 93 | `test_escalation_queue_accessible_to_clinician` | Clinician can access queue -> 200 |

#### Admin Endpoints (7 tests)

| # | Test | What It Verifies |
|---|---|---|
| 94 | `test_audit_log_requires_admin_role` | Caregiver -> 403 on audit log |
| 95 | `test_clinician_cannot_access_audit_log` | Clinician -> 403 on audit log |
| 96 | `test_admin_can_get_audit_log` | Admin -> 200 with total and entries |
| 97 | `test_audit_log_pagination` | limit=5 returns <= 5 entries |
| 98 | `test_audit_log_filter_by_decision` | Filter by decision=ALLOW returns only ALLOW entries |
| 99 | `test_admin_can_get_metrics` | Admin -> 200 with all 7 metric keys |
| 100 | `test_metrics_have_required_fields` | Each metric has value, threshold, pass fields |

#### Role Isolation (3 tests)

| # | Test | What It Verifies |
|---|---|---|
| 101 | `test_metrics_requires_admin` | Non-admin -> 403 on metrics |
| 102 | `test_clinician_sessions_include_abstained` | Abstained sessions appear in clinician list |
| 103 | `test_role_access_parametrized` | 10 parametrized role x endpoint combinations verify correct 200/403 responses |

#### Dev Endpoints (3 tests)

| # | Test | What It Verifies |
|---|---|---|
| 104 | `test_get_scenarios` | GET /dev/scenarios returns >= 10 scenarios including S-01 |
| 105 | `test_scenario_has_required_fields` | Every scenario has a `label` field |
| 106 | `test_get_approved_vocab` | GET /dev/approved-vocab returns list of 3 strings |

---

### Governance Event Coverage Matrix

The table below maps each governance event to the test(s) that cover it. Every
deterministic decision branch in the pipeline has at least one test.

| Governance Event | Agent | Tests |
|---|---|---|
| Identity verification (missing user_id) | Ethics | #8 |
| Identity verification (invalid role) | Ethics | #9 |
| Consent absent | Ethics | #2, #67, #68, #88 |
| Consent expired | Ethics | #3 |
| Consent malformed date | Ethics | #4 |
| Operation out of scope | Ethics | #5 |
| Role-based data access denied | Ethics | #10 |
| Role-based data access allowed | Ethics | #11 |
| Mid-session consent scope narrowing | Ethics | #6, #81 |
| Consent latency recording | Ethics | #7, #64 |
| LLM compliance review (fallback) | Ethics | #54, #55 |
| Audio SNR below threshold -> disable | Bias | #13, #69, #61 |
| Audio SNR at threshold -> keep | Bias | #14 |
| Age boundary warning (low) | Bias | #15, #80 |
| Age boundary warning (high) | Bias | #16 |
| Age in range -> no warning | Bias | #17 |
| Age unknown -> no warning | Bias | #18 |
| Required data missing -> ABSTAIN | Bias | #19 |
| Sufficient data -> COMPLETE | Bias | #20 |
| Inherits disabled modalities | Bias | #21 |
| All modalities pass default | Bias | #12 |
| Approved model accepted | Model | #22 |
| Unknown model rejected | Model | #23, #78 |
| Highest version selected | Model | #24 |
| Model doesn't support modalities | Model | #25 |
| Model unsigned -> verification fail | Model | #26 |
| Model status != deployed | Model | #27 |
| Inference result populated | Model | #28, #29 |
| High confidence -> REPORT | Confidence | #30 |
| Low confidence -> ABSTAIN | Confidence | #31 |
| Force abstain flag | Confidence | #32, #59, #76 |
| Model rejected -> forced ABSTAIN | Confidence | #33 |
| Insufficient modalities | Confidence | #34 |
| High cross-modal inconsistency (variance > tau_var) | Confidence | #35 |
| Low cross-modal inconsistency | Confidence | #36 |
| Repeated uncertainty -> escalation | Confidence | #37, #38, #77 |
| Confidence scores propagated | Confidence | #39 |
| Abstention written to DB | Confidence | #40 |
| No abstention on pass | Confidence | #41 |
| Complete status | Reporting | #42, #56, #70 |
| Caregiver vocab compliance | Reporting | #43, #71, #79 |
| Clinician report structured | Reporting | #44, #62 |
| Abstention report type | Reporting | #45 |
| Role escalation blocked | Reporting | #46, #65, #75 |
| Clinician report None when blocked | Reporting | #47 |
| Diagnostic language removed | Reporting | #48 |
| Audit metadata in clinician report | Reporting | #49 |
| Abstention caregiver message | Reporting | #50, #74 |
| Clinician gets full report | Reporting | #51 |
| Graph nodes present | Pipeline | #52 |
| Graph compiled | Pipeline | #53 |
| All agents ran on happy path | Pipeline | #57 |
| Ethics short-circuit | Pipeline | #58, #67 |
| Bias ABSTAIN skips downstream | Pipeline | #66 |
| Cross-modal conflict -> abstain | Pipeline | #60, #73 |
| LLM reasoning dict present | Pipeline | #63 |
| Role access control (10 combos) | API | #103 |
| Session CRUD | API | #82, #83, #84 |
| Consent CRUD | API | #85, #86 |
| Submit pipeline | API | #87, #88, #89 |
| Clinician endpoints gated | API | #90, #91, #92, #93 |
| Admin endpoints gated | API | #94, #95, #96, #97, #98, #99, #100, #101 |
| Dev scenarios endpoint | API | #104, #105, #106 |
