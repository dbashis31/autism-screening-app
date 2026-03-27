"""
Unit tests for each governance agent node in isolation.

Each agent is called directly with a minimal PipelineState — no FastAPI,
no DB, no Anthropic API key required. Tests verify the deterministic
rule-based behaviour of every decision branch.
"""

import datetime
import pytest

from backend.agents.state import PipelineState
from backend.agents.ethics_consent import ethics_consent_node
from backend.agents.bias_applicability import bias_applicability_node
from backend.agents.model_selection import model_selection_node
from backend.agents.confidence_abstention import confidence_abstention_node
from backend.agents.explanation_reporting import explanation_reporting_node
from backend.agents.constants import APPROVED_CAREGIVER_VOCAB, APPROVED_MODEL_REGISTRY


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_state(**overrides) -> PipelineState:
    """Build a minimal valid PipelineState for unit testing."""
    audit: list[dict] = []

    def mock_log(agent, sid, decision, reason, details=None):
        audit.append({"agent": agent, "decision": decision, "reason": reason})

    abstentions: list[dict] = []

    def mock_write_abs(cid, sid, reason):
        abstentions.append({"child_id": cid, "session_id": sid, "reason": reason,
                            "created_at": datetime.datetime.utcnow().isoformat()})

    def mock_get_history(cid):
        return [a for a in abstentions if a["child_id"] == cid]

    base: PipelineState = {
        "scenario": {
            "session_id": "unit-test-001",
            "child_id": "CHILD-UNIT",
            "role": "caregiver",
            "model_id": "model-v2.1-signed",
            "modalities": ["audio", "video", "text", "questionnaire"],
            "audio_snr_db": 20,
            "child_age_months": 36,
            "report_type": "standard",
            "requested_operation": "inference",
            "consent_record": {
                "permitted_ops": ["inference"],
                "expiry_date": "2028-12-31",
            },
            "cross_modal_conflict": False,
            "force_abstain": False,
            "confidence_scores": {
                "audio": 0.88, "video": 0.91, "text": 0.77, "questionnaire": 0.81
            },
            "consent_scope_change": None,
        },
        "log_fn":  mock_log,
        "db_ops": {
            "get_abstention_history": mock_get_history,
            "write_abstention":       mock_write_abs,
        },
        "agent_outputs":          {},
        "llm_reasoning":          {},
        "blocked":                False,
        "block_reason":           None,
        "enabled_modalities":     [],
        "applicability_warnings": [],
        "model_rejected":         False,
        "abstaining":             False,
        "abstention_reason":      None,
        "confidence_scores":      {},
        "caregiver_report":       None,
        "clinician_report":       None,
        "pipeline_status":        "pending",
        "consent_latency_ms":     0.0,
    }
    # Deep-merge scenario overrides
    for k, v in overrides.items():
        if k == "scenario":
            base["scenario"] = {**base["scenario"], **v}
        else:
            base[k] = v
    return base


# ── Agent 1: Ethics & Consent ──────────────────────────────────────────────────

class TestEthicsConsentAgent:
    def test_allows_valid_consent(self):
        result = ethics_consent_node(_make_state())
        ec = result["agent_outputs"]["ethics_consent"]
        assert ec["status"] == "ALLOWED"
        assert result["blocked"] is False

    def test_blocks_when_no_consent(self):
        state = _make_state(scenario={"consent_record": None})
        result = ethics_consent_node(state)
        assert result["blocked"] is True
        assert result["block_reason"] == "consent_absent_or_expired"

    def test_blocks_expired_consent(self):
        state = _make_state(scenario={
            "consent_record": {
                "permitted_ops": ["inference"],
                "expiry_date": "2020-01-01",   # past date
            }
        })
        result = ethics_consent_node(state)
        assert result["blocked"] is True

    def test_blocks_malformed_expiry(self):
        state = _make_state(scenario={
            "consent_record": {
                "permitted_ops": ["inference"],
                "expiry_date": "not-a-date",
            }
        })
        result = ethics_consent_node(state)
        assert result["blocked"] is True

    def test_blocks_operation_out_of_scope(self):
        state = _make_state(scenario={
            "requested_operation": "longitudinal_tracking",
            "consent_record": {
                "permitted_ops": ["inference"],  # does not include longitudinal_tracking
                "expiry_date": "2028-12-31",
            },
        })
        result = ethics_consent_node(state)
        assert result["blocked"] is True
        assert result["block_reason"] == "operation_out_of_scope"

    def test_scope_change_disables_modalities(self):
        state = _make_state(scenario={
            "consent_scope_change": {"removed_modalities": ["audio"]}
        })
        result = ethics_consent_node(state)
        ec = result["agent_outputs"]["ethics_consent"]
        assert "audio" in ec["disabled_modalities"]
        assert result["blocked"] is False

    def test_consent_latency_recorded(self):
        result = ethics_consent_node(_make_state())
        assert result["consent_latency_ms"] > 0


# ── Agent 2: Bias & Applicability ─────────────────────────────────────────────

class TestBiasApplicabilityAgent:
    def _run(self, **scenario_overrides):
        state = _make_state(scenario=scenario_overrides)
        # ethics_consent_node must run first to populate agent_outputs["ethics_consent"]
        state = ethics_consent_node(state)
        return bias_applicability_node(state)

    def test_all_modalities_pass_by_default(self):
        result = self._run()
        assert "audio" in result["enabled_modalities"]
        assert len(result["enabled_modalities"]) == 4

    def test_audio_excluded_when_snr_below_threshold(self):
        result = self._run(audio_snr_db=10)
        assert "audio" not in result["enabled_modalities"]
        assert any("audio_degraded" in w for w in result["applicability_warnings"])

    def test_audio_kept_when_snr_at_threshold(self):
        result = self._run(audio_snr_db=15)
        # 15 is the threshold; rule is snr < 15, so 15 is kept
        assert "audio" in result["enabled_modalities"]

    def test_age_boundary_warning_low(self):
        result = self._run(child_age_months=18)
        assert any("age_applicability" in w for w in result["applicability_warnings"])

    def test_age_boundary_warning_high(self):
        result = self._run(child_age_months=72)
        assert any("age_applicability" in w for w in result["applicability_warnings"])

    def test_no_age_warning_in_range(self):
        result = self._run(child_age_months=36)
        assert not any("age_applicability" in w for w in result["applicability_warnings"])

    def test_unknown_age_no_warning(self):
        result = self._run(child_age_months=None)
        assert not any("age_applicability" in w for w in result["applicability_warnings"])


# ── Agent 3: Model Selection ───────────────────────────────────────────────────

class TestModelSelectionAgent:
    def _run(self, model_id="model-v2.1-signed"):
        state = _make_state(scenario={"model_id": model_id})
        state = ethics_consent_node(state)
        state = bias_applicability_node(state)
        return model_selection_node(state)

    def test_approved_models_accepted(self):
        for model_id in APPROVED_MODEL_REGISTRY:
            result = self._run(model_id=model_id)
            assert result["model_rejected"] is False

    def test_unknown_model_rejected(self):
        result = self._run(model_id="model-v99.0-unsigned")
        assert result["model_rejected"] is True
        ms = result["agent_outputs"]["model_selection"]
        assert ms["status"] == "REJECTED"


# ── Agent 4: Confidence & Abstention ──────────────────────────────────────────

class TestConfidenceAbstentionAgent:
    def _run(self, **scenario_overrides):
        state = _make_state(scenario=scenario_overrides)
        state = ethics_consent_node(state)
        state = bias_applicability_node(state)
        state = model_selection_node(state)
        return confidence_abstention_node(state)

    def test_proceeds_on_high_confidence(self):
        result = self._run()
        assert result["abstaining"] is False
        assert result["agent_outputs"]["confidence_abstention"]["status"] == "REPORT"

    def test_abstains_on_cross_modal_conflict(self):
        result = self._run(cross_modal_conflict=True)
        assert result["abstaining"] is True
        assert result["abstention_reason"] == "inter_modal_conflict"

    def test_abstains_when_force_abstain(self):
        result = self._run(force_abstain=True)
        assert result["abstaining"] is True
        assert result["abstention_reason"] == "insufficient_confidence_data"

    def test_abstains_on_low_confidence(self):
        result = self._run(confidence_scores={
            "audio": 0.40,   # below 0.65
            "video": 0.91,
            "text": 0.77,
            "questionnaire": 0.81,
        })
        assert result["abstaining"] is True
        assert result["abstention_reason"] == "low_confidence"

    def test_abstains_with_insufficient_modalities(self):
        # Only 1 modality with scores — below minimum of 2
        result = self._run(
            modalities=["video"],
            confidence_scores={"video": 0.91},
        )
        assert result["abstaining"] is True
        assert result["abstention_reason"] == "insufficient_modalities"

    def test_abstains_when_model_rejected(self):
        state = _make_state(scenario={"model_id": "model-v3.0-unsigned"})
        state = ethics_consent_node(state)
        state = bias_applicability_node(state)
        state = model_selection_node(state)
        result = confidence_abstention_node(state)
        assert result["abstaining"] is True
        assert result["abstention_reason"] == "model_not_approved"

    def test_confidence_scores_stored_in_state(self):
        result = self._run()
        assert len(result["confidence_scores"]) >= 2

    def test_abstention_written_to_db(self):
        """Abstention should call write_abstention via db_ops."""
        written: list = []
        state = _make_state(scenario={"force_abstain": True})
        state["db_ops"]["write_abstention"] = lambda cid, sid, r: written.append(r)
        state = ethics_consent_node(state)
        state = bias_applicability_node(state)
        state = model_selection_node(state)
        confidence_abstention_node(state)
        assert len(written) == 1


# ── Agent 5: Explanation & Reporting ──────────────────────────────────────────

class TestExplanationReportingAgent:
    def _run_full(self):
        state = _make_state()
        state = ethics_consent_node(state)
        state = bias_applicability_node(state)
        state = model_selection_node(state)
        state = confidence_abstention_node(state)
        return explanation_reporting_node(state)

    def _run_abstention(self):
        state = _make_state(scenario={"force_abstain": True})
        state = ethics_consent_node(state)
        state = bias_applicability_node(state)
        state = model_selection_node(state)
        state = confidence_abstention_node(state)
        return explanation_reporting_node(state)

    def _run_role_escalation(self):
        state = _make_state(scenario={"report_type": "clinician_report"})
        state = ethics_consent_node(state)
        state = bias_applicability_node(state)
        state = model_selection_node(state)
        state = confidence_abstention_node(state)
        return explanation_reporting_node(state)

    def test_complete_path_status(self):
        result = self._run_full()
        assert result["pipeline_status"] == "complete"

    def test_caregiver_report_in_approved_vocab(self):
        result = self._run_full()
        assert result["caregiver_report"] in APPROVED_CAREGIVER_VOCAB

    def test_clinician_report_is_structured(self):
        result = self._run_full()
        cr = result["clinician_report"]
        assert isinstance(cr, dict)
        assert "type" in cr
        assert cr["type"] == "full_diagnostic_support"

    def test_abstention_report_type(self):
        result = self._run_abstention()
        assert result["pipeline_status"] == "abstained"
        cr = result["clinician_report"]
        assert cr["type"] == "abstention"

    def test_role_escalation_blocked(self):
        result = self._run_role_escalation()
        assert result["blocked"] is True
        assert result["pipeline_status"] == "blocked"
        # Caregiver still gets a vocab-compliant message
        assert result["caregiver_report"] in APPROVED_CAREGIVER_VOCAB

    def test_clinician_report_none_when_blocked(self):
        result = self._run_role_escalation()
        assert result["clinician_report"] is None
