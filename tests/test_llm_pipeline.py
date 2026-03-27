"""
LangGraph pipeline graph tests — verifies graph structure, state flow,
and fallback behaviour without an Anthropic API key.
"""

import pytest
from backend.pipeline import _GRAPH, run_pipeline
from backend.agents.state import PipelineState
from backend.agents.llm import call_llm, call_llm_json


# ── Graph structure ────────────────────────────────────────────────────────────

class TestGraphStructure:
    def test_all_nodes_present(self):
        nodes = set(_GRAPH.get_graph().nodes.keys())
        expected = {
            "__start__", "__end__",
            "ethics_consent", "bias_applicability",
            "model_selection", "confidence_abstention",
            "explanation_reporting",
        }
        assert expected == nodes

    def test_graph_is_compiled(self):
        # Should be a CompiledGraph, not a bare StateGraph
        assert hasattr(_GRAPH, "invoke")


# ── LLM fallback behaviour ─────────────────────────────────────────────────────

class TestLLMFallback:
    """With no ANTHROPIC_API_KEY set, all LLM calls return None and agents
    fall back to rule-based output. These tests verify that fallback path."""

    def test_call_llm_returns_none_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Clear the lru_cache so the env change takes effect
        from backend.agents.llm import _get_client
        _get_client.cache_clear()
        result = call_llm("system", "user")
        assert result is None
        _get_client.cache_clear()

    def test_call_llm_json_returns_none_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from backend.agents.llm import _get_client
        _get_client.cache_clear()
        result = call_llm_json("system", "user")
        assert result is None
        _get_client.cache_clear()


# ── Full graph end-to-end with mocked DB ──────────────────────────────────────

import datetime

def _mock_state(**scenario_overrides) -> PipelineState:
    audit: list = []
    def mock_log(agent, sid, decision, reason, details=None):
        audit.append((agent, decision, reason))

    abstentions: list = []
    def mock_write(cid, sid, reason):
        abstentions.append({"child_id": cid, "session_id": sid, "reason": reason,
                            "created_at": datetime.datetime.utcnow().isoformat()})
    def mock_get(cid):
        return [a for a in abstentions if a["child_id"] == cid]

    scenario = {
        "session_id": "graph-test-001",
        "child_id": "GRAPH-CHILD",
        "role": "caregiver",
        "model_id": "model-v2.1-signed",
        "modalities": ["audio", "video", "text", "questionnaire"],
        "audio_snr_db": 20,
        "child_age_months": 36,
        "report_type": "standard",
        "requested_operation": "inference",
        "consent_record": {"permitted_ops": ["inference"], "expiry_date": "2028-12-31"},
        "cross_modal_conflict": False,
        "force_abstain": False,
        "confidence_scores": {"audio": 0.88, "video": 0.91, "text": 0.77, "questionnaire": 0.81},
        "consent_scope_change": None,
        **scenario_overrides,
    }
    return {
        "scenario": scenario,
        "log_fn": mock_log,
        "db_ops": {"get_abstention_history": mock_get, "write_abstention": mock_write},
        "agent_outputs": {}, "llm_reasoning": {},
        "blocked": False, "block_reason": None,
        "enabled_modalities": [], "applicability_warnings": [],
        "model_rejected": False, "abstaining": False,
        "abstention_reason": None, "confidence_scores": {},
        "caregiver_report": None, "clinician_report": None,
        "pipeline_status": "pending", "consent_latency_ms": 0.0,
    }


class TestGraphEndToEnd:
    def test_happy_path_complete(self):
        result = _GRAPH.invoke(_mock_state())
        assert result["pipeline_status"] == "complete"
        assert result["caregiver_report"] is not None
        assert result["clinician_report"] is not None

    def test_all_agents_ran_on_happy_path(self):
        result = _GRAPH.invoke(_mock_state())
        outputs = result["agent_outputs"]
        for agent in ["ethics_consent", "bias_applicability",
                      "model_selection", "confidence_abstention",
                      "explanation_reporting"]:
            assert agent in outputs, f"Missing agent output: {agent}"

    def test_no_consent_stops_at_ethics(self):
        result = _GRAPH.invoke(_mock_state(consent_record=None))
        assert result["pipeline_status"] == "blocked"
        # Only ethics_consent should have run
        assert list(result["agent_outputs"].keys()) == ["ethics_consent"]

    def test_force_abstain_path(self):
        result = _GRAPH.invoke(_mock_state(force_abstain=True))
        assert result["pipeline_status"] == "abstained"
        assert result["abstention_reason"] == "insufficient_confidence_data"

    def test_cross_modal_conflict_abstains(self):
        result = _GRAPH.invoke(_mock_state(cross_modal_conflict=True))
        assert result["pipeline_status"] == "abstained"
        assert result["abstention_reason"] == "inter_modal_conflict"

    def test_low_snr_excludes_audio(self):
        result = _GRAPH.invoke(_mock_state(
            audio_snr_db=8,
            confidence_scores={"video": 0.88, "text": 0.77, "questionnaire": 0.81},
        ))
        assert "audio" not in result["enabled_modalities"]

    def test_clinician_report_is_dict(self):
        result = _GRAPH.invoke(_mock_state())
        assert isinstance(result["clinician_report"], dict)

    def test_llm_reasoning_keys_present(self):
        result = _GRAPH.invoke(_mock_state())
        # Even without API key, llm_reasoning dict should exist (values may be empty)
        assert isinstance(result["llm_reasoning"], dict)

    def test_consent_latency_greater_than_zero(self):
        result = _GRAPH.invoke(_mock_state())
        assert result["consent_latency_ms"] > 0

    def test_role_escalation_blocked(self):
        result = _GRAPH.invoke(_mock_state(report_type="clinician_report"))
        assert result["pipeline_status"] == "blocked"
        assert result["block_reason"] == "role_not_authorized_for_clinician_report"
