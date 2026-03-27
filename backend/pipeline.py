"""
LangGraph governance pipeline orchestrator.

Replaces the old imperative run_pipeline() with a typed StateGraph where
each node is an LLM-augmented governance agent. The graph is compiled once
at module import and reused for every request.

Graph topology
──────────────
ethics_consent
  │  BLOCKED → END
  └─ ALLOWED → bias_applicability
                 └─ model_selection
                      └─ confidence_abstention   (writes abstention to DB here)
                           └─ explanation_reporting
                                └─ END

Public API
──────────
  run_pipeline(scenario: dict, db: DBSession) -> dict
  (drop-in replacement for the previous version)
"""

from typing import Literal

from sqlalchemy.orm import Session as DBSession
from langgraph.graph import END, StateGraph

from .audit import get_abstention_history, make_log_fn, write_abstention
from .agents.state import PipelineState
from .agents.ethics_consent import ethics_consent_node
from .agents.bias_applicability import bias_applicability_node
from .agents.model_selection import model_selection_node
from .agents.confidence_abstention import confidence_abstention_node
from .agents.explanation_reporting import explanation_reporting_node


# ── Routing functions (pure state reads — no side effects) ─────────────────────

def _route_after_ethics(
    state: PipelineState,
) -> Literal["bias_applicability", "__end__"]:
    """If ethics/consent blocked the session, skip remaining agents."""
    return "__end__" if state["blocked"] else "bias_applicability"


# ── Graph compiler ─────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(PipelineState)

    g.add_node("ethics_consent",        ethics_consent_node)
    g.add_node("bias_applicability",    bias_applicability_node)
    g.add_node("model_selection",       model_selection_node)
    g.add_node("confidence_abstention", confidence_abstention_node)
    g.add_node("explanation_reporting", explanation_reporting_node)

    g.set_entry_point("ethics_consent")

    # Ethics → conditional branch
    g.add_conditional_edges("ethics_consent", _route_after_ethics)

    # Linear chain for stages 2-5
    g.add_edge("bias_applicability",    "model_selection")
    g.add_edge("model_selection",       "confidence_abstention")
    g.add_edge("confidence_abstention", "explanation_reporting")
    g.add_edge("explanation_reporting", END)

    return g.compile()


# Compile once at import time — thread-safe for read-only invocation
_GRAPH = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_pipeline(scenario: dict, db: DBSession) -> dict:
    """
    Run the 5-agent LangGraph governance pipeline for one screening session.

    Parameters
    ----------
    scenario : dict
        Assembled submission payload including session_id, child_id, consent_record,
        modalities, confidence_scores, and all governance flags.
    db : DBSession
        Active SQLAlchemy session used for audit logging and abstention history.

    Returns
    -------
    dict
        Pipeline result with pipeline_status, caregiver_report, clinician_report,
        enabled_modalities, applicability_warnings, confidence_scores, agent_outputs,
        llm_reasoning, abstention_reason, consent_latency_ms.
    """
    sid      = scenario["session_id"]
    child_id = scenario.get("child_id", "unknown")

    # Wrap DB operations as callables so agents don't hold a direct DB reference
    db_ops = {
        "get_abstention_history": lambda cid: get_abstention_history(db, cid),
        "write_abstention":       lambda cid, s, r: write_abstention(db, cid, s, r),
    }

    initial: PipelineState = {
        # Inputs
        "scenario":               scenario,
        "log_fn":                 make_log_fn(db, sid),
        "db_ops":                 db_ops,
        # Accumulators
        "agent_outputs":          {},
        "llm_reasoning":          {},
        # Routing
        "blocked":                False,
        "block_reason":           None,
        # Stage outputs
        "enabled_modalities":     [],
        "applicability_warnings": [],
        "model_rejected":         False,
        "abstaining":             False,
        "abstention_reason":      None,
        "confidence_scores":      {},
        # Reports
        "caregiver_report":       None,
        "clinician_report":       None,
        # Summary
        "pipeline_status":        "pending",
        "consent_latency_ms":     0.0,
    }

    final: PipelineState = _GRAPH.invoke(initial)

    return {
        "pipeline_status":        final["pipeline_status"],
        "block_reason":           final.get("block_reason"),
        "agent_outputs":          final["agent_outputs"],
        "llm_reasoning":          final["llm_reasoning"],
        "caregiver_report":       final.get("caregiver_report"),
        "clinician_report":       final.get("clinician_report"),
        "enabled_modalities":     final["enabled_modalities"],
        "applicability_warnings": final["applicability_warnings"],
        "confidence_scores":      final["confidence_scores"],
        "abstention_reason":      final.get("abstention_reason"),
        "consent_latency_ms":     final["consent_latency_ms"],
    }
