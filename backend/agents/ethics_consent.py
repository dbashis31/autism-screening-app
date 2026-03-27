"""
Ethics & Consent Agent (LangGraph node).

Responsibility: Gate ALL downstream processing on valid caregiver consent.
This is the first and hardest gate — if consent is absent, expired, or
out-of-scope the pipeline stops immediately.

LLM role: After the deterministic gate passes, Claude reviews the consent
record for nuanced compliance concerns and produces a brief audit summary
that is stored in the audit log and surfaced in the clinician report.
"""

import datetime
import time

from .llm import call_llm_json
from .state import PipelineState


# ── Helpers ────────────────────────────────────────────────────────────────────

def _consent_expired(record: dict) -> bool:
    expiry_str = record.get("expiry_date")
    if not expiry_str:
        return False
    try:
        return datetime.date.fromisoformat(expiry_str) < datetime.date.today()
    except ValueError:
        return True  # malformed date → treat as expired


# ── LangGraph node ─────────────────────────────────────────────────────────────

def ethics_consent_node(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    scenario = state["scenario"]
    log_fn   = state["log_fn"]
    sid      = scenario["session_id"]
    consent  = scenario.get("consent_record")

    # ── 1. Hard gate: consent absent or expired ────────────────────────────────
    if consent is None or _consent_expired(consent):
        latency = (time.perf_counter() - t0) * 1000
        log_fn("ethics_consent", sid, "BLOCK", "consent_absent_or_expired")
        ec_out = {
            "status": "BLOCKED",
            "reason": "consent_absent_or_expired",
            "latency_ms": latency,
            "disabled_modalities": [],
            "llm_review": None,
        }
        return {
            **state,
            "blocked": True,
            "block_reason": "consent_absent_or_expired",
            "pipeline_status": "blocked",
            "consent_latency_ms": latency,
            "agent_outputs": {**state["agent_outputs"], "ethics_consent": ec_out},
        }

    # ── 2. Hard gate: requested operation not in permitted scope ───────────────
    requested_op  = scenario.get("requested_operation", "inference")
    permitted_ops = consent.get("permitted_ops", [])
    if requested_op not in permitted_ops:
        latency = (time.perf_counter() - t0) * 1000
        log_fn("ethics_consent", sid, "BLOCK", "operation_out_of_scope",
               {"requested": requested_op, "permitted": permitted_ops})
        ec_out = {
            "status": "BLOCKED",
            "reason": "operation_out_of_scope",
            "latency_ms": latency,
            "disabled_modalities": [],
            "llm_review": None,
        }
        return {
            **state,
            "blocked": True,
            "block_reason": "operation_out_of_scope",
            "pipeline_status": "blocked",
            "consent_latency_ms": latency,
            "agent_outputs": {**state["agent_outputs"], "ethics_consent": ec_out},
        }

    # ── 3. Mid-session scope narrowing ────────────────────────────────────────
    scope_change = scenario.get("consent_scope_change")
    disabled: list[str] = []
    if scope_change:
        disabled = scope_change.get("removed_modalities", [])
        log_fn("ethics_consent", sid, "SCOPE_CHANGE", "mid_session_consent_update",
               {"removed_modalities": disabled})

    # ── 4. LLM: nuanced consent compliance review ──────────────────────────────
    llm_result = call_llm_json(
        system=(
            "You are a clinical ethics governance agent for a paediatric autism screening AI. "
            "Your task is to review a consent record and flag any nuanced compliance concerns "
            "that rule-based checks may miss (e.g., scope mismatch, near-expiry risk, "
            "ambiguous permitted operations). "
            "Respond with a JSON object containing exactly these keys:\n"
            "  \"compliance_level\": \"full\" | \"partial\" | \"concern\"\n"
            "  \"audit_summary\": one sentence for the audit log\n"
            "  \"clinician_note\": one sentence surfaced in the clinician report "
            "(use null if no concern)\n"
            "  \"flags\": list of short flag strings (may be empty)"
        ),
        user=(
            f"permitted_ops: {permitted_ops}\n"
            f"expiry_date: {consent.get('expiry_date')}\n"
            f"requested_operation: {requested_op}\n"
            f"scope_change: {scope_change}\n"
            f"today: {datetime.date.today().isoformat()}"
        ),
        max_tokens=256,
    )

    llm_audit_summary  = None
    llm_clinician_note = None
    llm_flags: list[str] = []
    if llm_result:
        llm_audit_summary  = llm_result.get("audit_summary")
        llm_clinician_note = llm_result.get("clinician_note")
        llm_flags          = llm_result.get("flags", [])

    latency = (time.perf_counter() - t0) * 1000
    log_fn("ethics_consent", sid, "ALLOW", "consent_valid",
           {"llm_audit_summary": llm_audit_summary or "rule-based-only",
            "llm_flags": llm_flags})

    ec_out = {
        "status": "ALLOWED",
        "reason": "consent_valid",
        "latency_ms": latency,
        "disabled_modalities": disabled,
        "consent_scope_updated": bool(scope_change),
        "llm_compliance_level": llm_result.get("compliance_level") if llm_result else None,
        "llm_audit_summary": llm_audit_summary,
        "llm_clinician_note": llm_clinician_note,
        "llm_flags": llm_flags,
    }
    return {
        **state,
        "consent_latency_ms": latency,
        "llm_reasoning": {
            **state["llm_reasoning"],
            "ethics_consent": llm_audit_summary or "",
        },
        "agent_outputs": {**state["agent_outputs"], "ethics_consent": ec_out},
    }
