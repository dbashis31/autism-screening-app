"""
Model Selection Agent (LangGraph node).

Responsibility: Verify the model is signed and in the approved registry.
A rejected model does NOT block the pipeline — it forces abstention in
the next stage (confidence_abstention) so the session is logged and the
clinician is notified.

LLM role: Perform a security provenance review on the model identifier —
flag unusual naming patterns (unsigned, pre-release, unversioned) that
the registry whitelist alone may not anticipate for future model IDs.
"""

from .constants import APPROVED_MODEL_REGISTRY
from .llm import call_llm_json
from .state import PipelineState


def model_selection_node(state: PipelineState) -> PipelineState:
    scenario = state["scenario"]
    log_fn   = state["log_fn"]
    sid      = scenario["session_id"]
    model_id = scenario.get("model_id", "model-v2.1-signed")

    # ── Deterministic registry whitelist check ─────────────────────────────────
    if model_id not in APPROVED_MODEL_REGISTRY:
        log_fn("model_selection", sid, "REJECT", "model_not_in_approved_registry",
               {"model_id": model_id})
        ms_out = {
            "status": "REJECTED",
            "reason": "model_unsigned_or_not_approved",
            "model_id": model_id,
            "llm_assessment": None,
        }
        return {
            **state,
            "model_rejected": True,
            "agent_outputs": {**state["agent_outputs"], "model_selection": ms_out},
        }

    # ── LLM: security provenance reasoning ────────────────────────────────────
    llm_result = call_llm_json(
        system=(
            "You are a model governance and security agent for a clinical AI deployment. "
            "Assess the security posture and provenance of a model identifier that has "
            "already passed a registry whitelist check. Consider: version naming conventions, "
            "signing status implied by the identifier, whether the version is current or "
            "legacy, any red flags in the naming pattern. "
            "Respond with JSON containing exactly:\n"
            "  \"security_level\": \"high\" | \"medium\" | \"low\"\n"
            "  \"provenance_assessment\": 1-2 sentences for the audit log\n"
            "  \"flags\": list of concern strings (may be empty)\n"
            "  \"approved\": true (always true here — model passed registry check)"
        ),
        user=(
            f"Model ID: {model_id}\n"
            f"Approved registry: {sorted(APPROVED_MODEL_REGISTRY)}\n"
            f"Registry check result: PASSED"
        ),
        max_tokens=256,
    )

    llm_assessment = None
    llm_security_level = "unknown"
    llm_flags: list[str] = []
    if llm_result:
        llm_assessment     = llm_result.get("provenance_assessment")
        llm_security_level = llm_result.get("security_level", "unknown")
        llm_flags          = llm_result.get("flags", [])

    log_fn("model_selection", sid, "EXECUTE", "model_approved",
           {"model_id": model_id,
            "llm_security_level": llm_security_level,
            "llm_assessment": llm_assessment or "rule-based-only"})

    ms_out = {
        "status": "APPROVED",
        "model_id": model_id,
        "llm_security_level": llm_security_level,
        "llm_provenance_assessment": llm_assessment,
        "llm_flags": llm_flags,
    }
    return {
        **state,
        "model_rejected": False,
        "llm_reasoning": {
            **state["llm_reasoning"],
            "model_selection": llm_assessment or "",
        },
        "agent_outputs": {**state["agent_outputs"], "model_selection": ms_out},
    }
