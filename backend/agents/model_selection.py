"""
Model Selection & Execution Agent (LangGraph node).

Responsibility: Select, verify, and execute an approved model for inference.

Steps (matching Algorithm 3 flowchart):
  1. Identify required modality type T from D
  2. Query model registry MR for approved models supporting T
  3. Gate: Approved model available? → No → DENY
  4. Select model M based on modality, version, and approval flag
  5. Verify model signature, version, and deployment status
  6. Gate: Verification passed? → No → DENY
  7. Execute inference using selected model M
  Output: Return Prediction P and Confidence Score C

LLM role: Perform a security provenance review on the model identifier —
flag unusual naming patterns (unsigned, pre-release, unversioned) that
the registry whitelist alone may not anticipate for future model IDs.
"""

from .constants import APPROVED_MODEL_REGISTRY
from .llm import call_llm_json
from .state import PipelineState


def _find_models_for_modalities(
    required_modalities: set[str],
) -> list[tuple[str, dict]]:
    """Query the registry for models that support all required modalities."""
    candidates = []
    for model_id, meta in APPROVED_MODEL_REGISTRY.items():
        if required_modalities.issubset(meta["modalities"]):
            candidates.append((model_id, meta))
    candidates.sort(key=lambda x: x[1]["version"], reverse=True)
    return candidates


def _verify_model(model_id: str, meta: dict) -> tuple[bool, str | None]:
    """Verify model signature, version, and deployment status."""
    if not meta.get("signed"):
        return False, "model_not_signed"
    if meta.get("status") != "deployed":
        return False, f"model_status_{meta.get('status', 'unknown')}"
    if not meta.get("version"):
        return False, "model_version_missing"
    return True, None


def _execute_inference(
    model_id: str, scenario: dict, enabled_modalities: list[str],
) -> dict:
    """
    Execute inference using selected model M.

    In production this calls the ML serving layer. Here we read
    pre-computed confidence scores from the scenario payload, which is
    the standard integration pattern for governance-wrapped pipelines
    where inference runs externally and results are validated here.
    """
    scores = scenario.get("confidence_scores", {})
    prediction = scenario.get("prediction", {})
    active_scores = {m: scores[m] for m in enabled_modalities if m in scores}

    overall = (
        sum(active_scores.values()) / len(active_scores)
        if active_scores else 0.0
    )

    return {
        "model_id": model_id,
        "prediction": prediction,
        "confidence_scores": active_scores,
        "overall_confidence": round(overall, 4),
    }


# ── LangGraph node ─────────────────────────────────────────────────────────────

def model_selection_node(state: PipelineState) -> PipelineState:
    scenario = state["scenario"]
    log_fn   = state["log_fn"]
    sid      = scenario["session_id"]
    enabled  = state["enabled_modalities"]

    # ── Step 1: Identify required modality types from D ───────────────────────
    required_modalities = set(enabled)

    # ── Step 2: Query model registry for approved models supporting T ─────────
    explicit_model = scenario.get("model_id")
    candidates = _find_models_for_modalities(required_modalities)

    # ── Gate: Approved model available? ───────────────────────────────────────
    if explicit_model:
        matched = [
            (mid, meta) for mid, meta in candidates if mid == explicit_model
        ]
        if not matched:
            if explicit_model not in APPROVED_MODEL_REGISTRY:
                reason = "model_not_in_approved_registry"
            else:
                reason = "model_does_not_support_required_modalities"
            log_fn("model_selection", sid, "REJECT", reason,
                   {"model_id": explicit_model,
                    "required_modalities": sorted(required_modalities)})
            ms_out = {
                "status": "REJECTED",
                "reason": reason,
                "model_id": explicit_model,
                "llm_assessment": None,
            }
            return {
                **state,
                "model_rejected": True,
                "agent_outputs": {
                    **state["agent_outputs"], "model_selection": ms_out,
                },
            }
        candidates = matched

    if not candidates:
        log_fn("model_selection", sid, "REJECT",
               "no_approved_model_for_modalities",
               {"required_modalities": sorted(required_modalities)})
        ms_out = {
            "status": "REJECTED",
            "reason": "no_approved_model_for_modalities",
            "model_id": None,
            "llm_assessment": None,
        }
        return {
            **state,
            "model_rejected": True,
            "agent_outputs": {
                **state["agent_outputs"], "model_selection": ms_out,
            },
        }

    # ── Step 4: Select model M (highest version among candidates) ─────────────
    selected_id, selected_meta = candidates[0]

    # ── Step 5 + Gate: Verify model signature, version, deployment status ─────
    verified, verify_reason = _verify_model(selected_id, selected_meta)
    if not verified:
        log_fn("model_selection", sid, "REJECT", "model_verification_failed",
               {"model_id": selected_id, "reason": verify_reason})
        ms_out = {
            "status": "REJECTED",
            "reason": verify_reason,
            "model_id": selected_id,
            "llm_assessment": None,
        }
        return {
            **state,
            "model_rejected": True,
            "agent_outputs": {
                **state["agent_outputs"], "model_selection": ms_out,
            },
        }

    # ── LLM: security provenance reasoning ────────────────────────────────────
    llm_result = call_llm_json(
        system=(
            "You are a model governance and security agent for a clinical AI deployment. "
            "Assess the security posture and provenance of a model identifier that has "
            "already passed a registry whitelist check and signature verification. Consider: "
            "version naming conventions, signing status, whether the version is current or "
            "legacy, any red flags in the naming pattern. "
            "Respond with JSON containing exactly:\n"
            "  \"security_level\": \"high\" | \"medium\" | \"low\"\n"
            "  \"provenance_assessment\": 1-2 sentences for the audit log\n"
            "  \"flags\": list of concern strings (may be empty)\n"
            "  \"approved\": true (always true here — model passed all checks)"
        ),
        user=(
            f"Model ID: {selected_id}\n"
            f"Version: {selected_meta['version']}\n"
            f"Signed: {selected_meta['signed']}\n"
            f"Status: {selected_meta['status']}\n"
            f"Supported modalities: {sorted(selected_meta['modalities'])}\n"
            f"Required modalities: {sorted(required_modalities)}\n"
            f"All registry models: {sorted(APPROVED_MODEL_REGISTRY.keys())}"
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

    # ── Step 7: Execute inference using selected model M ──────────────────────
    inference_result = _execute_inference(selected_id, scenario, enabled)

    log_fn("model_selection", sid, "EXECUTE", "model_approved_inference_complete",
           {"model_id": selected_id,
            "overall_confidence": inference_result["overall_confidence"],
            "llm_security_level": llm_security_level,
            "llm_assessment": llm_assessment or "rule-based-only"})

    ms_out = {
        "status": "APPROVED",
        "model_id": selected_id,
        "model_version": selected_meta["version"],
        "inference_result": inference_result,
        "llm_security_level": llm_security_level,
        "llm_provenance_assessment": llm_assessment,
        "llm_flags": llm_flags,
    }
    return {
        **state,
        "model_rejected": False,
        "confidence_scores": inference_result["confidence_scores"],
        "llm_reasoning": {
            **state["llm_reasoning"],
            "model_selection": llm_assessment or "",
        },
        "agent_outputs": {**state["agent_outputs"], "model_selection": ms_out},
    }
