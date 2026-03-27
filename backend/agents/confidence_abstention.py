"""
Confidence & Abstention Agent (LangGraph node).

Responsibility: Decide whether to produce a screening result or abstain.
Deterministic rules: confidence threshold (0.65), ≥2 active modalities,
cross-modal conflict flag, force_abstain flag.

LLM role: When the deterministic path says REPORT (proceed), Claude
interprets the full confidence score *pattern* — inter-modality agreement,
clinical plausibility, history of prior abstentions — and adds a clinical
interpretation note for the clinician report. This gives clinicians richer
context than a raw number.

DB operations: abstention writes and escalation checks are done here so
the routing edge after this node remains a pure state read.
"""

from .llm import call_llm_json
from .state import PipelineState


def confidence_abstention_node(state: PipelineState) -> PipelineState:
    scenario  = state["scenario"]
    log_fn    = state["log_fn"]
    db_ops    = state["db_ops"]
    sid       = scenario["session_id"]
    child_id  = scenario.get("child_id", "unknown")
    enabled   = state["enabled_modalities"]

    # ── Helper: record abstention in DB and trigger escalation if needed ───────
    def _record_abstention(reason: str) -> None:
        db_ops["write_abstention"](child_id, sid, reason)
        history = db_ops["get_abstention_history"](child_id)
        count = len(history)
        if count >= 2:
            log_fn(
                "human_in_the_loop", sid,
                "ESCALATION_QUEUED",
                f"consecutive_abstentions_{count}",
                {"child_id": child_id, "abstention_count": count},
            )

    # ── Rule 1: model was rejected by model_selection agent ───────────────────
    if state["model_rejected"]:
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "model_rejected_forced_abstention")
        _record_abstention("model_not_approved")
        ca_out = {"status": "ABSTAIN", "reason": "model_not_approved"}
        return {
            **state,
            "abstaining": True, "abstention_reason": "model_not_approved",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── Rule 2: force_abstain flag ────────────────────────────────────────────
    if scenario.get("force_abstain"):
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "insufficient_confidence_data")
        _record_abstention("insufficient_confidence_data")
        ca_out = {"status": "ABSTAIN", "reason": "insufficient_confidence_data"}
        return {
            **state,
            "abstaining": True, "abstention_reason": "insufficient_confidence_data",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── Rule 3: cross-modal conflict ──────────────────────────────────────────
    if scenario.get("cross_modal_conflict"):
        scores = scenario.get("confidence_scores", {})
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "inter_modal_conflict_detected", {"scores": scores})
        _record_abstention("inter_modal_conflict")
        ca_out = {"status": "ABSTAIN", "reason": "inter_modal_conflict",
                  "conflict_scores": scores}
        return {
            **state,
            "abstaining": True, "abstention_reason": "inter_modal_conflict",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── Rule 4: per-modality confidence threshold ─────────────────────────────
    scores = scenario.get("confidence_scores", {})
    active = {m: scores[m] for m in enabled if m in scores}

    low_conf = [m for m, s in active.items() if s < 0.65]
    if low_conf:
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "low_confidence_modalities",
               {"low": low_conf, "scores": active})
        _record_abstention("low_confidence")
        ca_out = {"status": "ABSTAIN", "reason": "low_confidence",
                  "affected_modalities": low_conf, "scores": active}
        return {
            **state,
            "abstaining": True, "abstention_reason": "low_confidence",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── Rule 5: minimum modality count ───────────────────────────────────────
    if len(active) < 2:
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "insufficient_modalities", {"active_count": len(active)})
        _record_abstention("insufficient_modalities")
        ca_out = {"status": "ABSTAIN", "reason": "insufficient_modalities",
                  "active_count": len(active)}
        return {
            **state,
            "abstaining": True, "abstention_reason": "insufficient_modalities",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── All rules passed — LLM interprets the confidence pattern ─────────────
    history = db_ops["get_abstention_history"](child_id)

    llm_result = call_llm_json(
        system=(
            "You are a clinical confidence evaluation agent for a paediatric autism "
            "screening AI. All deterministic thresholds have been met. Your task is to "
            "interpret the *pattern* of confidence scores across modalities and provide "
            "a clinical interpretation for the clinician report. Consider: inter-modality "
            "agreement (high agreement = reliable), any notable discordance, history of "
            "prior abstentions for this child, and any applicability concerns. "
            "Respond with JSON containing exactly:\n"
            "  \"confidence_interpretation\": 2-3 sentences interpreting the score pattern\n"
            "  \"inter_modality_agreement\": \"high\" | \"moderate\" | \"low\"\n"
            "  \"clinical_note\": one sentence to include in the clinician report\n"
            "  \"reliability_flag\": null or a short concern string if confidence is marginal"
        ),
        user=(
            f"Active modality scores: {active}\n"
            f"Abstention threshold: 0.65 per modality\n"
            f"Applicability warnings: {state['applicability_warnings']}\n"
            f"Prior abstentions for this child: {len(history)}\n"
            f"Bias risk narrative: "
            f"{state['agent_outputs'].get('bias_applicability', {}).get('llm_risk_narrative', 'N/A')}"
        ),
        max_tokens=384,
    )

    llm_interpretation = None
    llm_agreement      = "unknown"
    llm_clinical_note  = None
    llm_reliability    = None
    if llm_result:
        llm_interpretation = llm_result.get("confidence_interpretation")
        llm_agreement      = llm_result.get("inter_modality_agreement", "unknown")
        llm_clinical_note  = llm_result.get("clinical_note")
        llm_reliability    = llm_result.get("reliability_flag")

    log_fn("confidence_abstention", sid, "REPORT",
           "confidence_thresholds_met",
           {"scores": active,
            "llm_agreement": llm_agreement,
            "llm_interpretation": llm_interpretation or "rule-based-only"})

    ca_out = {
        "status": "REPORT",
        "confidence_scores": active,
        "llm_confidence_interpretation": llm_interpretation,
        "llm_inter_modality_agreement": llm_agreement,
        "llm_clinical_note": llm_clinical_note,
        "llm_reliability_flag": llm_reliability,
    }
    return {
        **state,
        "abstaining": False,
        "confidence_scores": active,
        "llm_reasoning": {
            **state["llm_reasoning"],
            "confidence_abstention": llm_interpretation or "",
        },
        "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
    }
