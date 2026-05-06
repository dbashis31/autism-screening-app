"""
Confidence & Abstention Agent (LangGraph node).

Responsibility: Decide whether to produce a screening result or abstain.

Steps (matching Algorithm 4 flowchart):
  1. Evaluate prediction confidence score C
  2. Evaluate consistency across available signals S
  3. Gate: C < τ_conf? → ABSTAIN
  4. Gate: Inconsistency > τ_var? → ABSTAIN
  5. Gate: Repeated uncertainty? → Escalate for clinician review → ABSTAIN
  6. Record confidence and abstention decision in audit log → PASS

Additional code gates (beyond flowchart):
  - model_rejected → forced ABSTAIN
  - force_abstain flag → ABSTAIN
  - minimum modality count < 2 → ABSTAIN

LLM role: When the deterministic path says REPORT (proceed), Claude
interprets the full confidence score *pattern* — inter-modality agreement,
clinical plausibility, history of prior abstentions — and adds a clinical
interpretation note for the clinician report.

DB operations: abstention writes and escalation checks are done here so
the routing edge after this node remains a pure state read.
"""

import statistics

from .constants import CONFIDENCE_THRESHOLD, INCONSISTENCY_THRESHOLD, MIN_MODALITY_COUNT
from .llm import call_llm_json
from .state import PipelineState


def _compute_inconsistency(scores: dict[str, float]) -> float:
    """Compute cross-modal variance as an inconsistency measure."""
    if len(scores) < 2:
        return 0.0
    vals = list(scores.values())
    return statistics.variance(vals)


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

    def _check_escalation() -> bool:
        history = db_ops["get_abstention_history"](child_id)
        count = len(history)
        if count >= 2:
            log_fn(
                "human_in_the_loop", sid,
                "ESCALATION_QUEUED",
                f"consecutive_abstentions_{count}",
                {"child_id": child_id, "abstention_count": count},
            )
            return True
        return False

    # ── Rule: model was rejected by model_selection agent ─────────────────────
    if state["model_rejected"]:
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "model_rejected_forced_abstention")
        _record_abstention("model_not_approved")
        _check_escalation()
        ca_out = {"status": "ABSTAIN", "reason": "model_not_approved"}
        return {
            **state,
            "abstaining": True, "abstention_reason": "model_not_approved",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── Rule: force_abstain flag ──────────────────────────────────────────────
    if scenario.get("force_abstain"):
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "insufficient_confidence_data")
        _record_abstention("insufficient_confidence_data")
        _check_escalation()
        ca_out = {"status": "ABSTAIN", "reason": "insufficient_confidence_data"}
        return {
            **state,
            "abstaining": True, "abstention_reason": "insufficient_confidence_data",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── Step 1: Evaluate prediction confidence score C ────────────────────────
    scores = state.get("confidence_scores") or {}
    active = {m: scores[m] for m in enabled if m in scores}

    # ── Rule: minimum modality count ──────────────────────────────────────────
    if len(active) < MIN_MODALITY_COUNT:
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "insufficient_modalities", {"active_count": len(active)})
        _record_abstention("insufficient_modalities")
        _check_escalation()
        ca_out = {"status": "ABSTAIN", "reason": "insufficient_modalities",
                  "active_count": len(active)}
        return {
            **state,
            "abstaining": True, "abstention_reason": "insufficient_modalities",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── Gate: C < τ_conf? → ABSTAIN (per-modality confidence threshold) ──────
    low_conf = [m for m, s in active.items() if s < CONFIDENCE_THRESHOLD]
    if low_conf:
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "low_confidence_modalities",
               {"low": low_conf, "scores": active, "threshold": CONFIDENCE_THRESHOLD})
        _record_abstention("low_confidence")
        _check_escalation()
        ca_out = {"status": "ABSTAIN", "reason": "low_confidence",
                  "affected_modalities": low_conf, "scores": active}
        return {
            **state,
            "abstaining": True, "abstention_reason": "low_confidence",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── Step 2 + Gate: Inconsistency > τ_var? → ABSTAIN ──────────────────────
    inconsistency = _compute_inconsistency(active)
    if inconsistency > INCONSISTENCY_THRESHOLD:
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "inter_modal_inconsistency_exceeds_threshold",
               {"inconsistency": round(inconsistency, 4),
                "threshold": INCONSISTENCY_THRESHOLD, "scores": active})
        _record_abstention("inter_modal_inconsistency")
        _check_escalation()
        ca_out = {"status": "ABSTAIN", "reason": "inter_modal_inconsistency",
                  "inconsistency": round(inconsistency, 4),
                  "threshold": INCONSISTENCY_THRESHOLD, "scores": active}
        return {
            **state,
            "abstaining": True, "abstention_reason": "inter_modal_inconsistency",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── Gate: Repeated uncertainty? → Escalate → ABSTAIN ──────────────────────
    history = db_ops["get_abstention_history"](child_id)
    if len(history) >= 2:
        log_fn("confidence_abstention", sid, "ABSTAIN",
               "repeated_uncertainty_escalation",
               {"child_id": child_id, "prior_abstentions": len(history)})
        log_fn("human_in_the_loop", sid, "ESCALATION_QUEUED",
               f"repeated_uncertainty_{len(history)}_prior_abstentions",
               {"child_id": child_id, "abstention_count": len(history)})
        _record_abstention("repeated_uncertainty")
        ca_out = {"status": "ABSTAIN", "reason": "repeated_uncertainty",
                  "prior_abstention_count": len(history), "escalated": True}
        return {
            **state,
            "abstaining": True, "abstention_reason": "repeated_uncertainty",
            "agent_outputs": {**state["agent_outputs"], "confidence_abstention": ca_out},
        }

    # ── All rules passed — LLM interprets the confidence pattern ─────────────
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
            f"Cross-modal inconsistency (variance): {round(inconsistency, 4)} "
            f"(threshold: {INCONSISTENCY_THRESHOLD})\n"
            f"Confidence threshold: {CONFIDENCE_THRESHOLD} per modality\n"
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
           {"scores": active, "inconsistency": round(inconsistency, 4),
            "llm_agreement": llm_agreement,
            "llm_interpretation": llm_interpretation or "rule-based-only"})

    ca_out = {
        "status": "REPORT",
        "confidence_scores": active,
        "inconsistency": round(inconsistency, 4),
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
