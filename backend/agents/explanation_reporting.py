"""
Explanation & Reporting Agent (LangGraph node).

Responsibility: Generate role-gated, vocabulary-compliant output.

Steps (matching Algorithm 5 flowchart):
  1. Determine user role U
  2. Gate: Caregiver → generate non-diagnostic summary
  3. Gate: Clinician → generate structured decision-support report
  4. Remove diagnostic language from caregiver output (explicit filter)
  5. Include confidence indicators and abstention status when applicable
  6. Attach relevant audit metadata for traceability
  7. Record report-generation event in audit log
  Output: Return Role-Appropriate Report R

Additional code features (beyond flowchart):
  - Role escalation block: caregiver requesting clinician_report → BLOCK
  - Abstention vs full report branching

LLM role: Claude synthesises evidence from all prior agent outputs (consent review,
bias risk, model security, confidence interpretation) into a structured diagnostic
support report for the clinician. The report is calibrated to the outcome
(abstention vs full screening) and always recommends clinician review.
"""

import json
import re

from .constants import APPROVED_CAREGIVER_VOCAB, DIAGNOSTIC_TERMS
from .llm import call_llm_json
from .state import PipelineState


# ── Step 4: Diagnostic language filter ────────────────────────────────────────

def _remove_diagnostic_language(text: str) -> str:
    """Remove or replace diagnostic terms from caregiver-facing text."""
    result = text
    for term in sorted(DIAGNOSTIC_TERMS, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub("developmental", result)
    return result


# ── LLM prompt builders ──────────────────────────────────────────────────────

def _build_report_context(state: PipelineState) -> dict:
    """Collects all upstream agent reasoning into a single context dict for Claude."""
    scenario  = state["scenario"]
    ca_out    = state["agent_outputs"].get("confidence_abstention", {})
    ba_out    = state["agent_outputs"].get("bias_applicability", {})
    ec_out    = state["agent_outputs"].get("ethics_consent", {})
    ms_out    = state["agent_outputs"].get("model_selection", {})
    return {
        "session_id":            scenario["session_id"],
        "child_id":              scenario.get("child_id"),
        "child_age_months":      scenario.get("child_age_months"),
        "enabled_modalities":    state["enabled_modalities"],
        "confidence_scores":     state["confidence_scores"],
        "applicability_warnings": state["applicability_warnings"],
        "abstaining":            state["abstaining"],
        "abstention_reason":     state.get("abstention_reason"),
        "model_id":              scenario.get("model_id"),
        "prior_agent_reasoning": {
            "consent_review":      ec_out.get("llm_audit_summary"),
            "consent_flags":       ec_out.get("llm_flags", []),
            "bias_risk_level":     ba_out.get("llm_risk_level"),
            "bias_narrative":      ba_out.get("llm_risk_narrative"),
            "model_security":      ms_out.get("llm_provenance_assessment"),
            "confidence_interp":   ca_out.get("llm_confidence_interpretation"),
            "modality_agreement":  ca_out.get("llm_inter_modality_agreement"),
            "clinical_note":       ca_out.get("llm_clinical_note"),
            "reliability_flag":    ca_out.get("llm_reliability_flag"),
        },
    }


def _generate_abstention_report(state: PipelineState) -> dict:
    """Claude generates a structured abstention clinician report."""
    context    = _build_report_context(state)
    llm_result = call_llm_json(
        system=(
            "You are an expert clinical AI reporting agent for a paediatric autism "
            "screening platform. The screening model has ABSTAINED — it declined to "
            "produce a result. Generate a structured clinician report explaining why "
            "and what the clinician should do next. The report must be factual, "
            "evidence-based, and actionable. Always end with a clear recommendation.\n"
            "Respond with JSON containing exactly:\n"
            "  \"type\": \"abstention\"\n"
            "  \"abstention_summary\": 2-3 sentences summarising why the model abstained\n"
            "  \"data_quality_assessment\": 1-2 sentences on available data quality\n"
            "  \"applicability_concerns\": list of concern strings (from bias/consent agents)\n"
            "  \"recommendations\": list of 2-4 specific next-step strings for the clinician\n"
            "  \"uncertainty_bounds\": \"N/A (abstention)\"\n"
            "  \"generated_by\": \"claude-llm\""
        ),
        user=json.dumps(context, default=str),
        max_tokens=600,
    )

    if llm_result and isinstance(llm_result, dict):
        llm_result.setdefault("generated_by", "claude-llm")
        return llm_result

    return {
        "type": "abstention",
        "abstention_summary": state.get("abstention_reason") or "see_confidence_agent_log",
        "data_quality_assessment": "Insufficient data quality or modality coverage.",
        "applicability_concerns": state["applicability_warnings"],
        "recommendations": ["Schedule follow-up screening with complete modality set."],
        "uncertainty_bounds": "N/A (abstention)",
        "generated_by": "rule-based",
    }


def _generate_full_report(state: PipelineState) -> dict:
    """Claude generates a comprehensive diagnostic support report."""
    context    = _build_report_context(state)
    llm_result = call_llm_json(
        system=(
            "You are an expert clinical AI reporting agent for a paediatric autism "
            "screening platform. Generate a comprehensive diagnostic support report "
            "for the clinician based on the multimodal screening result. This is a "
            "SUPPORT tool — always emphasise that clinical judgement is required.\n"
            "The report must synthesise: confidence scores, modality evidence, "
            "bias/applicability concerns, model security posture, and all prior "
            "agent reasoning passed in the context.\n"
            "Respond with JSON containing exactly:\n"
            "  \"type\": \"full_diagnostic_support\"\n"
            "  \"screening_summary\": 2-3 sentences summarising the overall result\n"
            "  \"modality_interpretation\": object mapping each modality to a 1-sentence interpretation\n"
            "  \"confidence_assessment\": 2 sentences on score reliability and agreement\n"
            "  \"applicability_notes\": list of strings (may be empty)\n"
            "  \"clinical_recommendations\": list of 2-4 specific next-step strings\n"
            "  \"uncertainty_bounds\": string (e.g. '95% CI' or specific range)\n"
            "  \"limitations\": 1-2 sentences on what the model cannot determine\n"
            "  \"generated_by\": \"claude-llm\""
        ),
        user=json.dumps(context, default=str),
        max_tokens=900,
    )

    if llm_result and isinstance(llm_result, dict):
        llm_result.setdefault("generated_by", "claude-llm")
        llm_result["modalities_used"] = state["enabled_modalities"]
        return llm_result

    return {
        "type": "full_diagnostic_support",
        "screening_summary": "Screening complete with sufficient modality coverage.",
        "modalities_used": state["enabled_modalities"],
        "confidence_assessment": "All active modalities met the confidence threshold.",
        "applicability_notes": state["applicability_warnings"],
        "clinical_recommendations": ["Review confidence scores and contact caregiver for follow-up."],
        "uncertainty_bounds": "95% CI",
        "limitations": "This tool supports — does not replace — clinician assessment.",
        "generated_by": "rule-based",
    }


# ── LangGraph node ─────────────────────────────────────────────────────────────

def explanation_reporting_node(state: PipelineState) -> PipelineState:
    scenario = state["scenario"]
    log_fn   = state["log_fn"]
    sid      = scenario["session_id"]
    role        = scenario.get("role", "caregiver")
    report_type = scenario.get("report_type", "standard")
    abstaining  = state["abstaining"]

    # ── Role escalation block ─────────────────────────────────────────────────
    if role == "caregiver" and report_type == "clinician_report":
        log_fn("explanation_reporting", sid, "BLOCK",
               "unauthorized_role_escalation",
               {"role": role, "requested": report_type})
        caregiver_text = _remove_diagnostic_language(APPROVED_CAREGIVER_VOCAB[2])
        er_out = {
            "status": "BLOCKED",
            "reason": "role_not_authorized_for_clinician_report",
            "caregiver_report": caregiver_text,
            "clinician_report": None,
        }
        return {
            **state,
            "blocked": True,
            "block_reason": "role_not_authorized_for_clinician_report",
            "pipeline_status": "blocked",
            "caregiver_report": caregiver_text,
            "clinician_report": None,
            "agent_outputs": {**state["agent_outputs"], "explanation_reporting": er_out},
        }

    # ── Step 1: Determine user role U ─────────────────────────────────────────
    is_clinician = role == "clinician"

    # ── Abstention path ───────────────────────────────────────────────────────
    if abstaining:
        clinician_report = _generate_abstention_report(state) if is_clinician else None
        # Step 2/4: Caregiver gets approved vocab with diagnostic language removed
        caregiver_text = _remove_diagnostic_language(APPROVED_CAREGIVER_VOCAB[1])

        log_fn("explanation_reporting", sid, "ABSTENTION_REPORT",
               "abstention_output_generated",
               {"role": role,
                "generated_by": clinician_report.get("generated_by") if clinician_report else "vocab-only",
                "abstention_reason": state.get("abstention_reason")})

        # Step 5: Include confidence indicators and abstention status
        er_out = {
            "status": "ABSTENTION",
            "role": role,
            "caregiver_report": caregiver_text,
            "clinician_report": clinician_report,
            "abstention_reason": state.get("abstention_reason"),
            "confidence_scores": state["confidence_scores"],
        }
        return {
            **state,
            "pipeline_status": "abstained",
            "caregiver_report": caregiver_text,
            "clinician_report": clinician_report,
            "agent_outputs": {**state["agent_outputs"], "explanation_reporting": er_out},
        }

    # ── Full report path ──────────────────────────────────────────────────────
    # Step 3: Clinician gets structured decision-support report
    clinician_report = _generate_full_report(state) if is_clinician else None
    # Step 2/4: Caregiver gets approved vocab with diagnostic language removed
    caregiver_text = _remove_diagnostic_language(APPROVED_CAREGIVER_VOCAB[0])

    # Step 6: Attach audit metadata for traceability
    if clinician_report and isinstance(clinician_report, dict):
        clinician_report["audit_metadata"] = {
            "session_id": sid,
            "enabled_modalities": state["enabled_modalities"],
            "applicability_warnings": state["applicability_warnings"],
            "agent_chain": list(state["agent_outputs"].keys()),
            "llm_reasoning_keys": list(state["llm_reasoning"].keys()),
        }

    # Step 7: Record report-generation event in audit log
    log_fn("explanation_reporting", sid, "REPORT_GENERATED", "output_complete",
           {"role": role, "modalities": state["enabled_modalities"],
            "generated_by": clinician_report.get("generated_by") if clinician_report else "vocab-only"})

    er_out = {
        "status": "REPORTED",
        "role": role,
        "caregiver_report": caregiver_text,
        "clinician_report": clinician_report,
    }
    return {
        **state,
        "pipeline_status": "complete",
        "caregiver_report": caregiver_text,
        "clinician_report": clinician_report,
        "agent_outputs": {**state["agent_outputs"], "explanation_reporting": er_out},
    }
