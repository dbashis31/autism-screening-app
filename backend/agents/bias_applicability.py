"""
Bias & Applicability Agent (LangGraph node).

Responsibility: Validate modality quality and demographic applicability.

Steps (matching Algorithm 2 flowchart):
  1. Check whether D contains required modality-specific inputs
  2. Assess input completeness, quality, and demographic metadata
  3. Compare M against known model applicability boundaries
  4. Estimate bias or applicability risk score B
  5. Gate: Required data missing? → ABSTAIN
  6. Gate: B > τ_bias? → ABSTAIN
  7. Record applicability and bias decision in audit log → PASS

LLM role: Assess *combined* bias risk that emerges from the interaction of
multiple factors (e.g., low-SNR audio + age boundary + limited modalities).
Returns a numeric risk_score used for the τ_bias threshold gate, plus
plain-English warning narratives for clinicians.
"""

from .constants import (
    AGE_RANGE_MONTHS,
    AUDIO_SNR_THRESHOLD_DB,
    BIAS_RISK_THRESHOLD,
    MIN_MODALITY_COUNT,
)
from .llm import call_llm_json
from .state import PipelineState


def bias_applicability_node(state: PipelineState) -> PipelineState:
    scenario = state["scenario"]
    log_fn   = state["log_fn"]
    sid      = scenario["session_id"]

    # Inherit any modalities already disabled by the ethics agent
    ec_out        = state["agent_outputs"].get("ethics_consent", {})
    extra_disabled = ec_out.get("disabled_modalities", [])

    enabled: list[str] = list(
        scenario.get("modalities", ["audio", "video", "text", "questionnaire"])
    )
    for mod in extra_disabled:
        if mod in enabled:
            enabled.remove(mod)

    rule_warnings: list[str] = []

    # ── Step 1: Check whether D contains required modality-specific inputs ────
    submitted_data = scenario.get("submitted_data", {})
    missing_modalities: list[str] = []
    for mod in list(enabled):
        if mod not in submitted_data or not submitted_data[mod]:
            missing_modalities.append(mod)
            enabled.remove(mod)

    if missing_modalities:
        log_fn("bias_applicability", sid, "MODALITY_DISABLED",
               "missing_modality_data",
               {"missing": missing_modalities})
        rule_warnings.append(f"missing_modality_data:{missing_modalities}")

    # ── Step 2: Assess input quality — audio SNR threshold ────────────────────
    snr = scenario.get("audio_snr_db")
    if snr is not None and snr < AUDIO_SNR_THRESHOLD_DB:
        if "audio" in enabled:
            enabled.remove("audio")
        log_fn("bias_applicability", sid, "MODALITY_DISABLED",
               "audio_snr_below_threshold",
               {"audio_snr_db": snr, "threshold_db": AUDIO_SNR_THRESHOLD_DB})
        rule_warnings.append(f"audio_degraded_excluded:snr={snr}dB<{AUDIO_SNR_THRESHOLD_DB}dB_threshold")

    # ── Step 3: Compare against model applicability boundaries (age) ──────────
    age = scenario.get("child_age_months")
    min_age, max_age = AGE_RANGE_MONTHS
    if age is not None and (age <= min_age or age >= max_age):
        log_fn("bias_applicability", sid, "WARNING", "age_edge_case",
               {"age_months": age, "validated_range": f"{min_age}–{max_age} months"})
        rule_warnings.append(f"age_applicability_warning:{age}mo_at_boundary")

    # ── Gate: Required data missing? → ABSTAIN ────────────────────────────────
    if len(enabled) < MIN_MODALITY_COUNT:
        log_fn("bias_applicability", sid, "ABSTAIN",
               "required_data_missing",
               {"enabled_modalities": enabled, "minimum_required": MIN_MODALITY_COUNT,
                "missing": missing_modalities})
        ba_out = {
            "status": "ABSTAIN",
            "reason": "required_data_missing",
            "enabled_modalities": enabled,
            "warnings": rule_warnings,
            "llm_risk_level": None,
            "llm_risk_narrative": None,
            "llm_recommendation": None,
        }
        return {
            **state,
            "enabled_modalities": enabled,
            "applicability_warnings": rule_warnings,
            "bias_abstain": True,
            "abstaining": True,
            "abstention_reason": "required_data_missing",
            "pipeline_status": "abstained",
            "agent_outputs": {**state["agent_outputs"], "bias_applicability": ba_out},
        }

    # ── Step 4: LLM estimates combined bias / applicability risk score B ──────
    llm_result = call_llm_json(
        system=(
            "You are a bias and applicability governance agent for a paediatric autism "
            "screening AI. Assess the *combined* risk arising from the interaction of "
            "modality availability, signal quality, and demographic factors. "
            "Consider: reduced modality set leading to one-sided evidence, age boundary "
            "effects on model calibration, compounding of multiple risk factors. "
            "Respond with JSON containing exactly:\n"
            "  \"risk_score\": float between 0.0 and 1.0 (probability of bias affecting result)\n"
            "  \"risk_level\": \"low\" | \"medium\" | \"high\"\n"
            "  \"risk_narrative\": 1-2 sentence plain-English summary for clinicians\n"
            "  \"additional_warnings\": list of short warning strings (may be empty)\n"
            "  \"recommendation\": one sentence on how to handle this screening"
        ),
        user=(
            f"Enabled modalities after filtering: {enabled}\n"
            f"Audio SNR: {snr} dB (threshold: {AUDIO_SNR_THRESHOLD_DB} dB, None = not submitted)\n"
            f"Child age: {age} months (validated range: {min_age}–{max_age} months, None = unknown)\n"
            f"Rule-based warnings already raised: {rule_warnings}\n"
            f"Applicability warnings from consent agent: "
            f"{state['agent_outputs'].get('ethics_consent', {}).get('llm_flags', [])}"
        ),
        max_tokens=320,
    )

    warnings = list(rule_warnings)
    llm_narrative      = None
    llm_risk_level     = "unknown"
    llm_risk_score     = 0.0
    llm_recommendation = None
    if llm_result:
        llm_narrative      = llm_result.get("risk_narrative")
        llm_risk_level     = llm_result.get("risk_level", "unknown")
        llm_recommendation = llm_result.get("recommendation")
        extra_warnings     = llm_result.get("additional_warnings", [])
        if isinstance(extra_warnings, list):
            warnings.extend(extra_warnings)
        try:
            llm_risk_score = float(llm_result.get("risk_score", 0.0))
        except (TypeError, ValueError):
            llm_risk_score = 0.0

    # ── Gate: B > τ_bias? → ABSTAIN ───────────────────────────────────────────
    if llm_risk_score > BIAS_RISK_THRESHOLD:
        log_fn("bias_applicability", sid, "ABSTAIN",
               "bias_risk_exceeds_threshold",
               {"risk_score": llm_risk_score, "threshold": BIAS_RISK_THRESHOLD,
                "risk_level": llm_risk_level})
        ba_out = {
            "status": "ABSTAIN",
            "reason": "bias_risk_exceeds_threshold",
            "enabled_modalities": enabled,
            "warnings": warnings,
            "llm_risk_score": llm_risk_score,
            "llm_risk_level": llm_risk_level,
            "llm_risk_narrative": llm_narrative,
            "llm_recommendation": llm_recommendation,
        }
        return {
            **state,
            "enabled_modalities": enabled,
            "applicability_warnings": warnings,
            "bias_abstain": True,
            "abstaining": True,
            "abstention_reason": "bias_risk_exceeds_threshold",
            "pipeline_status": "abstained",
            "llm_reasoning": {
                **state["llm_reasoning"],
                "bias_applicability": llm_narrative or "",
            },
            "agent_outputs": {**state["agent_outputs"], "bias_applicability": ba_out},
        }

    # ── Step 7: Record and PASS ───────────────────────────────────────────────
    log_fn("bias_applicability", sid, "COMPLETE", "applicability_check_done",
           {"enabled_modalities": enabled, "warnings": warnings,
            "llm_risk_level": llm_risk_level, "llm_risk_score": llm_risk_score,
            "llm_narrative": llm_narrative or "rule-based-only"})

    ba_out = {
        "status": "COMPLETE",
        "enabled_modalities": enabled,
        "warnings": warnings,
        "llm_risk_score": llm_risk_score,
        "llm_risk_level": llm_risk_level,
        "llm_risk_narrative": llm_narrative,
        "llm_recommendation": llm_recommendation,
    }
    return {
        **state,
        "enabled_modalities": enabled,
        "applicability_warnings": warnings,
        "bias_abstain": False,
        "llm_reasoning": {
            **state["llm_reasoning"],
            "bias_applicability": llm_narrative or "",
        },
        "agent_outputs": {**state["agent_outputs"], "bias_applicability": ba_out},
    }
