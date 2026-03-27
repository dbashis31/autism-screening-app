"""
Bias & Applicability Agent (LangGraph node).

Responsibility: Validate modality quality and demographic applicability.
Deterministic rules: SNR threshold, age boundary.

LLM role: Assess *combined* bias risk that emerges from the interaction of
multiple factors (e.g., low-SNR audio + age boundary + limited modalities).
Generates plain-English warning narratives for clinicians and may surface
additional risk flags the rule set cannot anticipate.
"""

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

    # ── Rule 1: audio SNR threshold ───────────────────────────────────────────
    snr = scenario.get("audio_snr_db")
    if snr is not None and snr < 15:
        if "audio" in enabled:
            enabled.remove("audio")
        log_fn("bias_applicability", sid, "MODALITY_DISABLED",
               "audio_snr_below_threshold",
               {"audio_snr_db": snr, "threshold_db": 15})
        rule_warnings.append(f"audio_degraded_excluded:snr={snr}dB<15dB_threshold")

    # ── Rule 2: age boundary check ────────────────────────────────────────────
    age = scenario.get("child_age_months")
    if age is not None and (age <= 18 or age >= 72):
        log_fn("bias_applicability", sid, "WARNING", "age_edge_case",
               {"age_months": age, "validated_range": "18–72 months"})
        rule_warnings.append(f"age_applicability_warning:{age}mo_at_boundary")

    # ── LLM: combined bias & applicability risk assessment ────────────────────
    llm_result = call_llm_json(
        system=(
            "You are a bias and applicability governance agent for a paediatric autism "
            "screening AI. Assess the *combined* risk arising from the interaction of "
            "modality availability, signal quality, and demographic factors. "
            "Consider: reduced modality set leading to one-sided evidence, age boundary "
            "effects on model calibration, compounding of multiple risk factors. "
            "Respond with JSON containing exactly:\n"
            "  \"risk_level\": \"low\" | \"medium\" | \"high\"\n"
            "  \"risk_narrative\": 1-2 sentence plain-English summary for clinicians\n"
            "  \"additional_warnings\": list of short warning strings (may be empty)\n"
            "  \"recommendation\": one sentence on how to handle this screening"
        ),
        user=(
            f"Enabled modalities after filtering: {enabled}\n"
            f"Audio SNR: {snr} dB (threshold: 15 dB, None = not submitted)\n"
            f"Child age: {age} months (validated range: 18–72 months, None = unknown)\n"
            f"Rule-based warnings already raised: {rule_warnings}\n"
            f"Applicability warnings from consent agent: "
            f"{state['agent_outputs'].get('ethics_consent', {}).get('llm_flags', [])}"
        ),
        max_tokens=320,
    )

    warnings = list(rule_warnings)
    llm_narrative    = None
    llm_risk_level   = "unknown"
    llm_recommendation = None
    if llm_result:
        llm_narrative      = llm_result.get("risk_narrative")
        llm_risk_level     = llm_result.get("risk_level", "unknown")
        llm_recommendation = llm_result.get("recommendation")
        extra_warnings     = llm_result.get("additional_warnings", [])
        if isinstance(extra_warnings, list):
            warnings.extend(extra_warnings)

    log_fn("bias_applicability", sid, "COMPLETE", "applicability_check_done",
           {"enabled_modalities": enabled, "warnings": warnings,
            "llm_risk_level": llm_risk_level,
            "llm_narrative": llm_narrative or "rule-based-only"})

    ba_out = {
        "status": "COMPLETE",
        "enabled_modalities": enabled,
        "warnings": warnings,
        "llm_risk_level": llm_risk_level,
        "llm_risk_narrative": llm_narrative,
        "llm_recommendation": llm_recommendation,
    }
    return {
        **state,
        "enabled_modalities": enabled,
        "applicability_warnings": warnings,
        "llm_reasoning": {
            **state["llm_reasoning"],
            "bias_applicability": llm_narrative or "",
        },
        "agent_outputs": {**state["agent_outputs"], "bias_applicability": ba_out},
    }
