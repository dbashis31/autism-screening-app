"""
LangGraph pipeline state shared across all governance agents.

The state is a TypedDict that flows through the graph. Since we do NOT
use a LangGraph checkpointer, non-serialisable objects (db_ops callables,
log_fn) are safe to include — they live only in memory for the duration
of a single request.
"""

from typing import Any, TypedDict


class PipelineState(TypedDict):
    # ── Inputs (set once at graph entry) ──────────────────────────────────────
    scenario: dict           # original submission payload + session metadata
    log_fn: Any              # DB-backed audit logger callable
    db_ops: dict             # {"get_abstention_history": fn, "write_abstention": fn}

    # ── Accumulated outputs ───────────────────────────────────────────────────
    agent_outputs: dict      # keyed by agent name
    llm_reasoning: dict      # keyed by agent name → LLM rationale text

    # ── Routing flags ─────────────────────────────────────────────────────────
    blocked: bool
    block_reason: str | None

    # ── Per-stage outputs ─────────────────────────────────────────────────────
    enabled_modalities: list[str]
    applicability_warnings: list[str]
    model_rejected: bool
    abstaining: bool
    abstention_reason: str | None
    confidence_scores: dict

    # ── Final report fields ────────────────────────────────────────────────────
    caregiver_report: str | None
    clinician_report: dict | None

    # ── Pipeline summary ──────────────────────────────────────────────────────
    pipeline_status: str        # pending | blocked | abstained | complete
    consent_latency_ms: float
