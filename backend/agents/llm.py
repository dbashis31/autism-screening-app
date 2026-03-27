"""
Anthropic Claude client for governance agents.

All agents call call_llm() for LLM-augmented reasoning.
If ANTHROPIC_API_KEY is not set, every call returns None and agents
fall back to their deterministic rule-based outputs — no crash, no
change in governance semantics.
"""

import json
import logging
import os
from functools import lru_cache

import anthropic

logger = logging.getLogger(__name__)

# Override via ANTHROPIC_MODEL env var; default to fastest capable model
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


@lru_cache(maxsize=1)
def _get_client() -> anthropic.Anthropic | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set — LLM reasoning disabled, "
            "all agents will use rule-based fallback outputs."
        )
        return None
    return anthropic.Anthropic(api_key=api_key)


def call_llm(system: str, user: str, max_tokens: int = 512) -> str | None:
    """
    Call Claude and return the text response.
    Returns None on any failure (missing key, network error, timeout).
    Callers must handle None and fall back gracefully.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
    except Exception as exc:
        logger.warning("LLM call failed (%s) — using rule-based fallback", exc)
        return None


def call_llm_json(system: str, user: str, max_tokens: int = 512) -> dict | None:
    """
    Call Claude and parse the response as JSON.
    Returns the parsed dict, or None if the call fails or the response
    is not valid JSON.
    """
    raw = call_llm(system, user, max_tokens)
    if raw is None:
        return None
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("LLM response was not valid JSON: %s", text[:200])
        return None
