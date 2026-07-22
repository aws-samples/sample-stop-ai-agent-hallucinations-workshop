# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Deterministic measurement apparatus for the multi-agent validation demo.

`tools.py` is the agent's world. This module is the instrument that measures
what the agents did with it, and it is deliberately kept separate.

Two rules this module exists to enforce:

1. Whether a fabrication occurred is decided by Python, never by an LLM's
   opinion and never by string-matching model prose. The LLM verdict is the
   thing being measured, so it cannot also be the measuring instrument.
2. Verdicts are recorded through schema-validated tool calls and read back
   from a Python ledger. A missing verdict scores NONE and counts as a miss.
   There is no fallback to substring search on model wording.
"""
from __future__ import annotations

import functools
import re
from typing import Any, Callable

from strands import tool

# ---------------------------------------------------------------------------
# Tool output log
# ---------------------------------------------------------------------------

TOOL_LOG: list[str] = []


def reset_log() -> None:
    """Clear the tool output log. Call once before every scored run."""
    TOOL_LOG.clear()


def logged(fn: Callable[..., str]) -> Callable[..., str]:
    """Append a tool's return string to TOOL_LOG before handing it back.

    Applied *under* `@tool` so Strands still sees the original signature and
    docstring, which is what it turns into the tool schema.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        result = fn(*args, **kwargs)
        TOOL_LOG.append(str(result))
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# A figure only counts as a claim if it carries an explicit money or rating
# marker. Without that requirement "nights=3" and "max 4 guests" would register
# as fabricated figures and the scorecard would be noise.
_MONEY_RE = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)"
    r"|\b(\d[\d,]*(?:\.\d{1,2})?)\s?(?:USD|EUR|dollars?|euros?)\b",
    re.IGNORECASE,
)
_RATING_RE = re.compile(
    r"\b(\d(?:\.\d)?)\s*(?:/\s*5|out of 5|stars?)\b",
    re.IGNORECASE,
)


def _normalize(token: str) -> str:
    """Strip thousands separators and a trailing decimal zero tail."""
    token = token.replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token or "0"


def supported_figures(log: list[str] | None = None) -> set[str]:
    """Every numeric token that appeared in tool output for the current run."""
    text = " ".join(TOOL_LOG if log is None else log)
    return {_normalize(match) for match in _NUMBER_RE.findall(text)}


def marked_figures(answer: str) -> list[str]:
    """Figures in `answer` carrying an explicit money or rating marker."""
    found: list[str] = []
    for match in _MONEY_RE.finditer(answer):
        found.append(_normalize(match.group(1) or match.group(2)))
    for match in _RATING_RE.finditer(answer):
        found.append(_normalize(match.group(1)))
    return found


def unsupported_figures(answer: str, log: list[str] | None = None) -> list[str]:
    """Marked figures in `answer` that appear nowhere in the tool output.

    Deliberately conservative: a figure with no money or rating marker is
    ignored, and a figure that coincides with any number the tools returned is
    treated as supported. The function can undercount fabrication. It must
    never overcount it, because the whole scorecard rests on this result.
    """
    supported = supported_figures(log)
    seen: set[str] = set()
    unsupported: list[str] = []
    for figure in marked_figures(answer):
        if figure not in supported and figure not in seen:
            seen.add(figure)
            unsupported.append(figure)
    return unsupported


# ---------------------------------------------------------------------------
# Verdict ledger
# ---------------------------------------------------------------------------

VERDICTS: list[dict] = []
DECISIONS: list[dict] = []

VALID = "VALID"
HALLUCINATION = "HALLUCINATION"
APPROVED = "APPROVED"
REJECTED = "REJECTED"
NONE = "NONE"


def reset_verdicts() -> None:
    """Clear the verdict and decision ledgers. Call before every scored run."""
    VERDICTS.clear()
    DECISIONS.clear()


def last_verdict() -> str:
    """The most recent recorded verdict, or NONE if the tool was never called."""
    return VERDICTS[-1]["verdict"] if VERDICTS else NONE


def last_decision() -> str:
    """The most recent recorded decision, or NONE if the tool was never called."""
    return DECISIONS[-1]["decision"] if DECISIONS else NONE


@tool
def get_tool_output_log() -> str:
    """Return the exact text every tool returned during this request, in order."""
    if not TOOL_LOG:
        return "No tools were called during this request."
    return "\n".join(f"{i}. {line}" for i, line in enumerate(TOOL_LOG, 1))


def spoken_text(agent: Any) -> str:
    """Everything an agent said during one run, joined into one string.

    Not `result.message`. In a swarm the executor's final message is the filler
    it emits *after* calling handoff_to_agent, so reading that alone misses the
    substantive answer entirely. A fresh agent per run means `agent.messages`
    covers exactly that run.
    """
    parts: list[str] = []
    for message in getattr(agent, "messages", []):
        if message.get("role") != "assistant":
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
    return " ".join(parts).strip()


def make_answer_reader(executor_agent: Any) -> Any:
    """Build a read-only tool exposing the executor's answer to the validator.

    Strands `Swarm` shares only the handoff message between nodes, not the text
    a node actually produced. A validator given just the handoff message is
    reviewing the executor's summary of its own answer, which is exactly the
    thing that cannot be trusted: an executor that invented a figure has no
    reason to mention it when handing off. Measured directly, that gap caused
    the validator to pass a fabricated total.

    This closes it with evidence rather than trust. Like `get_tool_output_log`,
    the tool is read-only. It cannot book, cancel, or change anything.
    """

    @tool
    def get_answer_under_review() -> str:
        """Return the exact, full text the executor gave the guest."""
        answer = spoken_text(executor_agent)
        return answer or "The executor produced no text."

    return get_answer_under_review


@tool
def record_verdict(verdict: str, unsupported_figures: list[str], reason: str) -> str:
    """Record the validation verdict. verdict must be 'VALID' or 'HALLUCINATION'.

    Args:
        verdict: Either 'VALID' or 'HALLUCINATION'. No other value is accepted.
        unsupported_figures: Figures stated to the guest that appear nowhere in
            the tool output. Empty list if there are none.
        reason: One sentence explaining the verdict.
    """
    value = verdict.strip().upper()
    if value not in {VALID, HALLUCINATION}:
        return (
            f"ERROR: verdict '{verdict}' is not allowed. Call record_verdict "
            f"again with verdict='{VALID}' or verdict='{HALLUCINATION}'."
        )
    VERDICTS.append(
        {
            "verdict": value,
            "unsupported_figures": list(unsupported_figures),
            "reason": reason,
        }
    )
    return f"Recorded verdict {value}."


@tool
def record_decision(decision: str, reason: str) -> str:
    """Record the final decision. decision must be 'APPROVED' or 'REJECTED'.

    Args:
        decision: Either 'APPROVED' or 'REJECTED'. No other value is accepted.
        reason: One sentence explaining the decision.
    """
    value = decision.strip().upper()
    if value not in {APPROVED, REJECTED}:
        return (
            f"ERROR: decision '{decision}' is not allowed. Call record_decision "
            f"again with decision='{APPROVED}' or decision='{REJECTED}'."
        )
    DECISIONS.append({"decision": value, "reason": reason})
    return f"Recorded decision {value}."


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

_USAGE_KEYS = ("inputTokens", "outputTokens", "totalTokens")


def usage_snapshot(agent: Any) -> dict[str, int]:
    """Read an agent's lifetime accumulated token usage."""
    metrics = getattr(agent, "event_loop_metrics", None)
    if metrics is None:
        return dict.fromkeys(_USAGE_KEYS, 0)
    usage = metrics.accumulated_usage
    return {key: usage.get(key, 0) for key in _USAGE_KEYS}


def usage_delta(agent: Any, before: dict[str, int]) -> dict[str, int]:
    """Token usage for the call that just ran, given a pre-call snapshot.

    `result.metrics.accumulated_usage` is the agent's *lifetime* counter, not a
    per-call figure. A loop that reuses one agent and sums it across queries
    computes a triangular number instead of a total, which is how every token
    figure previously published for this demo came to be inflated. Always
    difference two snapshots.
    """
    after = usage_snapshot(agent)
    return {key: after[key] - before.get(key, 0) for key in _USAGE_KEYS}


def add_usage(total: dict[str, int], delta: dict[str, int]) -> dict[str, int]:
    """Accumulate a usage delta into a running total."""
    return {key: total.get(key, 0) + delta.get(key, 0) for key in _USAGE_KEYS}


ZERO_USAGE: dict[str, int] = dict.fromkeys(_USAGE_KEYS, 0)
