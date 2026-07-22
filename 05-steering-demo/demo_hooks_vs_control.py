# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Demo: Strands Hooks vs Agent Control — Block vs Self-Correct

Compares two guardrail approaches on the SAME booking scenario:
  Test 1 — Hooks: MaxGuestsHook blocks violations with cancel_tool
  Test 2 — Agent Control: AgentControlSteeringHandler steers the agent to self-correct

Same tools, same model, same query. Only the guardrail layer changes.

Model provider: Amazon Bedrock, via the Strands default. This matches every other
demo in the workshop, so no provider-specific API key is required — AWS credentials
are the only credential this demo needs.

Based on:
  - https://strandsagents.com/blog/strands-agents-with-agent-control/
  - https://strandsagents.com/docs/community/plugins/agent-control/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from dotenv import load_dotenv

load_dotenv()

from strands import Agent
from strands.hooks import HookProvider, HookRegistry, BeforeToolCallEvent

from tools import ALL_TOOLS, reset_state

CONTROLS_FILE = Path(__file__).with_name("controls.yaml")
DEFAULT_SERVER_URL = "http://127.0.0.1:8000"

# Cap on how many times a single agent invocation may be steered.
#
# A steer control matches on LLM output. This one matches a guest count above 10 —
# which the agent's own corrective reply almost always restates ("splitting your 15
# guests across 2 rooms"), re-firing the control. The evaluator runs on RE2, which
# has no lookahead, so the pattern cannot be narrowed to exclude the correction.
# Bounding the retries is therefore the fix: one corrective nudge, then proceed.
# Unbounded, the model reads the repeated injections as a prompt-injection attack
# and does the very thing the control was meant to prevent. See README, "Why
# steering is bounded".
MAX_STEERS = 1

# Number of guests requested. This single constant feeds the booking query and the
# ledger classifier so the requested total and the split-bookings check can never
# drift apart. It sits above the 10-guest cap the controls enforce.
GUESTS = 15

# Dates are computed relative to today so a fixed date cannot rot into the past.
CHECK_IN = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
CHECK_OUT = (datetime.now() + timedelta(days=32)).strftime("%Y-%m-%d")
QUERY = f"Book AnyCompany Lisbon Resort for {GUESTS} guests from {CHECK_IN} to {CHECK_OUT}"

# System prompt that makes the LLM describe the booking before calling the tool.
# This is needed so the steer control can detect the guest count in the LLM text output.
PROMPT = (
    "You are a hotel booking assistant. "
    "When booking, first describe what you will book (hotel, guests, dates) "
    "then call the tool."
)


# ── Preflight: is the Agent Control control plane actually available? ─────────

@dataclass(frozen=True)
class ControlPlane:
    """How this run will source its Agent Control controls."""

    mode: str  # "server" or "local"
    server_url: str
    detail: str


def _http_status(url: str, timeout: float) -> int | None:
    """GET *url*, returning the HTTP status, or None if nothing answered."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _identify_service(server_url: str, timeout: float) -> str | None:
    """Return a foreign service's name if *server_url* is not Agent Control.

    Port 8000 is a very common local development port. Checking only that the
    socket is open, or only that /health returns 200, will happily accept an
    unrelated service and then fail confusingly deep inside the SDK. Agent
    Control serves its API under /api/v1/agents; a foreign service 404s there.
    """
    try:
        with urllib.request.urlopen(f"{server_url}/", timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
        if isinstance(body, dict):
            name = body.get("service") or body.get("name")
            if isinstance(name, str):
                return name
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass
    return None


def resolve_control_plane(server_url: str, allow_local: bool, timeout: float = 3.0) -> ControlPlane:
    """Decide where controls come from, failing fast with actionable guidance.

    A running Agent Control server is the demo's real prerequisite. If none is
    reachable, this exits non-zero with instructions rather than continuing into a
    stack trace, unless --local-controls was passed to opt into the development
    workaround described in `_inject_local_controls`.
    """
    health = _http_status(f"{server_url}/health", timeout)
    agents_route = _http_status(f"{server_url}/api/v1/agents", timeout)

    # 404 on the Agent Control API surface means whatever answered is not Agent Control.
    if health == 200 and agents_route is not None and agents_route != 404:
        return ControlPlane("server", server_url, f"Agent Control server at {server_url}")

    if health is None and agents_route is None:
        reason = f"no server is listening at {server_url}"
    else:
        foreign = _identify_service(server_url, timeout)
        named = f" ({foreign})" if foreign else ""
        reason = (
            f"something is listening at {server_url}{named}, but it is not an "
            f"Agent Control server — /api/v1/agents returned {agents_route}"
        )

    if not allow_local or not CONTROLS_FILE.is_file():
        _fail_fast(server_url, reason, allow_local)

    return ControlPlane("local", server_url, reason)


def _inject_local_controls() -> int:
    """Load controls.yaml directly into SDK state. Development workaround only.

    `agent_control.init()` advertises a `controls_file=` parameter and its docstring
    promises "auto-discover and load local controls.yaml as fallback". In
    agent-control-sdk 8.3.0 that parameter is accepted and then **ignored** — the
    package contains no YAML loading code at all. There is therefore no supported
    serverless mode, and the only way to exercise steering without a server is to
    write the control set into private SDK state, which is what this does.

    This is a stand-in for the server's *response*, not for the server. Control
    evaluation, steering, and the Guide injection all run through real SDK code.
    But it depends on a private attribute, it bypasses server-side policy
    resolution, and controls here declare `execution: sdk` while `setup_controls.py`
    registers them as `execution: server`. Results obtained this way are NOT
    evidence about the server path and must not be reported as such.
    """
    import yaml
    from agent_control._state import state

    with CONTROLS_FILE.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    raw_controls = data.get("controls", [])
    state.server_controls = [
        {
            "id": index + 1,
            "name": control["name"],
            "control": {k: v for k, v in control.items() if k != "name"},
        }
        for index, control in enumerate(raw_controls)
    ]
    return len(state.server_controls)


def _fail_fast(server_url: str, reason: str, allow_local: bool) -> None:
    """Print exactly what is missing and how to fix it, then exit non-zero."""
    print(
        "\n"
        "══════════════════════════════════════════════════════════════════════\n"
        "  CANNOT RUN — no Agent Control server\n"
        "══════════════════════════════════════════════════════════════════════\n"
        f"\nReason: {reason}.\n"
        "\nTest 2 of this demo steers the agent using controls served by an Agent\n"
        "Control server. That server is a hard prerequisite and is not bundled with\n"
        "this workshop or with agent-control-sdk.\n"
        "\nTo fix:\n"
        "\n"
        "  1. Start the Agent Control server:\n"
        "       Setup instructions: https://github.com/agentcontrol/agent-control\n"
        "  2. Register this demo's controls on it:\n"
        "       uv run setup_controls.py\n"
        "  3. Verify it is up:\n"
        "       curl 127.0.0.1:8000/health\n"
        "\n"
        "If your server listens somewhere other than the default, point the demo at\n"
        "it before re-running:\n"
        "       export AGENT_CONTROL_URL=http://<host>:<port>\n",
        file=sys.stderr,
    )
    if allow_local and not CONTROLS_FILE.is_file():
        print(
            f"--local-controls was passed, but {CONTROLS_FILE.name} does not exist next\n"
            f"to this script, so there is nothing to load. Expected at:\n  {CONTROLS_FILE}\n",
            file=sys.stderr,
        )
    else:
        print(
            "Test 1 (the hooks half of this comparison) needs no server. To exercise\n"
            "the steering half without one, --local-controls loads controls.yaml\n"
            "directly into the SDK. That is a development workaround, not the real\n"
            "server path, and its results are not evidence the server path works.\n",
            file=sys.stderr,
        )
    print(
        "This demo needs NO model-provider API key. It runs on Amazon Bedrock via the\n"
        "Strands default, so AWS credentials are the only credential required.\n",
        file=sys.stderr,
    )
    raise SystemExit(2)


# ── Approach 1: Hooks (block) ────────────────────────────────────────────────

class MaxGuestsHook(HookProvider):
    """Blocks bookings with more than 10 guests via cancel_tool."""

    def __init__(self) -> None:
        self.blocked = 0

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.check)

    def check(self, event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] != "book_hotel":
            return
        guests = event.tool_use["input"].get("guests", 1)
        if guests > 10:
            self.blocked += 1
            event.cancel_tool = f"BLOCKED: {guests} guests exceeds maximum of 10"


def run_test_1_hooks() -> dict:
    """Test 1: Hooks approach — block and fail."""
    print("\n" + "=" * 70)
    print("TEST 1: HOOKS (block with cancel_tool)")
    print("=" * 70)
    print(f"Query: {QUERY}\n")

    reset_state()  # isolate this test from any earlier run
    hook = MaxGuestsHook()
    agent = Agent(system_prompt=PROMPT, tools=ALL_TOOLS, hooks=[hook])

    start = time.time()
    response = agent(QUERY)
    elapsed = time.time() - start
    output = str(response)

    print(f"\n⏱️  {elapsed:.1f}s")
    print(f"🔧 Hook blocked: {hook.blocked} call(s)")

    if response.metrics:
        usage = response.metrics.accumulated_usage
        print(f"💰 Tokens: {usage['inputTokens']} in, {usage['outputTokens']} out, {usage['totalTokens']} total")

    # Structural check against the booking ledger, scored exactly as Test 2 is.
    # A hook that fires once but leaves 8 + 7 guests booked did not hard-block, and
    # the shared scorer says so rather than trusting the block counter alone.
    outcome, guest_counts = report_ledger(tools_state_bookings())

    return {
        "time": elapsed,
        "outcome": outcome,
        "blocked": hook.blocked,
        "bookings": guest_counts,
    }


def tools_state_bookings() -> list[dict]:
    """Bookings created during this run, excluding the pre-seeded fixture."""
    from tools import STATE

    return [b for bid, b in STATE["bookings"].items() if bid != "BK001"]


# ── Shared scoring: classify a run by its booking ledger ─────────────────────

def score_ledger(bookings: list[dict]) -> tuple[str, list[int]]:
    """Classify a run from what actually reached the ledger, not from model prose.

    BOTH tests score with this one function on purpose. If each arm used its own
    rule, the comparison would measure the scoring method instead of the guardrail
    layer. Concretely: an agent that is blocked once and then books 8 + 7 guests
    has circumvented the 10-guest rule, and must not be scored as a hard block
    merely because its arm only consulted a block counter.

    Returns the outcome name and the guest counts per booking, largest first.
    """
    guest_counts = sorted((b["guests"] for b in bookings), reverse=True)

    if any(g > 10 for g in guest_counts):
        return "failed-open", guest_counts
    if len(guest_counts) >= 2 and sum(guest_counts) == GUESTS:
        return "split-bookings", guest_counts
    if guest_counts:
        return "partial", guest_counts
    return "no-booking", guest_counts


def report_ledger(bookings: list[dict]) -> tuple[str, list[int]]:
    """Print the ledger and its classification, and return both."""
    outcome, guest_counts = score_ledger(bookings)
    print(f"📒 Bookings created: {len(bookings)} — guests per booking: {guest_counts}")

    if outcome == "failed-open":
        over_limit = [g for g in guest_counts if g > 10]
        print(f"❌ FAILED OPEN — booking(s) for {over_limit} guests exceeded the maximum of 10")
    elif outcome == "split-bookings":
        print(f"✅ Agent self-corrected — split into {len(guest_counts)} rooms ({' + '.join(map(str, guest_counts))} guests)")
    elif outcome == "partial":
        print(f"⚠️  Agent booked within the limit but did not accommodate all {GUESTS} guests")
    else:
        print("🚫 No booking completed")

    return outcome, guest_counts


# ── Approach 2: Agent Control (steer + self-correct) ─────────────────────────

def run_test_2_agent_control(plane: ControlPlane) -> dict:
    """Test 2: Agent Control — steer agent to self-correct."""
    print("\n" + "=" * 70)
    print("TEST 2: AGENT CONTROL (steer with Guide)")
    print("=" * 70)
    print(f"Controls: {plane.detail}")
    print(f"Query: {QUERY}\n")

    reset_state()  # isolate this test from Test 1
    try:
        import agent_control
        from agent_control.integrations.strands import (
            AgentControlPlugin,
            AgentControlSteeringHandler,
        )
        from agent_control.control_decorators import ControlViolationError
        from strands.experimental.steering import Proceed
        from strands.hooks import AfterToolCallEvent
    except ImportError as exc:
        print(f"❌ Missing dependency: {exc}")
        print("   Run: uv pip install -r requirements.txt")
        return {"time": 0, "outcome": "skipped", "steered": 0}

    class BoundedSteeringHandler(AgentControlSteeringHandler):
        """Stops steering after MAX_STEERS so an unsatisfiable control cannot loop.

        A steer control matches on LLM output. If its evaluator also matches the
        corrective reply the agent writes back, the control can never be satisfied
        and the agent is steered indefinitely. Left unbounded, the model eventually
        reads the repeated injections as an attack, ignores them, and performs the
        very action the control was meant to prevent. Bounding the retries makes
        the failure loud and safe instead of silent.
        """

        def __init__(self, agent_name: str, max_steers: int = MAX_STEERS) -> None:
            super().__init__(agent_name=agent_name, enable_logging=False)
            self.max_steers = max_steers
            self.cap_reached = False

        async def steer_after_model(self, **kwargs):
            if self.steers_applied >= self.max_steers:
                self.cap_reached = True
                return Proceed(reason=f"steer cap of {self.max_steers} reached")
            return await super().steer_after_model(**kwargs)

    agent_control.init(
        agent_name="booking-guardrails-demo",
        server_url=plane.server_url,
        policy_refresh_interval_seconds=0,
    )

    if plane.mode == "local":
        _inject_local_controls()

    loaded = agent_control.get_server_controls()
    if not loaded:
        print("❌ No controls loaded — the guardrail layer would be inert, so this")
        print("   test would prove nothing. Refusing to report a result.")
        print(f"   Control source: {plane.detail}")
        raise SystemExit(2)
    print(f"✅ {len(loaded)} control(s) loaded\n")

    # Plugin handles DENY controls at tool level
    plugin = AgentControlPlugin(
        agent_name="booking-guardrails-demo",
        event_control_list=[BeforeToolCallEvent, AfterToolCallEvent],
        enable_logging=False,
    )

    # Steering handler handles STEER controls at LLM output level
    steering = BoundedSteeringHandler(agent_name="booking-guardrails-demo")

    agent = Agent(system_prompt=PROMPT, tools=ALL_TOOLS, plugins=[plugin, steering])

    start = time.time()
    try:
        response = agent(QUERY)
        elapsed = time.time() - start
    except ControlViolationError as exc:
        elapsed = time.time() - start
        print(f"\n⏱️  {elapsed:.1f}s")
        print(f"🚫 Denied by control: {exc}")
        return {"time": elapsed, "outcome": "denied", "steered": steering.steers_applied}

    print(f"\n⏱️  {elapsed:.1f}s")
    print(f"🔄 Steered: {steering.steers_applied} time(s)")

    if response.metrics:
        usage = response.metrics.accumulated_usage
        print(f"💰 Tokens: {usage['inputTokens']} in, {usage['outputTokens']} out, {usage['totalTokens']} total")

    if steering.cap_reached:
        print(
            f"ℹ️  Steer cap of {MAX_STEERS} enforced — the control matched again after the\n"
            "    agent had already corrected, and the repeat steer was suppressed."
        )

    # Structural check against the booking ledger, scored exactly as Test 1 is.
    outcome, guest_counts = report_ledger(tools_state_bookings())

    return {
        "time": elapsed,
        "steered": steering.steers_applied,
        "outcome": outcome,
        "bookings": guest_counts,
    }


# ── Comparison ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-controls",
        action="store_true",
        help=(
            "Development workaround: load controls.yaml directly into the SDK when no "
            "Agent Control server is reachable. Not the server path; results are not "
            "evidence the server path works."
        ),
    )
    args = parser.parse_args()

    server_url = os.getenv("AGENT_CONTROL_URL", DEFAULT_SERVER_URL)
    plane = resolve_control_plane(server_url, allow_local=args.local_controls)

    print("=" * 70)
    print("  HOOKS vs AGENT CONTROL")
    print("  Same query, same tools, same model — different guardrail approach")
    print("=" * 70)

    if plane.mode == "local":
        print(
            f"\n⚠️  DEVELOPMENT MODE — no Agent Control server ({plane.detail}).\n"
            f"    controls.yaml is being loaded straight into SDK state because\n"
            "    agent-control-sdk 8.3.0 accepts init(controls_file=...) and then\n"
            "    ignores it — the package has no YAML loader. Evaluation and steering\n"
            "    are real, but server-side policy resolution is bypassed and these\n"
            "    controls declare `execution: sdk` where setup_controls.py registers\n"
            "    them as `execution: server`.\n"
            "    Do NOT treat this run as verification of the server path."
        )

    r1 = run_test_1_hooks()
    r2 = run_test_2_agent_control(plane)

    print(f"\n{'Approach':<35} {'Time':>8} {'Outcome':>20}")
    print("-" * 65)
    print(f"{'Hooks (cancel_tool)':<35} {r1['time']:>6.1f}s {r1['outcome']:>20}")
    print(f"{'Agent Control (steer)':<35} {r2['time']:>6.1f}s {r2['outcome']:>20}")

    # The demo's headline claim: hooks hard-block, Agent Control self-corrects.
    #
    # Hooks hard-blocked only if the ledger is empty AND the hook is what emptied it.
    # An empty ledger with zero blocks means the agent never tried, which proves
    # nothing about the guardrail. A non-empty ledger — including a self-corrected
    # 8 + 7 split — is not a hard block, however many times the hook fired.
    hooks_hard_blocked = r1["outcome"] == "no-booking" and r1["blocked"] > 0
    claim_holds = hooks_hard_blocked and r2["outcome"] == "split-bookings"
    qualifier = "" if plane.mode == "server" else " (development mode — NOT the server path)"
    print()
    if claim_holds:
        print(f"✅ CLAIM HOLDS{qualifier} — hooks hard-blocked; Agent Control steered")
        print("   the agent to self-correct and complete the booking.")
        return 0
    print(f"❌ CLAIM DOES NOT HOLD{qualifier} — see outcomes above.")
    print("   Expected: hooks='no-booking' with at least one hook block,")
    print("             agent-control='split-bookings'.")
    print(f"   Got:      hooks='{r1['outcome']}' with {r1['blocked']} hook block(s),")
    print(f"             agent-control='{r2['outcome']}'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
