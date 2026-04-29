# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Demo: Strands Hooks vs Agent Control — Block vs Self-Correct

Compares two guardrail approaches on the SAME booking scenario:
  Test 1 — Hooks: MaxGuestsHook blocks violations with cancel_tool
  Test 2 — Agent Control: AgentControlSteeringHandler steers the agent to self-correct

Same tools, same model, same query. Only the guardrail layer changes.

Based on:
  - https://strandsagents.com/blog/strands-agents-with-agent-control/
  - https://strandsagents.com/docs/community/plugins/agent-control/
"""

import os
import time
import yaml

os.environ["OTEL_SDK_DISABLED"] = "true"

from dotenv import load_dotenv

load_dotenv()

from strands import Agent
from strands.hooks import HookProvider, HookRegistry, BeforeToolCallEvent

from tools import ALL_TOOLS

_CONTROLS_FILE = os.path.join(os.path.dirname(__file__), "controls.yaml")


def _inject_local_controls() -> None:
    """Load controls.yaml and inject into agent_control state when no server is available."""
    try:
        import agent_control
        from agent_control._state import state

        with open(_CONTROLS_FILE) as f:
            data = yaml.safe_load(f)

        raw_controls = data.get("controls", [])
        server_controls = [
            {"id": i + 1, "name": c["name"], "control": {k: v for k, v in c.items() if k != "name"}}
            for i, c in enumerate(raw_controls)
        ]
        state.server_controls = server_controls
    except Exception as e:
        print(f"⚠️  Could not load local controls: {e}")

# Model configuration — Amazon Bedrock (default, requires AWS credentials)
# Strands Agents uses Bedrock by default. No extra import needed.
# To use a specific Bedrock model, pass the model ID as a string:
#   MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
#
# To use a different provider (e.g., OpenAI), install the extra and configure:
#   pip install "strands-agents[openai]"
#   from strands.models.openai import OpenAIModel
#   MODEL = OpenAIModel(model_id="gpt-4o-mini")
#   (requires OPENAI_API_KEY env var — get one at https://platform.openai.com/api-keys)
#
# See all providers: https://strandsagents.com/docs/user-guide/concepts/model-providers/

QUERY = "Book AnyCompany Lisbon Resort for 15 guests from 2026-05-01 to 2026-05-03"

# System prompt that makes the LLM describe the booking before calling the tool.
# This is needed so the steer control can detect "15 guests" in the LLM text output.
PROMPT = (
    "You are a hotel booking assistant. "
    "When booking, first describe what you will book (hotel, guests, dates) "
    "then call the tool."
)


# ── Approach 1: Hooks (block) ────────────────────────────────────────────────

class MaxGuestsHook(HookProvider):
    """Blocks bookings with more than 10 guests via cancel_tool."""

    def __init__(self):
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


def run_test_1_hooks():
    """Test 1: Hooks approach — block and fail."""
    print("\n" + "=" * 70)
    print("TEST 1: HOOKS (block with cancel_tool)")
    print("=" * 70)
    print(f"Query: {QUERY}\n")

    hook = MaxGuestsHook()
    agent = Agent(system_prompt=PROMPT, tools=ALL_TOOLS, hooks=[hook])

    start = time.time()
    response = agent(QUERY)
    elapsed = time.time() - start
    output = str(response)

    print(f"\n⏱️  {elapsed:.1f}s")
    print(f"🔧 Hook blocked: {hook.blocked} call(s)")

    # Check if the booking actually went through with 15 guests
    if "SUCCESS" in output and "15 guests" in output:
        print("❌ Agent bypassed the hook (unexpected)")
        return {"time": elapsed, "outcome": "bypassed"}
    elif hook.blocked > 0:
        print("🚫 Agent was BLOCKED — reported failure or asked user to change")
        return {"time": elapsed, "outcome": "blocked"}
    else:
        print("⚠️  Agent found a workaround")
        return {"time": elapsed, "outcome": "workaround"}


# ── Approach 2: Agent Control (steer + self-correct) ─────────────────────────

def run_test_2_agent_control():
    """Test 2: Agent Control — steer agent to self-correct."""
    print("\n" + "=" * 70)
    print("TEST 2: AGENT CONTROL (steer with Guide)")
    print("=" * 70)
    print(f"Query: {QUERY}\n")

    try:
        import agent_control
        from agent_control.integrations.strands import (
            AgentControlPlugin,
            AgentControlSteeringHandler,
        )
        from agent_control.control_decorators import ControlViolationError
        from strands.hooks import AfterToolCallEvent
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: uv pip install 'agent-control-sdk[strands-agents]'")
        return {"time": 0, "outcome": "skipped"}

    server_url = os.getenv("AGENT_CONTROL_URL", "http" + "://" + "127.0.0.1:8000")

    try:
        agent_control.init(
            agent_name="booking-guardrails-demo",
            server_url=server_url,
            policy_refresh_interval_seconds=0,
        )
    except Exception as e:
        if "409" not in str(e):
            pass  # server unavailable — will inject local controls below

    # If no controls loaded from server, inject from local controls.yaml
    if not agent_control.get_server_controls():
        _inject_local_controls()

    # Plugin handles DENY controls at tool level
    plugin = AgentControlPlugin(
        agent_name="booking-guardrails-demo",
        event_control_list=[BeforeToolCallEvent, AfterToolCallEvent],
        enable_logging=False,
    )

    # Steering handler handles STEER controls at LLM output level
    steering = AgentControlSteeringHandler(
        agent_name="booking-guardrails-demo",
        enable_logging=False,
    )

    agent = Agent(system_prompt=PROMPT, tools=ALL_TOOLS, plugins=[plugin, steering])

    start = time.time()
    try:
        response = agent(QUERY)
        elapsed = time.time() - start
        output = str(response)

        steered = steering.steers_applied
        print(f"\n⏱️  {elapsed:.1f}s")
        print(f"🔄 Steered: {steered} time(s)")

        bk_count = output.count("BK0")
        if bk_count >= 2:
            print("✅ Agent self-corrected — split into 2 rooms (10 + 5 guests)")
            return {"time": elapsed, "steered": steered, "outcome": "split-bookings"}
        elif "SUCCESS" in output or "BK" in output:
            print("⚠️  Agent completed booking but did not split into 2 rooms")
            return {"time": elapsed, "steered": steered, "outcome": "self-corrected"}
        else:
            print(f"⚠️  Response: {output[:200]}")
            return {"time": elapsed, "steered": steered, "outcome": "unclear"}

    except ControlViolationError as e:
        elapsed = time.time() - start
        print(f"\n⏱️  {elapsed:.1f}s")
        print(f"🚫 Denied by control: {e}")
        return {"time": elapsed, "outcome": "denied"}
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n⏱️  {elapsed:.1f}s")
        print(f"❌ Error: {type(e).__name__}: {str(e)[:200]}")
        return {"time": elapsed, "outcome": "error"}


# ── Comparison ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  HOOKS vs AGENT CONTROL")
    print("  Same query, same tools — different guardrail approach")
    print("=" * 70)

    r1 = run_test_1_hooks()
    r2 = run_test_2_agent_control()

    print(f"\n{'Approach':<35} {'Time':>8} {'Outcome':>20}")
    print("-" * 65)
    print(f"{'Hooks (cancel_tool)':<35} {r1['time']:>6.1f}s {r1['outcome']:>20}")
    print(f"{'Agent Control (steer)':<35} {r2['time']:>6.1f}s {r2['outcome']:>20}")
