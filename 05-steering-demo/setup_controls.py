# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Setup Agent Control server with booking safety controls.

Important: Run this setup script once before running the demo:
    python setup_controls.py

Requires Agent Control server running.
See: https://github.com/agentcontrol/agent-control

Creates 2 controls:
  1. steer-max-guests  — STEER at LLM output: guides agent to reduce guests when > 10
  2. deny-no-payment   — DENY at tool level: blocks confirmation without payment

Key architecture:
  - STEER controls evaluate LLM output (text) → AgentControlSteeringHandler → Guide()
  - DENY controls evaluate tool input (params) → AgentControlPlugin → RuntimeError

See: https://docs.agentcontrol.dev/
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agent_control import AgentControlClient, agents, controls
from agent_control import Agent as ACAgent

AGENT_NAME = "booking-guardrails-demo"
LOCALHOST_DEFAULT = "http" + "://" + "127.0.0.1:8000"
SERVER_URL = os.getenv("AGENT_CONTROL_URL", LOCALHOST_DEFAULT)

CONTROLS = [
    # Control 1: STEER at LLM output level — too many guests
    # The agent describes the booking before calling the tool.
    # If the description mentions > 10 guests, steering guides a correction.
    {
        "name": "steer-max-guests",
        "definition": {
            "description": "Guide agent to reduce guest count when exceeding maximum of 10",
            "enabled": True,
            "execution": "server",
            "scope": {
                "step_types": ["llm"],
                "stages": ["post"],
            },
            "selector": {"path": "output"},
            # Evaluated with RE2, which supports no lookahead or backreferences.
            # This pattern therefore also matches the agent's own corrective reply
            # whenever that reply restates the original total, so the control cannot
            # be made self-excluding by narrowing the regex. The demo bounds steer
            # retries instead — see MAX_STEERS in demo_hooks_vs_control.py.
            "evaluator": {
                "name": "regex",
                "config": {"pattern": r"(1[1-9]|[2-9]\d)\s*guest"},
            },
            "action": {
                "decision": "steer",
                "message": "Guest count exceeds maximum of 10",
                "steering_context": {
                    "message": (
                        "The booking exceeds the hotel maximum of 10 guests per room. "
                        "Split the reservation across multiple rooms: fill each room with "
                        "10 guests until all guests are accommodated, so a party of 15 "
                        "becomes one room of 10 and one room of 5. Call book_hotel once "
                        "per room now, in a single turn, before writing any further "
                        "explanation. Book each room exactly once — do not repeat a "
                        "booking you have already made. Once every room is booked, tell "
                        "the user how many rooms were reserved and the occupancy of each."
                    )
                },
            },
            "tags": ["booking", "steer", "capacity"],
        },
    },
    # Control 2: DENY at tool level — confirm without payment
    {
        "name": "deny-no-payment",
        "definition": {
            "description": "Block booking confirmation without prior payment",
            "enabled": True,
            "execution": "server",
            "scope": {
                "step_types": ["tool"],
                "stages": ["pre"],
                "step_names": ["confirm_booking"],
            },
            "selector": {"path": "input"},
            "evaluator": {
                "name": "regex",
                "config": {"pattern": r"BK\d{3}"},
            },
            "action": {
                "decision": "deny",
                "message": "Payment must be processed before confirming a booking",
            },
            "tags": ["booking", "deny", "payment"],
        },
    },
]


async def setup():
    """Create agent and controls on Agent Control server."""
    print(f"Connecting to Agent Control server at {SERVER_URL}...")

    async with AgentControlClient(base_url=SERVER_URL) as client:
        try:
            health = await client.health_check()
            print(f"Server status: {health.get('status', 'unknown')}")
        except Exception as e:
            print(f"Server not available: {e}")
            print("\nStart the server first:")
            print("  See: https://github.com/agentcontrol/agent-control")
            return

        agent = ACAgent(
            agent_name=AGENT_NAME,
            agent_description="Booking guardrails demo — Hooks vs Agent Control",
        )
        try:
            await agents.register_agent(client, agent, steps=[])
            print(f"Agent registered: {AGENT_NAME}")
        except Exception:
            print(f"Agent already exists: {AGENT_NAME}")

        control_ids = []
        for spec in CONTROLS:
            name = spec["name"]
            definition = spec["definition"]
            action = definition["action"]["decision"].upper()

            try:
                result = await controls.create_control(client, name=name, data=definition)
                control_id = result["control_id"]
            except Exception as e:
                if "409" in str(e):
                    controls_list = await controls.list_controls(client, name=name, limit=1)
                    if controls_list["controls"]:
                        control_id = controls_list["controls"][0]["id"]
                        await controls.set_control_data(client, control_id, definition)
                    else:
                        raise
                else:
                    raise

            control_ids.append(control_id)
            print(f"  [{action:>5}] {name} (ID: {control_id})")

        for control_id in control_ids:
            try:
                await agents.add_agent_control(client, AGENT_NAME, control_id)
            except Exception as e:
                print(f"  Warning: could not attach control {control_id}: {e}")

        print(f"\nSetup complete — {len(control_ids)} controls attached to {AGENT_NAME}")
        print("\nRun the demo:")
        print("  uv run demo_hooks_vs_control.py")


if __name__ == "__main__":
    asyncio.run(setup())
