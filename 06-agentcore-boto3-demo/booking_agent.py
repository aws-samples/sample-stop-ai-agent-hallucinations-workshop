# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Hotel Booking Agent — AgentCore Runtime entry point.

The agent connects to the AgentCore Gateway over MCP (Model Context Protocol) to
discover its tools: the Gateway fronts the booking/payment/graph Lambdas and does
the semantic tool routing, so no tools are defined inline here.

Best practice: the model, MCP connection, tools and Agent are built ONCE at
module load (cold start) and reused across invocations. The handler only runs
the prompt — it does not rebuild the agent or reopen the Gateway connection on
every call.
"""

import os
from datetime import datetime

import boto3
from bedrock_agentcore import BedrockAgentCoreApp
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.hooks.events import BeforeToolCallEvent
from strands.hooks.registry import HookProvider, HookRegistry
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient

GATEWAY_URL = os.environ["GATEWAY_URL"]
_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


# --- Hard guardrails (hooks — cannot be bypassed by the LLM) ---
class BookingGuardrailsHook(HookProvider):
    """Framework-level guardrails that must NEVER be bypassed: payment before
    confirmation (financial integrity) and the cancellation window (contractual).
    All softer rules are steering via the validate_booking_rules tool.
    """

    def __init__(self):
        self._bookings = boto3.resource("dynamodb", region_name=_region).Table(
            os.environ["BOOKINGS_TABLE"]
        )

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self._validate)

    def _validate(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use["name"]
        params = event.tool_use.get("input", {})
        if "confirm" in tool_name:
            self._validate_confirmation(event, params)
        elif "cancel" in tool_name:
            self._validate_cancellation(event, params)

    def _validate_confirmation(self, event, params):
        booking_id = params.get("booking_id", "")
        if not booking_id:
            event.cancel_tool = "BLOCKED: booking_id is required."
            return
        booking = self._bookings.get_item(Key={"booking_id": booking_id}).get("Item")
        if not booking:
            event.cancel_tool = f"BLOCKED: Booking '{booking_id}' not found."
            return
        if booking["status"] != "PAID":
            event.cancel_tool = (
                f"BLOCKED: Booking is '{booking['status']}'. Payment must be processed "
                "before confirmation. Ask the user if they want to proceed with payment."
            )

    def _validate_cancellation(self, event, params):
        booking_id = params.get("booking_id", "")
        if not booking_id:
            event.cancel_tool = "BLOCKED: booking_id is required."
            return
        booking = self._bookings.get_item(Key={"booking_id": booking_id}).get("Item")
        if not booking:
            event.cancel_tool = f"BLOCKED: Booking '{booking_id}' not found."
            return
        if booking["status"] == "CANCELLED":
            event.cancel_tool = "BLOCKED: Booking is already cancelled."
            return
        try:
            ci = datetime.fromisoformat(booking["check_in"])
            if (ci - datetime.now()).days < 2:
                event.cancel_tool = (
                    "BLOCKED: Cannot cancel within 48 hours of check-in. "
                    "Inform the user to contact support for exceptions."
                )
        except (ValueError, TypeError):
            pass


SYSTEM_PROMPT = (
    "You are a hotel booking assistant. Help users search, book, pay, confirm, "
    "and cancel hotel reservations.\n\n"
    "RULES:\n"
    "- ALWAYS call validate_booking_rules BEFORE book_hotel, confirm_booking, or cancel_booking.\n"
    "- If validation returns FAIL with a STEER instruction, follow it exactly: fix the "
    "parameters, retry, and tell the user what was not possible AND what you did instead. "
    "Pattern: 'X is not available, but Y is. I adjusted to Y.'\n"
    "- If a tool call is BLOCKED by the system, inform the user — you cannot override it.\n"
    "- For payment, ask the user if they want to proceed (simulated).\n"
    "- Follow the flow: search -> validate -> book -> pay -> validate -> confirm."
)


# --- Build the agent ONCE at cold start (not per invocation) ---
# The Gateway (authorizerType=NONE) is reached over MCP streamable-http; the
# connection is opened here and kept open for the process lifetime.
_mcp_client = MCPClient(lambda: streamablehttp_client(GATEWAY_URL))
_mcp_client.start()
_tools = _mcp_client.list_tools_sync()

_agent = Agent(
    model=BedrockModel(region_name=_region),
    tools=_tools,
    system_prompt=SYSTEM_PROMPT,
    hooks=[BookingGuardrailsHook()],
    context_manager="auto",
)

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload, context=None):
    """Run one prompt through the pre-built agent."""
    prompt = payload if isinstance(payload, str) else payload.get("prompt", "")
    result = _agent(prompt)

    tools_used = []
    metrics = getattr(result, "metrics", None)
    if metrics is not None and hasattr(metrics, "tool_metrics"):
        tools_used = list(metrics.tool_metrics.keys())

    return {"response": str(result), "tools_used": tools_used}


if __name__ == "__main__":
    app.run()
