# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Hotel Booking Agent — AgentCore Runtime entry point.

Connects to AgentCore Gateway via MCP (Model Context Protocol) to access tools.
The Gateway handles semantic tool routing — the agent does not define tools inline.
"""

import os
from datetime import datetime

import boto3
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client

# Model configuration — Amazon Bedrock (default, requires AWS credentials)
# Strands Agents uses Bedrock by default. No extra import needed.
#
# To use a different provider (e.g., OpenAI), install the extra and configure:
#   pip install "strands-agents[openai]"
#   from strands.models.openai import OpenAIModel
#   model = OpenAIModel(model_id="gpt-4o-mini", client_args={"api_key": api_key})
#   (store the API key in AWS Secrets Manager and retrieve it at runtime)
#
# See all providers: https://strandsagents.com/docs/user-guide/concepts/model-providers/

# --- Configuration from CDK environment variables ---

GATEWAY_URL = os.environ["GATEWAY_URL"]

_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


# --- Hard guardrails (hooks — cannot be bypassed by the LLM) ---

from strands.hooks.events import BeforeToolCallEvent
from strands.hooks.registry import HookProvider, HookRegistry


class BookingGuardrailsHook(HookProvider):
    """Hard guardrails enforced at the framework level.

    Only critical business rules that must NEVER be bypassed:
    - Payment before confirmation (financial integrity)
    - Cancellation window (contractual obligation)

    All other rules are handled by validate_booking_rules as steering —
    the agent self-corrects based on STEER messages from DynamoDB.
    """

    def __init__(self):
        self._dynamodb = boto3.resource("dynamodb", region_name=_region)
        self._bookings = self._dynamodb.Table(os.environ["BOOKINGS_TABLE"])

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
                f"BLOCKED: Booking is '{booking['status']}'. "
                "Payment must be processed before confirmation. "
                "Ask the user if they want to proceed with payment."
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


# --- Agent setup ---

SYSTEM_PROMPT = (
    "You are a hotel booking assistant. Help users search, book, pay, "
    "confirm, and cancel hotel reservations.\n\n"
    "RULES:\n"
    "- ALWAYS call validate_booking_rules BEFORE book_hotel, confirm_booking, or cancel_booking.\n"
    "- If validation returns FAIL with a STEER instruction, follow the STEER guidance exactly: "
    "fix the parameters, retry the action, and always tell the user what was not possible AND "
    "what you did instead. Pattern: 'X is not available, but Y is. I adjusted to Y.'\n"
    "- If a tool call is BLOCKED by the system, inform the user — you cannot override it.\n"
    "- For payment, ask the user if they want to proceed (simulated).\n"
    "- Follow the flow: search -> validate -> book -> pay -> validate -> confirm."
)

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload, context=None):
    """Entry point for AgentCore Runtime invocations."""
    model = BedrockModel(region_name=_region)
    hooks = [BookingGuardrailsHook()]

    mcp_client = MCPClient(lambda: streamablehttp_client(GATEWAY_URL))

    with mcp_client:
        tools = mcp_client.list_tools_sync()
        agent = Agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT, hooks=hooks)

        prompt = payload if isinstance(payload, str) else payload.get("prompt", "")
        result = agent(prompt)
        return str(result)


if __name__ == "__main__":
    app.run()
