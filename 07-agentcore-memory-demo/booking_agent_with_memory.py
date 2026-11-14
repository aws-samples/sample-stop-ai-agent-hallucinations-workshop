# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Hotel Booking Agent with Long-term Memory — AgentCore Runtime entry point.

Connects to AgentCore Gateway via MCP (Model Context Protocol) to access tools.
This version uses AgentCore Memory with STM + LTM (long-term memory across sessions via actor ID).

Compared to booking_agent.py, this version:
- Imports AgentCore memory integration classes
- Extracts actor ID from custom HTTP header
- Configures AgentCoreMemorySessionManager
- Enables cross-session memory recall
"""

import os
from datetime import datetime

import boto3
from bedrock_agentcore import BedrockAgentCoreApp, RequestContext
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.executors import SequentialToolExecutor
from strands.tools.mcp.mcp_client import MCPClient

# --- Configuration from environment variables ---

GATEWAY_URL = os.environ["GATEWAY_URL"]
MEMORY_ID = os.environ["BEDROCK_AGENTCORE_MEMORY_ID"]  # Memory resource created by the notebook
_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

# HTTP header name for actor ID (normalized to lowercase by AgentCore)
CUSTOM_HEADER_NAME = 'x-amzn-bedrock-agentcore-runtime-custom-actor-id'


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
    "You are a hotel booking assistant with long-term memory. Help users search, book, pay, "
    "confirm, and cancel hotel reservations. Remember user preferences across sessions.\n\n"
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

# One MCP client for the process. Passing it in tools=[...] lets Strands manage
# the MCP session lifecycle (Strands docs, "MCP tools" > Managed Integration).
# The Gateway uses AWS_IAM inbound auth, so requests are signed with SigV4 using the
# runtime's execution role (Strands docs, "MCP transports" > AWS IAM).
_mcp_client = MCPClient(lambda: aws_iam_streamablehttp_client(
    endpoint=GATEWAY_URL, aws_region=_region, aws_service="bedrock-agentcore"
))

# AgentCoreMemorySessionManager is bound to one actor and one session, and only
# one agent per session is supported, so keep one agent per (actor, session).
_agents = {}

def get_or_create_agent(actor_id: str, session_id: str):
    """Get the agent for this actor and session, creating it on first use."""
    key = (actor_id, session_id)

    if key not in _agents:
        # Configure memory with user facts and preferences
        memory_config = AgentCoreMemoryConfig(
            memory_id=MEMORY_ID,
            session_id=session_id,
            actor_id=actor_id,
            retrieval_config={
                f"/users/{actor_id}/facts": RetrievalConfig(top_k=3, relevance_score=0.3),
                f"/users/{actor_id}/preferences": RetrievalConfig(top_k=3, relevance_score=0.3)
            }
        )
        session_manager = AgentCoreMemorySessionManager(memory_config, _region)
        app.logger.info(f"Memory enabled for actor={actor_id}, session={session_id}")

        _agents[key] = Agent(
            model=BedrockModel(region_name=_region),
            tools=[_mcp_client],
            system_prompt=SYSTEM_PROMPT,
            hooks=[BookingGuardrailsHook()],
            # pay -> confirm in one turn must run in order, so the guardrail sees the payment
            tool_executor=SequentialToolExecutor(),
            session_manager=session_manager,
            context_manager="auto"
        )

    return _agents[key]


@app.entrypoint
def invoke(payload, context: RequestContext = None):
    """Entry point for AgentCore Runtime invocations."""
    app.logger.info("Payload: %s", payload)
    app.logger.info("Context: %s", context)

    if not MEMORY_ID:
        return {"error": "Memory not configured. Set BEDROCK_AGENTCORE_MEMORY_ID environment variable."}

    # Extract actor ID from the custom header (compared case-insensitively)
    # The runtime forwards the header only if it is in the runtime's requestHeaderAllowlist
    actor_id = 'default-user'
    request_headers = getattr(context, 'request_headers', None) or {}
    for key, value in request_headers.items():
        if key.lower() == CUSTOM_HEADER_NAME:
            actor_id = value
    app.logger.info(f"Actor ID from header '{CUSTOM_HEADER_NAME}': {actor_id}")

    session_id = context.session_id if context and hasattr(context, 'session_id') else None
    app.logger.info(f"Using actor_id='{actor_id}', session_id='{session_id}'")

    # Get or create agent (lazy loading)
    agent = get_or_create_agent(actor_id, session_id)

    prompt = payload if isinstance(payload, str) else payload.get("prompt", "")
    result = agent(prompt)

    # Extract tool calls from Strands Agent metrics
    # result.metrics.tool_metrics is a dict where keys are tool names
    tools_used = []
    if hasattr(result, 'metrics') and hasattr(result.metrics, 'tool_metrics'):
        tool_metrics = result.metrics.tool_metrics
        tools_used = list(tool_metrics.keys())

        # Log for debugging - verify MCP Gateway tools are tracked
        app.logger.info(f"Tool metrics found: {len(tools_used)} tools")
        for tool_name, metrics in tool_metrics.items():
            app.logger.info(f"  Tool: {tool_name}, Calls: {metrics.call_count if hasattr(metrics, 'call_count') else 'unknown'}")
    else:
        app.logger.warning("No tool metrics found in result")

    return {
        "response": result.message.get('content', [{}])[0].get('text', str(result)),
        "tools_used": tools_used
    }


if __name__ == "__main__":
    app.run()
