# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Neurosymbolic validation using Strands Hooks
Replaces validation logic inside tools with centralized hook
"""
from strands import Agent, tool
from strands.hooks import HookProvider, HookRegistry, BeforeToolCallEvent
from datetime import datetime
from rules import BOOKING_RULES, CONFIRMATION_RULES, CANCELLATION_RULES, validate

STATE = {
    "bookings": {"BK001": {"hotel": "AnyCompany Lisbon Resort", "check_in": "2026-02-15", "guests": 2}},
    "payments": {}
}

class NeurosymbolicHook(HookProvider):
    """Validates tool calls against symbolic rules before execution"""
    
    def __init__(self, state: dict):
        self.state = state
        self.rules = {
            "book_hotel": BOOKING_RULES,
            "confirm_booking": CONFIRMATION_RULES,
            "cancel_booking": CANCELLATION_RULES,
        }
    
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.validate)
    
    def validate(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use["name"]
        if tool_name not in self.rules:
            return
        
        ctx = self._build_context(tool_name, event.tool_use["input"])
        passed, violations = validate(self.rules[tool_name], ctx)
        
        if not passed:
            event.cancel_tool = f"BLOCKED: {', '.join(violations)}"
    
    def _build_context(self, tool_name: str, params: dict) -> dict:
        if tool_name == "book_hotel":
            try:
                ci = datetime.strptime(params["check_in"], "%Y-%m-%d")
                co = datetime.strptime(params["check_out"], "%Y-%m-%d")
                return {
                    "check_in": ci,
                    "check_out": co,
                    "guests": params.get("guests", 1),
                    "days_until_checkin": (ci - datetime.now()).days
                }
            except (ValueError, KeyError):
                # Return context that fails validation
                return {
                    "check_in": None,
                    "check_out": None,
                    "guests": 999,
                    "days_until_checkin": -999
                }
        elif tool_name == "confirm_booking":
            return {"payment_verified": params["booking_id"] in self.state["payments"]}
        elif tool_name == "cancel_booking":
            booking = self.state["bookings"].get(params["booking_id"])
            if booking:
                ci = datetime.strptime(booking["check_in"], "%Y-%m-%d")
                return {
                    "booking_id": params["booking_id"],
                    "days_until_checkin": (ci - datetime.now()).days
                }
            return {"booking_id": None}
        return {}

# Clean tools without validation logic
@tool
def book_hotel(hotel: str, check_in: str, check_out: str, guests: int = 1) -> str:
    """Book a hotel room."""
    return f"SUCCESS: Booked {hotel} for {guests} guests, {check_in} to {check_out}"

@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel an existing booking."""
    return f"SUCCESS: Cancelled booking {booking_id}"

@tool
def process_payment(amount: float, booking_id: str) -> str:
    """Process payment for a booking."""
    if booking_id not in STATE["bookings"]:
        return "ERROR: Booking not found"
    STATE["payments"][booking_id] = amount
    return f"SUCCESS: Processed ${amount} for {booking_id}"

@tool
def confirm_booking(booking_id: str) -> str:
    """Confirm a booking."""
    return f"SUCCESS: Confirmed {booking_id}"

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

print("="*70)
print("NEUROSYMBOLIC VALIDATION WITH STRANDS HOOKS")
print("="*70)
print("\nKey Benefits:")
print("  ✓ Tools are clean - no validation logic mixed in")
print("  ✓ Centralized validation in one hook")
print("  ✓ Symbolic rules enforced at execution time")
print("  ✓ LLM cannot bypass rules\n")
print("="*70)

hook = NeurosymbolicHook(STATE)
agent = Agent(
    tools=[book_hotel, cancel_booking, process_payment, confirm_booking],
    hooks=[hook],
)

tests = [
    ("Confirm booking BK001", "Should block - no payment"),
    ("Book AnyCompany Lisbon Resort for 15 people from 2026-03-20 to 2026-03-25", "Should block - max 10 guests"),
    ("Book AnyCompany Lisbon Resort for 2 guests from 2026-03-20 to 2026-03-25", "Should succeed"),
]

for query, expected in tests:
    print(f"\n📝 {query}")
    print(f"   Expected: {expected}")
    result = agent(query)
    output = str(result)
    
    if "BLOCKED" in output or ("cannot" in output.lower() and "must" in output.lower()):
        print("   ✅ Blocked by symbolic rules")
    elif "SUCCESS" in output:
        print("   ✅ Executed successfully")
    else:
        print("   ⚠️  Check output")

print("\n" + "="*70)
print("CONCLUSION: Hooks provide clean separation of concerns")
print("="*70)
