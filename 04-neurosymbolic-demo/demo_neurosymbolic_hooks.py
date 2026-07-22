# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Neurosymbolic validation using Strands Hooks
Replaces validation logic inside tools with centralized hook
"""
import sys
from strands import Agent, tool
from strands.hooks import HookProvider, HookRegistry, BeforeToolCallEvent
from datetime import datetime, timedelta
from rules import BOOKING_RULES, CONFIRMATION_RULES, CANCELLATION_RULES, validate

# Dates are always computed relative to today. Never hardcode a date here —
# a fixed date silently rots into the past and the advance_booking rule
# then blocks the scenario that is meant to succeed.
DAYS_AHEAD = 30
STAY_NIGHTS = 5
CHECK_IN = (datetime.now() + timedelta(days=DAYS_AHEAD)).strftime("%Y-%m-%d")
CHECK_OUT = (datetime.now() + timedelta(days=DAYS_AHEAD + STAY_NIGHTS)).strftime("%Y-%m-%d")

STATE = {
    "bookings": {"BK001": {"hotel": "AnyCompany Lisbon Resort", "check_in": CHECK_IN, "guests": 2}},
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
        self.blocked_calls: list[dict] = []  # structural record of what the hook cancelled

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.validate)

    def reset(self) -> None:
        """Clear the blocked-call log between scenarios."""
        self.blocked_calls = []

    def validate(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use["name"]
        if tool_name not in self.rules:
            return

        ctx = self._build_context(tool_name, event.tool_use["input"])
        passed, violations = validate(self.rules[tool_name], ctx)

        if not passed:
            reason = f"BLOCKED: {', '.join(violations)}"
            event.cancel_tool = reason
            self.blocked_calls.append({
                "tool": tool_name,
                "params": event.tool_use["input"],
                "reason": reason,
            })

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
#   MODEL = "us.anthropic.claude-sonnet-5"
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

SCENARIOS = [
    {
        "query": "Confirm booking BK001",
        "expected": "BLOCKED",
        "rule": "Payment must be verified before confirmation",
    },
    {
        "query": f"Book AnyCompany Lisbon Resort for 15 people from {CHECK_IN} to {CHECK_OUT}",
        "expected": "BLOCKED",
        "rule": "Maximum 10 guests per booking",
    },
    {
        "query": f"Book AnyCompany Lisbon Resort for 2 guests from {CHECK_IN} to {CHECK_OUT}",
        "expected": "ALLOWED",
        "rule": "All rules pass",
    },
]

results = []

for scenario in SCENARIOS:
    print(f"\n📝 {scenario['query']}")
    print(f"   Expected: {scenario['expected']} — {scenario['rule']}")

    # Assert on hook state, never on the model's wording: the LLM paraphrases
    # its refusals, so grepping the prose for "BLOCKED" is a false-negative machine.
    hook.reset()
    agent(scenario["query"])
    blocked = len(hook.blocked_calls) > 0
    correct = blocked == (scenario["expected"] == "BLOCKED")

    if blocked:
        call = hook.blocked_calls[0]
        print(f"   🛑 Hook cancelled {call['tool']}({call['params']})")
        print(f"      Reason: {call['reason']}")
    else:
        print("   🟢 Allowed — every symbolic rule passed")
    print(f"   {'✅ correct' if correct else '❌ wrong'}")

    results.append({"blocked": blocked, "correct": correct, **scenario})

blocked_expected = sum(1 for r in results if r["expected"] == "BLOCKED")
blocked_correct = sum(1 for r in results if r["expected"] == "BLOCKED" and r["blocked"])
allowed_expected = sum(1 for r in results if r["expected"] == "ALLOWED")
allowed_correct = sum(1 for r in results if r["expected"] == "ALLOWED" and not r["blocked"])
total_correct = sum(1 for r in results if r["correct"])

print("\n" + "="*70)
print("SCORECARD")
print("="*70)
print(f"  TOTAL                            {total_correct}/{len(results)} correct")
print(f"  Hook blocked {blocked_correct}/{blocked_expected} invalid operations")
print(f"  Hook allowed {allowed_correct}/{allowed_expected} valid operations")

# Regression guard. The allow path is the one that rots: reintroduce a hardcoded
# date and advance_booking blocks the valid booking, failing this loudly.
failures = []
if total_correct != len(results):
    failures.append(f"{len(results) - total_correct} scenario(s) did not match the expected outcome")
if blocked_correct < 1:
    failures.append("no operation was blocked — the symbolic rules never fired")
if allowed_correct < 1:
    failures.append("no valid operation was allowed — check for a hardcoded (now past) date")

if failures:
    print("\n❌ FAILED:")
    for failure in failures:
        print(f"   • {failure}")
    print("="*70)
    sys.exit(1)

print("\n✅ Hooks enforce symbolic rules the LLM cannot bypass.")
print("CONCLUSION: Hooks provide clean separation of concerns")
print("="*70)
