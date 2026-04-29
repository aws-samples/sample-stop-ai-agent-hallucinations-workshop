# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Booking tools — clean, no validation logic.

Same tools used by both approaches (Hooks and Agent Control).
Validation is handled externally:
  - Hooks approach: NeurosymbolicHook intercepts via BeforeToolCallEvent.cancel_tool
  - Agent Control: AgentControlPlugin + AgentControlSteeringHandler via server-managed policies

See: https://strandsagents.com/docs/user-guide/concepts/tools/custom-tools/
"""

from strands import tool

STATE = {
    "bookings": {
        "BK001": {"hotel": "AnyCompany Lisbon Resort", "check_in": "2026-04-15", "guests": 2, "total": 400},
    },
    "payments": {},
}


@tool
def book_hotel(hotel: str, check_in: str, check_out: str, guests: int = 1) -> str:
    """Book a hotel room.

    Args:
        hotel: Hotel name
        check_in: Check-in date (YYYY-MM-DD)
        check_out: Check-out date (YYYY-MM-DD)
        guests: Number of guests
    """
    booking_id = f"BK{len(STATE['bookings']) + 1:03d}"
    STATE["bookings"][booking_id] = {
        "hotel": hotel, "check_in": check_in, "check_out": check_out,
        "guests": guests, "total": guests * 100,
    }
    return f"SUCCESS: Booking {booking_id} — {hotel}, {guests} guests, {check_in} to {check_out}"


@tool
def process_payment(amount: float, booking_id: str) -> str:
    """Process payment for a booking.

    Args:
        amount: Payment amount in USD
        booking_id: Booking ID to pay for
    """
    if booking_id not in STATE["bookings"]:
        return f"ERROR: Booking '{booking_id}' not found"
    STATE["payments"][booking_id] = amount
    return f"SUCCESS: Processed ${amount:.2f} for {booking_id}"


@tool
def confirm_booking(booking_id: str) -> str:
    """Confirm a booking after payment.

    Args:
        booking_id: Booking ID to confirm
    """
    return f"SUCCESS: Confirmed {booking_id}"


ALL_TOOLS = [book_hotel, process_payment, confirm_booking]
