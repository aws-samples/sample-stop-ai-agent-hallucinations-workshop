# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Symbolic Rules Engine - Constrains LLM decisions with verifiable logic
"""
from dataclasses import dataclass
from typing import Callable

@dataclass
class Rule:
    name: str
    condition: Callable[[dict], bool]
    message: str

# Condition functions
def valid_dates_check(ctx: dict) -> bool:
    return ctx.get("check_in") and ctx.get("check_out") and ctx["check_in"] < ctx["check_out"]

def max_guests_check(ctx: dict) -> bool:
    return ctx.get("guests", 1) <= 10

def advance_booking_check(ctx: dict) -> bool:
    return ctx.get("days_until_checkin", 0) >= 1

def payment_verified_check(ctx: dict) -> bool:
    return ctx.get("payment_verified", False)

def cancellation_window_check(ctx: dict) -> bool:
    return ctx.get("days_until_checkin", 0) >= 2

def booking_exists_check(ctx: dict) -> bool:
    return ctx.get("booking_id") is not None

# Business rules that MUST be enforced
BOOKING_RULES = [
    Rule(
        name="valid_dates",
        condition=valid_dates_check,
        message="Check-in must be before check-out"
    ),
    Rule(
        name="max_guests",
        condition=max_guests_check,
        message="Maximum 10 guests per booking"
    ),
    Rule(
        name="advance_booking",
        condition=advance_booking_check,
        message="Must book at least 1 day in advance"
    ),
]

CONFIRMATION_RULES = [
    Rule(
        name="payment_before_confirm",
        condition=payment_verified_check,
        message="Payment must be verified before confirmation"
    ),
]

CANCELLATION_RULES = [
    Rule(
        name="cancellation_window",
        condition=cancellation_window_check,
        message="Cannot cancel within 48 hours of check-in"
    ),
    Rule(
        name="booking_exists",
        condition=booking_exists_check,
        message="No booking found to cancel"
    ),
]

def validate(rules: list[Rule], context: dict) -> tuple[bool, list[str]]:
    """Run all rules, return (passed, violations)"""
    violations = [r.message for r in rules if not r.condition(context)]
    return len(violations) == 0, violations
