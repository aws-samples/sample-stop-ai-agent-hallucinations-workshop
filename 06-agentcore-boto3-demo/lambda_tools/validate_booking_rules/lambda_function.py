# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda: validate_booking_rules — reads steering rules from DynamoDB and evaluates them."""

import json
import os
from datetime import datetime

import boto3

dynamodb = boto3.resource("dynamodb")
bookings_table = dynamodb.Table(os.environ["BOOKINGS_TABLE"])
steering_rules_table = dynamodb.Table(os.environ["STEERING_RULES_TABLE"])

OPERATORS = {
    "gt": lambda val, thr: val > thr,
    "lt": lambda val, thr: val < thr,
    "gte": lambda val, thr: val >= thr,
    "lte": lambda val, thr: val <= thr,
    "eq": lambda val, thr: val == thr,
    "ne": lambda val, thr: val != thr,
}


def _get_rules(action):
    response = steering_rules_table.scan(
        FilterExpression="#a = :action AND #e = :enabled",
        ExpressionAttributeNames={"#a": "action", "#e": "enabled"},
        ExpressionAttributeValues={":action": action, ":enabled": True},
    )
    return response.get("Items", [])


def _build_context(action, params):
    ctx = {}
    if action == "book":
        try:
            ci = datetime.fromisoformat(params.get("check_in", ""))
            co = datetime.fromisoformat(params.get("check_out", ""))
            ctx["nights"] = (co - ci).days
            ctx["days_until_checkin"] = (ci - datetime.now()).days
        except (ValueError, TypeError):
            ctx["nights"] = -1
            ctx["days_until_checkin"] = -1
        ctx["guests"] = params.get("guests", 0)
    elif action in ("confirm", "cancel"):
        b = bookings_table.get_item(Key={"booking_id": params.get("booking_id", "")}).get("Item")
        if not b:
            return {"_error": f"Booking '{params.get('booking_id')}' not found."}
        ctx["booking_status"] = b["status"]
        if action == "cancel":
            try:
                ci = datetime.fromisoformat(b["check_in"])
                ctx["days_until_checkin"] = (ci - datetime.now()).days
            except (ValueError, TypeError):
                ctx["days_until_checkin"] = 0
    return ctx


def _evaluate(rules, context):
    violations = []
    for rule in rules:
        field = rule["condition_field"]
        op = rule["operator"]
        threshold = rule["threshold"]
        value = context.get(field)
        if value is None:
            continue
        if isinstance(threshold, (int, float)) or str(threshold).lstrip("-").isdigit():
            threshold = int(threshold)
            value = int(value) if not isinstance(value, str) else value
        op_fn = OPERATORS.get(op)
        if op_fn and op_fn(value, threshold):
            violations.append(rule)
    return violations


def handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    action = body.get("action", "")

    if action not in ("book", "confirm", "cancel"):
        return {"statusCode": 400, "body": f"FAIL: Unknown action '{action}'."}

    rules = _get_rules(action)
    if not rules:
        return {"statusCode": 200, "body": f"PASS: No rules configured for '{action}'. Proceed."}

    ctx = _build_context(action, body)
    if "_error" in ctx:
        return {"statusCode": 404, "body": f"FAIL: {ctx['_error']}"}

    violated = _evaluate(rules, ctx)
    if not violated:
        return {"statusCode": 200, "body": f"PASS: All {len(rules)} rules passed for '{action}'. Proceed."}

    lines = [f"- {v['fail_message']}\n  STEER: {v['steer_message']}" for v in violated]
    return {"statusCode": 200, "body": f"FAIL: {len(violated)} rule(s) violated for '{action}':\n" + "\n".join(lines)}
