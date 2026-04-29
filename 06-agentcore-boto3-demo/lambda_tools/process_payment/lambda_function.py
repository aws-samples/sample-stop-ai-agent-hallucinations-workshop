# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda: process_payment — marks a PENDING booking as PAID."""

import json
import os
from datetime import datetime

import boto3

dynamodb = boto3.resource("dynamodb")
bookings_table = dynamodb.Table(os.environ["BOOKINGS_TABLE"])


def handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    booking_id = body.get("booking_id", "")
    amount = body.get("amount", 0)

    b = bookings_table.get_item(Key={"booking_id": booking_id}).get("Item")
    if not b:
        return {"statusCode": 404, "body": f"ERROR: Booking '{booking_id}' not found."}

    if b["status"] != "PENDING":
        return {"statusCode": 409, "body": f"ERROR: Booking is '{b['status']}', expected PENDING."}

    expected = int(b["total_amount"])
    if amount != expected:
        return {"statusCode": 400, "body": f"ERROR: Amount ${amount} != total ${expected}."}

    bookings_table.update_item(
        Key={"booking_id": booking_id},
        UpdateExpression="SET #s = :paid, updated_at = :now",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":paid": "PAID", ":now": datetime.now().isoformat()},
    )

    return {"statusCode": 200, "body": f"SUCCESS: ${amount} paid for {booking_id}. Status: PAID."}
