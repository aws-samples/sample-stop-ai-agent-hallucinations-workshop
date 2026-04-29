# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda: confirm_booking — marks a PAID booking as CONFIRMED."""

import json
import os
from datetime import datetime

import boto3

dynamodb = boto3.resource("dynamodb")
bookings_table = dynamodb.Table(os.environ["BOOKINGS_TABLE"])


def handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    booking_id = body.get("booking_id", "")

    b = bookings_table.get_item(Key={"booking_id": booking_id}).get("Item")
    if not b:
        return {"statusCode": 404, "body": f"ERROR: Booking '{booking_id}' not found."}

    if b["status"] != "PAID":
        return {"statusCode": 409, "body": f"ERROR: Booking is '{b['status']}'. Must be PAID first."}

    bookings_table.update_item(
        Key={"booking_id": booking_id},
        UpdateExpression="SET #s = :confirmed, updated_at = :now",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":confirmed": "CONFIRMED", ":now": datetime.now().isoformat()},
    )

    return {"statusCode": 200, "body": f"SUCCESS: Booking {booking_id} CONFIRMED."}
