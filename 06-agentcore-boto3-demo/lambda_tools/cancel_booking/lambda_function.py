# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda: cancel_booking — cancels a booking and returns room to inventory."""

import json
import os
from datetime import datetime

import boto3

dynamodb = boto3.resource("dynamodb")
hotels_table = dynamodb.Table(os.environ["HOTELS_TABLE"])
bookings_table = dynamodb.Table(os.environ["BOOKINGS_TABLE"])


def handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    booking_id = body.get("booking_id", "")

    b = bookings_table.get_item(Key={"booking_id": booking_id}).get("Item")
    if not b:
        return {"statusCode": 404, "body": f"ERROR: Booking '{booking_id}' not found."}

    if b["status"] == "CANCELLED":
        return {"statusCode": 409, "body": "ERROR: Already cancelled."}

    bookings_table.update_item(
        Key={"booking_id": booking_id},
        UpdateExpression="SET #s = :cancelled, updated_at = :now",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":cancelled": "CANCELLED", ":now": datetime.now().isoformat()},
    )

    hotels_table.update_item(
        Key={"hotel_id": b["hotel_id"]},
        UpdateExpression="SET available_rooms = available_rooms + :one",
        ExpressionAttributeValues={":one": 1},
    )

    return {"statusCode": 200, "body": f"SUCCESS: {booking_id} cancelled. Room returned to {b['hotel_name']}."}
