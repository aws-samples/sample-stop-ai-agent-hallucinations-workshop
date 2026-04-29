# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda: get_booking — retrieves booking details from DynamoDB."""

import json
import os

import boto3

dynamodb = boto3.resource("dynamodb")
bookings_table = dynamodb.Table(os.environ["BOOKINGS_TABLE"])


def handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    booking_id = body.get("booking_id", "")

    b = bookings_table.get_item(Key={"booking_id": booking_id}).get("Item")
    if not b:
        return {"statusCode": 404, "body": f"ERROR: Booking '{booking_id}' not found."}

    return {
        "statusCode": 200,
        "body": (
            f"Booking {b['booking_id']}:\n  Hotel: {b['hotel_name']}\n"
            f"  Guest: {b['guest_name']}\n  Dates: {b['check_in']} to {b['check_out']}\n"
            f"  Guests: {int(b['guests'])}\n  Total: ${int(b['total_amount'])}\n"
            f"  Status: {b['status']}"
        ),
    }
