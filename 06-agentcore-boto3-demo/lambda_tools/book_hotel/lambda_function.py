# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda: book_hotel — creates a PENDING reservation in DynamoDB."""

import json
import os
import uuid
from datetime import datetime

import boto3

dynamodb = boto3.resource("dynamodb")
hotels_table = dynamodb.Table(os.environ["HOTELS_TABLE"])
bookings_table = dynamodb.Table(os.environ["BOOKINGS_TABLE"])


def handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    hotel_id = body.get("hotel_id", "")
    guest_name = body.get("guest_name", "")
    check_in = body.get("check_in", "")
    check_out = body.get("check_out", "")
    guests = body.get("guests", 1)

    hotel = hotels_table.get_item(Key={"hotel_id": hotel_id}).get("Item")
    if not hotel:
        return {"statusCode": 404, "body": f"ERROR: Hotel '{hotel_id}' not found."}

    if int(hotel.get("available_rooms", 0)) <= 0:
        return {"statusCode": 409, "body": f"ERROR: No rooms available at {hotel['name']}."}

    max_g = int(hotel.get("max_guests_per_room", 2))
    if guests > max_g:
        return {"statusCode": 400, "body": f"ERROR: Max {max_g} guests. You requested {guests}."}

    try:
        nights = (datetime.fromisoformat(check_out) - datetime.fromisoformat(check_in)).days
    except (ValueError, TypeError):
        return {"statusCode": 400, "body": "ERROR: Invalid date format. Use YYYY-MM-DD."}

    if nights <= 0:
        return {"statusCode": 400, "body": "ERROR: Check-out must be after check-in."}

    total = int(hotel["price_per_night"]) * nights
    bid = f"BK-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now().isoformat()

    bookings_table.put_item(Item={
        "booking_id": bid, "hotel_id": hotel_id, "hotel_name": hotel["name"],
        "guest_name": guest_name, "check_in": check_in, "check_out": check_out,
        "guests": guests, "nights": nights, "price_per_night": int(hotel["price_per_night"]),
        "total_amount": total, "status": "PENDING", "created_at": now, "updated_at": now,
    })

    hotels_table.update_item(
        Key={"hotel_id": hotel_id},
        UpdateExpression="SET available_rooms = available_rooms - :one",
        ExpressionAttributeValues={":one": 1},
    )

    return {
        "statusCode": 200,
        "body": (
            f"SUCCESS: Booking {bid}\n  Hotel: {hotel['name']}\n  Guest: {guest_name}\n"
            f"  Dates: {check_in} to {check_out} ({nights} nights)\n"
            f"  Total: ${total}\n  Status: PENDING"
        ),
    }
