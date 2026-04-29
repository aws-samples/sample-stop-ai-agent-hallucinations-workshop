# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda: search_available_hotels — scans Hotels table with filters."""

import json
import os

import boto3

dynamodb = boto3.resource("dynamodb")
hotels_table = dynamodb.Table(os.environ["HOTELS_TABLE"])


def handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    city = body.get("city", "")
    country = body.get("country", "")
    max_price = body.get("max_price", 0)
    min_stars = body.get("min_stars", 0)

    if not any([city, country, max_price, min_stars]):
        return {"statusCode": 400, "body": "ERROR: Provide at least one filter"}

    filter_parts, expr_values, expr_names = [], {}, {}

    if city:
        filter_parts.append("contains(#city, :city)")
        expr_names["#city"] = "city"
        expr_values[":city"] = city.title()
    if country:
        filter_parts.append("contains(#country, :country)")
        expr_names["#country"] = "country"
        expr_values[":country"] = country.title()
    if max_price > 0:
        filter_parts.append("price_per_night <= :max_price")
        expr_values[":max_price"] = max_price
    if min_stars > 0:
        filter_parts.append("stars >= :min_stars")
        expr_values[":min_stars"] = min_stars

    filter_parts.append("available_rooms > :zero")
    expr_values[":zero"] = 0

    scan_kwargs = {
        "FilterExpression": " AND ".join(filter_parts),
        "ExpressionAttributeValues": expr_values,
    }
    if expr_names:
        scan_kwargs["ExpressionAttributeNames"] = expr_names

    hotels = hotels_table.scan(**scan_kwargs).get("Items", [])
    if not hotels:
        return {"statusCode": 200, "body": "No available hotels found matching your criteria."}

    lines = [
        f"- {h['name']} ({h['city']}, {h['country']}): {h['stars']} stars, ${int(h['price_per_night'])}/night, {int(h['available_rooms'])} rooms | ID: {h['hotel_id']}"
        for h in hotels
    ]
    return {"statusCode": 200, "body": f"Found {len(hotels)} hotel(s):\n" + "\n".join(lines)}
