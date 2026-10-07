# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Booking store — reservations live in a simple shared JSON file.

Neo4j is READ-ONLY (hotel facts). Writes — the bookings — go here instead, so a
single JSON file is the one source of truth that every agent reads and writes
through this tool. This also works across multi-agent systems (demo 03), where
per-agent `agent.state` would NOT be shared.

Production-correct: the store is behind a small class with a lock. Swap the
JSON backend for Amazon DynamoDB by reimplementing `_load`/`_save` (e.g.
`table.get_item` / `table.put_item`) — the tool interface and agent code stay
identical.
"""
import json
import os
import threading
from datetime import datetime

from strands import tool


class BookingStore:
    """A tiny reservation store backed by a JSON file (swappable to DynamoDB)."""

    def __init__(self, path="bookings.json"):
        self._path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not os.path.exists(self._path):
            return {}
        with open(self._path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # --- tools (class methods decorated with @tool; bind these when building the agent) ---

    @tool
    def book_hotel(self, hotel: str, guest: str, nights: int = 1) -> dict:
        """Create a hotel reservation and return its confirmation id.

        Args:
            hotel: Hotel name to book.
            guest: Guest full name.
            nights: Number of nights (default 1).
        """
        with self._lock:
            data = self._load()
            booking_id = f"BK{len(data) + 1:04d}"
            data[booking_id] = {
                "hotel": hotel, "guest": guest, "nights": nights,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            self._save(data)
        return {"status": "success", "content": [{"json": {"booking_id": booking_id, **data[booking_id]}}]}

    @tool
    def get_booking(self, booking_id: str) -> dict:
        """Look up a reservation by its confirmation id.

        Args:
            booking_id: The booking id, e.g. "BK0001".
        """
        booking = self._load().get(booking_id)
        if not booking:
            return {"status": "error", "content": [{"text": f"No booking {booking_id}"}]}
        return {"status": "success", "content": [{"json": {"booking_id": booking_id, **booking}}]}
