# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tools and the neurosymbolic hook for demo 04 (used by chat.py).

These are the same objects defined and explained inline in the notebook; they
are copied here so `chat.py` can import them.

The demo contrasts two ways of enforcing business rules:

  * PROMPT-ONLY  — rules written in the system prompt. The LLM can ignore them.
  * NEUROSYMBOLIC — the SAME rules as deterministic Python in a Strands
    HookProvider. `BeforeToolCallEvent` fires before every tool runs; the hook
    inspects the real tool input, and on a violation sets `event.cancel_tool`,
    so the tool never executes. The LLM physically cannot bypass it.

Neo4j is the READ-ONLY source of truth for hotel facts (does the hotel exist?
what is its room capacity?). Bookings — the writes — go to a JSON booking store,
never to Neo4j. So one rule validates against Neo4j (read) and the tools that
mutate state write to the store.
"""
import json
import os
import threading
from datetime import datetime

from neo4j import GraphDatabase, RoutingControl
from strands import tool
from strands.hooks import HookProvider, HookRegistry, BeforeToolCallEvent


# --- Read-only hotel knowledge graph (Neo4j) ---------------------------------

class HotelGraph:
    """Read-only access to the hotel knowledge graph via one managed driver."""

    def __init__(self, uri=None, user=None, password=None):
        self._driver = GraphDatabase.driver(
            uri or os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
            auth=(
                user or os.getenv("NEO4J_USER", "neo4j"),
                password or os.getenv("NEO4J_PASSWORD", "password"),
            ),
        )

    def close(self):
        self._driver.close()

    def capacity(self, hotel: str) -> int | None:
        """Max guests the hotel can seat in one room = max Room.maxOccupancy.

        Returns None if the hotel does not exist in the graph (vs 0 for a hotel
        with no rooms). Read routing, parameterized Cypher — cannot write.
        """
        records, _, _ = self._driver.execute_query(
            "MATCH (h:Hotel) WHERE toLower(h.name) = toLower($name) "
            "OPTIONAL MATCH (h)-[:HAS_ROOM]->(r:Room) "
            "RETURN count(DISTINCT h) AS hotels, max(r.maxOccupancy) AS cap",
            name=hotel, routing_=RoutingControl.READ,
        )
        row = records[0]
        if not row["hotels"]:
            return None
        cap = row["cap"]
        return int(cap) if cap is not None else 0


# --- Booking store (JSON writes; swappable to DynamoDB) ----------------------

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

    def is_paid(self, booking_id: str) -> bool:
        booking = self._load().get(booking_id)
        return bool(booking and booking.get("paid"))

    @tool
    def book_hotel(self, hotel: str, guest: str, check_in: str, check_out: str,
                   guests: int = 1) -> dict:
        """Create a hotel reservation (unpaid) and return its confirmation id.

        Args:
            hotel: Hotel name to book.
            guest: Guest full name.
            check_in: Check-in date, YYYY-MM-DD.
            check_out: Check-out date, YYYY-MM-DD.
            guests: Number of guests (default 1).
        """
        with self._lock:
            data = self._load()
            booking_id = f"BK{len(data) + 1:04d}"
            data[booking_id] = {
                "hotel": hotel, "guest": guest, "guests": guests,
                "check_in": check_in, "check_out": check_out, "paid": False,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            self._save(data)
        return {"status": "success", "content": [{"json": {"booking_id": booking_id, **data[booking_id]}}]}

    @tool
    def process_payment(self, booking_id: str, amount: float) -> dict:
        """Record a payment against a booking (marks it paid).

        Args:
            booking_id: The booking id, e.g. "BK0001".
            amount: Amount paid in USD.
        """
        with self._lock:
            data = self._load()
            if booking_id not in data:
                return {"status": "error", "content": [{"text": f"No booking {booking_id}"}]}
            data[booking_id]["paid"] = True
            data[booking_id]["amount"] = amount
            self._save(data)
        return {"status": "success", "content": [{"json": {"booking_id": booking_id, "paid": True, "amount": amount}}]}

    @tool
    def confirm_booking(self, booking_id: str) -> dict:
        """Confirm a reservation so the guest receives their final itinerary.

        Args:
            booking_id: The booking id, e.g. "BK0001".
        """
        booking = self._load().get(booking_id)
        if not booking:
            return {"status": "error", "content": [{"text": f"No booking {booking_id}"}]}
        return {"status": "success", "content": [{"json": {"booking_id": booking_id, "confirmed": True, **booking}}]}


# --- The neurosymbolic hook: business rules as deterministic Python ----------

MAX_GUESTS = 10  # company policy, independent of any single hotel's capacity


class NeurosymbolicHook(HookProvider):
    """Enforce booking rules in code before any tool runs.

    One `BeforeToolCallEvent` callback inspects the real tool input and, on a
    rule violation, sets `event.cancel_tool` to a message. Strands then skips
    the tool and feeds that message back to the LLM as the tool result — the
    rule is enforced regardless of what the model intended.

    Rules:
      book_hotel      — hotel must exist in Neo4j; guests within the hotel's
                        room capacity AND <= company max; check_in < check_out.
      confirm_booking — the booking must be paid first (checked in the store).
    """

    def __init__(self, graph: HotelGraph, store: BookingStore):
        self._graph = graph
        self._store = store

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.enforce)

    def enforce(self, event: BeforeToolCallEvent) -> None:
        name = event.tool_use["name"]
        params = event.tool_use["input"]
        violations = self._check(name, params)
        if violations:
            event.cancel_tool = "BLOCKED by business rules: " + "; ".join(violations)
            print(f"[HOOK] 🚫 {name} -> {event.cancel_tool}")

    def _check(self, name: str, params: dict) -> list[str]:
        if name == "book_hotel":
            return self._check_booking(params)
        if name == "confirm_booking":
            return self._check_confirm(params)
        return []

    def _check_booking(self, params: dict) -> list[str]:
        violations: list[str] = []
        hotel = params.get("hotel", "")
        guests = int(params.get("guests", 1))

        capacity = self._graph.capacity(hotel)  # read-only Neo4j lookup
        if capacity is None:
            violations.append(f"hotel '{hotel}' does not exist in the knowledge graph")
        elif guests > capacity:
            violations.append(f"{guests} guests exceed the hotel's room capacity of {capacity}")

        if guests > MAX_GUESTS:
            violations.append(f"maximum {MAX_GUESTS} guests per booking")

        ci, co = params.get("check_in"), params.get("check_out")
        if not (ci and co and ci < co):
            violations.append("check-in must be before check-out")
        return violations

    def _check_confirm(self, params: dict) -> list[str]:
        booking_id = params.get("booking_id", "")
        if not self._store.is_paid(booking_id):
            return [f"payment must be verified before confirming {booking_id}"]
        return []
