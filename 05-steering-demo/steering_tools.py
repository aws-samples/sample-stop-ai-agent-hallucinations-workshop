# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Demo 05 — hard block vs native steering, around one real rule:

    a room holds at most MAX_PER_ROOM (10) guests.

Two reactions to the same violating request ("book 15 guests"):

  * BlockOversizeRoomHook — a lifecycle hook that CANCELS the tool call
    (BeforeToolCallEvent.cancel_tool). The task simply fails; the agent is told
    "no" with no way forward.

  * MaxGuestsSteering — a native Strands SteeringHandler. It reads the REAL
    tool input (`tool_use["input"]["guests"]`), and when it is too big returns a
    Guide telling the agent HOW to fix it: split into ceil(guests / MAX_PER_ROOM)
    rooms of up to MAX_PER_ROOM each and book each room. The agent self-corrects
    and completes the booking. No regex on narrated text, no external server.

Both read the structured tool input, never free-text narration. The rule-from-
config variant (RuleSteering) shows the same handler with the limit loaded from
an external source (rules.json / dict) — rules can live outside code without an
Agent Control server.
"""
import os

from neo4j import GraphDatabase, RoutingControl
from strands import tool
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry
from strands.vended_plugins.steering import (
    Guide,
    Proceed,
    SteeringHandler,
    ToolSteeringAction,
)

MAX_PER_ROOM = 10


# ── Read-only Neo4j: confirm a hotel exists before we book it ────────────────
class HotelCheck:
    """One managed driver, READ routing only — Neo4j never writes here."""

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

    @tool
    def hotel_exists(self, hotel: str) -> dict:
        """Check that a hotel exists in the knowledge graph before booking it.

        Args:
            hotel: Exact hotel name.
        """
        recs, _, _ = self._driver.execute_query(
            "MATCH (h:Hotel {name:$hotel}) RETURN h.name AS name LIMIT 1",
            hotel=hotel, routing_=RoutingControl.READ,
        )
        if not recs:
            return {"status": "error", "content": [{"text": f"Unknown hotel: {hotel}"}]}
        return {"status": "success", "content": [{"json": {"hotel": recs[0]["name"]}}]}


def _split(guests: int, cap: int = MAX_PER_ROOM) -> list[int]:
    """Split a party into full rooms of `cap`, plus a remainder. Generic for any n.

    e.g. 15 → [10, 5], 23 → [10, 10, 3]. ceil(guests/cap) rooms, totalling guests.
    """
    full, rem = divmod(guests, cap)
    return [cap] * full + ([rem] if rem else [])


# ── Approach A: hard block (the problem) ─────────────────────────────────────
class BlockOversizeRoomHook(HookProvider):
    """Cancel any book_hotel call that exceeds MAX_PER_ROOM. The task fails."""

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self._block)

    def _block(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use
        if tool_use.get("name") != "book_hotel":
            return
        guests = (tool_use.get("input") or {}).get("guests", 1)
        if guests and guests > MAX_PER_ROOM:
            print(f"[HOOK] ⛔ Blocked: {guests} guests exceeds {MAX_PER_ROOM}/room")
            event.cancel_tool = (
                f"Rejected: {guests} guests exceeds the {MAX_PER_ROOM}-per-room limit."
            )


# ── Approach B: native steering (the fix) ────────────────────────────────────
class MaxGuestsSteering(SteeringHandler):
    """Guide the agent to split an oversize party into rooms, then let it book."""

    name = "max-guests"

    async def steer_before_tool(self, *, agent, tool_use, **kwargs) -> ToolSteeringAction:
        if tool_use.get("name") != "book_hotel":
            return Proceed()
        guests = (tool_use.get("input") or {}).get("guests", 1)
        if guests and guests > MAX_PER_ROOM:
            rooms = _split(guests)
            print(f"[STEERING] 🧭 {guests} guests > {MAX_PER_ROOM}/room → split into {rooms}")
            return Guide(
                reason=(
                    f"A room holds at most {MAX_PER_ROOM} guests, but you requested "
                    f"{guests}. Split the party into {len(rooms)} rooms of "
                    f"{rooms} guests and call book_hotel once per room."
                )
            )
        return Proceed()


# ── Variant: the rule lives in config, not code ──────────────────────────────
class RuleSteering(SteeringHandler):
    """Same native steering, but the limit is loaded from an external source.

    Shows that steering rules can live outside the code (a rules.json / dict,
    a config service, ...) — no Agent Control server required to run the demo.
    """

    name = "rule-steering"

    def __init__(self, rules: dict):
        super().__init__()
        self._cap = int(rules["max_guests_per_room"])

    async def steer_before_tool(self, *, agent, tool_use, **kwargs) -> ToolSteeringAction:
        if tool_use.get("name") != "book_hotel":
            return Proceed()
        guests = (tool_use.get("input") or {}).get("guests", 1)
        if guests and guests > self._cap:
            rooms = _split(guests, self._cap)
            return Guide(
                reason=(
                    f"Policy (from config): max {self._cap} guests per room. Split the "
                    f"{guests} guests into {len(rooms)} rooms of {rooms} and book each."
                )
            )
        return Proceed()
