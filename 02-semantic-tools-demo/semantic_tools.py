# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Demo 02 — a large, overlapping tool pool and a semantic-selection harness.

Two parts, both reused by the notebook and chat.py:

1. A pool of ~30 travel tools with REAL overlap (many get_*/search_*/check_*
   variants). Some are real — hotels via read-only Neo4j, weather via the free
   Open-Meteo API (no key), bookings via the JSON store. The rest are realistic
   stand-ins (flights, currency, visas, ...) that return plausible data: the
   point of this demo is tool SELECTION, not each tool's internals.

2. `SemanticToolHook` — the harness. Instead of Python code picking tools before
   building the agent, this HookProvider lets the agent own all tools and, on
   every request (`BeforeInvocationEvent`), trims the agent's live tool_registry
   to the FAISS top-k most relevant to the user's question. Fewer, relevant
   tools = fewer tokens and less chance a small model picks the wrong one.
"""
import json
import os
import urllib.parse
import urllib.request

import boto3
import faiss
import numpy as np
from neo4j import GraphDatabase, RoutingControl
from strands import tool
from strands.hooks import BeforeInvocationEvent, HookProvider, HookRegistry

REGION = os.getenv("AWS_REGION", "us-east-1")

# ── Real tool 1: hotels via read-only Neo4j ──────────────────────────────────
_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
)


@tool
def search_hotels_by_city(city: str) -> dict:
    """Find hotels located in a given city, with their guest rating.

    Args:
        city: City name, e.g. "Barcelona".
    """
    recs, _, _ = _driver.execute_query(
        "MATCH (h:Hotel) WHERE h.city = $city "
        "RETURN h.name AS name, h.guestRating AS rating ORDER BY rating DESC",
        city=city, routing_=RoutingControl.READ,
    )
    rows = [dict(r) for r in recs]
    return {"status": "success", "content": [{"json": {"hotels": rows or "none found"}}]}


@tool
def get_hotel_room_rates(hotel: str) -> dict:
    """Get how much rooms cost at a specific hotel — nightly price/rate in USD per room type.

    Use this for questions about price, cost, how much a room is, or room rates.

    Args:
        hotel: Exact hotel name.
    """
    recs, _, _ = _driver.execute_query(
        "MATCH (h:Hotel {name:$hotel})-[:HAS_ROOM]->(r:Room) "
        "RETURN r.type AS type, r.rate AS rate ORDER BY rate",
        hotel=hotel, routing_=RoutingControl.READ,
    )
    rows = [dict(r) for r in recs]
    return {"status": "success", "content": [{"json": {"rooms": rows or "none found"}}]}


@tool
def get_hotel_amenities(hotel: str) -> dict:
    """Get the amenities a hotel offers (pool, spa, WiFi, ...). No prices.

    Args:
        hotel: Exact hotel name.
    """
    recs, _, _ = _driver.execute_query(
        "MATCH (h:Hotel {name:$hotel})-[:OFFERS_AMENITY]->(a:Amenity) RETURN a.name AS amenity",
        hotel=hotel, routing_=RoutingControl.READ,
    )
    return {"status": "success", "content": [{"json": {"amenities": [r["amenity"] for r in recs] or "none found"}}]}


# ── Real tool 2: weather via Open-Meteo (free, no API key) ────────────────────
def _geocode(city):
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({"name": city, "count": 1})
    res = json.load(urllib.request.urlopen(url, timeout=10)).get("results")
    return (res[0]["latitude"], res[0]["longitude"]) if res else (None, None)


@tool
def get_current_weather(city: str) -> dict:
    """Get CURRENT weather conditions (temperature now) for a city.

    Args:
        city: City name.
    """
    lat, lon = _geocode(city)
    if lat is None:
        return {"status": "error", "content": [{"text": f"Unknown city: {city}"}]}
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
    cur = json.load(urllib.request.urlopen(url, timeout=10))["current"]
    return {"status": "success", "content": [{"json": {"city": city, "temperature_c": cur["temperature_2m"]}}]}


@tool
def get_weather_forecast(city: str) -> dict:
    """Get the multi-day weather FORECAST (upcoming days' highs/lows) for a city.

    Args:
        city: City name.
    """
    lat, lon = _geocode(city)
    if lat is None:
        return {"status": "error", "content": [{"text": f"Unknown city: {city}"}]}
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&daily=temperature_2m_max,temperature_2m_min&forecast_days=5")
    daily = json.load(urllib.request.urlopen(url, timeout=10))["daily"]
    return {"status": "success", "content": [{"json": {
        "city": city, "highs_c": daily["temperature_2m_max"], "lows_c": daily["temperature_2m_min"]}}]}


# ── Real tool 3: bookings via the JSON store ─────────────────────────────────
from booking_store import BookingStore

_store = BookingStore()
book_hotel = _store.book_hotel          # @tool-decorated below via registration
get_booking = _store.get_booking


# ── Extra travel tools (realistic stand-ins) ────────────────────────────────
# A real travel agent has dozens of tools. To reproduce that at scale — and the
# confusion a big, overlapping toolset causes a small model — we add ~33 more
# travel tools here. They return plausible placeholder data because THIS demo is
# about tool SELECTION (which tool the model picks), not each tool's internals.
# Each entry is  name: one-line description  (the description is the tool's
# docstring, which is what the semantic search and the model read).
def _make_tool(name, description):
    @tool(name=name)
    def f(query: str = "") -> dict:
        return {"status": "success", "content": [{"json": {"result": f"({name} result)"}}]}
    f.__doc__ = description + "\n\nArgs:\n    query: the request details."
    return f


EXTRA_TOOL_DESCRIPTIONS = {
    # hotels (overlap with the 3 real Neo4j hotel tools)
    "search_hotels_by_rating": "Search hotels filtered by minimum guest rating.",
    "search_hotels_by_budget": "Search hotels within a nightly budget range.",
    "get_hotel_reviews": "Read guest reviews and written feedback for a hotel.",
    "check_hotel_availability": "Check whether a hotel has rooms free on a date.",
    "compare_hotel_prices": "Compare nightly prices across hotels in a city.",
    "get_hotel_contact": "Get a hotel's phone number and email address.",
    "get_hotel_policies": "Get a hotel's check-in, cancellation and pet policies.",
    "cancel_booking": "Cancel an existing hotel reservation by its booking id.",
    "modify_booking": "Change the dates or guest count of an existing reservation.",
    # flights
    "search_flights": "Search available flights between two cities.",
    "get_flight_prices": "Compare flight ticket prices between cities.",
    "get_flight_status": "Check whether a flight is on time or delayed.",
    "get_flight_details": "Aircraft type, duration and route for a flight.",
    "check_flight_availability": "How many seats remain on a flight.",
    "book_flight": "Book a flight ticket for a passenger.",
    "cancel_flight": "Cancel a booked flight ticket.",
    "get_baggage_allowance": "Baggage allowance and fees for a flight/airline.",
    # ground transport
    "search_car_rentals": "Search rental cars available in a city.",
    "book_car_rental": "Book a rental car for given dates.",
    "get_public_transport": "Public transport options and routes in a city.",
    "estimate_taxi_fare": "Estimate a taxi fare between two points in a city.",
    # money
    "convert_currency": "Convert an amount of money between two currencies.",
    "get_exchange_rate": "Get the current exchange rate between two currencies.",
    "estimate_trip_budget": "Estimate a total trip budget from its components.",
    # docs & info
    "get_visa_requirements": "Visa and passport requirements for a destination.",
    "get_travel_advisory": "Government safety travel advisory for a country.",
    "get_vaccination_requirements": "Required vaccinations to enter a country.",
    "get_timezone": "Current local time and timezone for a city.",
    "get_points_of_interest": "Tourist attractions and sights in a city.",
    "get_restaurant_recommendations": "Restaurant recommendations in a city.",
    "translate_phrase": "Translate a short phrase into another language.",
    "get_emergency_numbers": "Local emergency phone numbers for a country.",
    "get_tipping_customs": "Tipping customs and etiquette for a country.",
}
_extra_tools = [_make_tool(name, desc) for name, desc in EXTRA_TOOL_DESCRIPTIONS.items()]

# The full tool pool: 7 real tools (Neo4j hotels + Open-Meteo weather + bookings)
# plus the ~33 extra travel tools above — a realistic, overlapping ~40-tool set.
ALL_TOOLS = [
    search_hotels_by_city, get_hotel_room_rates, get_hotel_amenities,
    get_current_weather, get_weather_forecast,
    book_hotel, get_booking,
    *_extra_tools,
]


# ── The harness: semantic tool selection as a hook ───────────────────────────
class SemanticToolHook(HookProvider):
    """Trim the agent's tools to the top-k most relevant to each request.

    The agent is created with ALL tools. Before each invocation this hook reads
    the user's message, finds the top-k tools by FAISS similarity over tool
    name+description, and replaces the agent's tool_registry with just those.
    """

    def __init__(self, tools, top_k=3):
        self.tools = tools
        self.top_k = top_k
        self._bedrock = boto3.client("bedrock-runtime", region_name=REGION)
        texts = [f"{t.tool_name}: {t.tool_spec['description']}" for t in tools]
        vecs = np.array([self._embed(t) for t in texts], dtype="float32")
        self._index = faiss.IndexFlatL2(vecs.shape[1])
        self._index.add(vecs)
        self.last_selected = []

    def _embed(self, text):
        r = self._bedrock.invoke_model(
            modelId="amazon.nova-2-multimodal-embeddings-v1:0",
            body=json.dumps({"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX", "embeddingDimension": 1024,
                "text": {"truncationMode": "END", "value": text[:8000]}}}),
            contentType="application/json", accept="application/json")
        return json.loads(r["body"].read())["embeddings"][0]["embedding"]

    def select(self, query):
        _, idx = self._index.search(np.array([self._embed(query)], dtype="float32"), self.top_k)
        return [self.tools[i] for i in idx[0]]

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self._trim)

    def _trim(self, event: BeforeInvocationEvent) -> None:
        msgs = event.messages or getattr(event.agent, "messages", [])
        query = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                query = "".join(b.get("text", "") for b in m.get("content", []))
                break
        if not query:
            return
        selected = self.select(query)
        self.last_selected = [t.tool_name for t in selected]
        reg = event.agent.tool_registry
        reg.registry.clear()
        reg.dynamic_tools.clear()
        for t in selected:
            reg.register_tool(t)
