# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared configuration for the Graph-RAG demo.

Holds the three things `build_graph.py`, `build_graph_lite.py`,
`load_vector_data_lite.py`, `travel_agent_demo.py` and `test_graphrag.ipynb`
must agree on:

1. The **pinned extraction schema**. Without it `SimpleKGPipeline` lets the LLM
   invent a fresh set of labels for every chunk, so one document yields
   `Address` nodes and another yields `RoomType`/`BedConfiguration`. The
   notebook's `query_knowledge_graph` docstring promises a fixed contract to the
   agent, and the graph has to actually honour it.
2. **Neo4j connection settings**. `NEO4J_USERNAME` defaults to `"neo4j"`, which
   is what Aura provisions; the older `NEO4J_USER` spelling is not read. There is
   no default for `NEO4J_PASSWORD` — it is required, and a missing value raises at
   import rather than silently sending a bad credential the way a baked-in default
   password would.
3. The **lite document sample**, stratified by city so the lite path exercises
   the same notebook questions (Paris, Cairo) as the full path.
"""

import os
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Neo4j connection
# ---------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    raise RuntimeError(
        "NEO4J_PASSWORD is not set. Export it (see .env.example) before running "
        "the Graph-RAG demo; there is no default, so a missing password fails "
        "loudly here instead of silently sending a bad credential to Neo4j."
    )


def neo4j_auth() -> tuple[str, str]:
    """Return the (username, password) pair for the Neo4j driver."""
    return NEO4J_USERNAME, NEO4J_PASSWORD


# ---------------------------------------------------------------------------
# Pinned extraction schema
# ---------------------------------------------------------------------------

# Every property name is snake_case, and location lives on `Hotel.address`
# rather than in a separate `Address` node, because that is what the agent is
# told to expect.
GRAPH_SCHEMA: dict[str, object] = {
    "node_types": [
        {
            "label": "Hotel",
            "description": "A hotel property. One per source document.",
            "properties": [
                {"name": "name", "type": "STRING", "description": "Full hotel name."},
                {
                    "name": "address",
                    "type": "STRING",
                    "description": (
                        "Full street address including city and country. "
                        "Never model the address as its own node."
                    ),
                },
                {
                    "name": "guest_rating",
                    "type": "FLOAT",
                    "description": "Guest rating out of 5, e.g. 4.6 from '4.6/5.0'.",
                },
                {"name": "total_rooms", "type": "INTEGER"},
                {"name": "email", "type": "STRING"},
                {"name": "phone", "type": "STRING"},
            ],
        },
        {
            "label": "Room",
            "description": "A room category offered by a hotel.",
            "properties": [
                {
                    "name": "type",
                    "type": "STRING",
                    "description": "Room category, e.g. 'Standard Room', 'Suite'.",
                },
                {
                    "name": "bed_configuration",
                    "type": "STRING",
                    "description": "Bed layout, e.g. 'One king bed'.",
                },
                {"name": "max_occupancy", "type": "INTEGER"},
                {
                    "name": "min_rate",
                    "type": "FLOAT",
                    "description": "Lower bound of the nightly rate range.",
                },
                {
                    "name": "max_rate",
                    "type": "FLOAT",
                    "description": "Upper bound of the nightly rate range.",
                },
            ],
        },
        {
            "label": "Amenity",
            "description": (
                "A facility or feature the hotel offers, e.g. 'Swimming Pool', "
                "'Spa', 'Fitness Center', 'WiFi'. Only create one when the "
                "document says the hotel actually has it."
            ),
            "properties": [
                {"name": "name", "type": "STRING"},
                {"name": "description", "type": "STRING"},
                {"name": "fee", "type": "STRING"},
            ],
        },
        {
            "label": "Policy",
            "description": "A hotel rule, e.g. cancellation or pet policy.",
            "properties": [
                {"name": "name", "type": "STRING"},
                {"name": "description", "type": "STRING"},
            ],
        },
        {
            "label": "Service",
            "description": "A service the hotel provides, e.g. airport shuttle.",
            "properties": [
                {"name": "name", "type": "STRING"},
                {"name": "description", "type": "STRING"},
                {"name": "cost", "type": "STRING"},
                {"name": "hours", "type": "STRING"},
                {"name": "is_available", "type": "BOOLEAN"},
                {"name": "is_complimentary", "type": "BOOLEAN"},
            ],
        },
    ],
    "relationship_types": [
        {"label": "HAS_ROOM"},
        {"label": "OFFERS_AMENITY"},
        {"label": "HAS_POLICY"},
        {"label": "PROVIDES_SERVICE"},
    ],
    "patterns": [
        ("Hotel", "HAS_ROOM", "Room"),
        ("Hotel", "OFFERS_AMENITY", "Amenity"),
        ("Hotel", "HAS_POLICY", "Policy"),
        ("Hotel", "PROVIDES_SERVICE", "Service"),
    ],
    # The whole point of pinning: refuse anything outside the contract.
    "additional_node_types": False,
    "additional_relationship_types": False,
    "additional_patterns": False,
}

SCHEMA_NODE_LABELS = ("Hotel", "Room", "Amenity", "Policy", "Service")

# Labels produced by earlier unpinned runs. Their presence after a build means
# the schema did not hold.
OFF_SCHEMA_LABELS = (
    "Address",
    "RoomType",
    "BedConfiguration",
    "Fee",
    "PaymentMethod",
    "ContactMethod",
    "ContactInfo",
    "Location",
    "City",
    "Country",
    "Attraction",
)

# Source documents top out at ~7.4 KB. A chunk this size keeps each hotel in a
# single chunk, so the hotel's name, address and rating are extracted together
# with its rooms and amenities instead of being split across two prompts.
CHUNK_SIZE = 12000
CHUNK_OVERLAP = 0

# A whole hotel in one chunk produces a large extraction payload. The 4096
# default truncates the JSON mid-object and the chunk is dropped.
EXTRACTION_MAX_TOKENS = 16000


# ---------------------------------------------------------------------------
# Document selection
# ---------------------------------------------------------------------------

# The notebook asks about Paris and Cairo by name, so the lite sample has to
# contain them. `sorted(...)[:30]` is alphabetical and stops at Boston.
REQUIRED_CITIES = ("paris", "cairo")


def _city_of(filename: str) -> str:
    """Extract the city from a `hotel-<city>-<nnn>.txt` filename."""
    parts = Path(filename).stem.split("-")
    return "-".join(parts[1:-1]) if len(parts) > 2 else Path(filename).stem


def select_lite_files(data_dir: str | Path, max_docs: int) -> list[str]:
    """Return a city-stratified sample of `max_docs` FAQ filenames.

    Every document for each city in `REQUIRED_CITIES` comes first, so the
    Paris average and the Cairo multi-hop query have more than one hotel to
    work with. The remainder is filled round-robin across the other cities
    rather than alphabetically, so the sample spans the corpus.
    """
    by_city: dict[str, list[str]] = defaultdict(list)
    for path in sorted(Path(data_dir).glob("*.txt")):
        by_city[_city_of(path.name)].append(path.name)

    picked = [name for city in REQUIRED_CITIES for name in by_city.get(city, [])]
    others = [city for city in sorted(by_city) if city not in REQUIRED_CITIES]

    depth = max((len(names) for names in by_city.values()), default=0)
    for i in range(depth):
        for city in others:
            if len(picked) >= max_docs:
                return picked[:max_docs]
            names = by_city[city]
            if i < len(names):
                picked.append(names[i])

    return picked[:max_docs]
