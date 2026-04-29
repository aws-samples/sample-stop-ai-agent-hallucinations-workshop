# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Neo4j query tools for the hotel knowledge graph.

Used by:
- 01-graphrag-demo: direct Graph-RAG comparison
- 02-semantic-tools-demo: real hotel data for semantic filtering accuracy
"""

import os
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def search_hotels_by_country(country: str, min_rating: float = 0.0) -> str:
    """Search hotels in a specific country with minimum rating from Neo4j."""
    driver = _get_driver()
    query = """
    MATCH (h)
    WHERE any(l IN labels(h) WHERE l CONTAINS 'Hotel' OR l = 'Hotel')
    AND (h.address CONTAINS $country OR h.name CONTAINS $country)
    AND coalesce(h.guestRating, 0) >= $min_rating
    RETURN h.name AS name, h.address AS address, h.guestRating AS rating, h.totalRooms AS rooms
    ORDER BY h.guestRating DESC
    LIMIT 10
    """
    with driver.session() as session:
        result = session.run(query, country=country, min_rating=min_rating)
        records = list(result)
    driver.close()

    if not records:
        return f"No hotels found in {country} with rating >= {min_rating}"

    output = f"Found {len(records)} hotels in {country}:\n"
    for r in records:
        output += f"- {r['name']} ({r['address']}): {r['rating']}/5.0, {r['rooms']} rooms\n"
    return output


def get_top_rated_hotels(limit: int = 5) -> str:
    """Get top-rated hotels from Neo4j knowledge graph."""
    driver = _get_driver()
    query = """
    MATCH (h)
    WHERE any(l IN labels(h) WHERE l CONTAINS 'Hotel' OR l = 'Hotel')
    AND h.guestRating IS NOT NULL
    RETURN h.name AS name, h.address AS address, h.guestRating AS rating, h.totalRooms AS rooms
    ORDER BY h.guestRating DESC
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(query, limit=limit)
        records = list(result)
    driver.close()

    if not records:
        return "No hotels found in database"

    output = f"Top {len(records)} rated hotels:\n"
    for i, r in enumerate(records, 1):
        output += f"{i}. {r['name']} ({r['address']}): {r['rating']}/5.0\n"
    return output
