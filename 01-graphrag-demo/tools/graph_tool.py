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


def query_hotel_knowledge_graph(cypher_query: str) -> str:
    """Execute a Cypher query against the hotel knowledge graph.

    Node labels: Hotel, Room, Amenity, Policy, Service
    Hotel properties: name, address, guest_rating, total_rooms, email, phone
    Room properties: type, bed_configuration, max_occupancy, min_rate, max_rate
    Amenity properties: name, description, fee
    Policy properties: name, description
    Service properties: name, description, cost, hours, is_available, is_complimentary

    Relationships: (Hotel)-[:HAS_ROOM]->(Room), (Hotel)-[:OFFERS_AMENITY]->(Amenity),
                   (Hotel)-[:HAS_POLICY]->(Policy), (Hotel)-[:PROVIDES_SERVICE]->(Service)

    Location is in Hotel.address property. Use: WHERE h.address CONTAINS 'Cairo'
    IMPORTANT: All property names use snake_case (e.g., guest_rating NOT guestRating)
    """
    driver = _get_driver()
    with driver.session() as session:
        try:
            result = session.run(cypher_query)
            records = list(result)
            if not records:
                return "No results found."
            output = f"Found {len(records)} results:\n"
            for record in records[:15]:
                output += f"  {dict(record.items())}\n"
            return output
        except Exception as e:
            return f"Query error: {str(e)}"
        finally:
            driver.close()


# Keep old functions for backward compatibility but marked as deprecated
def search_hotels_by_country(country: str, min_rating: float = 0.0) -> str:
    """[DEPRECATED] Use query_hotel_knowledge_graph instead.

    Search hotels in a specific country with minimum rating from Neo4j."""
    query = f"""
    MATCH (h:Hotel)
    WHERE h.address CONTAINS '{country}' OR h.name CONTAINS '{country}'
    AND coalesce(h.guest_rating, 0) >= {min_rating}
    RETURN h.name AS name, h.address AS address, h.guest_rating AS rating, h.total_rooms AS rooms
    ORDER BY h.guest_rating DESC
    LIMIT 10
    """
    return query_hotel_knowledge_graph(query)


def get_top_rated_hotels(limit: int = 5) -> str:
    """[DEPRECATED] Use query_hotel_knowledge_graph instead.

    Get top-rated hotels from Neo4j knowledge graph."""
    query = f"""
    MATCH (h:Hotel)
    WHERE h.guest_rating IS NOT NULL
    RETURN h.name AS name, h.address AS address, h.guest_rating AS rating, h.total_rooms AS rooms
    ORDER BY h.guest_rating DESC
    LIMIT {limit}
    """
    return query_hotel_knowledge_graph(query)
