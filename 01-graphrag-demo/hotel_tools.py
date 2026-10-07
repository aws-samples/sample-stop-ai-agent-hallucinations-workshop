# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Hotel tools for demo 01 (used by chat.py).

These are the same tools defined and explained inline in the notebook; they are
copied here so `chat.py` can import them. Demo 01 compares two retrieval styles:

  * VectorFAQs.search_faqs — standard RAG via FAISS vector similarity.
  * HotelGraph.query_hotels — Graph-RAG via read-only Neo4j Cypher.

Neo4j is the READ-ONLY source of truth for hotel facts — never used to book.
Queries run in READ routing mode, so the graph tool physically cannot write.
The docstrings are the contract the LLM reads; the graph schema they describe is
the one build_graph.py produces with its fixed schema.
"""
import os

from neo4j import GraphDatabase, RoutingControl
from strands import tool


class VectorFAQs:
    """Standard RAG: FAISS vector similarity over the hotel FAQ documents.

    Loads the index once (built by load_vector_data.py). Returns the top-k most
    similar FAQ chunks — which can look relevant even when the exact answer is
    not present, the failure mode demo 01 contrasts against Graph-RAG.
    """

    def __init__(self, index_path="faqs_vector.index", docs_path="faqs_docs.json",
                 region=None):
        import faiss
        import json
        self._faiss = faiss
        self._index = faiss.read_index(index_path)
        with open(docs_path, encoding="utf-8") as f:
            self._docs = json.load(f)
        import boto3
        self._bedrock = boto3.client(
            "bedrock-runtime", region_name=region or os.getenv("AWS_REGION", "us-east-1")
        )

    def _embed(self, text):
        import json
        import numpy as np
        resp = self._bedrock.invoke_model(
            modelId="amazon.nova-2-multimodal-embeddings-v1:0",
            body=json.dumps({"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX", "embeddingDimension": 1024,
                "text": {"truncationMode": "END", "value": text[:8000]}}}),
            contentType="application/json", accept="application/json",
        )
        return np.array([json.loads(resp["body"].read())["embeddings"][0]["embedding"]],
                        dtype="float32")

    @tool
    def search_faqs(self, query: str) -> dict:
        """Search hotel FAQs by vector similarity (standard RAG).

        Returns the 3 most similar FAQ passages. Note: similarity always returns
        the closest passages even if they don't actually contain the answer, so
        this is best for open-ended Q&A, not exact counts or aggregations.

        Args:
            query: The user's natural-language question.
        """
        _, idx = self._index.search(self._embed(query), 3)
        hits = [{"source": self._docs[i]["filename"],
                 "excerpt": self._docs[i]["text"][:500]} for i in idx[0]]
        return {"status": "success", "content": [{"json": {"results": hits}}]}


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

    @tool
    def query_hotels(self, cypher_query: str) -> dict:
        """Run a READ-ONLY Cypher query against the hotel knowledge graph.

        Use this to look up hotels and compute exact answers (counts, averages,
        filters) instead of guessing. The graph is the source of truth; if a
        query returns nothing, the honest answer is "no results" — do not invent.

        Schema (all property names are camelCase):
          (Hotel {name, address, city, country, guestRating, totalRooms, phone, email})
          (Room {type, rate, maxOccupancy})          rate = nightly price in USD (numeric)
          (Amenity {name, description})               e.g. name = "Outdoor Swimming Pool"
          (Policy {name, description})
          (Service {name, description})

        Relationships:
          (Hotel)-[:HAS_ROOM]->(Room)
          (Hotel)-[:OFFERS_AMENITY]->(Amenity)
          (Hotel)-[:HAS_POLICY]->(Policy)
          (Hotel)-[:PROVIDES_SERVICE]->(Service)

        Location lives in Hotel.city / Hotel.country (filter with
        `WHERE h.city = 'Paris'`). Match amenities case-insensitively, e.g.
        `WHERE toLower(a.name) CONTAINS 'pool'`.

        Args:
            cypher_query: A single read-only Cypher statement.

        Returns:
            {"status": "success", "content": [...]} with up to 25 rows, or
            {"status": "error", "content": [{"text": "..."}]} on failure.
        """
        try:
            records, _, _ = self._driver.execute_query(
                cypher_query, routing_=RoutingControl.READ
            )
            rows = [dict(r) for r in records[:25]]
            if not rows:
                return {"status": "success", "content": [{"text": "No results found."}]}
            return {"status": "success", "content": [{"json": {"results": rows}}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Query error: {e}"}]}
