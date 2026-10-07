# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Build the hotel knowledge graph with a FIXED schema.

Run this ONCE to create the local Neo4j graph. You only need it OUTSIDE a
Workshop Studio event — inside the WS the graph is already restored from a dump:

    AWS_PROFILE=<your-profile> AWS_REGION=us-east-1 python build_graph.py

Default is LITE (30 docs, ~10-15 min) for fast iteration. For the full 300-doc
graph set FULL=True below (or env BUILD_FULL=1); the full build takes ~2 hours.

Why a FIXED schema? `SimpleKGPipeline` can let the LLM invent the schema per
document (non-deterministic → junk labels, duplicate relationships, inconsistent
property types). Passing an explicit `schema=` constrains extraction so the graph
is deterministic and "add a hotel" is a safe, repeatable operation — which is the
whole point of demo 01.

Everything this build needs lives in THIS file: the schema, the Amazon Bedrock
wrappers that neo4j-graphrag requires, and the build loop. Full 300-doc build:
set MAX_DOCS = None (see build_graph.py).

"""
import asyncio
import json
import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import boto3
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.base import LLMInterface, LLMResponse

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DATA_DIR = "data"
FULL = os.getenv("BUILD_FULL") == "1"   # lite (30 docs) by default; set True / BUILD_FULL=1 for all 300
MAX_DOCS = None if FULL else 30

# ── Fixed schema (A2): the LLM may only extract THESE labels, props and rels ──
SCHEMA = {
    "node_types": [
        {"label": "Hotel", "properties": [
            {"name": "name", "type": "STRING"}, {"name": "address", "type": "STRING"},
            {"name": "city", "type": "STRING"}, {"name": "country", "type": "STRING"},
            {"name": "guestRating", "type": "FLOAT"}, {"name": "totalRooms", "type": "INTEGER"},
            {"name": "phone", "type": "STRING"}, {"name": "email", "type": "STRING"}]},
        {"label": "Room", "properties": [
            {"name": "type", "type": "STRING"}, {"name": "rate", "type": "FLOAT"},
            {"name": "maxOccupancy", "type": "INTEGER"}]},
        {"label": "Amenity", "properties": [
            {"name": "name", "type": "STRING"}, {"name": "description", "type": "STRING"}]},
        {"label": "Policy", "properties": [
            {"name": "name", "type": "STRING"}, {"name": "description", "type": "STRING"}]},
        {"label": "Service", "properties": [
            {"name": "name", "type": "STRING"}, {"name": "description", "type": "STRING"}]},
    ],
    "relationship_types": ["HAS_ROOM", "OFFERS_AMENITY", "HAS_POLICY", "PROVIDES_SERVICE"],
    "patterns": [
        ("Hotel", "HAS_ROOM", "Room"), ("Hotel", "OFFERS_AMENITY", "Amenity"),
        ("Hotel", "HAS_POLICY", "Policy"), ("Hotel", "PROVIDES_SERVICE", "Service"),
    ],
}


# ── Amazon Bedrock wrappers that neo4j-graphrag's pipeline requires ───────────
class BedrockLLM(LLMInterface):
    """Claude Sonnet 4 via the Bedrock Converse API (entity extraction)."""

    def __init__(self, model_id="global.anthropic.claude-sonnet-4-6", region=AWS_REGION):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def _converse(self, text, system_instruction=None):
        kwargs = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": text}]}],
            "inferenceConfig": {"temperature": 0, "maxTokens": 4096},
        }
        if system_instruction:
            kwargs["system"] = [{"text": system_instruction}]
        resp = self.client.converse(**kwargs)
        return LLMResponse(content=resp["output"]["message"]["content"][0]["text"])

    def invoke(self, input, message_history=None, system_instruction=None):
        return self._converse(input, system_instruction)

    async def ainvoke(self, input, message_history=None, system_instruction=None):
        return await asyncio.to_thread(self._converse, input, system_instruction)


class BedrockEmbeddings(Embedder):
    """Amazon Nova 2 Multimodal Embeddings (1024-dim) for chunk vectors."""

    def __init__(self, model_id="amazon.nova-2-multimodal-embeddings-v1:0", region=AWS_REGION):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def embed_query(self, text):
        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX", "embeddingDimension": 1024,
                "text": {"truncationMode": "END", "value": text[:8000]}}}),
            contentType="application/json", accept="application/json",
        )
        return json.loads(resp["body"].read())["embeddings"][0]["embedding"]


async def build():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("Clearing existing graph...")
    driver.execute_query("MATCH (n) DETACH DELETE n")

    pipeline = SimpleKGPipeline(
        llm=BedrockLLM(), driver=driver, embedder=BedrockEmbeddings(),
        schema=SCHEMA, from_pdf=False, perform_entity_resolution=True,
    )

    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".txt"))
    if MAX_DOCS:
        files = files[:MAX_DOCS]
    print(f"Building graph from {len(files)} documents with a FIXED schema...\n")

    errors = 0
    for i, filename in enumerate(files, 1):
        with open(os.path.join(DATA_DIR, filename), encoding="utf-8") as f:
            text = f.read()
        try:
            print(f"[{i}/{len(files)}] {filename}...", end=" ", flush=True)
            await asyncio.wait_for(pipeline.run_async(text=text), timeout=120)
            print("OK")
        except Exception as e:
            errors += 1
            print(f"ERROR: {str(e)[:70]}")

    _verify(driver, len(files) - errors, len(files))
    driver.close()


def _verify(driver, built, total):
    """Confirm the graph matches the fixed schema — no junk, no duplicate rels."""
    print(f"\n{'='*60}\nBUILD COMPLETE ({built}/{total} docs)\n{'='*60}")
    allowed_labels = {"Hotel", "Room", "Amenity", "Policy", "Service", "Chunk", "Document"}
    allowed_rels = {"HAS_ROOM", "OFFERS_AMENITY", "HAS_POLICY", "PROVIDES_SERVICE",
                    "FROM_CHUNK", "FROM_DOCUMENT", "NEXT_CHUNK"}

    labels, _, _ = driver.execute_query(
        "MATCH (n) UNWIND labels(n) AS l WITH l WHERE NOT l STARTS WITH '__' "
        "RETURN l AS label, count(*) AS c ORDER BY c DESC")
    junk_labels = [r["label"] for r in labels if r["label"] not in allowed_labels]
    print("\nLabels:")
    for r in labels:
        print(f"  {r['label']}: {r['c']}" + ("  <-- UNEXPECTED" if r["label"] in junk_labels else ""))

    rels, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC")
    junk_rels = [r["t"] for r in rels if r["t"] not in allowed_rels]
    print("\nRelationships:")
    for r in rels:
        print(f"  {r['t']}: {r['c']}" + ("  <-- UNEXPECTED" if r["t"] in junk_rels else ""))

    bad, _, _ = driver.execute_query(
        "MATCH (r:Room) WHERE r.rate IS NULL OR r.priceRange IS NOT NULL RETURN count(r) AS c")
    bad_rooms = bad[0]["c"]

    print(f"\n{'='*60}")
    if junk_labels or junk_rels or bad_rooms:
        print(f"⚠️  Schema NOT clean: junk_labels={junk_labels} junk_rels={junk_rels} bad_rooms={bad_rooms}")
    else:
        print("✅ Schema is clean: only expected labels/rels, every Room has a numeric rate.")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(build())
