# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Travel Agent Demo: Traditional RAG vs Graph-RAG Comparison
"""
import os
os.environ['OTEL_SDK_DISABLED'] = 'true'

from dotenv import load_dotenv
load_dotenv()

from strands import Agent, tool
from neo4j import GraphDatabase
import faiss
import json
import boto3
import numpy as np

from graph_config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# Amazon Bedrock Nova 2 for embeddings
_bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

def _embed(text):
    """Embed text using Amazon Bedrock Nova 2 Multimodal Embeddings."""
    resp = _bedrock.invoke_model(
        modelId="amazon.nova-2-multimodal-embeddings-v1:0",
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 1024,
                "text": {"truncationMode": "END", "value": text[:8000]},
            },
        }),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    return np.array([result["embeddings"][0]["embedding"]], dtype="float32")

# Load vector store for traditional RAG
index = faiss.read_index("faqs_vector.index")
with open("faqs_docs.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

@tool
def search_faqs(query: str) -> str:
    """Search hotel FAQs using vector similarity (Traditional RAG)."""
    query_embedding = _embed(query)
    distances, indices = index.search(query_embedding, 3)
    
    results = []
    for idx in indices[0]:
        doc = documents[idx]
        results.append(f"[{doc['filename']}]\n{doc['text'][:500]}...")
    
    return "\n\n".join(results)

@tool
def query_knowledge_graph(cypher_query: str) -> str:
    """Execute a Cypher query against the hotel knowledge graph.
    
    Cypher is Neo4j's query language for graph databases. It uses pattern matching
    to query relationships between entities. Think of it like SQL for graphs.
    
    Example: MATCH (h:Hotel)-[:HAS_ROOM]->(r:Room) WHERE h.name = 'Marriott' RETURN r.max_rate

    Node labels: Hotel, Room, Amenity, Policy, Service
    Hotel properties: name, address, guest_rating, total_rooms, email, phone
    Room properties: type, bed_configuration, max_occupancy, min_rate, max_rate
    Amenity properties: name, description, fee
    Policy properties: name, description
    Service properties: name, description, cost, hours, is_available, is_complimentary

    Relationships:
    - (Hotel)-[:HAS_ROOM]->(Room)
    - (Hotel)-[:OFFERS_AMENITY]->(Amenity)
    - (Hotel)-[:HAS_POLICY]->(Policy)
    - (Hotel)-[:PROVIDES_SERVICE]->(Service)

    Location is in Hotel.address property (e.g. "789 Corniche el-Nil, Cairo 11519").
    To find hotels by location, use: WHERE h.address CONTAINS 'Cairo'
    IMPORTANT: All property names use snake_case (e.g. guest_rating NOT guestRating)
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    with driver.session() as session:
        try:
            result = session.run(cypher_query)
            records = list(result)
            
            if not records:
                return "No results found."
            
            output = f"Found {len(records)} results:\n"
            for record in records[:15]:
                row = {k: v for k, v in record.items()}
                output += f"  {row}\n"
            
            return output
        except Exception as e:
            return f"Query error: {str(e)}"
        finally:
            driver.close()

# Model configuration — Amazon Bedrock (default, requires AWS credentials)
# Strands Agents uses Bedrock by default. No extra import needed.
#
# To use a different provider (e.g., OpenAI), install the extra and configure:
#   pip install "strands-agents[openai]"
#   from strands.models.openai import OpenAIModel
#   MODEL = OpenAIModel(model_id="gpt-4o-mini")
#   (requires OPENAI_API_KEY env var — get one at https://platform.openai.com/api-keys)
#
# See all providers: https://strandsagents.com/docs/user-guide/concepts/model-providers/

def graph_stats():
    """Measure the corpus from the graph itself.

    Every count this demo prints is read back from Neo4j at runtime. Hardcoding
    them ("300 hotels", "175 with a pool") makes the summary drift silently the
    moment the graph is rebuilt from a different sample — and the whole claim of
    the demo is that Graph-RAG reports what is actually there.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            return session.run(
                """
                MATCH (h:Hotel)
                WITH count(h) AS hotels
                MATCH (p:Hotel)-[:OFFERS_AMENITY]->(a:Amenity)
                WHERE toLower(a.name) CONTAINS 'pool'
                WITH hotels, count(DISTINCT p) AS with_pool
                MATCH (paris:Hotel) WHERE paris.address CONTAINS 'Paris'
                RETURN hotels, with_pool,
                       count(paris) AS paris_hotels,
                       round(avg(paris.guest_rating), 2) AS paris_avg
                """
            ).single()
    finally:
        driver.close()


STATS = graph_stats()
HOTELS = STATS["hotels"]
WITH_POOL = STATS["with_pool"]
POOL_PCT = round(100 * WITH_POOL / HOTELS) if HOTELS else 0

# Traditional RAG Agent
rag_agent = Agent(
    name="RAG_Agent",
    system_prompt="You are a travel agent. Use vector search to find relevant FAQ information.",
    tools=[search_faqs],
    # model=MODEL  # Uncomment if using a custom model provider
)

# Graph-RAG Agent
graph_agent = Agent(
    name="GraphRAG_Agent",
    system_prompt="You are a travel agent. Use the knowledge base to answer questions accurately. You can run multiple queries to explore the data.",
    tools=[query_knowledge_graph],
    # model=MODEL  # Uncomment if using a custom model provider
)

print("="*70)
print("TRAVEL AGENT COMPARISON: Traditional RAG vs Graph-RAG")
print("="*70)

queries = [
    {
        "query": "What is the average guest rating of all hotels in Paris?",
        "insight": "RAG guesses from top 3 docs | Graph-RAG calculates exact AVG() across all Paris hotels"
    },
    {
        "query": "How many hotels in the database have a swimming pool?",
        "insight": f"RAG only sees top 3 docs, cannot count | Graph-RAG executes COUNT() across all {HOTELS} hotels in the graph"
    },
    {
        "query": "Which hotels in Cairo have both a spa and a swimming pool, and what are their guest ratings?",
        "insight": "RAG finds partial matches, cannot filter by multiple criteria | Graph-RAG traverses Hotel→Amenity→Amenity with AND logic"
    },
    {
        "query": "Tell me about hotels in Antarctica",
        "insight": "RAG may hallucinate plausible info | Graph-RAG returns 'No hotels found' (honest failure)"
    },
]

for test in queries:
    query = test["query"]
    print(f"\n{'='*70}")
    print(f"👤 Query: {query}")
    print("="*70)

    # Traditional RAG
    print("\n[TRADITIONAL RAG]")
    print("-" * 70)
    response = rag_agent(query)
    print(response.message['content'][0]['text'])

    # Graph-RAG
    print("\n[GRAPH-RAG]")
    print("-" * 70)
    response = graph_agent(query)
    print(response.message['content'][0]['text'])

    print(f"\n📊 {test['insight']}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Data Used (measured from the graph at runtime, not hardcoded):
  - {HOTELS} hotels in the knowledge graph
  - Paris: {STATS['paris_hotels']} hotels, average guest rating {STATS['paris_avg']}★
  - Swimming pools: {WITH_POOL} out of {HOTELS} hotels (~{POOL_PCT}%)
  - Knowledge graph: pinned schema (Hotel, Room, Amenity, Policy, Service)

Why Graph-RAG Reduces Hallucinations:
  1. Structured queries: Cypher forces precise logic (AVG, COUNT, WHERE...AND)
  2. Complete dataset access: Not limited to top-k vector matches
  3. Relationship awareness: (Hotel)-[:OFFERS_AMENITY]->(Amenity) explicit
  4. Honest failure: Empty result = "No hotels found", not fabricated data

Result: Graph-RAG eliminates hallucinations with verified structured data
""")
