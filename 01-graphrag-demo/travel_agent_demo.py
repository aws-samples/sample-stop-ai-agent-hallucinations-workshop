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

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

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
    
    Example: MATCH (h:Hotel)-[:HAS_ROOM]->(r:Room) WHERE h.name = 'Marriott' RETURN r.price
    
    Node labels: Hotel, Room, Amenity, Policy, Service
    
    Hotel properties: name, address, guestRating, totalRooms, email, phone
    Room properties: name (e.g. "Standard Room"), price, maxOccupancy
    Amenity properties: name (e.g. "Outdoor Swimming Pool", "WiFi")
    Policy properties: name (e.g. "Check-in Policy"), details
    
    Relationships:
    - (Hotel)-[:HAS_ROOM]->(Room)
    - (Hotel)-[:OFFERS_AMENITY]->(Amenity)
    - (Hotel)-[:HAS_POLICY]->(Policy)
    - (Hotel)-[:PROVIDES_SERVICE]->(Service)
    
    Location is in Hotel.address property (e.g. "789 Corniche el-Nil, Cairo 11519").
    To find hotels by location, use: WHERE h.address CONTAINS 'Cairo'
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
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
    # Test 1: Aggregation - RAG cannot compute, Graph-RAG can
    "What is the average guest rating across all hotels in Paris?",
    # Test 2: Precise counting - RAG guesses, Graph-RAG counts
    "How many hotels have a swimming pool as an amenity?",
    # Test 3: Multi-hop reasoning - RAG mixes data, Graph-RAG traverses
    "What are the room types and prices for the highest rated hotel?",
    # Test 4: Out-of-domain - RAG may hallucinate, Graph-RAG says no data
    "Tell me about hotels in Antarctica",
]

for query in queries:
    print(f"\n{'='*70}")
    print(f"👤 Query: {query}")
    print("="*70)
    
    # Traditional RAG
    print("\n[TRADITIONAL RAG - Vector Search]")
    print("-" * 70)
    response = rag_agent(query)
    print(response.message['content'][0]['text'][:300] + "...")
    
    # Graph-RAG
    print("\n[GRAPH-RAG - Knowledge Graph]")
    print("-" * 70)
    response = graph_agent(query)
    print(response.message['content'][0]['text'][:300] + "...")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
Traditional RAG: Semantic similarity, may miss context or hallucinate
Graph-RAG: Structured queries on extracted entities, precise answers
Result: Graph-RAG eliminates hallucinations with verified data
""")
