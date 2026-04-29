# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Build LITE knowledge graph from hotel FAQ documents using neo4j-graphrag.

LITE VERSION: Processes only 30 documents (10% of full dataset) for faster testing.
Full version takes ~2 hours, lite version takes ~10-15 minutes.
"""
import os
import asyncio
os.environ['OTEL_SDK_DISABLED'] = 'true'

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from bedrock_providers import BedrockLLM, BedrockEmbeddings

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# LITE: Process only first 30 documents
MAX_DOCS = 30


async def build_graph():
    # Clear existing graph
    print("Clearing existing graph...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    print("✅ Graph cleared\n")

    # LLM and embedder — Amazon Bedrock (no OpenAI API key needed)
    llm = BedrockLLM(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        temperature=0,
    )
    embedder = BedrockEmbeddings(
        model_id="amazon.nova-embed-text-v1:0",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

    # No hardcoded schema - the LLM automatically discovers entities from the text,
    # eliminating the need to manually define entity types in advance
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)),
        embedder=embedder,
        from_pdf=False,
        perform_entity_resolution=True,
    )

    # Load LITE subset of FAQ documents
    data_dir = "data"
    files = sorted(os.listdir(data_dir))[:MAX_DOCS]  # Only first 30
    total = len(files)
    print(f"🚀 LITE MODE: Processing {total} documents (10% of full dataset)...\n")

    errors = 0
    for i, filename in enumerate(files, 1):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        try:
            print(f"[{i}/{total}] Processing {filename}...")
            await kg_builder.run_async(text=text)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors += 1
    
    await kg_builder.close()

    print(f"\n{'='*60}")
    print(f"LITE GRAPH BUILD COMPLETE ({total - errors}/{total} docs processed)")
    print(f"{'='*60}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("""
            MATCH (n) 
            WHERE NOT 'Chunk' IN labels(n) AND NOT 'Document' IN labels(n)
            RETURN DISTINCT [l IN labels(n) WHERE l <> '__Entity__'][0] as label, count(*) as count 
            ORDER BY count DESC
        """)
        print("\nEntity types (auto-discovered):")
        for r in result:
            print(f"  :{r['label']}: {r['count']}")

        result = session.run("""
            MATCH ()-[r]->() 
            WHERE NOT type(r) IN ['PART_OF_DOCUMENT', 'NEXT_CHUNK', 'PART_OF_CHUNK', 'FROM_DOCUMENT', 'FROM_CHUNK']
            RETURN DISTINCT type(r) as rel, count(*) as count 
            ORDER BY count DESC
        """)
        print("\nRelationship types (auto-discovered):")
        for r in result:
            print(f"  :{r['rel']}: {r['count']}")

        # Test queries
        print("\n--- Test: Hotels in Egypt ---")
        result = session.run("""
            MATCH (h)-[*1..3]->(co)
            WHERE any(l IN labels(h) WHERE l CONTAINS 'Hotel' OR l = 'Hotel')
              AND any(l IN labels(co) WHERE l CONTAINS 'Country' OR l = 'Country')
              AND co.id =~ '(?i).*egypt.*'
            RETURN h.id as hotel, co.id as country
            LIMIT 5
        """)
        for r in result:
            print(f"  {r['hotel']} -> {r['country']}")

    driver.close()
    print("\n✅ LITE graph ready for queries!")


if __name__ == "__main__":
    asyncio.run(build_graph())
