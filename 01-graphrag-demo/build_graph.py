# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Build knowledge graph from hotel FAQ documents using neo4j-graphrag.

Uses LLM to AUTOMATICALLY extract entities and relationships.
No hardcoded schema - the LLM discovers entities from the text.
Same 300 documents as the FAISS vector store.
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
        model_id="amazon.nova-2-multimodal-embeddings-v1:0",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

    # No hardcoded schema - LLM discovers entities automatically
    # schema="EXTRACTED" (default): LLM analyzes text, generates schema, then extracts
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)),
        embedder=embedder,
        from_pdf=False,
        perform_entity_resolution=True,
    )

    # Load all 300 FAQ documents (same as FAISS)
    data_dir = "data"
    files = sorted(os.listdir(data_dir))
    total = len(files)
    print(f"Processing {total} documents...\n")

    errors = 0
    for i, filename in enumerate(files, 1):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"  [{i}/{total}] {filename}...", end=" ", flush=True)
        try:
            await asyncio.wait_for(kg_builder.run_async(text=text), timeout=90)
            print("✅")
        except asyncio.TimeoutError:
            errors += 1
            print("⏰ timeout")
        except Exception as e:
            errors += 1
            print(f"❌ {str(e)[:60]}")

    # Summary
    print(f"\n{'='*60}")
    print(f"GRAPH BUILD COMPLETE ({total - errors}/{total} docs processed)")
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
            AND co.name CONTAINS 'Egypt'
            RETURN h.name, co.name
            LIMIT 5
        """)
        for r in result:
            print(f"  {r['h.name']} -> {r['co.name']}")

        print("\n--- Test: Hotels in Paris ---")
        result = session.run("""
            MATCH (h)-[*1..2]->(c)
            WHERE any(l IN labels(h) WHERE l CONTAINS 'Hotel' OR l = 'Hotel')
            AND c.name CONTAINS 'Paris'
            RETURN h.name, c.name
            LIMIT 5
        """)
        for r in result:
            print(f"  {r['h.name']} -> {r['c.name']}")

    driver.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(build_graph())
