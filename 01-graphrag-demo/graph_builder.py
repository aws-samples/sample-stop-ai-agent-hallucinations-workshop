# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared knowledge-graph build machinery for the Graph-RAG demo.

`build_graph.py` and `build_graph_lite.py` differ only in which documents they
feed in. Everything else — the pinned schema, the canary check, the scoped
wipe, the verification queries — lives here so the two paths cannot drift apart
the way they previously did (one verified on `h.id`, the other on `h.name`,
and neither matched the notebook).

Build order matters and is deliberate:

    wipe -> canary (3 docs) -> verify typing -> wipe canary -> ingest all -> report

The wipe happens *before* the canary. The graph this script builds is
disposable — it is rebuilt from scratch on every run — so there is nothing
worth preserving across a failed build. Wiping first also means the canary
runs against an empty graph, so entity resolution has nothing to merge into
and the check reflects exactly what this run extracted.
"""

import asyncio
import os
from pathlib import Path

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from bedrock_providers import BedrockEmbeddings, BedrockLLM
from graph_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EXTRACTION_MAX_TOKENS,
    GRAPH_SCHEMA,
    NEO4J_URI,
    OFF_SCHEMA_LABELS,
    SCHEMA_NODE_LABELS,
    neo4j_auth,
)

# Must sit above the worst case of the Bedrock retry chain (see F16 /
# bedrock_providers.BEDROCK_CONFIG): 3 attempts x 45s + backoff = ~135s < 180s.
DOC_TIMEOUT_SECONDS = 180

# The canary samples several documents rather than one. LLM extraction is
# stochastic, so a single-document gate intermittently fails a healthy
# pipeline, and an attendee who hits that concludes the demo is broken.
CANARY_DOCS = 3

# Everything neo4j-graphrag writes carries this label, so the wipe can be
# scoped to this demo's own output instead of `MATCH (n) DETACH DELETE n`,
# which would also take out anything else sharing the instance.
KG_LABEL = "__KGBuilder__"


def connect() -> Driver:
    """Open a Neo4j driver using NEO4J_USERNAME / NEO4J_PASSWORD."""
    return GraphDatabase.driver(NEO4J_URI, auth=neo4j_auth())


def build_pipeline(driver: Driver) -> SimpleKGPipeline:
    """Construct the extraction pipeline with the schema pinned.

    Without `schema=`, `SimpleKGPipeline` asks the LLM to invent a schema per
    chunk, and the labels it invents do not match what the agent is told to
    query.
    """
    llm = BedrockLLM(
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        max_tokens=EXTRACTION_MAX_TOKENS,
    )
    embedder = BedrockEmbeddings(
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=GRAPH_SCHEMA,
        text_splitter=FixedSizeSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        ),
        from_pdf=False,
        perform_entity_resolution=True,
    )


def snapshot_chunk_ids(driver: Driver) -> set[str]:
    """Return the element IDs of every :Chunk currently in the graph.

    The canary is scoped by chunk rather than by a diff over all nodes because
    `perform_entity_resolution=True` merges a newly extracted entity into an
    existing node when one already matches. Re-ingesting a document the graph
    already holds therefore creates a `Chunk` but no new `Hotel`, and a
    node-level diff reads that as "extraction produced no Hotel" when in fact
    it produced one and deduplicated it. Chunks are never merged, so they are a
    stable handle on "what this run just extracted".
    """
    with driver.session() as session:
        return {
            record["id"]
            for record in session.run("MATCH (c:Chunk) RETURN elementId(c) AS id")
        }


def clear_demo_graph(driver: Driver) -> None:
    """Delete only the nodes this demo created."""
    with driver.session() as session:
        session.run(f"MATCH (n:`{KG_LABEL}`) DETACH DELETE n")


async def ingest(pipeline: SimpleKGPipeline, paths: list[Path]) -> int:
    """Run every document through the pipeline. Returns the error count."""
    total = len(paths)
    errors = 0
    for i, path in enumerate(paths, 1):
        text = path.read_text(encoding="utf-8")
        print(f"  [{i}/{total}] {path.name}...", end=" ", flush=True)
        try:
            await asyncio.wait_for(
                pipeline.run_async(text=text), timeout=DOC_TIMEOUT_SECONDS
            )
            print("✅")
        except asyncio.TimeoutError:
            errors += 1
            print("⏰ timeout")
        except Exception as exc:  # noqa: BLE001 - one bad doc must not stop the build
            errors += 1
            print(f"❌ {str(exc)[:80]}")
    return errors


def check_schema_held(driver: Driver, chunk_ids: set[str]) -> list[str]:
    """Return a list of problems with what the canary chunks extracted.

    Entities are reached by traversing `(:Chunk)<-[:FROM_CHUNK]-(entity)` from
    the chunks this run created, so the check is correct whether the entity was
    newly inserted or merged into an existing node by entity resolution.

    An empty list means extraction honoured the contract in
    `query_knowledge_graph`'s docstring.
    """
    problems: list[str] = []
    ids = list(chunk_ids)
    with driver.session() as session:
        labels = {
            record["label"]: record["count"]
            for record in session.run(
                """
                MATCH (c:Chunk)<-[:FROM_CHUNK]-(n)
                WHERE elementId(c) IN $ids
                UNWIND labels(n) AS label
                WITH label, count(*) AS count
                WHERE NOT label STARTS WITH '__'
                RETURN label, count
                """,
                ids=ids,
            )
        }
        print(f"  labels produced: {labels}")

        stray = sorted(set(labels) & set(OFF_SCHEMA_LABELS))
        if stray:
            problems.append(f"off-schema labels present: {stray}")

        if not labels.get("Hotel"):
            problems.append("no :Hotel node was extracted from the canary chunks")
            return problems

        # Extraction is stochastic: an LLM can miss a field on any single
        # document without the pipeline being broken. The gate is therefore
        # "at least one canary document extracted a complete Hotel", not
        # "every one did". The off-schema label check above stays strict,
        # because inventing an `Address` node is a schema failure rather than
        # a bad roll.
        hotels = list(
            session.run(
                """
                MATCH (c:Chunk)<-[:FROM_CHUNK]-(h:Hotel)
                WHERE elementId(c) IN $ids
                OPTIONAL MATCH (h)-[r]->(n)
                WHERE type(r) IN ['HAS_ROOM', 'OFFERS_AMENITY',
                                  'HAS_POLICY', 'PROVIDES_SERVICE']
                RETURN DISTINCT h.name AS name, h.address AS address,
                       h.guest_rating AS guest_rating, count(r) AS relationships
                """,
                ids=ids,
            )
        )

        conforming = []
        for hotel in hotels:
            missing = [
                field
                for field in ("name", "address", "guest_rating")
                if hotel[field] is None
            ]
            is_conforming = not missing and hotel["relationships"] > 0
            if is_conforming:
                conforming.append(hotel)
            status = "ok" if is_conforming else "INCOMPLETE"
            print(
                f"  hotel: {hotel['name']!r} rating={hotel['guest_rating']} "
                f"rels={hotel['relationships']} [{status}]"
            )

        print(f"  conforming hotels: {len(conforming)}/{len(hotels)}")
        if not conforming:
            problems.append(
                f"none of the {len(hotels)} canary hotels had name, address, "
                "guest_rating and a contracted relationship"
            )

    return problems


def count_documents(driver: Driver) -> int:
    """Return the number of :Document nodes in the graph.

    B17's invariant: after a clean build this equals the number of files that
    were processed. Anything else means two builds overlapped or a partial run
    was left behind, and the resulting graph looks plausible while being wrong.
    """
    with driver.session() as session:
        return session.run(
            "MATCH (d:Document) RETURN count(d) AS count"
        ).single()["count"]


def count_chunks(driver: Driver) -> int:
    """Return the number of :Chunk nodes in the graph."""
    with driver.session() as session:
        return session.run("MATCH (c:Chunk) RETURN count(c) AS count").single()["count"]


def report(driver: Driver) -> None:
    """Print the graph shape plus the three queries the notebook depends on."""
    with driver.session() as session:
        print("\nNode labels:")
        for record in session.run(
            """
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
            UNWIND [l IN labels(n) WHERE l IN $labels] AS label
            RETURN label, count(*) AS count
            ORDER BY count DESC
            """,
            labels=list(SCHEMA_NODE_LABELS),
        ):
            print(f"  :{record['label']}: {record['count']}")

        print("\nRelationship types:")
        for record in session.run(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN ['HAS_ROOM', 'OFFERS_AMENITY',
                              'HAS_POLICY', 'PROVIDES_SERVICE']
            RETURN type(r) AS rel, count(*) AS count
            ORDER BY count DESC
            """
        ):
            print(f"  :{record['rel']}: {record['count']}")

        print("\n--- Acceptance queries (these are what the notebook asks) ---")

        record = session.run(
            """
            MATCH (h:Hotel)
            WHERE toLower(h.address) CONTAINS 'paris'
            RETURN avg(h.guest_rating) AS avg_rating, count(h) AS hotels
            """
        ).single()
        print(
            f"  Aggregation  avg guest rating in Paris: {record['avg_rating']} "
            f"across {record['hotels']} hotels"
        )

        record = session.run(
            """
            MATCH (h:Hotel)-[:OFFERS_AMENITY]->(a:Amenity)
            WHERE toLower(a.name) CONTAINS 'pool'
            RETURN count(DISTINCT h) AS hotels
            """
        ).single()
        print(f"  Counting  hotels with a pool: {record['hotels']}")

        print("  Multi-hop  Cairo hotels with spa AND pool:")
        rows = session.run(
            """
            MATCH (h:Hotel)-[:OFFERS_AMENITY]->(spa:Amenity),
                  (h)-[:OFFERS_AMENITY]->(pool:Amenity)
            WHERE h.address CONTAINS 'Cairo'
              AND toLower(spa.name) CONTAINS 'spa'
              AND toLower(pool.name) CONTAINS 'pool'
            RETURN DISTINCT h.name AS name, h.guest_rating AS rating
            """
        )
        found = False
        for record in rows:
            found = True
            print(f"    {record['name']} — {record['rating']}")
        if not found:
            print("    (none)")


async def run_build(paths: list[Path], title: str) -> int:
    """Canary, verify, wipe, ingest, report. Returns a process exit code."""
    if not paths:
        print("No documents selected.")
        return 1

    print(f"{title}: {len(paths)} documents\n")
    driver = connect()
    try:
        print("Clearing this demo's previous graph...")
        clear_demo_graph(driver)
        print("✅ Cleared\n")

        canary = paths[:CANARY_DOCS]
        names = ", ".join(path.name for path in canary)
        print(f"Canary: extracting {names} before ingesting the rest...")
        baseline = snapshot_chunk_ids(driver)
        pipeline = build_pipeline(driver)
        await ingest(pipeline, canary)

        new_chunks = snapshot_chunk_ids(driver) - baseline
        if not new_chunks:
            print("\n❌ Canary produced no :Chunk — extraction did not run.")
            clear_demo_graph(driver)  # leave a clean, empty graph on failure
            return 1
        problems = check_schema_held(driver, new_chunks)
        if problems:
            print("\n❌ Canary failed. The graph was cleared; fix and re-run:")
            for problem in problems:
                print(f"  - {problem}")
            clear_demo_graph(driver)  # remove the canary's partial docs
            return 1
        print("✅ Canary passed: extraction matches the documented schema\n")

        print("Clearing the canary's documents before the full ingest...")
        clear_demo_graph(driver)
        print("✅ Cleared\n")

        errors = await ingest(pipeline, paths)
        # No `await pipeline.close()` here. `SimpleKGPipeline` defines no
        # `close()`, so that call raised `AttributeError` at the end of every
        # otherwise-successful build. It owns no resource needing release; the
        # driver is closed in the `finally` below.

        processed = len(paths) - errors
        print(f"\n{'=' * 60}")
        print(f"BUILD COMPLETE ({processed}/{len(paths)} docs processed)")
        print(f"{'=' * 60}")

        documents = count_documents(driver)
        chunks = count_chunks(driver)
        print(f"\n:Document nodes: {documents} (expected {processed})")
        print(f":Chunk nodes: {chunks} (expected {documents}, one chunk per document)")
        if documents != processed:
            print(
                "❌ Document count does not match the files processed. "
                "That means a concurrent build overlapped this one, or a "
                "partial run was left behind."
            )
            return 1

        report(driver)
        print("\n✅ Done!")
        return 0
    finally:
        driver.close()
