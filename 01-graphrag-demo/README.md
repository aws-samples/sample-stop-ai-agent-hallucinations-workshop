[< Back to Main README](../README.md)

# RAG vs Graph-RAG: Reducing Agent Hallucinations

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Strands Agents](https://img.shields.io/badge/Strands_Agents-1.58-00B4D8.svg?style=flat)](https://strandsagents.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph--RAG-4581C3.svg?style=flat&logo=neo4j)](https://neo4j.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue.svg?style=flat)](https://github.com/facebookresearch/faiss)

> Vector-only RAG makes AI agents fabricate counts and averages, and answer even when nothing matches. This demo compares standard RAG ([FAISS](https://github.com/facebookresearch/faiss), a vector similarity index) against Graph-RAG (a read-only [Neo4j](https://neo4j.com) knowledge graph) over the same hotel FAQ documents, so you can see which failure mode each approach produces on the same question.

![Two stacked bands over the same question, how many hotels have a pool. Vector RAG runs a cosine search over chunks, sees the top 3 of 300 hotels and estimates a number, so the total is a guess. Graph-RAG runs Cypher over every hotel, counts in the database and returns the exact count or no results, so the aggregation covers all the data and empty means empty](../images/rag-vs-graphrag.png)

## What This Demo Shows

Two agents answer the **same** question back-to-back so you see both behaviours together:

- **Standard RAG agent** → `search_faqs` → FAISS returns the top-3 most similar passages → the LLM summarizes them.
- **Graph-RAG agent** → `query_hotels` → the LLM writes a read-only Cypher query (Text2Cypher) → Neo4j computes the exact answer.

Three RAG failure modes, each with a query in the notebook that triggers it:

1. **Fabricated statistics**, the LLM invents plausible numbers from a few text chunks instead of computing them.
2. **Incomplete retrieval**, vector search returns only the top-k passages, missing data spread across many documents (e.g. 5 Las Vegas hotels when FAISS only returns 3 chunks).
3. **Out-of-domain fabrication**, when no relevant data exists, vector search still returns its closest matches and the LLM answers from them (e.g. "hotels in Antarctica").

Graph-RAG avoids these because the structured side of the graph gives it:

- **Native aggregations**, `AVG()`, `COUNT()` computed in the database, not guessed.
- **Relationship traversal**, Cypher follows exact paths (Hotel → Room, Hotel → Amenity).
- **Honest empty results**, when a query matches nothing, the tool returns "No results found" and the agent is told not to invent.

## It Is Hybrid Graph-RAG: Chunks vs Documents vs Entities

A common misconception is that Graph-RAG has "no embeddings". It does, the graph this demo builds is **hybrid**, with two layers:

| Node | What it is | How it's used |
|------|------------|---------------|
| **Document** | One node per source `.txt` file, the origin of the text. | Provenance. `(Chunk)-[:FROM_DOCUMENT]->(Document)`. |
| **Chunk** | A slice of a document's text, **each embedded as a vector**. | The embedding/vector layer, chained with `NEXT_CHUNK`. |
| **Entity** (Hotel, Room, Amenity, Policy, Service) | Structured facts extracted from the text by an LLM. | Queried with exact Cypher traversal, **no vectors**. |

The two agents use different layers of the same data: the RAG agent searches a standalone FAISS index built from the documents, while the Graph-RAG agent queries the structured entities with Cypher. **The anti-hallucination value comes from the structured side** (exact aggregation and honest empty results), not from the chunks. The chunks exist so the graph can also do vector/semantic retrieval when a question needs free-text passages.

## Why a Fixed Schema (and why "add a hotel" is safe)

`build_graph.py` uses `neo4j-graphrag`'s `SimpleKGPipeline`, but with an **explicit, fixed schema** instead of letting the LLM discover one per document.

- **LLM-discovered schema (no `schema=`)**: the model invents labels and property names on every run. Across 300 documents this is non-deterministic, you get junk labels, duplicate relationship types, and inconsistent property types (one room has a numeric `rate`, another a text `priceRange`). Queries that worked yesterday break today.
- **Fixed schema (this demo)**: the LLM may only extract the labels, properties, and relationships you declared. Extraction is constrained and deterministic, so **adding a new hotel is a safe, repeatable operation**, the new hotel lands in exactly the same shape as every other, and your Cypher keeps working.

The schema this demo pins:

```
(Hotel {name, address, city, country, guestRating, totalRooms, phone, email})
(Room {type, rate, maxOccupancy})          rate = nightly price in USD (numeric)
(Amenity {name, description})
(Policy {name, description})
(Service {name, description})

(Hotel)-[:HAS_ROOM]->(Room)
(Hotel)-[:OFFERS_AMENITY]->(Amenity)
(Hotel)-[:HAS_POLICY]->(Policy)
(Hotel)-[:PROVIDES_SERVICE]->(Service)
```

After the build, `build_graph.py` verifies the result: only the expected labels and relationships, and every `Room` with a numeric `rate`. All property names are **camelCase**.

> **Neo4j is READ-ONLY here.** The graph is the source of truth for hotel facts; it is never written to. The graph tool runs every query in READ routing mode, so it physically cannot mutate the graph. Writes (bookings) belong to a separate store, introduced in Demos 02–05.

## Architecture

![Two pipelines over the same hotel FAQ documents: vector RAG chunks and embeds them into a FAISS index and the agent sees only the 3 closest chunks, while Graph-RAG extracts entities and relationships with neo4j-graphrag into a Neo4j graph and the agent queries it with Text2Cypher](../images/graphrag-architecture.png)

Both the FAISS index and the graph's chunk vectors use the same Amazon Bedrock embedding model (Nova 2), and the LLM entity extraction uses Amazon Bedrock Claude.

## Quick Start

### Prerequisites

- Python 3.9+
- An AWS account with [Amazon Bedrock](https://aws.amazon.com/bedrock/) access in `us-east-1` (Claude + Nova 2 embeddings). **No external API key is needed**, Amazon Bedrock is the default provider.
- A Neo4j instance with the **APOC** plugin enabled.

### 1. Install Dependencies

```bash
cd 01-graphrag-demo
uv venv && uv pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your Neo4j connection. The `.env` file is gitignored, never commit real credentials.

```bash
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change-me
AWS_REGION=us-east-1
```

AWS credentials come from `aws configure` or an AWS profile, no key goes in `.env`.

### 3. Build the Data Stores (skip inside Workshop Studio)

**Inside a Workshop Studio event the Neo4j graph is already restored from a dump, skip the build entirely.** The data and dependencies are pre-loaded.

Running on your own? Build both stores once (uses Amazon Bedrock, so set your AWS profile/region in the shell first):

```bash
# Neo4j knowledge graph with the fixed schema (default: 30 docs / "lite", ~10–15 min)
AWS_PROFILE=<profile> AWS_REGION=us-east-1 python build_graph.py

# FAISS vector index for the standard-RAG side (fast)
AWS_PROFILE=<profile> AWS_REGION=us-east-1 python load_vector_data.py
```

Both scripts default to the **lite** base (30 documents) for fast iteration. For the full 300-document base, set `BUILD_FULL=1` (the full graph build takes roughly 2 hours because each document requires an LLM extraction call):

```bash
BUILD_FULL=1 python build_graph.py
BUILD_FULL=1 python load_vector_data.py
```

> The comparison questions in the notebook use **Las Vegas (5 hotels)** and ask how many hotels the answer is based on. The contrast is clearest on the **full 300-document base**, where many cities have more hotels than FAISS returns as top-k.

### 4. Run the Demo

```bash
# Notebook (recommended), open in VS Code, Kiro, or Jupyter
test_graphrag.ipynb

# Or the interactive REPL, ask one question, both agents answer
AWS_PROFILE=<profile> AWS_REGION=us-east-1 python chat.py
```

In `chat.py` each agent is created once and reused, so it keeps the conversation in memory: ask "How many hotels have a pool?" then a follow-up like "which ones?" and it remembers the previous turn.

## Files

| File | What it is |
|------|------------|
| `test_graphrag.ipynb` | The walkthrough: two agents (one tool each), the same question to both, then the fixed-schema check. Tools are defined inline to teach them. |
| `build_graph.py` | Builds the Neo4j graph with `SimpleKGPipeline` and a **fixed schema**. Schema and Amazon Bedrock wrapper classes live inline. Lite (30) by default, `BUILD_FULL=1` for 300. |
| `load_vector_data.py` | Builds the FAISS index for the standard-RAG side with Bedrock Nova 2 embeddings. Lite by default, `BUILD_FULL=1` for 300. |
| `hotel_tools.py` | `VectorFAQs.search_faqs` (FAISS) and `HotelGraph.query_hotels` (read-only Cypher), copied from the notebook so `chat.py` can import them. |
| `chat.py` | REPL that sends one question to both agents side by side. |
| `data/` | The hotel FAQ `.txt` documents. |

## Technologies

| Technology | Purpose |
|------------|---------|
| [Strands Agents](https://strandsagents.com) | AI agent framework |
| [Amazon Bedrock](https://aws.amazon.com/bedrock/) | LLM (Claude) for entity extraction + Nova 2 embeddings |
| [neo4j-graphrag](https://neo4j.com/docs/neo4j-graphrag-python/current/) | Knowledge-graph construction (`SimpleKGPipeline` with a fixed schema) |
| [Neo4j](https://neo4j.com) | Graph database (read-only in this demo) |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search for the standard-RAG side |

## Troubleshooting

**APOC not found:** APOC (Awesome Procedures On Cypher) is a Neo4j plugin needed for graph operations. Enable it in your Neo4j instance and restart the database.

**Graph build is slow:** Each document is one LLM extraction call (~20–30s). The lite base (30 docs) takes ~10–15 min; the full base (300 docs) ~2 hours. Build once.

**Bedrock access denied:** Ensure Claude and Nova 2 embeddings are enabled in `us-east-1` in the [Bedrock Model Access console](https://console.aws.amazon.com/bedrock/home#/modelaccess), and that your AWS profile/region are set.

The same Graph-RAG pattern (knowledge graph + Text2Cypher) can be implemented with any framework that supports custom tool calling.

---

## Frequently Asked Questions

### Does Graph-RAG have "no embeddings"?

No. The graph is **hybrid**: it has a vector layer (Chunk nodes, each embedded) and a structured layer (Hotel/Room/Amenity/... entities queried with Cypher). The anti-hallucination value in this demo comes from the structured layer, exact aggregation and honest empty results, not from removing embeddings.

### Do I need to define a schema for the knowledge graph?

This demo deliberately **does** define one. `SimpleKGPipeline` can let the LLM discover a schema per document, but that is non-deterministic and produces junk labels and inconsistent property types over hundreds of documents. Passing a fixed `schema=` constrains extraction so the graph is deterministic, which is what makes "add a new hotel" a safe, repeatable operation.

### How long does it take to build the graph?

The lite base (30 docs) takes roughly 10–15 minutes; the full base (300 docs) roughly 2 hours (one LLM extraction call per document). You build it once. Inside Workshop Studio you don't build it at all, it's restored from a dump.

### Does the demo write to Neo4j?

No. Neo4j is read-only here (hotel facts). The graph tool uses READ routing, so it cannot write. Bookings and other writes live in a separate store introduced in later demos.

---

## Further Reading

- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130), Microsoft Research on Graph-RAG.
- [RAG-KG-IL: A Multi-Agent Hybrid Framework for Reducing Hallucinations through RAG and Incremental Knowledge Graph Learning](https://arxiv.org/abs/2503.13514), case studies on health queries; a different domain and pipeline, so none of this demo's behaviour is derived from its numbers.
- [RAKG: Document-level Retrieval Augmented Knowledge Graph Construction](https://arxiv.org/abs/2504.09823v1), automated knowledge-graph construction from text, the same problem `SimpleKGPipeline` solves here.

---

## Navigation

- **Previous:** [Demo 00 - Getting Started](../00-getting-started/)
- **Next:** [Demo 02 - Semantic Tool Selection](../02-semantic-tools-demo/): reduce token waste and wrong tool picks with FAISS-based semantic filtering.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.
