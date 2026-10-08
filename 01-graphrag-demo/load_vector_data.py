# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Build the FAISS vector index for the "standard RAG" side of demo 01.

Demo 01 compares two retrieval strategies over the SAME hotel FAQs:
  * standard RAG  — FAISS vector similarity (this file). Returns the top-k most
    similar chunks, which can look relevant even when the exact answer isn't
    there — that is where fabrication starts.
  * Graph-RAG     — Neo4j traversal (build_graph.py). Computes exact answers and
    returns nothing when nothing matches.

Run this ONCE, outside a Workshop Studio event, alongside build_graph.py:

    AWS_PROFILE=<your-profile> AWS_REGION=us-east-1 python load_vector_data.py

Default is FULL (300 docs) to match the 300-hotel dump. Set BUILD_LITE=1 for 30.
Uses Amazon Bedrock Nova 2 embeddings (same model as the graph's chunk vectors).
"""
import json
import os
from pathlib import Path

import boto3
import faiss
import numpy as np

MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
REGION = os.environ.get("AWS_REGION", "us-east-1")
DIMENSIONS = 1024
# Default is the FULL 300-hotel dataset so the FAISS index matches the 300-hotel
# Neo4j dump restored in Workshop Studio. Set BUILD_LITE=1 for a quick 30-doc index.
LITE = os.getenv("BUILD_LITE") == "1"
MAX_DOCS = 30 if LITE else None


def _embed(texts, client):
    vectors = []
    for text in texts:
        resp = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": DIMENSIONS,
                    "text": {"truncationMode": "END", "value": text[:8000]},
                },
            }),
            contentType="application/json",
            accept="application/json",
        )
        vectors.append(json.loads(resp["body"].read())["embeddings"][0]["embedding"])
    return np.array(vectors, dtype="float32")


def _ensure_data():
    """Make sure data/*.txt exists. In Workshop Studio data/ is .gitignored and
    only hotel-faqs.zip ships, so extract it on first run (otherwise glob finds
    0 files and FAISS fails with 'Embedding 0 FAQ documents')."""
    data_dir = Path("data")
    if data_dir.exists() and any(data_dir.glob("*.txt")):
        return
    zip_path = Path("hotel-faqs.zip")
    if not zip_path.exists():
        raise FileNotFoundError(
            "data/ has no .txt files and hotel-faqs.zip is missing. "
            "Restore hotel-faqs.zip or populate data/."
        )
    import zipfile
    data_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    # The zip may contain a top-level folder; flatten *.txt into data/.
    for p in [q for q in data_dir.rglob("*.txt") if q.parent != data_dir]:
        p.rename(data_dir / p.name)


def build():
    _ensure_data()
    files = sorted(Path("data").glob("*.txt"))
    if MAX_DOCS:
        files = files[:MAX_DOCS]
    documents = [{"filename": f.name, "text": f.read_text(encoding="utf-8")} for f in files]
    print(f"Embedding {len(documents)} FAQ documents into FAISS...")

    client = boto3.client("bedrock-runtime", region_name=REGION)
    embeddings = _embed([d["text"] for d in documents], client)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "faqs_vector.index")
    with open("faqs_docs.json", "w", encoding="utf-8") as f:
        json.dump(documents, f)
    print(f"✅ FAISS index built: {len(documents)} docs, {DIMENSIONS} dims -> faqs_vector.index")


if __name__ == "__main__":
    build()
