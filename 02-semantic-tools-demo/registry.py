import os
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Tool Registry — FAISS semantic search over tools.

Default: Amazon Bedrock Nova 2 Multimodal Embeddings (requires AWS credentials).
Alternative: SentenceTransformer all-MiniLM-L6-v2 (runs locally, no AWS needed).

To use the local model instead, uncomment the SentenceTransformer section below
and comment out the Bedrock section. Install: pip install sentence-transformers
"""
import json
import faiss
import boto3
from typing import List, Callable

_client = None
_index = None
_tools = []

# --- Amazon Bedrock Nova 2 Embeddings (default) ---
MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
REGION = os.environ.get("AWS_REGION", "us-east-1")
DIMENSIONS = 1024

# --- Local model alternative (uncomment to use instead of Bedrock) ---
# from sentence_transformers import SentenceTransformer
# _local_model = None
# DIMENSIONS = 384
# def _embed_local(texts):
#     global _local_model
#     if _local_model is None:
#         _local_model = SentenceTransformer('all-MiniLM-L6-v2')
#     return _local_model.encode(texts).astype('float32')


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=REGION)
    return _client


def _embed(texts: List[str]) -> list:
    """Embed texts using Amazon Bedrock Nova 2 Multimodal Embeddings."""
    import numpy as np
    client = _get_client()
    vectors = []
    for text in texts:
        resp = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": DIMENSIONS,
                    "text": {"truncationMode": "END", "value": text},
                },
            }),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        vectors.append(result["embeddings"][0]["embedding"])
    return np.array(vectors, dtype="float32")


def build_index(tools: List[Callable]):
    """Build FAISS index from tool docstrings using Nova 2 embeddings."""
    global _index, _tools
    _tools = tools

    texts = [f"{t.__name__}: {t.__doc__}" for t in tools]
    embeddings = _embed(texts)

    _index = faiss.IndexFlatL2(embeddings.shape[1])
    _index.add(embeddings)

    print(f"Indexed {len(tools)} tools ({DIMENSIONS} dims, Nova 2 embeddings)")


def search_tools(query: str, top_k: int = 3) -> List[Callable]:
    """Find most relevant tools for a query."""
    emb = _embed([query])
    _, indices = _index.search(emb, top_k)
    return [_tools[i] for i in indices[0]]


def swap_tools(agent, new_tools: List[Callable]):
    """Swap tools in a live agent without losing conversation memory.

    Clears the agent's tool_registry and re-registers only the given tools.
    Since get_all_tools_config() is called each event loop cycle, the agent
    will see the new tools on the next call.
    """
    reg = agent.tool_registry
    reg.registry.clear()
    reg.dynamic_tools.clear()
    for t in new_tools:
        reg.register_tool(t)


def get_scores(query: str, top_k: int = 10) -> List[dict]:
    """Get tool scores for debugging."""
    emb = _embed([query])
    distances, indices = _index.search(emb, min(top_k, len(_tools)))
    return [
        {"name": _tools[i].__name__, "score": 1 / (1 + d), "doc": _tools[i].__doc__}
        for i, d in zip(indices[0], distances[0])
    ]
