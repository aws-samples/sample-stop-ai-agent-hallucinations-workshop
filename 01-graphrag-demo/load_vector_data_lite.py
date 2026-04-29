# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Build LITE FAISS vector store from hotel FAQ documents using Amazon Bedrock Nova 2.

LITE VERSION: Processes only 30 documents (10% of full dataset) for faster testing.
"""
import json
import os
import faiss
import boto3
from pathlib import Path

MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
REGION = os.environ.get("AWS_REGION", "us-east-1")
DIMENSIONS = 1024
MAX_DOCS = 30


def _embed_texts(texts):
    """Embed texts using Amazon Bedrock Nova 2 Multimodal Embeddings."""
    import numpy as np
    client = boto3.client("bedrock-runtime", region_name=REGION)
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
        result = json.loads(resp["body"].read())
        vectors.append(result["embeddings"][0]["embedding"])
    return np.array(vectors, dtype="float32")


def load_to_vector_store():
    documents = []
    data_dir = Path("data")

    for faq_file in sorted(data_dir.glob("*.txt"))[:MAX_DOCS]:
        with open(faq_file, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append({"filename": faq_file.name, "text": text})

    print(f"LITE MODE: Loading {len(documents)} FAQ documents (10% of full dataset)...")

    texts = [doc["text"] for doc in documents]
    embeddings = _embed_texts(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "faqs_vector_lite.index")
    with open("faqs_docs_lite.json", "w", encoding="utf-8") as f:
        json.dump(documents, f)

    print(f"LITE vector store created with {len(documents)} documents ({DIMENSIONS} dims)")


if __name__ == "__main__":
    load_to_vector_store()
