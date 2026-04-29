# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Amazon Bedrock providers for neo4j-graphrag.

Replaces OpenAI dependencies with:
- Amazon Nova 2 Multimodal Embeddings for embeddings
- Amazon Bedrock Claude for LLM entity extraction

No OpenAI API key required — uses AWS credentials only.
"""

import json
import os
import boto3
from neo4j_graphrag.llm.base import LLMInterface, LLMResponse
from neo4j_graphrag.embeddings.base import Embedder


class BedrockEmbeddings(Embedder):
    """Amazon Bedrock embeddings using Nova 2 Multimodal Embeddings."""

    def __init__(
        self,
        model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0",
        region_name: str = os.environ.get("AWS_REGION", "us-east-1"),
        dimensions: int = 1024,
    ):
        self.model_id = model_id
        self.dimensions = dimensions
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def embed_query(self, text: str) -> list[float]:
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": self.dimensions,
                    "text": {"truncationMode": "END", "value": text},
                },
            }),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["embeddings"][0]["embedding"]


class BedrockLLM(LLMInterface):
    """Amazon Bedrock LLM using Claude via the Converse API."""

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
        region_name: str = os.environ.get("AWS_REGION", "us-east-1"),
        temperature: float = 0,
        max_tokens: int = 4096,
    ):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, input: str, message_history=None, system_instruction=None) -> LLMResponse:
        messages = []

        if message_history:
            for msg in message_history:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", str(msg))
                messages.append({"role": role, "content": [{"text": content}]})

        messages.append({"role": "user", "content": [{"text": input}]})

        kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
            },
        }

        if system_instruction:
            kwargs["system"] = [{"text": system_instruction}]

        response = self.client.converse(**kwargs)
        content = response["output"]["message"]["content"][0]["text"]
        return LLMResponse(content=content)

    async def ainvoke(self, input: str, message_history=None, system_instruction=None) -> LLMResponse:
        return self.invoke(input, message_history, system_instruction)
