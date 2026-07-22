# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Amazon Bedrock providers for neo4j-graphrag.

Replaces OpenAI dependencies with:
- Amazon Nova 2 Multimodal Embeddings for embeddings
- Amazon Bedrock Claude for LLM entity extraction

No OpenAI API key required — uses AWS credentials only.
"""

import asyncio
import json
import os
from collections.abc import Sequence

import boto3
from botocore.config import Config
from neo4j_graphrag.llm.base import LLMInterface, LLMResponse
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage


# botocore defaults to a 60s read timeout and 5 attempts, so one hung call can
# burn 300s. graph_builder wraps each document in a 180s asyncio.wait_for, and
# that outer bound cannot govern a larger inner one: the timeout fires while the
# worker thread keeps running underneath, because a thread cannot be cancelled.
#
# max_attempts counts retries, so this resolves to 3 total attempts.
# Worst case is 3 x 45s plus backoff, roughly 140s, under the 180s bound.
# Keep that inequality true if you change either number.
BEDROCK_CONFIG = Config(read_timeout=45, retries={"max_attempts": 2})


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence, if the model added one.

    neo4j-graphrag parses entity-extraction responses as raw JSON, but Claude
    commonly wraps JSON in ```json ... ``` fences. Left in place, every chunk
    fails with "LLM response has improper format" and the graph comes out empty.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text

    # Only strip when the closing fence is also present. A truncated response
    # carrying an opening fence but no closing one would otherwise drop its
    # first line and hand back broken JSON.
    lines = stripped.splitlines()
    if len(lines) < 2 or lines[-1].strip() != "```":
        return text
    return "\n".join(lines[1:-1]).strip()


def _converse_messages(message_history) -> list[dict]:
    """Convert a neo4j-graphrag message history into Converse API messages.

    `LLMMessage` is a TypedDict, so history entries arrive as plain dicts.
    Attribute access silently misses on those: every turn would collapse to the
    dict's repr carried under role "user", relabelling assistant turns as user.
    A `MessageHistory` object holds the same dicts on `.messages`.
    """
    messages = getattr(message_history, "messages", message_history)

    converted = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", str(msg))
        converted.append({"role": role, "content": [{"text": content}]})
    return converted


class BedrockEmbeddings(Embedder):
    """Amazon Bedrock embeddings using Nova 2 Multimodal Embeddings."""

    def __init__(
        self,
        model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0",
        region_name: str | None = None,
        dimensions: int = 1024,
    ):
        # Resolve the region inside the body: an os.environ default argument is
        # evaluated once at import, before the caller can set AWS_REGION.
        if region_name is None:
            region_name = os.environ.get("AWS_REGION", "us-east-1")
        self.model_id = model_id
        self.dimensions = dimensions
        self.client = boto3.client(
            "bedrock-runtime", region_name=region_name, config=BEDROCK_CONFIG
        )

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
        model_id: str = "us.anthropic.claude-sonnet-5",
        region_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ):
        # Resolve the region inside the body: an os.environ default argument is
        # evaluated once at import, before the caller can set AWS_REGION.
        if region_name is None:
            region_name = os.environ.get("AWS_REGION", "us-east-1")
        self.model_id = model_id
        self.client = boto3.client(
            "bedrock-runtime", region_name=region_name, config=BEDROCK_CONFIG
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, input: str,
               message_history: Sequence[LLMMessage] | MessageHistory | None = None,
               system_instruction: str | None = None) -> LLMResponse:
        messages = _converse_messages(message_history) if message_history else []
        messages.append({"role": "user", "content": [{"text": input}]})

        inference_config = {"maxTokens": self.max_tokens}

        # Sonnet 5 rejects `temperature` outright ("deprecated for this model"),
        # so only send it when a caller explicitly asks for one.
        if self.temperature is not None:
            inference_config["temperature"] = self.temperature

        kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        if system_instruction:
            kwargs["system"] = [{"text": system_instruction}]

        response = self.client.converse(**kwargs)
        blocks = response["output"]["message"]["content"]

        # Sonnet 5 puts a reasoningContent block before the answer, so take the
        # first block that actually carries text instead of assuming index 0.
        content = next((b["text"] for b in blocks if "text" in b), None)
        if content is None:
            raise ValueError(f"No text block in Bedrock response: {[list(b) for b in blocks]}")

        return LLMResponse(content=_strip_code_fence(content))

    async def ainvoke(self, input: str, message_history=None, system_instruction=None) -> LLMResponse:
        # `invoke` is a blocking botocore round trip. Awaiting it inline would
        # block the event loop, so the `asyncio.wait_for` per-document timeout
        # in graph_builder could never fire. Hand it to a worker thread.
        return await asyncio.to_thread(
            self.invoke, input, message_history, system_instruction
        )
