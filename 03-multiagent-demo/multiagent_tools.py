# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Demo 03: multi-agent validation by consortium consistency.

Judgement questions ("best hotel for a honeymoon?") have no single database
answer, so a model guesses — confidently. Consortium consistency
(arXiv:2510.19507) asks the same question to several heterogeneous models and
measures agreement: consensus means confident, divergence means the models are
guessing (likely hallucination). A model that guesses alone is out-voted.

Built as a native Strands multi-agent Graph (parallel fan-out), not a Python
loop. We use Graph, not Swarm, because the models must answer INDEPENDENTLY and
in PARALLEL: Swarm shares context between agents (they would copy each other and
lose the diversity the consortium relies on), while Graph runs them as separate
parallel nodes whose results we read and score.

Neo4j (read-only) supplies the hotel context the models reason over.
No Llama (it can't be enabled in the workshop).
"""
import math
import os
from collections import Counter

from neo4j import GraphDatabase, RoutingControl
from strands import Agent
from strands.models import BedrockModel
from strands.multiagent import GraphBuilder

REGION = os.getenv("AWS_REGION", "us-east-1")

# Heterogeneous consortium: two model families, different training/architecture.
CONSORTIUM = {
    "claude-haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "nova-pro": "amazon.nova-pro-v1:0",
    "nova-lite": "us.amazon.nova-2-lite-v1:0",
}

ADVISOR_PROMPT = (
    "You are a travel agent that helps people with hotels. Answer by naming "
    "exactly one hotel from the list in the task and nothing else."
)

_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
)


def hotel_context(limit: int = 12) -> str:
    """Read-only Neo4j: a short hotel list the models reason over (shared context)."""
    recs, _, _ = _driver.execute_query(
        "MATCH (h:Hotel) RETURN h.name AS name, h.city AS city, h.guestRating AS rating "
        "ORDER BY rating DESC LIMIT $limit", limit=limit, routing_=RoutingControl.READ)
    return "\n".join(f"- {r['name']} ({r['city']}, rating {r['rating']})" for r in recs)


def build_consortium():
    """A Strands Graph that runs every consortium model in parallel on one task."""
    builder = GraphBuilder()
    for name, model_id in CONSORTIUM.items():
        builder.add_node(
            # callback_handler=None silences each agent's streaming output so the
            # notebook shows only the clean consortium summary, not raw reasoning.
            Agent(name=name, system_prompt=ADVISOR_PROMPT, callback_handler=None,
                  model=BedrockModel(model_id=model_id, region_name=REGION)),
            name,
        )
        builder.set_entry_point(name)
    return builder.build()


def consortium_verdict(graph_result, hotels: list[str]) -> dict:
    """Score the parallel model answers: cluster them and compute agreement/entropy."""
    def canonical(text):
        # Map each answer to the hotel it names (longest match). If the model
        # named no hotel (e.g. it refused or rambled), label it compactly so the
        # output stays readable and the non-answer forms its own cluster.
        hits = [h for h in hotels if h.lower() in text.lower()]
        return max(hits, key=len) if hits else "(no answer)"

    answers = {
        name: canonical(str(node.result))
        for name, node in graph_result.results.items()
    }
    tokens = {
        name: {
            "in": node.result.metrics.accumulated_usage["inputTokens"],
            "out": node.result.metrics.accumulated_usage["outputTokens"],
        }
        for name, node in graph_result.results.items()
    }
    votes = Counter(answers.values())
    n = sum(votes.values()) or 1
    entropy = -sum((c / n) * math.log2(c / n) for c in votes.values())
    entropy /= (math.log2(len(answers)) if len(answers) > 1 else 1)
    return {
        "answers": answers,
        "tokens": tokens,
        "total_in": graph_result.accumulated_usage["inputTokens"],
        "total_out": graph_result.accumulated_usage["outputTokens"],
        "clusters": dict(votes),
        "agreement": round(votes.most_common(1)[0][1] / n, 2),
        "entropy": round(entropy, 2),
        "verdict": "CONFIDENT (consensus)" if len(votes) == 1
                   else "LIKELY HALLUCINATION (models disagree, verify)",
    }


def ask(graph, question: str, context: str, hotels: list[str]) -> dict:
    """Run the consortium graph on one question and score the result."""
    result = graph(f"Hotels:\n{context}\n\nQuestion: {question}")
    return consortium_verdict(result, hotels)
