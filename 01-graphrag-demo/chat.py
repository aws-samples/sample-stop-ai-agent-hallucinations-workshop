# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Try it yourself — ask ONE question, both agents answer.

Mirrors the notebook's `compare()`: each question goes to BOTH agents so you see
standard RAG vs Graph-RAG side by side.
  * search_faqs   — standard RAG (FAISS, top-3 passages)
  * query_hotels  — Graph-RAG (read-only Neo4j, exact, honest "no results")

Each agent is created ONCE and reused, so it keeps the conversation in memory:
ask a follow-up like "and which ones?" and it remembers the previous turn.

Prerequisites (once, outside Workshop Studio):
    AWS_PROFILE=<profile> AWS_REGION=us-east-1 python build_graph.py
    AWS_PROFILE=<profile> AWS_REGION=us-east-1 python load_vector_data.py

Run:
    AWS_PROFILE=<profile> AWS_REGION=us-east-1 python chat.py
"""
import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from dotenv import load_dotenv
from strands import Agent

from hotel_tools import HotelGraph, VectorFAQs

load_dotenv()


def main():
    faqs = VectorFAQs()
    graph = HotelGraph()

    # Two agents, each created ONCE so each keeps its own conversation memory
    # across turns. Strands uses Amazon Bedrock by default — no model config.
    rag_agent = Agent(
        system_prompt="You are a travel agent that helps people with hotels. Keep answers concise — at most two sentences.",
        tools=[faqs.search_faqs],
    )
    graph_agent = Agent(
        system_prompt="You are a travel agent that helps people with hotels. Keep answers concise — at most two sentences.",
        tools=[graph.query_hotels],
    )

    print("Ask about hotels — both agents answer (Ctrl-C to exit).")
    print("Try: 'How many hotels have a pool?' then a follow-up like 'which ones?'\n")
    try:
        while True:
            q = input("you> ").strip()
            if not q:
                continue
            print("\n--- STANDARD RAG (FAISS, top-3) ---")
            rag_agent(q)
            print("\n--- GRAPH-RAG (Neo4j, exact) ---")
            graph_agent(q)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
