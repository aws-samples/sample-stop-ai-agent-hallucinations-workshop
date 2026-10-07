# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Try it yourself — a booking agent with the neurosymbolic hook enforcing rules.

Same hook and tools taught in the notebook: the LLM understands your request,
but `NeurosymbolicHook` validates every tool call before it runs. Try to break
a rule (book a hotel that doesn't exist, 15 guests, confirm before paying) and
watch the hook block it — no matter how you phrase it.

The agent is created ONCE and reused, so it keeps the conversation in memory:
book, then say "now pay $300 for it", then "confirm it".

Prerequisites:
    A Neo4j hotel graph (demo 01 builds it). Set NEO4J_PASSWORD in a .env file.

Run:
    AWS_PROFILE=<profile> AWS_REGION=us-east-1 python chat.py
"""
import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from dotenv import load_dotenv
from strands import Agent

from neurosymbolic_tools import BookingStore, HotelGraph, NeurosymbolicHook

load_dotenv()

SYSTEM_PROMPT = (
    "You are a travel agent that helps people with hotels. Keep answers concise — "
    "at most two sentences."
)


def main():
    graph = HotelGraph()
    store = BookingStore()
    hook = NeurosymbolicHook(graph, store)

    # Strands uses Amazon Bedrock by default — no model config needed.
    agent = Agent(
        system_prompt=SYSTEM_PROMPT,
        tools=[store.book_hotel, store.process_payment, store.confirm_booking],
        hooks=[hook],
    )

    print("Book hotels — the hook enforces the rules (Ctrl-C to exit).")
    print("Try: \"book Adobe Plaza Inn for Ana, 15 guests, 2026-03-20 to 2026-03-22\"\n")
    try:
        while True:
            q = input("you> ").strip()
            if q:
                agent(q)
                print()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
