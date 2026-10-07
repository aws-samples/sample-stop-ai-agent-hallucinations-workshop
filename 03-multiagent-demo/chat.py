# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Try the consortium yourself.

Ask a judgement question about hotels. The same question runs in parallel across
a heterogeneous consortium (Claude Haiku, Nova Pro, Nova Lite) as a Strands
Graph; agreement means confident, disagreement means a likely guess to verify.

Prerequisites: a running Neo4j with the hotel graph (demo 01's build_graph.py),
Amazon Bedrock with Claude and Nova enabled.

Run:
    AWS_PROFILE=<profile> AWS_REGION=us-east-1 python chat.py
"""
import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from dotenv import load_dotenv

from multiagent_tools import ask, build_consortium, hotel_context

load_dotenv()


def main():
    context = hotel_context()
    hotels = [l[2:].split(" (")[0] for l in context.splitlines() if l.startswith("- ")]
    graph = build_consortium()

    print("Consortium ready (Claude Haiku, Nova Pro, Nova Lite).")
    print("Ask a judgement question about hotels (Ctrl-C to exit).\n")
    try:
        while True:
            q = input("you> ").strip()
            if not q:
                continue
            v = ask(graph, q, context, hotels)
            print(f"  answers:  {v['answers']}")
            print(f"  tokens:   {v['tokens']} | consortium total: {v['total_in']}in/{v['total_out']}out")
            print(f"  clusters: {v['clusters']} | agreement: {v['agreement']} | entropy: {v['entropy']}")
            print(f"  => {v['verdict']}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    main()
