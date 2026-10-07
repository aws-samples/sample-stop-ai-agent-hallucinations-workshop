# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Try the semantic-tool-selection agent yourself.

Same pieces the notebook explains:
  * ALL_TOOLS       — a ~40-tool travel pool (real hotels/weather/booking + stand-ins)
  * SemanticToolHook — trims the agent to the top-3 relevant tools per request
  * FileSessionManager — persists the conversation so it survives restarts

Prerequisites: a running Neo4j with the hotel graph (see demo 01's build_graph.py).

Run:
    AWS_PROFILE=<profile> AWS_REGION=us-east-1 python chat.py
"""
import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel
from strands.session import FileSessionManager

from semantic_tools import ALL_TOOLS, SemanticToolHook

load_dotenv()

# A small model (Claude Haiku) makes tool confusion visible — and the semantic
# hook's benefit measurable. Strands uses Amazon Bedrock by default.
MODEL = BedrockModel(
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)


def main():
    hook = SemanticToolHook(ALL_TOOLS, top_k=3)
    # FileSessionManager persists the conversation to disk, so the agent
    # remembers across restarts (resume with the same session_id).
    session = FileSessionManager(
        session_id="semantic-tools-chat",
        storage_dir="./sessions/",
    )
    agent = Agent(
        model=MODEL,
        system_prompt="You are a travel agent that helps people with hotels. Keep answers concise — at most two sentences.",
        tools=ALL_TOOLS,
        hooks=[hook],
        session_manager=session,
    )

    print(f"Travel assistant ready — {len(ALL_TOOLS)} tools, trimmed to top-3 per question.")
    print("Memory persists across restarts. Ctrl-C to exit.\n")
    try:
        while True:
            q = input("you> ").strip()
            if not q:
                continue
            agent(q)
            print(f"\n  [selected tools: {hook.last_selected}]\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    main()
