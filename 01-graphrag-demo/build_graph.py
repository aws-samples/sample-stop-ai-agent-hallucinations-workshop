# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Build the full knowledge graph from all 300 hotel FAQ documents.

The extraction schema is pinned (see `graph_config.GRAPH_SCHEMA`) so the graph
matches the contract the agent is given in `query_knowledge_graph`'s docstring.
Letting the LLM discover a schema per chunk sounds appealing, but it produces a
different set of labels for every document and no query written against it can
be relied on.

Expect roughly two hours. `build_graph_lite.py` runs a stratified 30-document
sample in about fifteen minutes and answers the same notebook questions.
"""

import asyncio
import os
import sys
from pathlib import Path

os.environ["OTEL_SDK_DISABLED"] = "true"

from dotenv import load_dotenv

load_dotenv()

from graph_builder import run_build

DATA_DIR = Path("data")


def main() -> int:
    paths = sorted(DATA_DIR.glob("*.txt"))
    return asyncio.run(run_build(paths, "FULL BUILD"))


if __name__ == "__main__":
    sys.exit(main())
