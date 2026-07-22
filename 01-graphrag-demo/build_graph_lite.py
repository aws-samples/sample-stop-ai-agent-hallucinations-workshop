# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Build a LITE knowledge graph from a 30-document sample (~15 minutes).

The sample is stratified by city and always includes every Paris and Cairo
document, because those are the cities the notebook asks about. Taking the
first 30 files alphabetically only reached Boston, which made the Paris and
Cairo tests fail for want of data rather than for any interesting reason.

Same pinned schema and same verification as `build_graph.py`, so the notebook
runs unmodified against either graph.
"""

import asyncio
import os
import sys
from pathlib import Path

os.environ["OTEL_SDK_DISABLED"] = "true"

from dotenv import load_dotenv

load_dotenv()

from graph_builder import run_build
from graph_config import select_lite_files

DATA_DIR = Path("data")
MAX_DOCS = 30


def main() -> int:
    names = select_lite_files(DATA_DIR, MAX_DOCS)
    paths = [DATA_DIR / name for name in names]
    return asyncio.run(run_build(paths, "🚀 LITE BUILD"))


if __name__ == "__main__":
    sys.exit(main())
