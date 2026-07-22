# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Cleanup for the AgentCore workshop (Modules 6 and 7).

Usage::

    python3 cleanup.py             # dry run: show what would be deleted, and why
    python3 cleanup.py --yes       # actually delete

The default is a dry run. Deletion requires the explicit ``--yes`` flag so a
bare ``python3 cleanup.py`` never destroys resources by surprise.

The deletion logic lives in ``08-cleanup/workshop_cleanup.py`` so there is
exactly **one** teardown path to audit. This file used to carry its own copy,
and that copy deleted IAM roles by account-wide name prefix::

    cb_roles = iam.list_roles(PathPrefix="/")["Roles"]
    for role in cb_roles:
        if role["RoleName"].startswith("AmazonBedrockAgentCoreSDKCodeBuild"):
            ... iam.delete_role(RoleName=rn)

IAM is global, so that reached every region and destroyed five unrelated roles
that this workshop never created (``verify.md`` bug B6). Deletion is now scoped
by the tag ``WorkshopResource=stop-ai-agent-hallucinations`` and never by name
shape. Untagged resources are reported and left alone, and the process exits
non-zero rather than reporting a teardown that did not happen.
"""

from __future__ import annotations

import sys
from pathlib import Path

CLEANUP_MODULE_DIR = Path(__file__).resolve().parent.parent / "08-cleanup"
if str(CLEANUP_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(CLEANUP_MODULE_DIR))

from workshop_cleanup import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
