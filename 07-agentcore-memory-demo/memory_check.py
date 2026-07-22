"""Readiness check for the cross-session AgentCore Memory demo."""

import time
from typing import Any

import boto3


def _record_count(client: Any, memory_id: str, namespace: str) -> int:
    """Count all records in one actor namespace."""
    return sum(
        len(page.get("memoryRecordSummaries", []))
        for page in client.get_paginator("list_memory_records").paginate(
            memoryId=memory_id,
            namespace=namespace,
        )
    )


def wait_for_memory_records(
    memory_id: str,
    actor_id: str,
    region: str,
    timeout: float = 300,
    poll_interval: float = 10,
    client: Any | None = None,
) -> dict[str, int]:
    """Wait until AgentCore has extracted both facts and preferences."""
    memory_client = client or boto3.client("bedrock-agentcore", region_name=region)
    namespaces = {
        "facts": f"/users/{actor_id}/facts",
        "preferences": f"/users/{actor_id}/preferences",
    }
    deadline = time.monotonic() + timeout

    print("Waiting for AgentCore to extract long-term memory records...")
    while True:
        counts = {
            kind: _record_count(memory_client, memory_id, namespace)
            for kind, namespace in namespaces.items()
        }
        print(
            "  extracted records: "
            f"facts={counts['facts']}, preferences={counts['preferences']}"
        )
        if all(count > 0 for count in counts.values()):
            print("PASS  LTM records are ready for the cross-session check.")
            return counts

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "AgentCore did not extract both fact and preference records "
                f"within {timeout:g}s for actor {actor_id!r}. Last counts: {counts}"
            )
        time.sleep(min(poll_interval, remaining))
