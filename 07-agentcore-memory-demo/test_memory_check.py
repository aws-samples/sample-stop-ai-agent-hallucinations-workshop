"""Offline tests for the cross-session memory readiness check."""

import unittest

from memory_check import wait_for_memory_records


class _Paginator:
    def __init__(self, records_by_namespace):
        self.records_by_namespace = records_by_namespace

    def paginate(self, **kwargs):
        count = self.records_by_namespace.get(kwargs["namespace"], 0)
        yield {"memoryRecordSummaries": [{} for _ in range(count)]}


class _Client:
    def __init__(self, records_by_namespace):
        self.paginator = _Paginator(records_by_namespace)

    def get_paginator(self, operation):
        if operation != "list_memory_records":
            raise AssertionError(f"Unexpected operation: {operation}")
        return self.paginator


class TestWaitForMemoryRecords(unittest.TestCase):
    def test_returns_when_both_namespaces_have_records(self) -> None:
        actor_id = "user-test"
        client = _Client(
            {
                f"/users/{actor_id}/facts": 2,
                f"/users/{actor_id}/preferences": 1,
            }
        )

        counts = wait_for_memory_records(
            "memory-test",
            actor_id,
            "us-east-1",
            timeout=0,
            client=client,
        )

        self.assertEqual(counts, {"facts": 2, "preferences": 1})

    def test_times_out_when_a_namespace_is_empty(self) -> None:
        actor_id = "user-test"
        client = _Client({f"/users/{actor_id}/facts": 1})

        with self.assertRaisesRegex(TimeoutError, "preferences"):
            wait_for_memory_records(
                "memory-test",
                actor_id,
                "us-east-1",
                timeout=0,
                client=client,
            )


if __name__ == "__main__":
    unittest.main()
