# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Regression guards for the Amazon Bedrock providers.

Three already-landed fixes are pinned here:

* **F2** — :func:`_converse_messages` must preserve each turn's role and content.
  ``LLMMessage`` is a ``TypedDict``, so history entries arrive as plain dicts;
  the old attribute-access path silently relabelled every assistant turn as
  ``user``. This test round-trips a two-turn history and asserts the assistant
  turn survives as ``assistant``.
* **F3** — :meth:`BedrockLLM.ainvoke` hands the blocking botocore call to
  ``asyncio.to_thread`` so an outer ``asyncio.wait_for`` can actually fire.
  Before the fix the call ran inline on the event loop and the timeout could
  never cancel it. This test stubs a slow ``invoke`` and asserts the timeout
  raises rather than hanging.
* **F16** — both Bedrock clients are built with ``BEDROCK_CONFIG`` so each call
  is bounded (``read_timeout`` x ``total_max_attempts``) well under the
  per-document ``DOC_TIMEOUT_SECONDS`` budget. This test asserts the config
  reaches both clients and that the bounding invariant holds.

Run with::

    python -m unittest discover -s 01-graphrag-demo -v

or collect the whole file with ``pytest 01-graphrag-demo/test_bedrock_providers.py``.
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path

# The module under test lives beside this file rather than on an installed
# package path, so make the sibling directory importable regardless of how the
# suite is launched (``unittest`` discovery, a direct path, or pytest).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neo4j_graphrag.llm.base import LLMResponse  # noqa: E402
from neo4j_graphrag.message_history import InMemoryMessageHistory  # noqa: E402
from neo4j_graphrag.types import LLMMessage  # noqa: E402

from bedrock_providers import (  # noqa: E402
    BedrockEmbeddings,
    BedrockLLM,
    _converse_messages,
)

# ``DOC_TIMEOUT_SECONDS`` is imported lazily inside the one test that needs it:
# ``graph_builder`` pulls in ``graph_config``, which raises at import when
# ``NEO4J_PASSWORD`` is unset. Importing it here would fail collection for the
# whole file, including the F2 and F3 guards that never touch Neo4j.


class TestConverseMessages(unittest.TestCase):
    """F2 — a message history must round-trip with roles and content intact."""

    def test_typed_dict_history_preserves_roles_and_content(self) -> None:
        """A list of ``LLMMessage`` dicts survives conversion unchanged.

        ``LLMMessage`` is a ``TypedDict``, so entries are plain dicts. The old
        ``getattr(msg, "role", "user")`` path missed on dicts and relabelled the
        assistant turn as ``user``; this asserts it stays ``assistant``.
        """
        user_turn: LLMMessage = {"role": "user", "content": "book me a hotel"}
        assistant_turn: LLMMessage = {"role": "assistant", "content": "which city?"}

        converted = _converse_messages([user_turn, assistant_turn])

        self.assertEqual(
            converted,
            [
                {"role": "user", "content": [{"text": "book me a hotel"}]},
                {"role": "assistant", "content": [{"text": "which city?"}]},
            ],
        )
        # The heart of the regression: the second turn is still the assistant.
        self.assertEqual(converted[1]["role"], "assistant")
        self.assertEqual(converted[1]["content"][0]["text"], "which city?")

    def test_message_history_object_is_accepted(self) -> None:
        """The helper also reads a ``MessageHistory`` object via ``.messages``.

        ``_converse_messages`` does ``getattr(history, "messages", history)``,
        so a real ``InMemoryMessageHistory`` holding the same dicts must convert
        identically to the bare list.
        """
        history = InMemoryMessageHistory(
            messages=[
                {"role": "user", "content": "book me a hotel"},
                {"role": "assistant", "content": "which city?"},
            ]
        )

        converted = _converse_messages(history)

        self.assertEqual(
            converted,
            [
                {"role": "user", "content": [{"text": "book me a hotel"}]},
                {"role": "assistant", "content": [{"text": "which city?"}]},
            ],
        )
        self.assertEqual(converted[1]["role"], "assistant")


class TestAinvokeTimeout(unittest.TestCase):
    """F3 — ``ainvoke`` runs off the event loop so ``wait_for`` can cancel it."""

    def test_wait_for_times_out_instead_of_hanging(self) -> None:
        """A slow synchronous ``invoke`` must not defeat an outer timeout.

        ``ainvoke`` delegates to ``asyncio.to_thread``, so a 1s ``wait_for``
        around a 5s call raises ``TimeoutError`` in about a second. Before the
        fix the blocking call ran inline and this hung for the full 5s.
        """
        provider = BedrockLLM()

        def slow_invoke(
            _input: str,
            _message_history: object = None,
            _system_instruction: object = None,
        ) -> LLMResponse:
            time.sleep(5)
            return LLMResponse(content="too late")

        # Stub the blocking round trip only; ``ainvoke`` (the code under test)
        # is left untouched.
        provider.invoke = slow_invoke  # type: ignore[method-assign]

        # A private loop closed with ``wait=False`` keeps the test near 1s: we
        # do not block on the orphaned worker thread's remaining sleep.
        loop = asyncio.new_event_loop()
        started = time.monotonic()
        try:
            with self.assertRaises(asyncio.TimeoutError):
                loop.run_until_complete(
                    asyncio.wait_for(provider.ainvoke("hello"), timeout=1.0)
                )
        finally:
            loop.close()

        elapsed = time.monotonic() - started
        self.assertLess(
            elapsed,
            5.0,
            "wait_for should cancel around 1s, not wait out the full 5s sleep",
        )


class TestBedrockClientConfig(unittest.TestCase):
    """F16 — both clients carry a bounded config, and the bound holds."""

    def test_both_clients_apply_bedrock_config(self) -> None:
        for provider in (BedrockLLM(), BedrockEmbeddings()):
            config = provider.client.meta.config
            with self.subTest(provider=type(provider).__name__):
                self.assertEqual(config.read_timeout, 45)
                # botocore normalises ``max_attempts`` to ``total_max_attempts``
                # (initial try + retries), so 2 configured retries -> 3 total.
                self.assertEqual(config.retries["total_max_attempts"], 3)

    def test_worst_case_call_fits_inside_the_document_budget(self) -> None:
        """read_timeout x total_max_attempts must stay under DOC_TIMEOUT_SECONDS.

        This is the one-line invariant the config comment promises: 45 * 3 =
        135 < 180. If either number drifts, a single hung call can outlast the
        per-document ``asyncio.wait_for`` and the timeout stops meaning anything.

        The bound is read off a live client's normalised config, since botocore
        exposes the raw ``BEDROCK_CONFIG.retries`` as ``{"max_attempts": 2}`` and
        only resolves it to ``total_max_attempts`` once a client is built.
        """
        try:
            from graph_builder import DOC_TIMEOUT_SECONDS
        except RuntimeError as exc:
            self.skipTest(f"graph_builder needs Neo4j config: {exc}")

        config = BedrockLLM().client.meta.config
        read_timeout = config.read_timeout
        total_max_attempts = config.retries["total_max_attempts"]

        self.assertEqual(read_timeout * total_max_attempts, 135)
        self.assertLess(read_timeout * total_max_attempts, DOC_TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
