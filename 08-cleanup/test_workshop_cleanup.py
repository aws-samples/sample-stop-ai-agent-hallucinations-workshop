# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Regression guards for the workshop teardown.

The headline test is :meth:`TestTagScoping.test_untagged_unrelated_roles_are_never_selected`,
which pins bug B6: teardown once deleted IAM roles by account-wide name prefix
and destroyed five unrelated ``AmazonBedrockAgentCoreSDKCodeBuild*`` roles, two
of them in a different region. The fixture below reproduces that exact account
shape and asserts none of those roles is selected.

Run with::

    python -m unittest discover -s 08-cleanup -v
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from botocore.awsrequest import AWSResponse
from botocore.exceptions import ClientError

import workshop_cleanup as wc

#: String methods that decide membership by name shape rather than identity.
_PREFIX_MATCHERS = frozenset(
    {"startswith", "endswith", "removeprefix", "removesuffix"}
)


def _is_delete_attr(attr: str) -> bool:
    """True for method names that destroy the resource they name."""
    return attr.startswith(("delete", "terminate")) or attr == "unlink"


def _calls(node: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _contains_prefix_match(node: ast.AST) -> bool:
    return any(
        isinstance(call.func, ast.Attribute) and call.func.attr in _PREFIX_MATCHERS
        for call in _calls(node)
    )


def _contains_delete(node: ast.AST) -> bool:
    return any(
        isinstance(call.func, ast.Attribute) and _is_delete_attr(call.func.attr)
        for call in _calls(node)
    )


def _prefix_coupled_deletes(tree: ast.AST) -> bool:
    """True if any delete call is guarded by a prefix/suffix match.

    This is the exact shape of bug B6: a name-prefix test whose branch then
    deletes. A lone ``startswith`` used for ordinary string handling is not
    flagged, so the guard can safely sweep the whole repo.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            if _contains_prefix_match(node.test) and (
                any(_contains_delete(stmt) for stmt in node.body)
                or any(_contains_delete(stmt) for stmt in node.orelse)
            ):
                return True
        elif isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            guarded = any(
                _contains_prefix_match(cond)
                for gen in node.generators
                for cond in gen.ifs
            )
            if isinstance(node, ast.DictComp):
                produced = [node.key, node.value]
            else:
                produced = [node.elt]
            if guarded and any(_contains_delete(part) for part in produced):
                return True
    return False


def _strip_ipython_magics(source: str) -> str:
    """Replace line magics and shell escapes so a cell parses as plain Python."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!", "?")) or stripped.endswith("?"):
            lines.append("pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def _repo_root() -> Path:
    root = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
    return Path(root)


def _tracked(root: Path, pattern: str) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", pattern], text=True
    )
    return [line for line in output.splitlines() if line]


def _tracked_sources(root: Path) -> Iterator[tuple[str, str]]:
    """Yield (label, python-source) for every tracked .py and .ipynb code cell."""
    for rel in _tracked(root, "*.py"):
        yield rel, (root / rel).read_text(encoding="utf-8")
    for rel in _tracked(root, "*.ipynb"):
        notebook = json.loads((root / rel).read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = _strip_ipython_magics("".join(cell.get("source", [])))
            yield f"{rel} cell {index}", source

_TMPDIR: tempfile.TemporaryDirectory[str] | None = None


def setUpModule() -> None:
    """Point the local-config paths at a throwaway directory.

    ``wc.CONFIG_FILES`` holds real paths inside this repo. A non-dry test run
    calls the real ``Path.unlink`` on them, so without this the suite would
    delete a developer's live ``.bedrock_agentcore.yaml`` as a side effect of
    running the tests.
    """
    global _TMPDIR
    _TMPDIR = tempfile.TemporaryDirectory()
    wc.CONFIG_FILES = [str(Path(_TMPDIR.name) / "absent.bedrock_agentcore.yaml")]


def tearDownModule() -> None:
    if _TMPDIR is not None:
        _TMPDIR.cleanup()


TAG = {wc.WORKSHOP_TAG_KEY: wc.WORKSHOP_TAG_VALUE}
TAG_KV = [{"Key": wc.WORKSHOP_TAG_KEY, "Value": wc.WORKSHOP_TAG_VALUE}]

#: Roles that actually existed in the account when B6 fired. None was created by
#: this workshop; two are in a different region entirely. All are untagged.
COLLATERAL_ROLES = [
    "AmazonBedrockAgentCoreSDKCodeBuild-us-east-1-2660eddb3d",
    "AmazonBedrockAgentCoreSDKCodeBuild-us-east-1-3dce273434",
    "AmazonBedrockAgentCoreSDKCodeBuild-us-east-1-c423c61358",
    "AmazonBedrockAgentCoreSDKCodeBuild-us-west-2-7e178ca98a",
    "AmazonBedrockAgentCoreSDKCodeBuild-us-west-2-8556fc4504",
    # Pre-existing, verified live, shares the "workshop" name stem.
    "workshop-studio-execution-role",
]


def not_found(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "Fake")


class Recorder:
    """Collects every mutating call so tests can assert nothing was deleted."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


class _SinglePagePaginator:
    """Wrap a fake's ``list_*`` method as a one-page paginator.

    The source now reads every ``list_*`` result through ``get_paginator``. The
    fakes still expose the plain ``list_*`` methods, so a paginator that yields a
    single page built from that method keeps their behaviour identical while
    matching the real client's interface.
    """

    def __init__(self, client: Any, op: str) -> None:
        self._client = client
        self._op = op

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [getattr(self._client, self._op)(**kwargs)]


class FakeIam:
    def __init__(self, roles: dict[str, dict[str, str]], recorder: Recorder) -> None:
        # role name -> tags
        self.roles = roles
        self.rec = recorder

    def get_paginator(self, _op: str) -> FakeIam:
        return self

    def paginate(self) -> list[dict[str, Any]]:
        return [
            {
                "Roles": [
                    {"RoleName": name, "Arn": f"arn:aws:iam::123456789012:role/{name}"}
                    for name in self.roles
                ]
            }
        ]

    def list_role_tags(self, RoleName: str) -> dict[str, Any]:
        return {"Tags": [{"Key": k, "Value": v} for k, v in self.roles[RoleName].items()]}

    def get_role(self, RoleName: str) -> dict[str, Any]:
        if RoleName not in self.roles:
            raise not_found("NoSuchEntity")
        return {"Role": {"RoleName": RoleName}}

    def list_attached_role_policies(self, RoleName: str) -> dict[str, Any]:
        return {"AttachedPolicies": []}

    def list_role_policies(self, RoleName: str) -> dict[str, Any]:
        return {"PolicyNames": []}

    def detach_role_policy(self, **kwargs: Any) -> None:
        self.rec.record("detach_role_policy", **kwargs)

    def delete_role_policy(self, **kwargs: Any) -> None:
        self.rec.record("delete_role_policy", **kwargs)

    def delete_role(self, **kwargs: Any) -> None:
        self.rec.record("delete_role", **kwargs)
        # Deletion must actually take effect in the fake, otherwise the absence
        # probes added for B42 would poll forever against a resource the fake
        # keeps insisting still exists.
        self.roles.pop(kwargs["RoleName"], None)


class FakeAgentCore:
    def __init__(
        self,
        recorder: Recorder,
        memories: list[dict[str, Any]] | None = None,
        tags_by_arn: dict[str, dict[str, str]] | None = None,
        delete_memory_error: Exception | None = None,
    ) -> None:
        self.rec = recorder
        self.memories = memories or []
        self.tags_by_arn = tags_by_arn or {}
        self.delete_memory_error = delete_memory_error

    def get_paginator(self, op: str) -> _SinglePagePaginator:
        return _SinglePagePaginator(self, op)

    def list_memories(self) -> dict[str, Any]:
        return {"memories": self.memories}

    def list_agent_runtimes(self) -> dict[str, Any]:
        return {"agentRuntimes": []}

    def list_gateways(self) -> dict[str, Any]:
        return {"items": []}

    def list_gateway_targets(self, **_kw: Any) -> dict[str, Any]:
        return {"items": []}

    def list_tags_for_resource(self, resourceArn: str) -> dict[str, Any]:
        return {"tags": self.tags_by_arn.get(resourceArn, {})}

    def delete_memory(self, **kwargs: Any) -> None:
        self.rec.record("delete_memory", **kwargs)
        if self.delete_memory_error is not None:
            raise self.delete_memory_error
        # A delete that does not change the fake's state cannot be observed, and
        # observation is exactly what B42's fix requires.
        self.memories = [m for m in self.memories if m["id"] != kwargs["memoryId"]]

    # --- absence probes (B42) -------------------------------------------------

    def get_memory(self, memoryId: str, **_kw: Any) -> dict[str, Any]:
        match = next((m for m in self.memories if m["id"] == memoryId), None)
        if match is None:
            raise not_found("ResourceNotFoundException")
        return {"memory": match}

    def get_agent_runtime(self, **_kw: Any) -> dict[str, Any]:
        raise not_found("ResourceNotFoundException")

    def get_gateway(self, **_kw: Any) -> dict[str, Any]:
        raise not_found("ResourceNotFoundException")

    def get_gateway_target(self, **_kw: Any) -> dict[str, Any]:
        raise not_found("ResourceNotFoundException")


class FakeEmpty:
    """Every lookup reports 'absent'; every mutation is recorded."""

    def __init__(self, recorder: Recorder) -> None:
        self.rec = recorder

    def get_paginator(self, op: str) -> _SinglePagePaginator:
        return _SinglePagePaginator(self, op)

    def get_function(self, **_kw: Any) -> dict[str, Any]:
        raise not_found("ResourceNotFoundException")

    def list_layer_versions(self, **_kw: Any) -> dict[str, Any]:
        return {"LayerVersions": []}

    def get_layer_version(self, **_kw: Any) -> dict[str, Any]:
        raise not_found("ResourceNotFoundException")

    def describe_table(self, **_kw: Any) -> dict[str, Any]:
        raise not_found("ResourceNotFoundException")

    def describe_repositories(self, **_kw: Any) -> dict[str, Any]:
        raise not_found("RepositoryNotFoundException")

    def batch_get_projects(self, **_kw: Any) -> dict[str, Any]:
        return {"projects": []}

    def __getattr__(self, name: str) -> Any:
        def _mutating(**kwargs: Any) -> None:
            self.rec.record(name, **kwargs)

        return _mutating


def make_clients(
    recorder: Recorder,
    *,
    roles: dict[str, dict[str, str]] | None = None,
    agentcore: FakeAgentCore | None = None,
) -> wc.Clients:
    empty = FakeEmpty(recorder)
    return wc.Clients(
        dynamodb=empty,
        iam=FakeIam(roles or {}, recorder),
        lambda_=empty,
        agentcore=agentcore or FakeAgentCore(recorder),
        ecr=empty,
        codebuild=empty,
    )


class TestTagScoping(unittest.TestCase):
    """B6 — deletion must be scoped by tag, never by name prefix."""

    def test_untagged_unrelated_roles_are_never_selected(self) -> None:
        recorder = Recorder()
        clients = make_clients(recorder, roles={name: {} for name in COLLATERAL_ROLES})

        plan = wc.build_plan(clients)
        selected = {c.name for c in plan if c.will_delete}

        for role_name in COLLATERAL_ROLES:
            self.assertNotIn(
                role_name,
                selected,
                msg=f"{role_name} is untagged and unrelated; selecting it is bug B6",
            )

        # And prove it end to end: a real (non-dry) run must issue zero deletes.
        exit_code = wc.run(clients, dry_run=False)
        self.assertNotIn("delete_role", recorder.names)
        self.assertEqual(exit_code, 0, "an account of purely unrelated roles is a clean no-op")

    def test_tagged_role_is_selected_regardless_of_its_name(self) -> None:
        recorder = Recorder()
        clients = make_clients(
            recorder,
            roles={"some-unrelated-name": dict(TAG), **{n: {} for n in COLLATERAL_ROLES}},
        )
        plan = wc.build_plan(clients)
        selected = {c.name for c in plan if c.will_delete}
        self.assertEqual(selected, {"some-unrelated-name"})

        wc.run(clients, dry_run=False)
        self.assertIn(("delete_role", {"RoleName": "some-unrelated-name"}), recorder.calls)

    def test_untagged_workshop_role_blocks_and_exits_nonzero(self) -> None:
        recorder = Recorder()
        clients = make_clients(recorder, roles={wc.LAMBDA_ROLE_NAME: {}})

        plan = wc.build_plan(clients)
        blocked = [c for c in plan if c.selection is wc.Selection.UNTAGGED_BLOCKED]
        self.assertEqual([c.name for c in blocked], [wc.LAMBDA_ROLE_NAME])

        exit_code = wc.run(clients, dry_run=False)
        self.assertEqual(exit_code, 1, "an untagged workshop resource must fail loudly")
        self.assertNotIn("delete_role", recorder.names)

    def test_untaggable_kinds_are_pinned(self) -> None:
        # If this fails, someone widened the tag-gate exemption. Review it.
        self.assertEqual(
            wc.UNTAGGABLE_KINDS,
            frozenset({"lambda-layer-version", "local-config-file"}),
        )

    def test_source_calls_no_prefix_matching_methods(self) -> None:
        """Parse the module and assert no prefix/suffix match is ever *called*.

        An AST walk rather than a text search, so the docstring that quotes the
        old ``startswith`` bug does not trip it.
        """
        tree = ast.parse((Path(__file__).parent / "workshop_cleanup.py").read_text())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for banned in ("startswith", "endswith", "removeprefix", "removesuffix"):
            self.assertNotIn(
                banned,
                called,
                msg=f"{banned}() reintroduces name-based matching into the deletion path",
            )

    def test_source_never_lists_roles_by_path_prefix(self) -> None:
        source = (Path(__file__).parent / "workshop_cleanup.py").read_text()
        self.assertNotIn('PathPrefix="/"', source)

    def test_no_tracked_file_couples_delete_with_prefix_match(self) -> None:
        """Widened B6 guard (V7/V13): sweep the whole repo, not one file.

        The narrow guard above pins the deletion module. This one walks every
        tracked ``.py`` file and every code cell of every tracked ``.ipynb`` so
        a prefix-scoped delete cannot slip back into a sibling module or a
        notebook. It flags the dangerous coupling, a delete reachable inside a
        branch chosen by ``startswith``/``endswith``, rather than any lone
        string call, so ordinary string handling elsewhere is left alone.
        """
        root = _repo_root()
        analyzed = 0
        offenders: list[str] = []
        for label, source in _tracked_sources(root):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            analyzed += 1
            if _prefix_coupled_deletes(tree):
                offenders.append(label)

        self.assertGreater(
            analyzed, 20, "expected to analyze the repo's tracked Python sources"
        )
        self.assertEqual(
            offenders,
            [],
            msg=(
                "these tracked sources delete a resource chosen by a name-prefix "
                f"match, which is bug B6: {offenders}"
            ),
        )


class TestMemory(unittest.TestCase):
    """B5 — Memory must be matched on ``id``; there is no ``memoryName``."""

    #: Exactly the keys the live API returned. Any code reading ``memoryName``
    #: raises KeyError against this, which is the bug.
    LIVE_SHAPE = {
        "arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/workshop_HotelBookingMemory-AbC123",
        "id": "workshop_HotelBookingMemory-AbC123",
        "status": "ACTIVE",
        "createdAt": "2026-07-19",
        "updatedAt": "2026-07-19",
    }

    def test_live_summary_has_no_memory_name_field(self) -> None:
        self.assertNotIn("memoryName", self.LIVE_SHAPE)

    def test_id_matching(self) -> None:
        name = "workshop_HotelBookingMemory"
        self.assertTrue(wc.memory_id_matches_name(f"{name}-AbC123", name))
        self.assertTrue(wc.memory_id_matches_name(name, name))
        # Equality after stripping one suffix, not a prefix test.
        self.assertFalse(wc.memory_id_matches_name(f"{name}Extra-AbC123", name))
        self.assertFalse(wc.memory_id_matches_name("orchestrator_agent_mem-FGnAIw2mUS", name))

    def test_tagged_memory_is_deleted_by_id(self) -> None:
        recorder = Recorder()
        agentcore = FakeAgentCore(
            recorder,
            memories=[self.LIVE_SHAPE],
            tags_by_arn={self.LIVE_SHAPE["arn"]: dict(TAG)},
        )
        clients = make_clients(recorder, agentcore=agentcore)

        exit_code = wc.run(clients, dry_run=False)
        self.assertEqual(exit_code, 0)
        self.assertIn(
            ("delete_memory", {"memoryId": self.LIVE_SHAPE["id"]}), recorder.calls
        )

    def test_unrelated_memory_is_not_touched(self) -> None:
        recorder = Recorder()
        other = {
            "arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/orchestrator_agent_mem-FGnAIw2mUS",
            "id": "orchestrator_agent_mem-FGnAIw2mUS",
            "status": "ACTIVE",
        }
        clients = make_clients(recorder, agentcore=FakeAgentCore(recorder, memories=[other]))

        wc.run(clients, dry_run=False)
        self.assertNotIn("delete_memory", recorder.names)

    def test_memory_deletion_failure_is_not_swallowed(self) -> None:
        recorder = Recorder()
        agentcore = FakeAgentCore(
            recorder,
            memories=[self.LIVE_SHAPE],
            tags_by_arn={self.LIVE_SHAPE["arn"]: dict(TAG)},
            delete_memory_error=not_found("ThrottlingException"),
        )
        clients = make_clients(recorder, agentcore=agentcore)

        exit_code = wc.run(clients, dry_run=False)
        self.assertEqual(exit_code, 1, "a failed delete must exit non-zero, not report success")


class _MultiPagePaginator:
    """Yield a fixed list of pages, ignoring the paginate() arguments."""

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages

    def paginate(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return list(self._pages)


class PaginatedAgentCore(FakeAgentCore):
    """A fake whose ``list_memories`` result spans two pages.

    Only the first page would be read by a caller that ignores ``nextToken``.
    A workshop-tagged memory placed on the second page proves the discover
    function walks every page.
    """

    def __init__(
        self,
        recorder: Recorder,
        pages: list[dict[str, Any]],
        tags_by_arn: dict[str, dict[str, str]],
    ) -> None:
        flattened = [m for page in pages for m in page.get("memories", [])]
        super().__init__(recorder, memories=flattened, tags_by_arn=tags_by_arn)
        self._pages = pages

    def get_paginator(self, op: str) -> Any:
        if op == "list_memories":
            return _MultiPagePaginator(self._pages)
        return super().get_paginator(op)


class TestPagination(unittest.TestCase):
    """B50/V6 — discovery must read every page, not just the first."""

    def test_tagged_memory_on_second_page_is_discovered(self) -> None:
        recorder = Recorder()
        decoy = {
            "arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/orchestrator_agent_mem-Page1AA",
            "id": "orchestrator_agent_mem-Page1AA",
            "status": "ACTIVE",
        }
        target = {
            "arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/workshop_HotelBookingMemory-Page2BB",
            "id": "workshop_HotelBookingMemory-Page2BB",
            "status": "ACTIVE",
        }
        pages = [{"memories": [decoy]}, {"memories": [target]}]
        agentcore = PaginatedAgentCore(recorder, pages, {target["arn"]: dict(TAG)})
        clients = make_clients(recorder, agentcore=agentcore)

        plan = wc.build_plan(clients)
        tagged = [
            c
            for c in plan
            if c.kind == "agentcore-memory" and c.selection is wc.Selection.TAGGED
        ]
        self.assertEqual(
            [c.identifier for c in tagged],
            [target["id"]],
            "a workshop-tagged memory on page 2 must be found; reading only page 1 misses it",
        )


class TestAsyncDeletionWaits(unittest.TestCase):
    """B42 — 'still deleting' and 'failed' are different states.

    The old code reported deletion the moment the API accepted the request, and
    a resource that was merely mid-delete surfaced as exit 1. These tests pin
    both halves: a slow delete must succeed, and a delete that never lands must
    still fail.
    """

    def setUp(self) -> None:
        self._real_timeout = wc.WAIT_TIMEOUT
        wc.WAIT_TIMEOUT = 1.0  # keep the suite fast
        # Patch sleep through the module under test and let unittest restore it,
        # rather than rebinding time.sleep by hand (F6). addCleanup guarantees
        # restoration even if a test raises.
        sleep_patcher = patch("workshop_cleanup.time.sleep")
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def tearDown(self) -> None:
        wc.WAIT_TIMEOUT = self._real_timeout

    def _tagged_memory_clients(self, recorder: Recorder, agentcore: Any) -> wc.Clients:
        return make_clients(recorder, agentcore=agentcore)

    def test_resource_that_lingers_then_disappears_still_exits_zero(self) -> None:
        """A DELETING resource that eventually goes is a success, not a failure."""

        class SlowAgentCore(FakeAgentCore):
            def __init__(self, recorder: Recorder) -> None:
                super().__init__(
                    recorder,
                    memories=[TestMemory.LIVE_SHAPE],
                    tags_by_arn={TestMemory.LIVE_SHAPE["arn"]: dict(TAG)},
                )
                self.probes_before_gone = 3

            def delete_memory(self, **kwargs: Any) -> None:
                # Accept the request but keep reporting the resource for a while,
                # exactly as AgentCore does.
                self.rec.record("delete_memory", **kwargs)

            def get_memory(self, memoryId: str, **_kw: Any) -> dict[str, Any]:
                if self.probes_before_gone > 0:
                    self.probes_before_gone -= 1
                    return {"memory": {**TestMemory.LIVE_SHAPE, "status": "DELETING"}}
                raise not_found("ResourceNotFoundException")

        recorder = Recorder()
        clients = self._tagged_memory_clients(recorder, SlowAgentCore(recorder))

        exit_code = wc.run(clients, dry_run=False)
        self.assertEqual(
            exit_code, 0, "a resource observed DELETING and then gone is a clean teardown"
        )

    def test_resource_that_never_goes_away_exits_nonzero(self) -> None:
        """The exit-code contract must not be weakened: a real leak still fails."""

        class StuckAgentCore(FakeAgentCore):
            def __init__(self, recorder: Recorder) -> None:
                super().__init__(
                    recorder,
                    memories=[TestMemory.LIVE_SHAPE],
                    tags_by_arn={TestMemory.LIVE_SHAPE["arn"]: dict(TAG)},
                )

            def delete_memory(self, **kwargs: Any) -> None:
                self.rec.record("delete_memory", **kwargs)  # accepted, never lands

            def get_memory(self, memoryId: str, **_kw: Any) -> dict[str, Any]:
                return {"memory": {**TestMemory.LIVE_SHAPE, "status": "DELETING"}}

        recorder = Recorder()
        clients = self._tagged_memory_clients(recorder, StuckAgentCore(recorder))

        exit_code = wc.run(clients, dry_run=False)
        self.assertEqual(exit_code, 1, "a resource that never disappears is a real leak")

    def test_delete_already_in_flight_is_not_a_failure(self) -> None:
        """A re-run hitting ConflictException on a DELETING resource must not fail."""

        class ConflictingAgentCore(FakeAgentCore):
            def __init__(self, recorder: Recorder) -> None:
                super().__init__(
                    recorder,
                    memories=[TestMemory.LIVE_SHAPE],
                    tags_by_arn={TestMemory.LIVE_SHAPE["arn"]: dict(TAG)},
                )
                self.gone = False

            def delete_memory(self, **kwargs: Any) -> None:
                self.rec.record("delete_memory", **kwargs)
                raise not_found("ConflictException")

            def get_memory(self, memoryId: str, **_kw: Any) -> dict[str, Any]:
                if not self.gone:
                    self.gone = True
                    return {"memory": {**TestMemory.LIVE_SHAPE, "status": "DELETING"}}
                raise not_found("ResourceNotFoundException")

        recorder = Recorder()
        clients = self._tagged_memory_clients(recorder, ConflictingAgentCore(recorder))

        exit_code = wc.run(clients, dry_run=False)
        self.assertEqual(exit_code, 0, "ConflictException means in flight, not failed")

    def test_every_deletable_kind_is_assigned_a_tier(self) -> None:
        """A kind missing from DELETION_TIERS would be silently skipped."""
        tiered = {kind for _, kinds in wc.DELETION_TIERS for kind in kinds}
        handled = {
            node.pattern.value.value
            for node in ast.walk(
                ast.parse((Path(__file__).parent / "workshop_cleanup.py").read_text())
            )
            if isinstance(node, ast.match_case)
            and isinstance(node.pattern, ast.MatchValue)
            and isinstance(node.pattern.value, ast.Constant)
            and isinstance(node.pattern.value.value, str)
        }
        self.assertTrue(handled, "expected to find the match/case delete handlers")
        self.assertEqual(
            handled - tiered,
            set(),
            "every kind with a delete handler must appear in DELETION_TIERS",
        )


class TestDryRun(unittest.TestCase):
    def test_dry_run_issues_no_mutating_calls(self) -> None:
        recorder = Recorder()
        agentcore = FakeAgentCore(
            recorder,
            memories=[TestMemory.LIVE_SHAPE],
            tags_by_arn={TestMemory.LIVE_SHAPE["arn"]: dict(TAG)},
        )
        clients = make_clients(
            recorder,
            roles={"tagged-role": dict(TAG), **{n: {} for n in COLLATERAL_ROLES}},
            agentcore=agentcore,
        )

        plan = wc.build_plan(clients)
        self.assertTrue([c for c in plan if c.will_delete], "fixture should select something")

        wc.execute_plan(clients, plan, dry_run=True)
        self.assertEqual(recorder.calls, [], "dry run must not call a single mutating API")


class TestIamRetryConfig(unittest.TestCase):
    """F8 — the IAM client must ride out throttling, not abort the whole plan.

    ``discover_roles`` issues one ``list_role_tags`` per role in the account, so
    a busy account throttles it. Without a retry config that ``ClientError``
    escapes the discovery generator and kills ``build_plan`` before it prints
    anything. The adaptive config makes botocore absorb the throttle internally.
    """

    _THROTTLE_BODY = (
        b'<ErrorResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">'
        b"<Error><Type>Sender</Type><Code>Throttling</Code>"
        b"<Message>Rate exceeded</Message></Error>"
        b"<RequestId>req</RequestId></ErrorResponse>"
    )
    _SUCCESS_BODY = (
        b'<ListRolesResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">'
        b"<ListRolesResult><IsTruncated>false</IsTruncated><Roles/>"
        b"</ListRolesResult><ResponseMetadata><RequestId>ok</RequestId>"
        b"</ResponseMetadata></ListRolesResponse>"
    )

    @staticmethod
    def _http_response(status: int, body: bytes) -> AWSResponse:
        response = AWSResponse("https://iam.amazonaws.com/", status, {}, None)
        response._content = body
        return response

    def test_iam_client_absorbs_throttling(self) -> None:
        # Dummy credentials so request signing succeeds; the HTTP layer is faked
        # below, so nothing ever leaves the process.
        env = {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
        with patch.dict(os.environ, env, clear=False):
            iam = wc.Clients.build("us-east-1").iam

            # The config is the adaptive one F8 asks for, whose ceiling of 11
            # total attempts sits well above botocore's legacy default of 5, so
            # a burst of throttling is absorbed rather than escaping discovery.
            self.assertEqual(iam.meta.config.retries["mode"], "adaptive")
            self.assertEqual(iam.meta.config.retries["total_max_attempts"], 11)

            # Return a throttling response first, then success. Without the retry
            # config this call raises ThrottlingException instead of retrying.
            # Sleep is left real (one short adaptive backoff): faking it makes
            # adaptive mode's client-side rate limiter spin on the real clock,
            # which is slower than letting the single backoff elapse.
            sends = {"count": 0}

            def fake_send(_self: Any, _request: Any) -> AWSResponse:
                sends["count"] += 1
                if sends["count"] == 1:
                    return self._http_response(400, self._THROTTLE_BODY)
                return self._http_response(200, self._SUCCESS_BODY)

            with patch("botocore.httpsession.URLLib3Session.send", fake_send):
                result = iam.list_roles()

        self.assertEqual(result["Roles"], [])
        self.assertEqual(
            sends["count"],
            2,
            "the client should retry past the throttle and then succeed",
        )


class FakeBusyDynamo:
    """A DynamoDB table stuck in ``CREATING``: ``delete_table`` always rejects.

    ``delete_table`` on a table that is still being created raises
    ``ResourceInUseException`` — the table is fully present and the delete did
    not happen (F10).
    """

    def __init__(
        self, recorder: Recorder, table_name: str, tags: dict[str, str]
    ) -> None:
        self.rec = recorder
        self.table_name = table_name
        self.tags = tags
        self.arn = f"arn:aws:dynamodb:us-east-1:123456789012:table/{table_name}"

    def describe_table(self, TableName: str) -> dict[str, Any]:
        if TableName != self.table_name:
            raise not_found("ResourceNotFoundException")
        return {"Table": {"TableArn": self.arn, "TableStatus": "CREATING"}}

    def list_tags_of_resource(self, ResourceArn: str) -> dict[str, Any]:
        return {"Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()]}

    def delete_table(self, TableName: str) -> None:
        self.rec.record("delete_table", TableName=TableName)
        raise not_found("ResourceInUseException")


class TestResourceInUseRetry(unittest.TestCase):
    """F10 — ResourceInUseException means the delete did not happen; retry it.

    The old code classified it as an in-flight delete, added the table to
    ``deleted``, and polled it for the full timeout. The table was never on its
    way out, so that timed out on a resource that is still fully present.
    """

    def test_resource_in_use_is_retried_and_never_reported_deleted(self) -> None:
        recorder = Recorder()
        clients = make_clients(recorder)
        clients.dynamodb = FakeBusyDynamo(recorder, wc.HOTELS_TABLE, dict(TAG))

        plan = wc.build_plan(clients)
        selected = [c for c in plan if c.will_delete]
        self.assertEqual(
            [c.name for c in selected],
            [wc.HOTELS_TABLE],
            "the tagged CREATING table should be selected for deletion",
        )

        with patch("workshop_cleanup.time.sleep"):
            failures = wc.execute_plan(clients, plan, dry_run=False)

        # The delete was retried, not abandoned after the first rejection.
        delete_calls = [name for name in recorder.names if name == "delete_table"]
        self.assertEqual(len(delete_calls), wc._DELETE_MAX_ATTEMPTS)

        # And the busy table is a genuine ResourceInUseException failure — never
        # misreported as an in-flight delete that gets polled to absence. That
        # path would surface a "still present ... deadline expired" failure after
        # a single delete call, so this pins the corrected classification.
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].kind, "dynamodb-table")
        self.assertEqual(failures[0].name, wc.HOTELS_TABLE)
        self.assertIn("ResourceInUseException", failures[0].error)


if __name__ == "__main__":
    unittest.main()
