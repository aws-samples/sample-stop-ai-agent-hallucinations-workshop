# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tag-scoped teardown for the AgentCore workshop (Modules 6 and 7).

Safety contract
---------------
1. **Nothing is ever selected by name prefix.** A resource is deleted only when
   it carries the workshop tag ``WorkshopResource=stop-ai-agent-hallucinations``.
   The single exception is Lambda layer versions, which AWS does not allow to be
   tagged; those are matched by *exact* name and are listed explicitly in
   :data:`UNTAGGABLE_KINDS` so the exception is auditable.
2. **A resource that exists under a workshop name but carries no workshop tag is
   never deleted.** It is reported as ``UNTAGGED_BLOCKED`` and makes the run exit
   non-zero, because it is either someone else's resource or a deployment that
   forgot to tag.
3. **No failure is swallowed.** Every error is recorded and forces a non-zero
   exit. A cleanup script that lies about success is worse than one that crashes.
4. **Success is observed, never assumed.** Accepting a delete request is not the
   same as the resource being gone. Every tier is polled until the describe call
   reports the resource absent, and only then does the next tier start. ``DELETING``
   means keep waiting; absence means done; anything still standing when the
   timeout expires is a real failure and is named in the output. See bug B42.

Usage::

    python workshop_cleanup.py              # dry run by default: print the plan, touch nothing
    python workshop_cleanup.py --dry-run    # same as above, made explicit
    python workshop_cleanup.py --yes        # execute the plan and delete

Background: an earlier version of this teardown deleted IAM roles by
account-wide name prefix and destroyed five unrelated roles, two of them in a
different region. See ``verify.md`` bug B6.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

WORKSHOP_TAG_KEY = "WorkshopResource"
WORKSHOP_TAG_VALUE = "stop-ai-agent-hallucinations"

REGION = os.environ.get("AWS_REGION", "us-east-1")

HOTELS_TABLE = "workshop-Hotels"
BOOKINGS_TABLE = "workshop-Bookings"
STEERING_RULES_TABLE = "workshop-SteeringRules"
TABLE_NAMES = [HOTELS_TABLE, BOOKINGS_TABLE, STEERING_RULES_TABLE]

LAMBDA_ROLE_NAME = "workshop-LambdaExecutionRole"
AGENTCORE_ROLE_NAME = "workshop-AgentCoreExecutionRole"
ROLE_NAMES = [LAMBDA_ROLE_NAME, AGENTCORE_ROLE_NAME]

GATEWAY_NAME = "HotelBookingGateway"
RUNTIME_NAME = "HotelBookingAgent"
MEMORY_RUNTIME_NAME = "HotelBookingAgentWithMemory"
RUNTIME_NAMES = [RUNTIME_NAME, MEMORY_RUNTIME_NAME]
MEMORY_NAME = "workshop_HotelBookingMemory"

LAMBDA_TOOLS = [
    "search_available_hotels",
    "book_hotel",
    "get_booking",
    "process_payment",
    "confirm_booking",
    "cancel_booking",
    "validate_booking_rules",
    "query_knowledge_graph",
]
LAMBDA_FUNCTIONS = [f"hotel-booking-{tool}" for tool in LAMBDA_TOOLS]
LAMBDA_LAYER_NAME = "workshop-neo4j-driver"

ECR_REPOS = [f"bedrock-agentcore-{name.lower()}" for name in RUNTIME_NAMES]
CODEBUILD_PROJECTS = [f"{repo}-builder" for repo in ECR_REPOS]

#: Exact config files this workshop creates. Modules 6 and 7 each run the starter
#: toolkit from their own directory, and the toolkit writes ``.bedrock_agentcore.yaml``
#: into that directory.
#:
#: This was previously a glob list that included ``~/.bedrock_agentcore*.yaml``. HOME is
#: shared with every other AgentCore project on the machine, so running this teardown
#: destroyed unrelated local config. That is bug B6's mistake on the filesystem instead
#: of in IAM: matching by name shape across a shared namespace. See bug B45.
#:
#: Exact repo-relative paths only. Never a glob, and never anything under HOME.
_REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILES = [
    str(_REPO_ROOT / "06-agentcore-boto3-demo" / ".bedrock_agentcore.yaml"),
    str(_REPO_ROOT / "07-agentcore-memory-demo" / ".bedrock_agentcore.yaml"),
]


class Selection(StrEnum):
    """Why a candidate is or is not going to be deleted."""

    TAGGED = "TAGGED"
    UNTAGGED_BLOCKED = "UNTAGGED_BLOCKED"
    ABSENT = "ABSENT"
    UNTAGGABLE_EXACT_NAME = "UNTAGGABLE_EXACT_NAME"


#: Resource kinds AWS does not support tagging on. Deletion for these falls back
#: to an *exact* name match. Keep this list as short as the API allows; a unit
#: test pins its contents so the exception cannot quietly grow.
UNTAGGABLE_KINDS = frozenset({"lambda-layer-version", "local-config-file"})

#: Selections that authorise a delete call.
DELETABLE = frozenset({Selection.TAGGED, Selection.UNTAGGABLE_EXACT_NAME})

#: How long to wait for one tier of resources to actually disappear, in seconds.
WAIT_TIMEOUT = 600.0

#: Deletion order. Every tier is deleted, then polled to actual absence, before
#: the next tier is touched. The order is a dependency order, and getting it
#: wrong is exactly what produced bug B42: gateway targets were still deleting
#: when ``delete_gateway`` fired, and runtimes were still ``DELETING`` when the
#: roles they assume were removed.
DELETION_TIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # AgentCore plane first: runtimes and gateways are the things that hold
    # references to everything else.
    ("agentcore", ("agentcore-runtime", "agentcore-gateway", "agentcore-memory")),
    # Then what those runtimes called and stored.
    (
        "compute-and-data",
        (
            "lambda-function",
            "lambda-layer-version",
            "dynamodb-table",
            "ecr-repository",
            "codebuild-project",
        ),
    ),
    # IAM last of the AWS tiers: nothing may still be assuming these roles.
    ("iam", ("iam-role",)),
    # Local files are not AWS resources and cannot race.
    ("local", ("local-config-file",)),
)


@dataclass
class Candidate:
    """One resource the teardown considered, and the verdict on it."""

    kind: str
    name: str
    selection: Selection
    identifier: str | None = None
    arn: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    detail: str = ""

    @property
    def will_delete(self) -> bool:
        return self.selection in DELETABLE

    def describe(self) -> str:
        target = self.identifier or self.name
        reason = {
            Selection.TAGGED: (
                f"tagged {WORKSHOP_TAG_KEY}={WORKSHOP_TAG_VALUE}"
            ),
            Selection.UNTAGGED_BLOCKED: (
                "EXISTS but is NOT tagged — refusing to delete"
            ),
            Selection.ABSENT: "not found",
            Selection.UNTAGGABLE_EXACT_NAME: (
                "matched exactly, never by name shape or tag"
            ),
        }[self.selection]
        line = f"{'DELETE' if self.will_delete else 'SKIP  '}  {self.kind:<24} {target:<52} {reason}"
        return f"{line}\n{'':>10}{self.detail}" if self.detail else line


@dataclass
class Failure:
    kind: str
    name: str
    error: str


def has_workshop_tag(tags: dict[str, str]) -> bool:
    return tags.get(WORKSHOP_TAG_KEY) == WORKSHOP_TAG_VALUE


def _kv_list_to_dict(items: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    return {item[key]: item[value] for item in items}


def memory_id_matches_name(memory_id: str, memory_name: str) -> bool:
    """AgentCore Memory ids have the form ``<name>-<suffix>``.

    ``list_memories`` returns no name field at all — the ``MemorySummary`` shape
    is ``arn, id, status, createdAt, updatedAt, managedByResourceArn``. Reading
    ``m["memoryName"]`` raises ``KeyError`` on every item, which is bug B5.

    The comparison below strips exactly one trailing ``-<suffix>`` and requires
    the remainder to equal the name. It is an equality test, not a prefix test:
    ``workshop_HotelBookingMemoryExtra-abc`` does not match.
    """
    if memory_id == memory_name:
        return True
    head, sep, _suffix = memory_id.rpartition("-")
    return bool(sep) and head == memory_name


@dataclass
class Clients:
    """The AWS clients the teardown needs, grouped so tests can inject fakes."""

    dynamodb: Any
    iam: Any
    lambda_: Any
    agentcore: Any
    ecr: Any
    codebuild: Any
    # The region these clients were built for. Carried here so the plan header
    # cannot disagree with the clients that actually perform the deletions.
    region: str = REGION

    @classmethod
    def build(cls, region: str = REGION) -> Clients:
        return cls(
            dynamodb=boto3.client("dynamodb", region_name=region),
            # Adaptive retries absorb the IAM throttling that ``discover_roles``
            # can provoke: it calls ``list_role_tags`` once per role in the
            # account (see F8). Known limitation for large accounts: this stays
            # O(roles) API calls. The scalable fix is to migrate discovery onto
            # ``resourcegroupstaggingapi.get_resources(TagFilters=...)``, which
            # returns tagged resources in one paginated call; it is intentionally
            # not done here because workshop attendees run in sandbox accounts.
            iam=boto3.client(
                "iam",
                config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
            ),
            lambda_=boto3.client("lambda", region_name=region),
            agentcore=boto3.client("bedrock-agentcore-control", region_name=region),
            ecr=boto3.client("ecr", region_name=region),
            codebuild=boto3.client("codebuild", region_name=region),
            region=region,
        )


def _error_code_in(error: ClientError, *codes: str) -> bool:
    return error.response.get("Error", {}).get("Code") in codes


# --------------------------------------------------------------------------
# Discovery — every function below returns candidates without deleting anything
# --------------------------------------------------------------------------


def discover_memories(clients: Clients) -> Iterator[Candidate]:
    paginator = clients.agentcore.get_paginator("list_memories")
    memories = [
        memory
        for page in paginator.paginate()
        for memory in page.get("memories", [])
    ]
    matches = [m for m in memories if memory_id_matches_name(m["id"], MEMORY_NAME)]
    if not matches:
        yield Candidate("agentcore-memory", MEMORY_NAME, Selection.ABSENT)
        return
    for memory in matches:
        tags = clients.agentcore.list_tags_for_resource(
            resourceArn=memory["arn"]
        ).get("tags", {})
        yield Candidate(
            kind="agentcore-memory",
            name=MEMORY_NAME,
            selection=Selection.TAGGED if has_workshop_tag(tags) else Selection.UNTAGGED_BLOCKED,
            identifier=memory["id"],
            arn=memory["arn"],
            tags=tags,
            detail=f"id {memory['id']} -> name {memory['id'].rpartition('-')[0]}",
        )


def discover_runtimes(clients: Clients) -> Iterator[Candidate]:
    paginator = clients.agentcore.get_paginator("list_agent_runtimes")
    runtimes = [
        runtime
        for page in paginator.paginate()
        for runtime in page.get("agentRuntimes", [])
    ]
    by_name = {rt.get("agentRuntimeName"): rt for rt in runtimes}
    for name in RUNTIME_NAMES:
        runtime = by_name.get(name)
        if runtime is None:
            yield Candidate("agentcore-runtime", name, Selection.ABSENT)
            continue
        tags = clients.agentcore.list_tags_for_resource(
            resourceArn=runtime["agentRuntimeArn"]
        ).get("tags", {})
        yield Candidate(
            kind="agentcore-runtime",
            name=name,
            selection=Selection.TAGGED if has_workshop_tag(tags) else Selection.UNTAGGED_BLOCKED,
            identifier=runtime["agentRuntimeId"],
            arn=runtime["agentRuntimeArn"],
            tags=tags,
        )


def discover_gateways(clients: Clients) -> Iterator[Candidate]:
    paginator = clients.agentcore.get_paginator("list_gateways")
    gateways = [
        gateway
        for page in paginator.paginate()
        for gateway in page.get("items", [])
    ]
    match = next((g for g in gateways if g.get("name") == GATEWAY_NAME), None)
    if match is None:
        yield Candidate("agentcore-gateway", GATEWAY_NAME, Selection.ABSENT)
        return
    # GatewaySummary carries no arn, so the arn has to be fetched before tags
    # can be read.
    arn = clients.agentcore.get_gateway(gatewayIdentifier=match["gatewayId"])["gatewayArn"]
    tags = clients.agentcore.list_tags_for_resource(resourceArn=arn).get("tags", {})
    yield Candidate(
        kind="agentcore-gateway",
        name=GATEWAY_NAME,
        selection=Selection.TAGGED if has_workshop_tag(tags) else Selection.UNTAGGED_BLOCKED,
        identifier=match["gatewayId"],
        arn=arn,
        tags=tags,
    )


def discover_lambda_functions(clients: Clients) -> Iterator[Candidate]:
    for name in LAMBDA_FUNCTIONS:
        try:
            response = clients.lambda_.get_function(FunctionName=name)
        except ClientError as exc:
            if _error_code_in(exc, "ResourceNotFoundException"):
                yield Candidate("lambda-function", name, Selection.ABSENT)
                continue
            raise
        tags = response.get("Tags", {}) or {}
        yield Candidate(
            kind="lambda-function",
            name=name,
            selection=Selection.TAGGED if has_workshop_tag(tags) else Selection.UNTAGGED_BLOCKED,
            arn=response["Configuration"]["FunctionArn"],
            tags=tags,
        )


def discover_lambda_layers(clients: Clients) -> Iterator[Candidate]:
    try:
        paginator = clients.lambda_.get_paginator("list_layer_versions")
        versions = [
            version
            for page in paginator.paginate(LayerName=LAMBDA_LAYER_NAME)
            for version in page.get("LayerVersions", [])
        ]
    except ClientError as exc:
        if _error_code_in(exc, "ResourceNotFoundException"):
            versions = []
        else:
            raise
    if not versions:
        yield Candidate("lambda-layer-version", LAMBDA_LAYER_NAME, Selection.ABSENT)
        return
    for version in versions:
        yield Candidate(
            kind="lambda-layer-version",
            name=LAMBDA_LAYER_NAME,
            selection=Selection.UNTAGGABLE_EXACT_NAME,
            identifier=str(version["Version"]),
            detail="Lambda layer versions cannot carry tags; exact name match only",
        )


def discover_tables(clients: Clients) -> Iterator[Candidate]:
    for name in TABLE_NAMES:
        try:
            arn = clients.dynamodb.describe_table(TableName=name)["Table"]["TableArn"]
        except ClientError as exc:
            if _error_code_in(exc, "ResourceNotFoundException"):
                yield Candidate("dynamodb-table", name, Selection.ABSENT)
                continue
            raise
        tags = _kv_list_to_dict(
            clients.dynamodb.list_tags_of_resource(ResourceArn=arn).get("Tags", []),
            "Key",
            "Value",
        )
        yield Candidate(
            kind="dynamodb-table",
            name=name,
            selection=Selection.TAGGED if has_workshop_tag(tags) else Selection.UNTAGGED_BLOCKED,
            arn=arn,
            tags=tags,
        )


def discover_roles(clients: Clients) -> Iterator[Candidate]:
    """Select IAM roles **purely by tag**.

    This is the B6 fix. The old code did
    ``if role["RoleName"].startswith("AmazonBedrockAgentCoreSDKCodeBuild")``
    over every role in the account, which is global and reached other regions.
    Nothing here looks at name shape when deciding to delete; the workshop's own
    role names are only used to *warn* about an untagged leftover.
    """
    seen: set[str] = set()
    paginator = clients.iam.get_paginator("list_roles")
    for page in paginator.paginate():
        for role in page.get("Roles", []):
            role_name = role["RoleName"]
            tags = _kv_list_to_dict(
                clients.iam.list_role_tags(RoleName=role_name).get("Tags", []),
                "Key",
                "Value",
            )
            if not has_workshop_tag(tags):
                continue
            seen.add(role_name)
            yield Candidate(
                kind="iam-role",
                name=role_name,
                selection=Selection.TAGGED,
                arn=role.get("Arn"),
                tags=tags,
            )

    for role_name in ROLE_NAMES:
        if role_name in seen:
            continue
        try:
            clients.iam.get_role(RoleName=role_name)
        except ClientError as exc:
            if _error_code_in(exc, "NoSuchEntity", "NoSuchEntityException"):
                yield Candidate("iam-role", role_name, Selection.ABSENT)
                continue
            raise
        yield Candidate(
            kind="iam-role",
            name=role_name,
            selection=Selection.UNTAGGED_BLOCKED,
            detail="workshop role name but no workshop tag — tag it at creation or remove it by hand",
        )


def discover_ecr_repos(clients: Clients) -> Iterator[Candidate]:
    for name in ECR_REPOS:
        try:
            repo = clients.ecr.describe_repositories(repositoryNames=[name])["repositories"][0]
        except ClientError as exc:
            if _error_code_in(exc, "RepositoryNotFoundException"):
                yield Candidate("ecr-repository", name, Selection.ABSENT)
                continue
            raise
        tags = _kv_list_to_dict(
            clients.ecr.list_tags_for_resource(
                resourceArn=repo["repositoryArn"]
            ).get("tags", []),
            "Key",
            "Value",
        )
        yield Candidate(
            kind="ecr-repository",
            name=name,
            selection=Selection.TAGGED if has_workshop_tag(tags) else Selection.UNTAGGED_BLOCKED,
            arn=repo["repositoryArn"],
            tags=tags,
        )


def discover_codebuild_projects(clients: Clients) -> Iterator[Candidate]:
    found = {
        project["name"]: project
        for project in clients.codebuild.batch_get_projects(
            names=CODEBUILD_PROJECTS
        ).get("projects", [])
    }
    for name in CODEBUILD_PROJECTS:
        project = found.get(name)
        if project is None:
            # delete_project is idempotent and returns success for projects that
            # never existed, which is how the old code produced false
            # "Deleted ..." lines. Report absence instead of calling it.
            yield Candidate("codebuild-project", name, Selection.ABSENT)
            continue
        tags = _kv_list_to_dict(project.get("tags", []), "key", "value")
        yield Candidate(
            kind="codebuild-project",
            name=name,
            selection=Selection.TAGGED if has_workshop_tag(tags) else Selection.UNTAGGED_BLOCKED,
            arn=project.get("arn"),
            tags=tags,
        )


def discover_local_config(_clients: Clients) -> Iterator[Candidate]:
    # Report each config path independently: an absent one is reported ABSENT by
    # its own name, so a missing 07 file is not hidden behind a present 06 file
    # (or vice versa). The old code reported only CONFIG_FILES[0] when none
    # existed, naming the wrong path for the 07-only case.
    for path in CONFIG_FILES:
        if os.path.isfile(path):
            yield Candidate(
                kind="local-config-file",
                name=path,
                selection=Selection.UNTAGGABLE_EXACT_NAME,
                detail="local file at a fixed repo-relative path, not an AWS resource",
            )
        else:
            yield Candidate("local-config-file", path, Selection.ABSENT)


DISCOVERERS = (
    discover_memories,
    discover_runtimes,
    discover_gateways,
    discover_lambda_functions,
    discover_lambda_layers,
    discover_tables,
    discover_roles,
    discover_ecr_repos,
    discover_codebuild_projects,
    discover_local_config,
)


def build_plan(clients: Clients) -> list[Candidate]:
    """Enumerate every candidate. Read-only: this never deletes anything."""
    return [candidate for discover in DISCOVERERS for candidate in discover(clients)]


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Absence probes — "is it actually gone yet?"
#
# Each probe returns None when the resource is gone, or a short human-readable
# state string when it is still there. A state string is not a failure on its
# own: ``DELETING`` simply means poll again. Only a state that survives the
# timeout is a failure. Collapsing those two into one outcome is bug B42.
# --------------------------------------------------------------------------

#: Error codes every service uses to say "that does not exist".
_GONE_CODES = (
    "ResourceNotFoundException",
    "RepositoryNotFoundException",
    "NoSuchEntity",
    "NoSuchEntityException",
)


def _probe_state(clients: Clients, candidate: Candidate) -> str | None:
    """Return None if the resource is gone, else a description of its state."""
    try:
        match candidate.kind:
            case "agentcore-memory":
                memory = clients.agentcore.get_memory(memoryId=candidate.identifier)
                return f"status={memory.get('memory', {}).get('status', 'UNKNOWN')}"
            case "agentcore-runtime":
                runtime = clients.agentcore.get_agent_runtime(
                    agentRuntimeId=candidate.identifier
                )
                return f"status={runtime.get('status', 'UNKNOWN')}"
            case "agentcore-gateway":
                gateway = clients.agentcore.get_gateway(
                    gatewayIdentifier=candidate.identifier
                )
                return f"status={gateway.get('status', 'UNKNOWN')}"
            case "lambda-function":
                clients.lambda_.get_function(FunctionName=candidate.name)
                return "present"
            case "lambda-layer-version":
                clients.lambda_.get_layer_version(
                    LayerName=candidate.name, VersionNumber=int(candidate.identifier)
                )
                return "present"
            case "dynamodb-table":
                table = clients.dynamodb.describe_table(TableName=candidate.name)
                return f"status={table['Table'].get('TableStatus', 'UNKNOWN')}"
            case "iam-role":
                clients.iam.get_role(RoleName=candidate.name)
                return "present"
            case "ecr-repository":
                clients.ecr.describe_repositories(repositoryNames=[candidate.name])
                return "present"
            case "codebuild-project":
                # batch_get_projects reports absence in the payload, not by raising.
                found = clients.codebuild.batch_get_projects(
                    names=[candidate.name]
                ).get("projects", [])
                return "present" if found else None
            case "local-config-file":
                return "present" if Path(candidate.name).exists() else None
            case unknown:
                raise ValueError(f"no absence probe for kind {unknown!r}")
    except ClientError as exc:
        if _error_code_in(exc, *_GONE_CODES):
            return None
        raise


def wait_until_gone(
    clients: Clients,
    candidates: list[Candidate],
    *,
    timeout: float | None = None,
    label: str = "",
) -> list[Failure]:
    """Poll until every candidate is absent. Report only what was observed.

    Backs off from 2s to 15s. Returns a Failure per resource still standing when
    the timeout expires, naming the resource and the state it was last seen in.
    Returning an empty list means every resource was *observed* absent, which is
    the whole point: the previous code reported success the moment the API
    accepted the delete request.
    """
    # Read the module global at call time, not at def time, so tests can shorten it.
    timeout = WAIT_TIMEOUT if timeout is None else timeout
    # One shared deadline for the whole tier (F5). Candidates within a tier are
    # independent by construction, so a per-candidate deadline let a genuinely
    # stuck tier take len(candidates) x timeout to fail instead of one timeout.
    deadline = time.monotonic() + timeout
    failures: list[Failure] = []
    for candidate in candidates:
        target = candidate.identifier or candidate.name
        delay = 2.0
        state: str | None = "unknown"
        while time.monotonic() < deadline:
            try:
                state = _probe_state(clients, candidate)
            except (ClientError, BotoCoreError, ValueError) as exc:
                failures.append(
                    Failure(candidate.kind, candidate.name, f"probe failed: {exc}")
                )
                state = None  # stop polling; the failure is already recorded
                break
            if state is None:
                print(f"  gone    {candidate.kind} {target}")
                break
            time.sleep(delay)
            delay = min(delay * 1.5, 15.0)
        else:
            failures.append(
                Failure(
                    candidate.kind,
                    candidate.name,
                    f"still present when tier {label!r} deadline expired "
                    f"({timeout:.0f}s, {state}) — this is a real leak, not a race",
                )
            )
            print(
                f"  TIMEOUT {candidate.kind} {target} still present ({state})",
                file=sys.stderr,
            )
    return failures


def _delete_role(clients: Clients, role_name: str) -> None:
    attached = clients.iam.list_attached_role_policies(RoleName=role_name)
    for policy in attached.get("AttachedPolicies", []):
        clients.iam.detach_role_policy(RoleName=role_name, PolicyArn=policy["PolicyArn"])
    inline = clients.iam.list_role_policies(RoleName=role_name)
    for policy_name in inline.get("PolicyNames", []):
        clients.iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
    clients.iam.delete_role(RoleName=role_name)


def _wait_gateway_targets_gone(
    clients: Clients, gateway_id: str, target_ids: list[str], timeout: float
) -> None:
    """Block until every gateway target is absent.

    ``delete_gateway`` fails while any target is still deleting. The old code
    fired the two calls back to back and reported the resulting error as a
    cleanup failure, which is half of bug B42.
    """
    deadline = time.monotonic() + timeout
    delay = 2.0
    remaining = list(target_ids)
    while remaining and time.monotonic() < deadline:
        still_there = []
        for target_id in remaining:
            try:
                clients.agentcore.get_gateway_target(
                    gatewayIdentifier=gateway_id, targetId=target_id
                )
            except ClientError as exc:
                if _error_code_in(exc, *_GONE_CODES):
                    continue
                raise
            still_there.append(target_id)
        remaining = still_there
        if remaining:
            time.sleep(delay)
            delay = min(delay * 1.5, 15.0)
    if remaining:
        raise TimeoutError(
            f"gateway targets {remaining} still present after {timeout:.0f}s; "
            "refusing to call delete_gateway on a gateway that still has targets"
        )


def _delete_gateway(clients: Clients, gateway_id: str) -> None:
    paginator = clients.agentcore.get_paginator("list_gateway_targets")
    targets = [
        target
        for page in paginator.paginate(gatewayIdentifier=gateway_id)
        for target in page.get("items", [])
    ]
    target_ids = [target["targetId"] for target in targets]
    for target_id in target_ids:
        clients.agentcore.delete_gateway_target(
            gatewayIdentifier=gateway_id, targetId=target_id
        )
    # Observe the targets actually gone before touching the gateway.
    _wait_gateway_targets_gone(clients, gateway_id, target_ids, WAIT_TIMEOUT)
    clients.agentcore.delete_gateway(gatewayIdentifier=gateway_id)


def _delete(clients: Clients, candidate: Candidate) -> None:
    match candidate.kind:
        case "agentcore-memory":
            clients.agentcore.delete_memory(memoryId=candidate.identifier)
        case "agentcore-runtime":
            clients.agentcore.delete_agent_runtime(agentRuntimeId=candidate.identifier)
        case "agentcore-gateway":
            _delete_gateway(clients, candidate.identifier)
        case "lambda-function":
            clients.lambda_.delete_function(FunctionName=candidate.name)
        case "lambda-layer-version":
            clients.lambda_.delete_layer_version(
                LayerName=candidate.name, VersionNumber=int(candidate.identifier)
            )
        case "dynamodb-table":
            clients.dynamodb.delete_table(TableName=candidate.name)
        case "iam-role":
            _delete_role(clients, candidate.name)
        case "ecr-repository":
            clients.ecr.delete_repository(repositoryName=candidate.name, force=True)
        case "codebuild-project":
            clients.codebuild.delete_project(name=candidate.name)
        case "local-config-file":
            Path(candidate.name).unlink(missing_ok=True)
        case unknown:
            raise ValueError(f"no delete handler for kind {unknown!r}")


#: Error codes meaning "a delete is already in flight for this resource".
#: These are not failures. The resource is on its way out, and the wait below is
#: what decides whether it actually left. Reporting them as failures is the other
#: half of bug B42: a re-run saw runtimes in ``DELETING`` and exited 1 on a race.
_ALREADY_DELETING_CODES = ("ConflictException",)

#: Error codes meaning "the resource is busy, so the delete did NOT happen".
#: DynamoDB raises ``ResourceInUseException`` from ``delete_table`` on a table
#: still in ``CREATING`` state: the table is fully present and the request was
#: rejected. This is the opposite of an in-flight delete, so treating it as one
#: (adding it to ``deleted`` and polling for absence) waits out the whole timeout
#: on a resource that was never asked to leave. The right response is to wait for
#: the resource to settle and retry the delete. See F10.
_RETRYABLE_DELETE_CODES = ("ResourceInUseException",)

#: How many times to attempt a delete that keeps failing with a busy code, and
#: how long to wait between attempts.
_DELETE_MAX_ATTEMPTS = 5
_DELETE_RETRY_DELAY = 5.0


def _delete_with_busy_retry(clients: Clients, candidate: Candidate) -> None:
    """Delete a candidate, retrying while it reports itself busy.

    Retries only on :data:`_RETRYABLE_DELETE_CODES`, where the resource is still
    present and the delete did not happen (F10). Every other error, and a busy
    state that outlives the attempts, is raised for the caller to classify.
    """
    for attempt in range(1, _DELETE_MAX_ATTEMPTS + 1):
        try:
            _delete(clients, candidate)
            return
        except ClientError as exc:
            if not _error_code_in(exc, *_RETRYABLE_DELETE_CODES):
                raise
            if attempt == _DELETE_MAX_ATTEMPTS:
                raise
            target = candidate.identifier or candidate.name
            print(
                f"  busy    {candidate.kind} {target} not ready to delete "
                f"(attempt {attempt}/{_DELETE_MAX_ATTEMPTS}); retrying"
            )
            time.sleep(_DELETE_RETRY_DELAY)


def execute_plan(
    clients: Clients, plan: list[Candidate], *, dry_run: bool
) -> list[Failure]:
    """Delete in dependency order, waiting for actual absence between tiers.

    Errors are collected, never swallowed. A tier is not considered done until
    every resource in it has been *observed* absent, so a later tier can never
    race ahead of a resource that still depends on it.
    """
    failures: list[Failure] = []
    if dry_run:
        return failures

    selected = [c for c in plan if c.will_delete]
    known_kinds = {kind for _, kinds in DELETION_TIERS for kind in kinds}
    unhandled = {c.kind for c in selected} - known_kinds
    if unhandled:
        # Fail loudly rather than silently skipping a tier we forgot to list.
        raise ValueError(f"kinds missing from DELETION_TIERS: {sorted(unhandled)}")

    for label, kinds in DELETION_TIERS:
        tier = [c for c in selected if c.kind in kinds]
        if not tier:
            continue
        print(f"\n-- tier {label}: {len(tier)} resource(s)")
        deleted: list[Candidate] = []
        for candidate in tier:
            target = candidate.identifier or candidate.name
            try:
                _delete_with_busy_retry(clients, candidate)
            except ClientError as exc:
                if _error_code_in(exc, *_GONE_CODES):
                    print(f"  gone    {candidate.kind} {target} (already absent)")
                    continue
                if _error_code_in(exc, *_ALREADY_DELETING_CODES):
                    print(f"  pending {candidate.kind} {target} (delete already in flight)")
                    deleted.append(candidate)
                    continue
                failures.append(Failure(candidate.kind, candidate.name, str(exc)))
                print(f"  FAILED  {candidate.kind} {candidate.name}: {exc}", file=sys.stderr)
            except (BotoCoreError, OSError, ValueError) as exc:
                failures.append(Failure(candidate.kind, candidate.name, str(exc)))
                print(f"  FAILED  {candidate.kind} {candidate.name}: {exc}", file=sys.stderr)
            else:
                print(f"  delete requested: {candidate.kind} {target}")
                deleted.append(candidate)

        if deleted:
            print(f"  waiting for tier {label} to be observed gone...")
            failures.extend(wait_until_gone(clients, deleted, label=label))

    return failures


def print_plan(plan: list[Candidate], *, dry_run: bool, region: str) -> None:
    header = "DRY RUN — nothing will be deleted" if dry_run else "TEARDOWN PLAN"
    print("=" * 100)
    print(f"{header}   region={region}   gate={WORKSHOP_TAG_KEY}={WORKSHOP_TAG_VALUE}")
    print("=" * 100)
    for candidate in plan:
        print(candidate.describe())
    selected = [c for c in plan if c.will_delete]
    blocked = [c for c in plan if c.selection is Selection.UNTAGGED_BLOCKED]
    print("-" * 100)
    print(f"selected for deletion: {len(selected)}")
    print(f"blocked (present, untagged): {len(blocked)}")
    print(f"absent: {sum(1 for c in plan if c.selection is Selection.ABSENT)}")


def run(clients: Clients, *, dry_run: bool) -> int:
    plan = build_plan(clients)
    print_plan(plan, dry_run=dry_run, region=clients.region)

    failures: list[Failure] = []
    if not dry_run:
        print("\nexecuting...")
        failures = execute_plan(clients, plan, dry_run=False)

    blocked = [c for c in plan if c.selection is Selection.UNTAGGED_BLOCKED]
    if blocked:
        print("\nBLOCKED — these exist under workshop names but carry no workshop tag:", file=sys.stderr)
        for candidate in blocked:
            print(f"  {candidate.kind} {candidate.name}", file=sys.stderr)
        print(
            "Refusing to delete untagged resources. Tag them at creation time with "
            f"{WORKSHOP_TAG_KEY}={WORKSHOP_TAG_VALUE}, or remove them by hand after "
            "confirming they are yours.",
            file=sys.stderr,
        )
    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure.kind} {failure.name}: {failure.error}", file=sys.stderr)

    if blocked or failures:
        print("\nCLEANUP INCOMPLETE", file=sys.stderr)
        return 1
    print("\nCLEANUP COMPLETE")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print exactly what would be deleted and why, then exit without "
            "deleting; this is the default when neither flag is given"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "actually delete the tagged resources; without this flag the run is "
            "a dry run and touches nothing"
        ),
    )
    parser.add_argument("--region", default=REGION, help=f"AWS region (default {REGION})")
    args = parser.parse_args(argv)
    # Default and --dry-run both mean dry run. Only --yes authorises deletion,
    # and an explicit --dry-run always wins over --yes.
    dry_run = args.dry_run or not args.yes
    return run(Clients.build(args.region), dry_run=dry_run)


if __name__ == "__main__":
    sys.exit(main())
