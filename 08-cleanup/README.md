# Module 8: Cleanup

Delete the resources created in Modules 6 and 7.

## How to Run

```bash
python workshop_cleanup.py             # dry run by default: show the plan and the reasoning
python workshop_cleanup.py --dry-run   # the same read-only run, made explicit
python workshop_cleanup.py --yes       # execute it and delete the tagged resources
```

The default is a dry run. Deletion happens only when you pass `--yes`, so running the script with no
arguments never deletes anything.

Or open `cleanup.ipynb`, which runs the same code and shows the dry run before it deletes anything.
`../06-agentcore-boto3-demo/cleanup.py` is a thin wrapper around the same module, so there is exactly
one teardown path to audit.

Both exit **non-zero** if the teardown is incomplete, so they are safe to call from a script.

## What Gets Deleted

| Resource Type | Count | Description |
|--------------|-------|-------------|
| AgentCore Runtimes | 2 | HotelBookingAgent + HotelBookingAgentWithMemory |
| AgentCore Gateway | 1 | HotelBookingGateway with its Lambda targets |
| AgentCore Memory | 1 | workshop_HotelBookingMemory |
| Lambda Functions | 8 | 7 booking tools + 1 Neo4j query |
| Lambda Layer | 1 | Neo4j Python driver |
| DynamoDB Tables | 3 | Hotels, Bookings, SteeringRules |
| IAM Roles | 2 | Lambda + AgentCore execution roles |
| ECR Repositories | 2 | Container images for both agents |
| CodeBuild Projects | 2 | Build projects from the starter toolkit |

...but only the ones carrying the workshop tag. See below.

## Deletion is Scoped by Tag, Never by Name

Every resource is deleted on one condition: it carries

```
WorkshopResource=stop-ai-agent-hallucinations
```

Nothing is selected by name prefix. Module 6 and 7 apply this tag at creation; cleanup deletes only
what is tagged.

**Why this matters.** An earlier version of this teardown selected IAM roles like so:

```python
cb_roles = iam.list_roles(PathPrefix="/")["Roles"]
for role in cb_roles:
    if role["RoleName"].startswith("AmazonBedrockAgentCoreSDKCodeBuild"):
        ... iam.delete_role(RoleName=rn)
```

IAM is global, so that swept the entire account across every region and destroyed five roles this
workshop never created, two of them in a different region. A name prefix describes what a resource is
*called*. A tag records who *owns* it. Only the second is safe to delete on.

Measured on the development account: 426 IAM roles exist, 126 of them have names a prefix scheme
would plausibly match, and the tag gate selects **zero** of them.

### Untagged resources are reported, not deleted

If a resource exists under a workshop name but carries no workshop tag, cleanup refuses to delete it,
prints it under `BLOCKED`, and exits 1. That is deliberate. The resource is either someone else's or
came from a deployment that did not tag, and neither is a call a script should make for you. Tag it,
or delete it by hand once you have confirmed it is yours.

### An untagged workshop role blocks cleanup

If an IAM role carries a workshop name (`workshop-LambdaExecutionRole` or
`workshop-AgentCoreExecutionRole`) but no workshop tag, cleanup reports it under `BLOCKED` and exits 1
without deleting it. This is the same safety rule above, applied to roles. It affects anyone who
deployed Modules 6 or 7 before tagging landed (bug B20): those roles were created without the
`WorkshopResource` tag, so the current teardown refuses to remove them.

There are two ways to clear the block. Pick one.

1. **Tag the role, then re-run.** Adopt the role by applying the workshop tag, after which cleanup
   deletes it on the next run:

```bash
aws iam tag-role --role-name workshop-LambdaExecutionRole \
  --tags Key=WorkshopResource,Value=stop-ai-agent-hallucinations
python workshop_cleanup.py --yes
```

2. **Delete the role by hand in the console.** Open IAM in the AWS Console, confirm the role is yours
   and came from this workshop, detach its policies, and delete it.

Do either only for a role you have confirmed is yours. Tagging is a claim of ownership, and the next
cleanup run acts on it.

### Two residual deletion risks

Two paths delete without a per-name tag check. Both are deliberate and test-pinned, and both are named
here so you can recognize them:

- **The Lambda layer `workshop-neo4j-driver` is deleted by exact name.** AWS does not allow tags on
  Lambda layer versions, so this layer cannot be tag-gated and is matched by name alone. If you own an
  unrelated layer with the same name, every version of it is deleted. Rename your layer, or run this
  teardown in an account that holds no same-named layer.
- **A tagged IAM role is deleted regardless of its name.** `discover_roles` selects every role in the
  account carrying `WorkshopResource=stop-ai-agent-hallucinations`, whatever it is called. This is the
  safe default of tag over name, but it means a role you tagged with the workshop key for any other
  reason is also removed. Apply that tag only to resources you want this teardown to delete.

### The two documented exceptions

`UNTAGGABLE_KINDS` in `workshop_cleanup.py` lists the only things matched directly rather than by tag:

- **Lambda layer versions.** AWS does not support tags on them, so they are matched by exact name.
- **Local `.bedrock_agentcore.yaml` config files.** Not AWS resources. They are matched by fixed repo-relative path, one per demo directory, never by glob.

Both are matched exactly, never by prefix or glob. A unit test pins the contents of that set so the
exemption cannot quietly grow.

## Failures Are Loud

No exception is swallowed. The previous version hid a `KeyError` behind a bare `except`, printed a
clean bill of health, and left a billable AgentCore Memory resource running. The `KeyError` came from
reading `m["memoryName"]` in `list_memories()` output — that field does not exist. `MemorySummary`
is `arn, id, status, createdAt, updatedAt, managedByResourceArn`, and the memory is matched on `id`,
which has the form `<name>-<suffix>`.

Any delete that fails, and any untagged workshop resource found, produces a non-zero exit.

## Tests

```bash
python -m unittest discover -s . -v
```

18 tests, no AWS credentials needed. The clients are injected fakes. The headline test is
`test_untagged_unrelated_roles_are_never_selected`, which reproduces the exact account shape that
bug B6 damaged and asserts none of those roles is selected.

## What Does NOT Get Deleted

- **Neo4j infrastructure** — Code Editor EC2 or the Central Neo4j ECS stack from Module 1.
  Delete via AWS Console → CloudFormation → Delete Stack.
- **CloudWatch log groups** — retained so you can review the run.
- **The starter toolkit's shared CodeBuild role** (`AmazonBedrockAgentCoreSDKCodeBuild-*`). It is
  created by the toolkit, shared across projects, and costs nothing. Deleting it is what caused the
  original incident. If you want it gone, remove it by hand.
- **Anything untagged.**

## Reclaiming Resources From a Run Before Tagging Existed

Earlier versions of Modules 6 and 7 created resources without tagging them. Cleanup will refuse to
delete those, correctly: it cannot prove it created them. If you ran this workshop before, you may be
paying for orphans that this script reports and then leaves alone.

They are safe to remove, but you have to confirm they are yours first. Deletion here is by hand and by
eye, not by script, for the same reason the tag gate exists.

**Step 1. List what cleanup is refusing to touch.** The dry run names them under `BLOCKED`:

```bash
python workshop_cleanup.py --dry-run
```

**Step 2. Confirm each one predates your tagged deployment.** Creation time is the evidence that a
resource came from an older run rather than from something else you are running now:

```bash
aws dynamodb describe-table --table-name <name> --query 'Table.CreationDateTime'
aws lambda get-function-configuration --function-name <name> --query 'LastModified'
aws iam get-role --role-name <name> --query 'Role.CreateDate'
aws ecr describe-repositories --repository-names <name> --query 'repositories[0].createdAt'
```

**Step 3. Check the whole account, not just this region.** IAM is global. Lambda, DynamoDB, ECR,
CodeBuild and AgentCore are regional, so an old run in another region is invisible from this one.
Repeat with `--region` for every region you have used.

**Step 4. Delete by hand, one at a time, reading each name before you confirm it.**

Do not write a loop that deletes everything matching `hotel-booking-*` or `workshop-*`. That is
precisely the prefix-matching mistake documented above, and running it against your own account is how
five unrelated roles were destroyed. If a name looks close but you cannot account for the resource,
leave it and investigate.

**A shortcut worth knowing.** Instead of deleting an old resource, you can adopt it by applying the
workshop tag, after which cleanup will remove it normally:

```bash
aws dynamodb tag-resource --resource-arn <arn> \
  --tags Key=WorkshopResource,Value=stop-ai-agent-hallucinations
```

Only do this for resources you have confirmed came from this workshop. Tagging something is a claim of
ownership, and the next cleanup run will act on it.

**What you will probably find.** The starter toolkit's shared `AmazonBedrockAgentCoreSDKCodeBuild-*`
role and its `bedrock-agentcore-*-builder` CodeBuild projects survive teardown by design. They cost
nothing, they are shared across projects, and deleting the role is what caused the original incident.
Leaving them is the right outcome.

## Estimated Time

2-3 minutes. Enumerating IAM role tags across a large account adds roughly a minute to the plan step.
