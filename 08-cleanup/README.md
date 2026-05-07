# Module 8: Cleanup

Safely delete all resources created in Modules 6 and 7.

## What Gets Deleted

Running `cleanup.ipynb` removes:

| Resource Type | Count | Description |
|--------------|-------|-------------|
| AgentCore Runtimes | 2 | HotelBookingAgent + HotelBookingAgentWithMemory |
| AgentCore Gateway | 1 | HotelBookingGateway with 8 Lambda targets |
| AgentCore Memory | 1 | workshop_HotelBookingMemory |
| Lambda Functions | 8 | 7 booking tools + 1 Neo4j query |
| Lambda Layer | 1 | Neo4j Python driver |
| DynamoDB Tables | 3 | Hotels, Bookings, SteeringRules |
| IAM Roles | 2 | Lambda + AgentCore execution roles |
| ECR Repositories | 2 | Container images for both agents |
| CodeBuild Projects | 2 | Build projects from starter toolkit |

**Cost after cleanup:** $0 (no ongoing charges)

## What Does NOT Get Deleted

- **Neo4j infrastructure** — Code Editor EC2 or Central Neo4j ECS stack from Module 1
  - Delete via AWS Console → CloudFormation → Delete Stack
- **S3 buckets** — Workshop Studio assets bucket
- **CloudWatch Logs** — Log groups remain until manually deleted

## How to Run

```bash
cd 08-cleanup
jupyter notebook cleanup.ipynb
```

Execute all cells in order. Each step reports:
- ✅ Resource deleted successfully
- ℹ️  Resource not found (already deleted or never created)
- ⚠️  Deletion failed (error details shown)

## Safety Features

- **No `rm -rf` or destructive shell commands** — all deletion via boto3 API calls
- **Idempotent** — safe to run multiple times
- **Granular** — each resource type in a separate cell, run individually if needed
- **Error handling** — continues even if some resources don't exist

## Estimated Time

~2-3 minutes (AgentCore Runtime deletion takes the longest)

## When to Use

Run this notebook:
- ✅ After completing Modules 6 and 7
- ✅ When cleaning up a test deployment
- ✅ Before re-running Module 6 from scratch
- ❌ NOT while an agent is actively being used
- ❌ NOT if you want to keep the production deployment running
