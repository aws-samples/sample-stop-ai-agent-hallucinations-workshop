[< Back to Main README](../README.md)

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

No ongoing charges after cleanup.

## What Does NOT Get Deleted

- **Neo4j**: your own Neo4j instance from Module 1 (Docker, AuraDB or Neo4j Desktop). At AWS events it is part of the event stack and is removed with it
- **S3 buckets**: Workshop Studio assets bucket
- **CloudWatch Logs**: log groups remain until manually deleted

## How to Run

Open `cleanup.ipynb` and execute all cells in order. Each step reports:
- ✅ Resource deleted successfully
- ℹ️  Resource not found (already deleted or never created)
- ⚠️  Deletion failed (error details shown)

## Safety Features

- **No `rm -rf` or destructive shell commands**: all deletion goes through boto3 API calls
- **Idempotent**: safe to run multiple times
- **Granular**: each resource type in a separate cell, run individually if needed
- **Error handling**: continues even if some resources don't exist

## Estimated Time

~2-3 minutes (AgentCore Runtime deletion takes the longest)

## When to Use

Run this notebook:
- ✅ After completing Modules 6 and 7
- ✅ When cleaning up a test deployment
- ✅ Before re-running Module 6 from scratch
- ❌ NOT while an agent is actively being used
- ❌ NOT if you want to keep the production deployment running

---

## Navigation

- **Previous:** [Module 7: AgentCore Memory](../07-agentcore-memory-demo/)
