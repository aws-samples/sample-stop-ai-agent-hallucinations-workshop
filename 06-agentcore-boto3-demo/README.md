[< Back to Main README](../README.md)

# Production-Ready Booking Agent on Amazon Bedrock AgentCore (boto3 + Notebook)

Deploy all anti-hallucination techniques from the previous demos in this series (01-05) to production using [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/), [Amazon DynamoDB](https://aws.amazon.com/dynamodb/), [AWS Lambda](https://aws.amazon.com/lambda/), and a [Neo4j](https://neo4j.com) knowledge graph (at AWS events, a per-team Neo4j on Amazon ECS on AWS Fargate behind an internal Network Load Balancer in the VPC; self-paced, any Neo4j such as [AuraDB](https://neo4j.com/product/auradb/)), driven entirely from a single Jupyter notebook with boto3. No CDK required.

[![Python](https://img.shields.io/badge/Python-3.11+-green.svg?style=flat)](https://python.org)
[![AgentCore](https://img.shields.io/badge/Bedrock-AgentCore-orange.svg?style=flat&logo=amazon-aws)](https://aws.amazon.com/bedrock/agentcore/)
[![DynamoDB](https://img.shields.io/badge/DynamoDB-tables-orange.svg?style=flat&logo=amazon-aws)](https://aws.amazon.com/dynamodb/)
[![Lambda](https://img.shields.io/badge/AWS-Lambda-orange.svg?style=flat&logo=aws-lambda)](https://aws.amazon.com/lambda/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge_Graph-blue.svg?style=flat)](https://neo4j.com)

This demo uses [Strands Agents](https://github.com/strands-agents/harness-sdk) with [Amazon Bedrock](https://aws.amazon.com/bedrock/). Similar patterns apply with other agent frameworks that support AgentCore Runtime.

---

## What This Demo Shows

Demos 01-05 demonstrate techniques that significantly reduce hallucinations. This demo takes those techniques to production:

| Technique (from demos) | Production implementation |
|------------------------|--------------------------|
| **Semantic tool selection** (demo 02) | [AgentCore Gateway](https://aws.amazon.com/bedrock/agentcore/) with MCP (Model Context Protocol) semantic routing, no custom FAISS index needed |
| **Multi-agent validation** (demo 03) | Not deployed here: the consortium stays a notebook pattern for judgement questions. The booking flow has deterministic rules, which `validate_booking_rules` (backed by [Amazon DynamoDB](https://aws.amazon.com/dynamodb/)) and the hooks check instead |
| **Neurosymbolic guardrails** (demo 04) | Steering rules in [Amazon DynamoDB](https://aws.amazon.com/dynamodb/), so rules change without redeploying the agent |
| **Steering** (demo 05) | STEER messages in DynamoDB rules, so the agent self-corrects instead of failing |
| **Graph-RAG** (demo 01) | [Neo4j](https://neo4j.com) (per-team on Amazon ECS on AWS Fargate behind an internal NLB at AWS events; any Neo4j such as [AuraDB](https://neo4j.com/product/auradb/) self-paced) with a `query_knowledge_graph` [AWS Lambda](https://aws.amazon.com/lambda/) in the VPC (optional, auto-detected) |

---

## Workshop vs. Self-Paced

This demo adapts to your environment automatically.

**At an AWS event:** Neo4j runs on Amazon ECS on AWS Fargate behind an internal Network Load Balancer in your account. The notebook reads its DNS name (`NEO4J_HOST`), password secret ARN, security group and subnets from environment variables set in the notebook kernel, so no manual configuration is needed.

**Self-paced:** Set `NEO4J_HOST` manually in the notebook's configuration cell to your own Neo4j instance (AuraDB URI, local Docker, etc.). If no Neo4j is configured, the notebook deploys without the graph query tool, and the booking agent still works fully.

See [Neo4j Setup (self-paced)](#neo4j-setup-self-paced) for step-by-step instructions.

---

## What Gets Deployed

| Component | AWS Service | Purpose |
|-----------|-------------|---------|
| Data layer | [Amazon DynamoDB](https://aws.amazon.com/dynamodb/) (3 tables) | Hotels catalog, reservations, steering rules |
| Booking tools | [AWS Lambda](https://aws.amazon.com/lambda/) (7 functions) | search, book, get booking, pay, confirm, cancel, validate |
| Graph query tool | [AWS Lambda](https://aws.amazon.com/lambda/) (1 function, VPC) | Cypher queries to Neo4j knowledge graph (optional) |
| Tool routing | [AgentCore Gateway](https://aws.amazon.com/bedrock/agentcore/) (MCP) | Semantic tool discovery and invocation |
| Agent | [AgentCore Runtime](https://aws.amazon.com/bedrock/agentcore/) | Hosts the Strands agent with Amazon Bedrock |
| Guardrails | Hooks + DynamoDB rules | Hard blocks (hooks) + soft steering (STEER messages) |

---

## Quick Start

### Prerequisites

Before starting, make sure you have:

- **[Python](https://python.org/downloads) 3.11+** installed (the agent container uses your Python version, and `mcp-proxy-for-aws` needs 3.11+), with `pip` (optionally inside a virtual environment created with `python -m venv .venv`)
- **[AWS CLI](https://aws.amazon.com/cli/)** installed and configured with credentials for your account
- **Amazon Bedrock access**: Amazon Bedrock enables serverless models on first use; Anthropic models need a one-time use case submission per account (open an Anthropic model in the Amazon Bedrock model catalog and submit the use case details)

### Step 1: Install dependencies

```bash
cd 06-agentcore-boto3-demo
pip install -r requirements.txt
```

### Step 2: Open and run the deployment notebook

```bash
# Open in VS Code, Kiro, or any IDE with notebook support
code deploy_agentcore.ipynb
```

The notebook walks through every step in order:

| Step | What it does |
|------|-------------|
| 0 | Configure AWS clients and read the Neo4j settings from the kernel environment (workshop) or set them manually (self-paced) |
| 1 | Create DynamoDB tables (Hotels, Bookings, SteeringRules) |
| 2 | Seed 18 hotels across 18 cities worldwide |
| 3 | Seed 6 steering rules with STEER self-correction messages |
| 4 | Create IAM roles (Lambda execution + AgentCore execution) |
| 5 | Deploy 7 Lambda booking tools, then the Neo4j query Lambda with VPC config (if Neo4j detected) |
| 6 | Create AgentCore Gateway with MCP semantic routing and AWS IAM inbound authorization |
| 7 | Register Lambda tools as Gateway targets |
| 8 | Deploy Strands agent to AgentCore Runtime via `bedrock-agentcore-starter-toolkit` |
| 9 | Run the test scenarios against the live agent |

Every cell is **idempotent**: if a resource already exists, the cell finds it and continues. You can re-run any cell or restart from any point.

### Step 3: Test the deployed agent

After Step 8 completes, test via AWS CLI:

```bash
aws bedrock-agentcore invoke-agent-runtime \
    --agent-runtime-arn <RUNTIME_ARN from notebook output> \
    --payload '{"prompt": "Find hotels in Paris under $120"}' \
    --cli-binary-format raw-in-base64-out \
    --region us-east-1 /tmp/response.json && cat /tmp/response.json
```

Or continue running the test cells directly in the notebook (Step 9).

---

## Neo4j Setup (Self-Paced)

### What is Neo4j AuraDB?

[Neo4j](https://neo4j.com/) is a graph database that stores data as nodes and relationships instead of rows and columns. This lets the agent answer questions like "What amenities does the Grand Hotel have?" by traversing graph connections directly, instead of guessing from text chunks. See [demo 01](../01-graphrag-demo/) for a full explanation.

### Create a free Neo4j AuraDB instance

1. Go to [console.neo4j.io](https://console.neo4j.io) and create a free account
2. Click **New Instance** → **AuraDB Free**
3. Choose a name (e.g., `hotel-graphrag`) and region
4. **Download the credentials file** when prompted. It contains your URI, username, and password

### Configure in the notebook

In Step 0 of `deploy_agentcore.ipynb`, uncomment and set these values:

```python
NEO4J_HOST = "your-instance.databases.neo4j.io"   # From AuraDB console
NEO4J_SECRET_ARN = ""       # Create a Secrets Manager secret with your password
SECURITY_GROUP_ID = "sg-xxxxx"   # Your VPC security group
SUBNET_IDS = ["subnet-xxxxx"]    # Your VPC subnets (for Lambda VPC config)
```

> If no Neo4j is configured, the notebook skips the graph Lambda and deploys without it, and the booking agent still works fully for hotel search, booking, payment, and cancellation.

---

## Steering Rules

Rules live in [Amazon DynamoDB](https://aws.amazon.com/dynamodb/), not in code. Change agent behavior without redeploying:

```json
{
    "rule_id": "max-guests",
    "action": "book",
    "condition_field": "guests",
    "operator": "gt",
    "threshold": 10,
    "fail_message": "Guest count exceeds maximum of 10",
    "steer_message": "The booking exceeds the hotel maximum of 10 guests per room. Calculate how many rooms are needed (divide {guests} by 10, rounding up). Explain to the user that you are splitting the reservation into multiple rooms, then immediately call book_hotel multiple times: fill rooms with 10 guests each until all guests are accommodated. After all calls succeed, confirm the split reservation with room details (e.g., Reserved 2 rooms: 10 + 5 guests).",
    "enabled": true
}
```

The agent calls `validate_booking_rules` before every booking action. When a rule is violated, it receives the `steer_message`, an instruction on how to self-correct, instead of a hard failure.

**To change a rule** (takes effect immediately, no redeploy):

```bash
aws dynamodb update-item \
    --table-name workshop-SteeringRules \
    --key '{"rule_id": {"S": "max-guests"}}' \
    --update-expression "SET threshold = :t" \
    --expression-attribute-values '{":t": {"N": "8"}}'
```

---

## Test Scenarios

The test cells in Step 9, in notebook order:

| # | Prompt | Expected behavior |
|---|--------|-------------------|
| 5b | "Which hotels have a swimming pool and spa amenities?" | Amenities exist only in Neo4j → `query_knowledge_graph` (only if Neo4j was detected) |
| 1 | "Search for hotels in Lisbon" | Semantic routing → `search_available_hotels` → DynamoDB scan |
| 2 | "Book AnyCompany Paris Central for 15 guests from 2027-06-01 to 2027-06-05" | `validate_booking_rules` → STEER: split the party into several rooms instead of failing (also respecting the hotel's room size) |
| 3 | "Find me a 4-star hotel in London" → "Book AnyCompany London City Hotel for Ana Lima, 2 guests, ..." → "Process the payment for that booking and confirm it" (same session) | CONFIRMED; the agent keeps the booking ID within the session |
| 5 | "Find hotels under $100 per night" | Cross-country search → finds cheapest hotels |
| 6 | "Confirm booking BK-DOESNOTEXIST" | Hard hook `BookingGuardrailsHook` BLOCKS the call at the framework level (booking not found) |

---

## File Structure

```
06-agentcore-boto3-demo/
├── deploy_agentcore.ipynb      # Step-by-step deployment + testing notebook
├── booking_agent.py            # AgentCore Runtime entry point (Strands + BookingGuardrailsHook)
├── cleanup.py                  # Delete all workshop resources
├── agent_requirements.txt      # Runtime dependencies (deployed to AgentCore)
├── requirements.txt            # Local dependencies (notebook)
├── lambda_tools/
│   ├── search_available_hotels/    # DynamoDB scan with city, country, price, stars filters
│   ├── book_hotel/                 # Create PENDING reservation + decrement room count
│   ├── get_booking/                # Read booking status
│   ├── process_payment/            # PENDING → PAID
│   ├── confirm_booking/            # PAID → CONFIRMED
│   ├── cancel_booking/             # Cancel + return room to inventory
│   ├── validate_booking_rules/     # Evaluate steering rules from DynamoDB
│   └── query_knowledge_graph/      # Cypher queries to Neo4j (optional, VPC)
└── tool_schemas/
    └── tools.json              # Tool definitions registered with AgentCore Gateway
```

---

## Technologies

| Technology | Purpose |
|------------|---------|
| [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) | Runtime (agent hosting), Gateway (MCP semantic tool routing) |
| [Amazon DynamoDB](https://aws.amazon.com/dynamodb/) | Hotels catalog, bookings, and steering rules |
| [AWS Lambda](https://aws.amazon.com/lambda/) | Serverless tool functions (search, book, validate, query graph) |
| [Amazon Bedrock](https://aws.amazon.com/bedrock/) | LLM provider (Strands `BedrockModel`) |
| [bedrock-agentcore-starter-toolkit](https://pypi.org/project/bedrock-agentcore-starter-toolkit/) | Packages and deploys the agent container to AgentCore Runtime |
| [Strands Agents](https://github.com/strands-agents/harness-sdk) | Open-source agent framework (tool calling, lifecycle hooks) |
| [Neo4j](https://neo4j.com) | Graph database for the knowledge graph (optional): per-team on Amazon ECS on AWS Fargate at AWS events, any Neo4j such as [AuraDB](https://neo4j.com/product/auradb/) self-paced |
| [Cypher](https://neo4j.com/docs/cypher-manual/current/) | Neo4j query language for graph traversal |

---

## Observability

Amazon Bedrock AgentCore provides built-in observability when OpenTelemetry (OTel) dependencies are included in the agent package. The `agent_requirements.txt` includes:

```
strands-agents[otel]==1.58.0
bedrock-agentcore==1.25.1
aws-opentelemetry-distro==0.12.2
mcp==1.30.0
mcp-proxy-for-aws==1.8.0
boto3==1.43.120
```

With these dependencies, Amazon Bedrock AgentCore automatically instruments Strands Agents, capturing:
- **Invocation logs**: every `invoke_agent_runtime` call
- **Tool call traces**: which Lambda was invoked, input/output, latency
- **Error tracking**: failed tool calls, guardrail blocks

Logs appear in [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/) under `/aws/bedrock-agentcore/runtimes/`. See the [observability getting started guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-get-started.html) for details.

---

## Latency Benchmarks

Tool latency matters in production agents, because every tool call adds to end-to-end response time. These benchmarks measure real [AWS Lambda](https://aws.amazon.com/lambda/) execution duration from [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/) REPORT lines, not CLI round-trip time.

### DynamoDB-backed tools

| Function | Cold start | Warm p50 | Warm p90 | Memory used |
|----------|:----------:|:--------:|:--------:|:-----------:|
| `validate_booking_rules` | ~550-620 ms | ~25 ms | ~105 ms | 88 MB / 256 MB |
| `search_available_hotels` | ~540-660 ms | ~16 ms | ~43 ms | 88 MB / 256 MB |

**Key takeaways:**
- **Warm invocations are fast:** DynamoDB single-digit-ms reads translate to 10-30 ms Lambda execution for steering rule validation
- **Cold starts are predictable:** ~450-540 ms Init + ~110 ms execution = ~600 ms total, only on first invocation after idle period
- **Why this matters for steering:** `validate_booking_rules` runs before every booking action. At ~25 ms warm, it adds negligible latency while preventing hallucinated bookings that violate business rules

### Neo4j (optional Graph-RAG tool, measured against AuraDB)

| Function | Cold start | Warm p50 | Warm p90 | Memory used |
|----------|:----------:|:--------:|:--------:|:-----------:|
| `query_knowledge_graph` | ~1.9-2.0 s | ~13 ms | ~75 ms | 122 MB / 256 MB |

These numbers were measured against [Neo4j AuraDB](https://neo4j.com/product/auradb/) Free, which runs outside your VPC; at AWS events the Lambda reaches the per-team Neo4j on Amazon ECS on AWS Fargate through the internal NLB in the VPC, so your numbers will differ. The [neo4j Python driver](https://neo4j.com/docs/python-manual/current/) reuses TCP connections across warm invocations. After the first call, subsequent [Cypher](https://neo4j.com/docs/cypher-manual/current/) queries execute in 11-30 ms, comparable to DynamoDB. Cold starts are higher (~2s) due to driver initialization + TLS handshake + Secrets Manager lookup.

> **Benchmark methodology:** 10 consecutive invocations per function in `us-east-1`. Duration extracted from CloudWatch Lambda REPORT lines. Cold start = Init Duration + Duration. Warm = Duration only.

---

## Cleanup

Run `cleanup.py` to delete all resources created by the notebook:

```bash
python cleanup.py
```

This removes the resources of Modules 6 and 7: both AgentCore Runtimes, the AgentCore Gateway + targets, the AgentCore Memory, the 8 Lambda functions and the Neo4j driver layer, the 3 DynamoDB tables, the IAM roles (Lambda and AgentCore execution, plus the CodeBuild roles created by the starter toolkit), the S3 buckets (agent code and CodeBuild sources), the CodeBuild projects, the ECR repositories, this workshop's CloudWatch log groups, and the local starter toolkit config files.

> **Skip this if you are using an AWS-provided workshop account**, because it is cleaned up automatically.

---

## Troubleshooting

| Symptom | Resolution |
|---------|------------|
| `NoRegionError` | Set `AWS_DEFAULT_REGION=us-east-1` before running the notebook |
| Agent returns 500 error | Check [CloudWatch Logs](https://aws.amazon.com/cloudwatch/) under `/aws/bedrock-agentcore/runtimes/` |
| `ResourceNotFoundException` on Lambda | Re-run Step 5 in the notebook, the Lambda deploy may have timed out |
| Neo4j Lambda fails to connect | Verify `NEO4J_HOST` and `SECURITY_GROUP_ID` allow inbound on port 7687 |
| `AccessDeniedException` on Bedrock | Check your Amazon Bedrock permissions and submit the Anthropic use case details in the model catalog |
| IAM propagation errors | The notebook waits 10s after IAM creation, retry the failing step after 30s |

---

## Frequently Asked Questions

### What is Amazon Bedrock AgentCore and how does it work?

[Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) is an AWS managed service for hosting AI agents in production. It provides a **Runtime** (agent container hosting with auto-scaling) and a **Gateway** (MCP-based semantic routing to tools). The agent connects to the Gateway via MCP (Model Context Protocol), signing each request with SigV4 (the Gateway uses AWS IAM inbound authorization), which discovers and routes tool calls to Lambda functions, so no custom routing code is needed.

### Do I need AWS CDK to run this demo?

No. This demo uses `boto3` and the `bedrock-agentcore-starter-toolkit` to create all AWS resources directly from the Jupyter notebook (`deploy_agentcore.ipynb`). No CDK installation or bootstrap is required.

### How much does this deployment cost?

Pay-per-use; see the pricing pages for [Amazon DynamoDB](https://aws.amazon.com/dynamodb/pricing/), [AWS Lambda](https://aws.amazon.com/lambda/pricing/) and [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/pricing/). Self-paced, Neo4j AuraDB has a free tier. Run `cleanup.py` after the workshop to avoid ongoing charges.

### Can I change the LLM from Bedrock to another provider?

Yes. Change the `BedrockModel` in `booking_agent.py` to any provider supported by Strands Agents: Anthropic API, Ollama (local models). The tools, Lambda functions, and AgentCore Gateway remain unchanged, only the LLM call changes. See [Strands Model Providers](https://strandsagents.com/docs/user-guide/sdk/model-providers/) for configuration.

### How are the steering rules different from the hard guardrails?

There are two distinct mechanisms: **Hard guardrails** (`BookingGuardrailsHook` in `booking_agent.py`) enforce payment-before-confirm and 48h cancellation window at the framework level, so the LLM cannot bypass them. **Steering rules** (stored in DynamoDB, evaluated by `validate_booking_rules`) are softer: they return a STEER message guiding the agent to self-correct (e.g., "adjust guests to 10"). Steering rules can be changed in DynamoDB at any time without redeployment.

---

## Navigation

- **Previous:** [Demo 05: Steering](../05-steering-demo/)
- **Next:** [Module 7: AgentCore Memory](../07-agentcore-memory-demo/)
- **Start from the beginning:** [Demo 01: Graph-RAG vs RAG](../01-graphrag-demo/)

---

## Contributing

Contributions are welcome! See [CONTRIBUTING](../CONTRIBUTING.md) for more information.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.

> Last updated: November 2026 | Strands Agents 1.58 | Python 3.11+ | Amazon Bedrock AgentCore
