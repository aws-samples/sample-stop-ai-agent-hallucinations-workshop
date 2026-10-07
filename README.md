# How to Stop AI Agent Hallucinations: 5 Techniques + Production on Amazon Bedrock AgentCore

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Strands Agents](https://img.shields.io/badge/Strands_Agents-1.58-00B4D8.svg?style=flat)](https://strandsagents.com)
[![Amazon Bedrock](https://img.shields.io/badge/Amazon-Bedrock-FF9900.svg?style=flat&logo=amazon-aws)](https://aws.amazon.com/bedrock/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph--RAG-4581C3.svg?style=flat&logo=neo4j)](https://neo4j.com)
[![License: MIT-0](https://img.shields.io/badge/License-MIT--0-green.svg)](LICENSE)
[![Last Updated](https://img.shields.io/badge/Updated-March_2026-brightgreen.svg?style=flat)]()

**AI agent hallucinations** occur when agents fabricate statistics, pick wrong tools, ignore business rules, or claim success when operations fail. This workshop provides 5 hands-on techniques (Graph-RAG, semantic tool selection, multi-agent validation, neurosymbolic guardrails, and agent steering) plus two production deployment demos on Amazon Bedrock AgentCore.

> Based on the Dev.to series [Stop AI Agent Hallucinations: 4 Essential Techniques](https://dev.to/aws/stop-ai-agent-hallucinations-4-essential-techniques-2i94) and [5 Techniques to Stop AI Agent Hallucinations in Production](https://dev.to/aws/5-techniques-to-stop-ai-agent-hallucinations-in-production-oik).

Built with [Strands Agents](https://strandsagents.com) and Amazon Bedrock. The same patterns are general agent concepts and carry over to other agent frameworks.

![Two bands. Demos 01 to 05 are the five defenses you build locally: Graph-RAG computes exact aggregations in Neo4j and returns empty instead of fabricating an answer, semantic tool selection sends the top 3 relevant tool descriptions per query instead of all ~40, multi-agent validation asks a consortium of heterogeneous models the same judgement question and flags disagreement as likely hallucination, neurosymbolic guardrails put business rules in code and block violations, and native steering makes the agent self-correct and complete the booking. Demos 06 and 07 run that same code in production on Amazon Bedrock AgentCore, with AgentCore Runtime and Gateway, AWS Lambda, Amazon DynamoDB and a Neo4j knowledge graph, plus AgentCore Memory for recall across sessions](images/control-ai-agent-hallucinations-progressive-flow.png)

---

## Graph-RAG vs. Standard RAG: Why It Matters for Hallucinations

| Approach | Hallucination Risk | Retrieval Method | Best For |
|---|---|---|---|
| Standard RAG (vector) | High, because it returns the most *similar* chunks even when the exact answer isn't in them | Cosine similarity over embedded chunks | Open-ended Q&A |
| Graph-RAG (Neo4j) | Lower for structured questions, because answers are computed from entity relationships | Graph traversal + Cypher (exact `COUNT`/`AVG`/filters) | Structured domains (hotels, products, finance) |

> **Key insight:** vector search always returns the top-k most similar chunks, so for a question like "how many hotels have a pool?" it answers from ~3 chunks and *guesses* the total. Graph-RAG runs an exact Cypher aggregation over every hotel and returns an honest "no results" when nothing matches.

**It is hybrid, and the demo is honest about that.** The Neo4j build (demo 01) stores both: a structured entity graph (`Hotel`, `Room`, `Amenity`, …) *and* a vector layer of embedded text. Two node types matter:

- **`Document`**, one node per source FAQ file (the origin).
- **`Chunk`**, slices of a document, each stored with a vector embedding; this is the layer vector search queries.

The entities come from Cypher traversal (no vectors). So the anti-hallucination value of Graph-RAG is the **structured side**, exact aggregation and honest empty results, not the chunks. A **fixed extraction schema** keeps that structured side deterministic, so adding a new hotel is safe and the agent's Cypher always matches reality (see [demo 01](./01-graphrag-demo/)).

---

## What Does Each Demo Solve?

| # | Demo | What It Solves | Key Result | Stack |
|:-:|------|----------------|------------|-------|
| 01 | [Graph-RAG vs RAG](./01-graphrag-demo/) | Fabricated statistics, incomplete retrieval, out-of-domain hallucination | Exact aggregations computed in Neo4j; honest "no results" instead of a plausible lie | ![Neo4j](https://img.shields.io/badge/Neo4j-4581C3?style=flat&logo=neo4j&logoColor=white) ![FAISS](https://img.shields.io/badge/FAISS-blue?style=flat) |
| 02 | [Semantic Tool Selection](./02-semantic-tools-demo/) | Wrong tool picks and token waste with a large toolset (~40 tools) | A hook trims to the top-3 relevant tools per request: ~73% fewer tokens, accuracy equal-or-better | ![FAISS](https://img.shields.io/badge/FAISS-blue?style=flat) ![Hooks](https://img.shields.io/badge/Hooks-purple?style=flat) |
| 03 | [Multi-Agent Validation](./03-multiagent-demo/) | Confident guesses on judgement questions with no single right answer | Consortium of heterogeneous models; agreement = confident, divergence = likely hallucination | ![Bedrock](https://img.shields.io/badge/Claude_+_Nova-FF9900?style=flat&logo=amazon-aws) |
| 04 | [Neurosymbolic Guardrails](./04-neurosymbolic-demo/) | Agents ignoring business rules written only in prompts | Rules enforced in code via lifecycle hooks, validated against Neo4j | ![Hooks](https://img.shields.io/badge/Hooks-purple?style=flat) |
| 05 | [Agent Steering](./05-steering-demo/) | Hard-blocking stops the task instead of fixing it | Native Strands steering guides the agent to self-correct and complete the booking | ![Steering](https://img.shields.io/badge/Steering-orange?style=flat) |
| 06 | [Amazon Bedrock AgentCore](./06-agentcore-boto3-demo/) | Taking the techniques to production on AWS | Hotel booking agent on Amazon Bedrock AgentCore with Lambda + DynamoDB | ![AgentCore](https://img.shields.io/badge/Bedrock-AgentCore-FF9900?style=flat&logo=amazon-aws) ![DynamoDB](https://img.shields.io/badge/DynamoDB-4053D6?style=flat&logo=amazon-dynamodb) ![Lambda](https://img.shields.io/badge/Lambda-FF9900?style=flat&logo=aws-lambda) |
| 07 | [AgentCore Memory](./07-agentcore-memory-demo/) | An agent that forgets what the user told it in an earlier session | The deployed agent recalls preferences across sessions with AgentCore Memory | ![AgentCore](https://img.shields.io/badge/Bedrock-AgentCore-FF9900?style=flat&logo=amazon-aws) ![Memory](https://img.shields.io/badge/Memory-01A88D?style=flat) |

> **When is multi-agent the right tool?** Demo 03 uses a consortium of models for **judgement questions that have no single database answer**. For questions with a deterministic answer, prefer Graph-RAG (demo 01) or in-code guardrails (demo 04), they are cheaper and more reliable than a second agent. The honest boundary is part of the lesson.

---

## How Do the Demos Build on Each Other?

Each demo builds on the previous one. You can run any demo independently, but the learning path is designed to be progressive:

**Phase 1: Understand Failures.** Demos 01-02 show *why* agents fail: RAG hallucinations and tool selection errors.

**Phase 2: Detect and Prevent.** Demos 03-04 show *how to catch* failures: multi-agent validation and symbolic rule enforcement.

**Phase 3: Self-Correct.** Demo 05 shows *how to steer* agents to self-correct instead of blocking (native Strands steering).

**Phase 4: Deploy to Production.** Demo 06 shows *how to ship* all 5 techniques to production on AWS using Amazon Bedrock AgentCore, DynamoDB, and Lambda.

**Phase 5: Remember Across Sessions.** Demo 07 adds AgentCore Memory so the deployed agent recalls what the user told it in earlier sessions.

---

## How Do I Run These Demos?

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- AWS account with [Amazon Bedrock](https://aws.amazon.com/bedrock/) access (Claude Sonnet 4 enabled in your region)

### Run Any Demo

```bash
cd 01-graphrag-demo   # or any demo folder
uv venv && uv pip install -r requirements.txt

# Run the demo
uv run <main_script>.py
# Or open the .ipynb notebook in your IDE (VS Code, Kiro, or any editor with notebook support)
```

Each demo README has specific setup instructions and prerequisites.

### Neo4j Setup (Demo 01)

Demo 01 requires a Neo4j instance to build and query the knowledge graph.

**Running as part of a workshop:** A Neo4j Aura instance will be provided. You will receive the connection credentials (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`) at the start of the session. Add them to a `.env` file in `01-graphrag-demo/`.

**Running independently:** Create your own free Neo4j Aura instance:

1. Go to [console.neo4j.io](https://console.neo4j.io) and create a free **AuraDB** instance
2. Download the credentials file when prompted. It contains your URI, username, and password
3. Create a `.env` file in `01-graphrag-demo/`:
   ```
   NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=<your-password>
   ```
4. Enable the **APOC** plugin in your Aura instance settings
5. Run `build_graph_lite.py` (30 docs, ~15 min) or `build_graph.py` (300 docs, ~2 hours) to populate the graph

---

## Frequently Asked Questions

### What types of AI agent hallucinations does this repository address?

This repository addresses four main categories: **(1)** fabricated statistics, when RAG agents guess numbers instead of computing them; **(2)** wrong tool selection, when agents pick inappropriate tools from large toolsets; **(3)** business rule violations, when agents ignore constraints expressed only in prompts; and **(4)** undetected failures, when single agents claim success without validation.

### Can I use these patterns with frameworks other than Strands Agents?

Yes. The patterns (Graph-RAG, semantic tool filtering, multi-agent validation, neurosymbolic guardrails, steering controls) are framework-agnostic concepts. These demos use Strands Agents, but the same approaches carry over to other agent frameworks and to custom implementations. The key insight is architectural, not framework-specific.

### Do I need an AWS account to run the demos?

Yes. All demos use Amazon Bedrock (Claude Sonnet 4) as the default LLM provider. You need an AWS account with Bedrock access enabled in your region.

### How long does it take to run each demo?

Demos 02-05 run in under 5 minutes. Demo 01 has a lite mode (30 docs, ~15 minutes) and full mode (300 docs, ~2 hours) for building the knowledge graph.

### What LLM providers are supported?

All demos default to Amazon Bedrock (Claude Sonnet 4) but work with any provider supported by Strands Agents: Anthropic API, Ollama (local models). See [Strands Model Providers](https://strandsagents.com/docs/user-guide/sdk/model-providers/amazon-bedrock/) for configuration.

---

## Common Issues and How to Fix Them

**Bedrock access denied:** Ensure the model (`us.anthropic.claude-sonnet-4-5` or similar) is enabled in your region via the [Bedrock Model Access console](https://console.aws.amazon.com/bedrock/home#/modelaccess).

**Neo4j connection fails (demo 01):** Verify `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` are set in your `.env` file and that APOC is enabled on your Aura instance.

**OpenTelemetry warnings:** "Failed to detach context" warnings in demos 03-05 are harmless and do not affect functionality.

**Model alternatives:** Change the model in any demo by modifying the `BedrockModel(model_id=...)` call. See [Strands Model Providers](https://strandsagents.com/docs/user-guide/sdk/model-providers/amazon-bedrock/) for all supported options.

For demo-specific issues, check the troubleshooting section in each demo's README.

---

## References

1. Fuentes, E. (2025). *Stop AI Agent Hallucinations: 4 Essential Techniques.* Dev.to / AWS. [dev.to/aws/stop-ai-agent-hallucinations-4-essential-techniques-2i94](https://dev.to/aws/stop-ai-agent-hallucinations-4-essential-techniques-2i94)
2. Fuentes, E. (2025). *5 Techniques to Stop AI Agent Hallucinations in Production.* Dev.to / AWS. [dev.to/aws/5-techniques-to-stop-ai-agent-hallucinations-in-production-oik](https://dev.to/aws/5-techniques-to-stop-ai-agent-hallucinations-in-production-oik)
3. Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft Research. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
4. AWS. *Reducing hallucinations in large language models with custom intervention using Amazon Bedrock Agents.* [AWS Blog](https://aws.amazon.com/blogs/machine-learning/reducing-hallucinations-in-large-language-models/)

---

## Contributing

Contributions are welcome! See [CONTRIBUTING](CONTRIBUTING.md) for more information.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file for details.

> Last updated: October 2026 | Strands Agents 1.58 | Python 3.10+
