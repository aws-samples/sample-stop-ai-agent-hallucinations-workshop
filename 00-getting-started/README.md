[< Back to Main README](../README.md)

# Build AI Agents with Strands Agents and Amazon Bedrock: Workshop Primer

Get up and running with [Strands Agents](https://strandsagents.com) in under 30 minutes. This notebook covers every core concept used throughout the workshop — agents, tools, lifecycle hooks, and multi-agent swarms — with runnable examples and real explanations.

[![Python](https://img.shields.io/badge/Python-3.9+-green.svg?style=flat)](https://python.org)
[![Strands Agents](https://img.shields.io/badge/Strands_Agents-1.27+-00B4D8.svg?style=flat)](https://strandsagents.com)
[![Amazon Bedrock](https://img.shields.io/badge/Amazon-Bedrock-FF9900.svg?style=flat&logo=amazon-aws)](https://aws.amazon.com/bedrock/)

> **Already familiar with Strands Agents?** Skip to [`01-graphrag-demo/`](../01-graphrag-demo/) and come back here if you need a refresher on hooks or swarms.

---

## What This Notebook Covers

| Concept | What it does | Used in |
|---------|-------------|---------|
| `Agent` + system prompt | Creates an LLM-powered agent that reasons and acts | All demos |
| Model providers | Switch between Bedrock, Anthropic, Ollama, or any OpenAI-compatible endpoint | All demos |
| `@tool` decorator | Expose Python functions as tools the agent can call | All demos |
| `BeforeToolCallEvent` + `cancel_tool` | Block tool calls that violate rules — LLM cannot bypass | Demos 04, 05, 06 |
| `Swarm` | Multi-agent handoff chain (Executor → Validator → Critic) | Demo 03 |

---

## Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager

**At an AWS event:** AWS credentials and dependencies are pre-configured. Run the first notebook cell as-is — no setup needed.

**Self-paced:** Configure your AWS credentials and enable Bedrock model access:

```bash
aws configure   # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
```

Then enable the model in the [Bedrock Model Access console](https://console.aws.amazon.com/bedrock/home#/modelaccess) (`us.anthropic.claude-sonnet-4-5` or equivalent).

---

## Quick Start

```bash
cd 00-getting-started
uv venv && uv pip install -r requirements.txt
```

Open `getting_started_strands.ipynb` in your IDE (VS Code, Kiro, or any editor with notebook support) and run the cells in order.

---

## How It Works

### 1. Creating an Agent

An agent combines an LLM with a system prompt and optional tools. [Amazon Bedrock](https://aws.amazon.com/bedrock/) is the default model provider. At an AWS event, credentials are pre-configured. Self-paced users need to configure AWS credentials first (see Prerequisites above).

```python
from strands import Agent

agent = Agent(
    system_prompt="You are a helpful travel assistant.",
)
response = agent("What should I consider when booking a hotel in Lisbon?")
```

To specify a model explicitly:

```python
agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    system_prompt="You are a helpful assistant.",
)
```

See all supported providers: [Strands Model Providers](https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/)

---

### 2. Building Tools with `@tool`

Tools are Python functions decorated with `@tool`. The agent reads the function name and **docstring** to decide when and how to call each tool.

```python
from strands import Agent, tool

@tool
def search_hotels(city: str, max_price: int = 500) -> str:
    """Search for available hotels in a city under a maximum price per night."""
    return f"Found hotels in {city} under ${max_price}/night..."

agent = Agent(tools=[search_hotels], system_prompt="You are a booking assistant.")
agent("Find hotels in Lisbon under $100")
```

> **Docstrings are critical.** The agent uses them to match user queries to tools. A vague docstring leads to wrong tool selection — Demo 02 demonstrates this in depth.

See: [Strands Tools Documentation](https://strandsagents.com/docs/user-guide/concepts/tools/custom-tools/)

---

### 3. Lifecycle Hooks — Enforcing Rules the LLM Cannot Bypass

Hooks intercept the agent's execution at specific points. `BeforeToolCallEvent` fires after the LLM decides to call a tool but **before** the tool executes. Setting `event.cancel_tool` blocks the call entirely — the LLM receives the cancellation message instead of the tool result and cannot override it.

```python
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent

class MaxGuestsHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.check)

    def check(self, event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] == "book_room":
            guests = event.tool_use["input"].get("guests", 1)
            if guests > 10:
                event.cancel_tool = f"BLOCKED: {guests} guests exceeds maximum of 10"

agent = Agent(tools=[book_room], hooks=[MaxGuestsHook()], ...)
```

This pattern is the foundation of Demos 04, 05, and 06. See: [Strands Hooks Documentation](https://strandsagents.com/docs/user-guide/concepts/agents/hooks/)

---

### 4. Multi-Agent Swarms

A swarm is a group of agents that hand off work to each other in sequence. Each agent has a specific role. Strands manages the handoff chain automatically.

```python
from strands.multiagent import Swarm

executor  = Agent(name="Executor",  tools=[lookup_hotel], system_prompt="Look up hotel details.")
validator = Agent(name="Validator", system_prompt="Verify if the response is consistent.")
critic    = Agent(name="Critic",    system_prompt="Give a final verdict: VALID or SUSPICIOUS.")

swarm = Swarm(agents=[executor, validator, critic], max_handoffs=5)
swarm("What are the details for AnyCompany Lisbon Resort?")
```

See: [Strands Multi-Agent Documentation](https://strandsagents.com/docs/user-guide/concepts/multi-agent/)

---

## Concepts Summary

| Concept | Why it matters for hallucination prevention |
|---------|---------------------------------------------|
| `@tool` + clear docstrings | Accurate tool selection reduces tool-mismatch hallucinations (Demo 02) |
| `BeforeToolCallEvent` + `cancel_tool` | Enforces business rules the LLM cannot invent its way around (Demo 04) |
| STEER messages in `cancel_tool` | Instead of blocking, guide the agent to self-correct (Demo 05) |
| `Swarm` with Validator + Critic | Catches fabricated responses before they reach the user (Demo 03) |

---

## Frequently Asked Questions

### What is Strands Agents and how is it different from LangChain?

[Strands Agents](https://strandsagents.com) is an open-source Python framework for building AI agents. It focuses on simplicity: a single `Agent` class, `@tool` decorator, and hook system. Similar patterns exist in [LangGraph](https://langchain-ai.github.io/langgraph/), [AutoGen](https://microsoft.github.io/autogen/), and [CrewAI](https://www.crewai.com/) — the workshop concepts (tool calling, guardrails, multi-agent validation) apply to all of them.

### Why does the docstring matter for tool selection?

The agent uses the function name and docstring as the tool's description when deciding which tool to call. If two tools have similar or vague docstrings, the agent may pick the wrong one. Demo 02 measures this problem quantitatively (89% token reduction with semantic filtering) and shows how to fix it.

### Do I need an AWS account to run this notebook?

Yes, if running self-paced. You need an AWS account with [Amazon Bedrock](https://aws.amazon.com/bedrock/) access and the model enabled in the [Bedrock Model Access console](https://console.aws.amazon.com/bedrock/home#/modelaccess). At an AWS event, the account and credentials are provided — no setup required.

### Can I use a different LLM provider?

Yes. Change the `model` parameter to use any provider supported by Strands Agents: Anthropic API directly, Ollama (local models), or any OpenAI-compatible endpoint. See [Strands Model Providers](https://strandsagents.com/docs/user-guide/concepts/model-providers/) for configuration details.

### What is `cancel_tool` and how is it different from a prompt instruction?

`cancel_tool` is a framework-level block set in a `BeforeToolCallEvent` hook. It fires **before** the tool executes and **after** the LLM has already decided to call it. Unlike instructions in a system prompt (which the LLM can reason around or ignore), `cancel_tool` is enforced by the Strands framework — the LLM receives the block message and cannot retry the tool call for the same reason.

---

## Navigation

- **Next:** [Demo 01 — Graph-RAG vs RAG](../01-graphrag-demo/)

---

## Contributing

Contributions are welcome! See [CONTRIBUTING](../CONTRIBUTING.md) for more information.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.

> Last updated: April 2026 | Strands Agents 1.27+ | Python 3.9+
