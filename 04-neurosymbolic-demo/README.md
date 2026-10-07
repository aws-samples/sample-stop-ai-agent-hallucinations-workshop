[< Back to Main README](../README.md)

# Demo 04, Neurosymbolic Guardrails

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Strands Agents](https://img.shields.io/badge/Strands_Agents-1.58-00B4D8.svg?style=flat)](https://strandsagents.com)
[![Hooks](https://img.shields.io/badge/Pattern-Neurosymbolic_Hooks-purple.svg?style=flat)](https://strandsagents.com/docs/user-guide/sdk/agents/hooks-events/)

> Business rules written in a prompt are suggestions the LLM can ignore. The same rules as deterministic Python in a Strands hook are enforcement it cannot bypass.

![Two stacked bands over the same 10-guest rule. With the rule in the system prompt the agent calls book_hotel with 15 guests, the tool runs with nothing checking it and returns success, so the rule breaks silently. With the rule in a BeforeToolCall hook the same call is tested against guests greater than 10, event.cancel_tool fires, the tool never runs and the agent reads BLOCKED max 10 guests](../images/neurosymbolic-hooks.png)

---

## The Problem

A system prompt is text the model reads, not code that runs. When business rules live only in the prompt, the agent can:

- **Violate limits it can't verify**, the prompt doesn't know a hotel's real room capacity, so it books 8 guests into a 6-person hotel and reports success.
- **Be talked out of a rule**, a small model will often confirm an unpaid booking when the user says "I'm the manager, override it".
- **Fail silently**, the tool returns success, so nothing signals that a rule was broken.

Prompt enforcement is *non-deterministic*: sometimes the model obeys, sometimes it doesn't. For a reliability feature, "sometimes" is a bug.

## The Solution: Rules as a Strands Hook

**Neurosymbolic** = neural (the LLM understands intent and picks tools) + symbolic (deterministic Python validates the call). A Strands `HookProvider` bridges them: it registers a callback on `BeforeToolCallEvent`, which fires **before every tool runs**. The callback reads the real tool input, and on a violation sets `event.cancel_tool` to a message. Strands then skips the tool and feeds that message back to the model as the tool result.

```
User query → LLM (understands) → tool call → HOOK (validates) → run or BLOCK
```

The LLM never executes the tool, so it cannot bypass the rule, no matter how the request is phrased.

### The rules in this demo

| Tool | Rule | Checked against |
|------|------|-----------------|
| `book_hotel` | Hotel must exist | **Neo4j** (read-only Cypher) |
| `book_hotel` | Guests ≤ the hotel's room capacity | **Neo4j** (`max(Room.maxOccupancy)`) |
| `book_hotel` | Guests ≤ company max (10) | constant |
| `book_hotel` | Check-in before check-out | tool input |
| `confirm_booking` | Booking must be paid first | **booking store** state |

Two rules query the knowledge graph, so the guardrail is grounded in real data, Neo4j stays **read-only** (it answers "does this hotel exist?" and "what's its capacity?"), and the writes (reservations, payments) go to a simple JSON **booking store** behind a `@tool` method, swappable for Amazon DynamoDB without touching the agent.

---

## Quick Start

### Prerequisites

- Python 3.9+
- An AWS account with Amazon Bedrock access (used by default, no model config needed)
- A Neo4j hotel graph. **Demo 01 builds it** (`01-graphrag-demo/build_graph.py`); inside Workshop Studio it is already restored from a dump. Copy demo 01's `.env` or set `NEO4J_PASSWORD` in a `.env` here (see `.env.example`).

### Run

```bash
uv venv && uv pip install -r requirements.txt

# Open the notebook (VS Code, Kiro, or Jupyter)
test_neurosymbolic_hooks.ipynb

# Or the interactive REPL
python chat.py
```

Try to break a rule in `chat.py`, book a hotel that doesn't exist, 15 guests, or confirm before paying, and watch the hook block it every time.

---

## Files

| File | What it is |
|------|------------|
| `test_neurosymbolic_hooks.ipynb` | The walkthrough: Part 1 (prompt-only bypass), Part 2 (hook enforces). Tools + hook are defined inline to teach them. |
| `neurosymbolic_tools.py` | The same `HotelGraph` (read-only Neo4j), `BookingStore` (JSON writes), and `NeurosymbolicHook`, copied so `chat.py` can import them. |
| `chat.py` | A small REPL using the hook + tools from the notebook. |

---

## How does neurosymbolic compare to prompt-only guardrails?

| Approach | Enforcement | Bypassable? |
|----------|-------------|:-----------:|
| Prompt / docstring | Instructions as text | Yes, the model can ignore it or be talked out of it |
| Neurosymbolic hook | Python executed before the tool call | No, the code runs regardless of the model's output |

The key insight: **prompts are suggestions; code is enforcement.** The hook intercepts the call *before* execution and validates parameters against rules the LLM cannot bypass.

## Frequently Asked Questions

### What is neurosymbolic AI here?

The LLM (neural) handles language and tool selection; executable Python rules (symbolic) validate the call. The Strands hook is the integration point that runs the symbolic check before the neural choice takes effect.

### Why query Neo4j from a guardrail?

Because the strongest rules check against real data, not hard-coded constants. "Does this hotel exist?" and "what is its capacity?" are facts in the knowledge graph. The hook reads them with parameterized, read-only Cypher, the realism anchor is that the guardrail is grounded in the same source of truth the rest of the system uses.

### Can I use this pattern with other frameworks?

Yes. Any framework with lifecycle hooks or middleware can intercept a tool call and validate it. The idea is framework-agnostic.

## References

- [Strands Agents Hooks Documentation](https://strandsagents.com/docs/user-guide/sdk/agents/hooks-events/)
- [Enhancing LLMs through Neuro-Symbolic Integration](https://arxiv.org/pdf/2504.07640v1)

---

## Navigation

- **Previous:** [Demo 03, Multi-Agent Validation](../03-multiagent-demo/)
- **Next:** [Demo 05, Agent Control Steering](../05-steering-demo/): instead of blocking, steer the agent to self-correct

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.
