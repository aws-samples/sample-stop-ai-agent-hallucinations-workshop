[< Back to Main README](../README.md)

# AI Agent Guardrails That Self-Correct Instead of Block

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Strands Agents](https://img.shields.io/badge/Strands_Agents-1.27+-00B4D8.svg?style=flat)](https://strandsagents.com)
[![Agent Control](https://img.shields.io/badge/Agent_Control-Steer_&_Deny-orange.svg?style=flat)](https://github.com/agentcontrol/agent-control)

> Hooks are functions that run at specific points in an agent's lifecycle. In this demo, hooks intercept tool calls and block them using `cancel_tool` when a business rule is violated. The agent reports failure and the user must retry. Agent Control goes further: it **steers** the agent to fix the problem and complete the task, instead of failing.

![Two bands over the same request for 15 guests against the same 10-guest rule. The hook reads the tool input, sees guests=15, and cancel_tool returns a BLOCKED message in place of a booking, so no room is booked and someone has to ask again. The steer control reads the text the LLM wrote, matches any count from 11 up, and sends back the instruction to call book_hotel twice with 10 and then 5, so the user is told the reservation was split into two rooms at the same hotel](./images/hooks-vs-steering.png)

Based on: [Strands Agents with Agent Control](https://strandsagents.com/blog/strands-agents-with-agent-control/)

This demo uses Strands Agents and Agent Control. The guardrail patterns demonstrated (hooks, steering, symbolic rules) can be applied with other agent frameworks that support lifecycle hooks.

---

## The Problem with Blocking

[Demo 04 (Neurosymbolic Guardrails)](../04-neurosymbolic-demo/) demonstrates that hooks can enforce business rules at the tool level. When a rule is violated, `cancel_tool` blocks the call and the agent tells the user it cannot proceed.

But blocking alone has limitations. If a user requests 15 guests and the maximum is 10 per room, the booking can still go ahead as two rooms. With hooks alone, the agent asks the user to change their request instead, interrupting the flow.

## The Solution: Steer Instead of Block

![One turn, five steps. The user asks to book AnyCompany Lisbon Resort for 15 guests, the LLM describes the booking it is about to make, a regex in the steer control matches any guest count from 11 up in that output, the control sends back the instruction to call book_hotel twice with 10 and then 5 guests, and the agent tells the user the reservation was split into two rooms](./images/steering-loop.png)

[Agent Control](https://github.com/agentcontrol/agent-control) introduces **steer controls**: server-managed policies that guide the agent to self-correct when a violation is detected, instead of terminating the operation:

| Approach | 15 guests requested | Result |
|----------|-------------------|--------|
| **Hooks** | `cancel_tool` blocks `book_hotel` | Nothing is booked, the user has to ask again |
| **Steering** | Guidance: call `book_hotel` twice, with 10 then 5 | Two rooms booked, the split explained to the user |

## How It Differs from Hooks

| | Hooks ([Demo 04](../04-neurosymbolic-demo/)) | Agent Control (this demo) |
|---|---|---|
| Where rules live | Python code (`rules.py`) | Server (API or dashboard) |
| When a rule fails | `cancel_tool = "BLOCKED"` → agent fails | The control's steering context goes back to the agent → it calls the tool again as guided |
| To change a rule | Edit code, redeploy | API call or dashboard, no code changes |
| Integration | `HookProvider` + `hooks=[...]` | `Plugin` + `plugins=[...]` |
| Evaluators | Custom Python lambdas | regex (pattern matching), list (exact value matching), JSON schema (structure validation), AI via Galileo Luna-2 (semantic evaluation) |
| Scope | `BeforeToolCallEvent` only | LLM input/output, tool input/output, pre/post |

## The Tools

Three booking tools in `tools.py`, with no validation logic:

| Tool | What it does | Key behavior |
|------|-------------|--------------|
| `book_hotel(hotel, check_in, check_out, guests)` | Books a hotel room | Returns `"SUCCESS: Booking BK001..."`, no guest limit in the tool |
| `process_payment(amount, booking_id)` | Processes payment | Returns `"SUCCESS"` or `"ERROR: Booking not found"` |
| `confirm_booking(booking_id)` | Confirms a booking | Returns `"SUCCESS: Confirmed BK001"` |

The tools do NOT enforce the max-guests rule. That is the guardrail layer's job, either Hooks or Agent Control.

Agent Control integrates as a Plugin with two lines:

```python
# Hooks (existing approach, block):
agent = Agent(tools=[...], hooks=[MaxGuestsHook()])

# Agent Control (new approach, steer):
agent = Agent(tools=[...], plugins=[AgentControlPlugin(...), AgentControlSteeringHandler(...)])
```

## What We Test

Same query, same tools, same model. Only the guardrail changes:

| Test | Guardrail | Outcome |
|------|-----------|---------|
| 1. Hooks | `MaxGuestsHook` with `cancel_tool` | Agent is BLOCKED → asks user what to do |
| 2. Agent Control | `AgentControlSteeringHandler` | Agent splits the booking into two rooms, 10 guests and 5 → both bookings complete |

---

## Two Ways to Define Controls

| Mode | Best for | How it works |
|------|----------|-------------|
| **Server** (this demo) | Teams, production, dashboard management | Controls live on the Agent Control server, so you change them via API or dashboard without redeploying |
| **Local YAML** | Quick prototyping, single-developer projects | Controls defined in a `controls.yaml` file, no server needed: `agent_control.init(controls_file="controls.yaml")` |

This demo uses the **server approach**. See the [Agent Control docs](https://docs.agentcontrol.dev/) for YAML-based local mode or server setup instructions.

---

## Prerequisites

- Python 3.9+
- OpenAI API key, get one at https://platform.openai.com/api-keys (or use any [supported model provider](https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/) such as Amazon Bedrock or Anthropic)
- [Agent Control server](https://docs.agentcontrol.dev/) running locally (see [setup instructions](https://github.com/agentcontrol/agent-control))

---

## Quick Start

### 1. Start Agent Control server

Follow the [Agent Control setup instructions](https://github.com/agentcontrol/agent-control) to start the server locally.

```bash
# Verify it's running
# Replace <PORT> with the Agent Control server port (default: 8000)
curl 127.0.0.1:8000/health
```

### 2. Install dependencies

```bash
uv venv && uv pip install -r requirements.txt
```

### 3. Configure API key

```bash
# Create .env with your OpenAI key
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 4. Setup controls on the server

```bash
uv run setup_controls.py
```

### 5. Run the comparison

```bash
uv run test_hooks_vs_control.py
```

Or open `test_hooks_vs_control.ipynb` in your IDE (VS Code, Kiro, or any editor with notebook support).

---

## Controls Created by setup_controls.py

| Control | Type | Scope | What it does |
|---------|------|-------|-------------|
| `steer-max-guests` | STEER | LLM output (post) | Guides agent to call `book_hotel` twice, with 10 guests and then 5, and report the split |
| `deny-no-payment` | DENY | Tool input (pre) on `confirm_booking` | Blocks booking confirmation without payment |

---

## Expected Output

The two status lines the script prints for each test:

```
Test 1 — Hooks:          🚫 Agent was BLOCKED — reported failure or asked user to change
Test 2 — Agent Control:  ✅ Agent self-corrected — split into 2 rooms (10 + 5 guests)
```

Timings and token counts are printed alongside them and vary per run.

---

## Cleanup

Stop the Agent Control server following the [shutdown instructions](https://docs.agentcontrol.dev/).

---

## Files

| File | Purpose |
|------|---------|
| `tools.py` | Booking tools, no validation logic |
| `setup_controls.py` | Creates steer + deny controls on Agent Control server |
| `test_hooks_vs_control.py` | Runs both approaches on the same query, compares results |
| `test_hooks_vs_control.ipynb` | Interactive notebook version |
| `requirements.txt` | Dependencies |

---

## References

### Research
- [ATA: Autonomous Trustworthy Agents (2024)](https://arxiv.org/html/2510.16381v1): guardrail failure patterns in AI agents
- [Enhancing LLMs through Neuro-Symbolic Integration](https://arxiv.org/pdf/2504.07640v1): combining neural + symbolic reasoning

### Strands Agents
- [Strands Agents with Agent Control](https://strandsagents.com/blog/strands-agents-with-agent-control/): blog announcement
- [Agent Control Plugin](https://strandsagents.com/docs/community/plugins/agent-control/): Strands integration docs
- [Strands Hooks](https://strandsagents.com/docs/user-guide/concepts/agents/hooks/): `BeforeToolCallEvent`, `cancel_tool`
- [Strands Steering](https://strandsagents.com/docs/user-guide/concepts/plugins/steering/): `Guide`, `Proceed`, `SteeringHandler`
- [Strands Model Providers](https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/): swap to Amazon Bedrock, Anthropic, Ollama

### Agent Control
- [Agent Control GitHub](https://github.com/agentcontrol/agent-control): open source, Apache 2.0
- [Agent Control Docs](https://docs.agentcontrol.dev/): server setup and API reference

---

## Frequently Asked Questions

### What is the difference between Agent Control and Amazon Bedrock AgentCore?

They are different products. **Agent Control** is an open-source guardrail server that evaluates agent actions and returns steer/deny decisions, and it runs locally or on any infrastructure. **Amazon Bedrock AgentCore** is an AWS managed service for hosting and deploying agents in production with MCP routing, observability, and scaling. Demo 05 uses Agent Control for steering; [Demo 06](../06-agentcore-boto3-demo/) uses Amazon Bedrock AgentCore for production deployment.

### When should I use steering (Agent Control) instead of blocking (hooks)?

Use **hooks** (blocking) when the violation is a hard constraint that cannot be self-corrected, for example confirming a booking without payment. Use **steering** (Agent Control) when the agent can adjust and complete the task, for example splitting a 15-guest request into two rooms of 10 and 5 and telling the user. Steering reduces user friction because the task completes instead of failing.

### Can I use the steering pattern with other agent frameworks?

Yes. The steer-instead-of-block pattern is framework-agnostic. Agent Control integrates as a plugin with Strands Agents, but the concept (intercepting LLM output, evaluating it against rules, and injecting corrective guidance) can be implemented in any framework that supports middleware or output hooks.

---

## Navigation

- **Previous:** [Demo 04 - Neurosymbolic Guardrails](../04-neurosymbolic-demo/)
- **Next:** [Demo 06 - Amazon Bedrock AgentCore Production](../06-agentcore-boto3-demo/): deploy all techniques to production on AWS

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.
