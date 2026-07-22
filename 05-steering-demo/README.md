[< Back to Main README](../README.md)

# AI Agent Guardrails That Self-Correct Instead of Block

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Strands Agents](https://img.shields.io/badge/Strands_Agents-1.27+-00B4D8.svg?style=flat)](https://strandsagents.com)
[![Agent Control](https://img.shields.io/badge/Agent_Control-Steer_&_Deny-orange.svg?style=flat)](https://github.com/agentcontrol/agent-control)

> Hooks are functions that run at specific points in an agent's lifecycle. In this demo, hooks intercept tool calls and block them using `cancel_tool` when a business rule is violated. The agent reports failure and the user must retry. Agent Control goes further: it **steers** the agent to fix the problem and complete the task, instead of failing.

![Hooks (Block) vs Agent Control (Self-Correct) comparison](./images/hooks-vs-agent-control.jpg)

Based on: [Strands Agents with Agent Control](https://strandsagents.com/blog/strands-agents-with-agent-control/)

This demo uses Strands Agents and Agent Control. The guardrail patterns demonstrated here are hooks, steering, and symbolic rules. They can be applied with other agent frameworks that support lifecycle hooks.

---

> ## ⚠️ Read this before you start
>
> **This demo needs a running Agent Control server. Every other demo in this workshop needs only AWS credentials.**
>
> The server is a separate open-source product. It is not bundled with this workshop, and it is not part of the `agent-control-sdk` package that `requirements.txt` installs. You must install and start it yourself from [github.com/agentcontrol/agent-control](https://github.com/agentcontrol/agent-control) before Test 2 will run. Budget setup time for this.
>
> **What runs without the server:** Test 1, the hooks half of the comparison. It is the baseline the demo measures against, and it needs only AWS credentials.
>
> **What does not:** Test 2, the Agent Control steering half. That is the technique this demo teaches.
>
> Both the script and the notebook check for the server at startup and stop with instructions if it is missing, so a missing server produces a readable message rather than a stack trace.
>
> **No model-provider API key is required.** This demo runs on Amazon Bedrock through the Strands default model, the same as every other demo. Earlier versions asked for an `OPENAI_API_KEY`. That requirement is gone.

---

## The Problem with Blocking

[Demo 04 (Neurosymbolic Guardrails)](../04-neurosymbolic-demo/) demonstrates that hooks can enforce business rules at the tool level. When a rule is violated, `cancel_tool` blocks the call and the agent tells the user it cannot proceed.

But blocking alone has limitations. If a user requests 15 guests and the maximum is 10, the agent could adjust to 10 and complete the booking. Instead, with hooks alone, it asks the user to change their request, interrupting the flow.

## The Solution: Steer Instead of Block

![Agent Control steer flow: User Request → LLM → Agent Control server evaluates → Self-Correct → Final Response](./images/Agent-Control.jpg)

[Agent Control](https://github.com/agentcontrol/agent-control) introduces **steer controls** — server-managed policies that guide the agent to self-correct when a violation is detected, instead of terminating the operation:

| Approach | 15 guests requested | Result |
|----------|-------------------|--------|
| **Hooks** | BLOCKED | "Would you like to adjust?" The flow stops. |
| **Agent Control** | Guide("split across rooms") | Books BK002 with 10 guests and BK003 with 5. The flow completes. |

## How It Differs from Hooks

| | Hooks ([Demo 04](../04-neurosymbolic-demo/)) | Agent Control (this demo) |
|---|---|---|
| Where rules live | Python code (`rules.py`) | Server — API/dashboard |
| When a rule fails | `cancel_tool = "BLOCKED"` → agent fails | `Guide("split across rooms")` → agent retries corrected |
| To change a rule | Edit code, redeploy | API call or dashboard — no code changes |
| Integration | `HookProvider` + `hooks=[...]` | `Plugin` + `plugins=[...]` |
| Evaluators | Custom Python lambdas | regex (pattern matching), list (exact value matching), JSON schema (structure validation), AI via Galileo Luna-2 (semantic evaluation) |
| Scope | `BeforeToolCallEvent` only | LLM input/output, tool input/output, pre/post |

## The Tools

Three booking tools in `tools.py` — clean, no validation logic:

| Tool | What it does | Key behavior |
|------|-------------|--------------|
| `book_hotel(hotel, check_in, check_out, guests)` | Books a hotel room | Returns `"SUCCESS: Booking BK001..."` — no guest limit in the tool |
| `process_payment(amount, booking_id)` | Processes payment | Returns `"SUCCESS"` or `"ERROR: Booking not found"` |
| `confirm_booking(booking_id)` | Confirms a booking | Returns `"SUCCESS: Confirmed BK001"` |

The tools do NOT enforce the max-guests rule. That is the guardrail layer's job — either Hooks or Agent Control.

Agent Control integrates as a Plugin with two lines:

```python
# Hooks (existing approach — block):
agent = Agent(tools=[...], hooks=[MaxGuestsHook()])

# Agent Control (new approach — steer):
agent = Agent(tools=[...], plugins=[AgentControlPlugin(...), AgentControlSteeringHandler(...)])
```

## What We Test

Same query, same tools, same model — only the guardrail changes:

| Test | Guardrail | Outcome | Needs the server |
|------|-----------|---------|---|
| 1 — Hooks | `MaxGuestsHook` with `cancel_tool` | Agent is BLOCKED → asks user what to do | No |
| 2 — Agent Control | `AgentControlSteeringHandler` with `Guide()` | Agent splits into 2 rooms, 10 + 5 guests → booking completes | **Yes** |

Both tests use the Strands default Bedrock model. Neither passes a `model=` argument, which is what makes "same model" true rather than aspirational.

---

## Two Ways to Define Controls

| Mode | Best for | How it works |
|------|----------|-------------|
| **Server** (this demo) | Teams, production, dashboard management | Controls live on the Agent Control server. Change them via API or dashboard without redeploying. |
| **Local YAML** | Quick prototyping, single-developer projects | Controls defined in a `controls.yaml` file, evaluated in-process with no server. |

This demo uses the **server approach**. See the [Agent Control docs](https://docs.agentcontrol.dev/) for server setup.

> **Local YAML mode does not work in `agent-control-sdk` 8.3.0, the version this demo pins.**
>
> `agent_control.init()` accepts a `controls_file=` argument and its docstring promises to "auto-discover and load local `controls.yaml` as fallback". The parameter is accepted and then ignored. The installed package contains no YAML loading code at all, so `init(controls_file=...)` silently loads nothing and leaves the guardrail layer inert.
>
> A `controls.yaml` ships here anyway. It is the source of truth for what `setup_controls.py` registers on the server, and `demo_hooks_vs_control.py --local-controls` can load it directly into SDK state for local development. That flag is a workaround for the gap above, not a supported mode. It bypasses server-side policy resolution, and the controls in the file declare `execution: sdk` while `setup_controls.py` registers them as `execution: server`. **Results from `--local-controls` are not evidence that the server path works.**

---

## Prerequisites

| # | Requirement | Needed for | Notes |
|---|---|---|---|
| 1 | Python 3.9+ | everything | |
| 2 | AWS credentials with Amazon Bedrock access | everything | The only credential this demo uses. Both tests run on the Strands default model. |
| 3 | **A running [Agent Control server](https://github.com/agentcontrol/agent-control)** | **Test 2 only** | **Hard prerequisite.** Separate product, installed and started by you. Not bundled with this workshop and not part of `agent-control-sdk`. |

No model-provider API key is required. This demo does not use OpenAI.

---

## Quick Start

### 1. Start the Agent Control server

Install and start it by following the [Agent Control setup instructions](https://github.com/agentcontrol/agent-control). Nothing in this workshop installs it for you.

```bash
# Verify it is running and is actually Agent Control.
# /health alone is not enough: port 8000 is a common local development port,
# and an unrelated service answering there will pass a naive check.
curl 127.0.0.1:8000/health
curl -o /dev/null -w '%{http_code}\n' 127.0.0.1:8000/api/v1/agents   # 404 means it is NOT Agent Control
```

If your server listens elsewhere, point the demo at it:

```bash
export AGENT_CONTROL_URL=http://<host>:<port>
```

### 2. Install dependencies

```bash
uv venv && uv pip install -r requirements.txt
```

### 3. Set up controls on the server

```bash
uv run setup_controls.py
```

### 4. Run the comparison

```bash
uv run demo_hooks_vs_control.py
```

Or open `test_hooks_vs_control.ipynb` in your IDE. VS Code, Kiro, and any other editor with notebook support all work.

The script exits `2` with setup instructions if no Agent Control server is reachable, `1` if the demo's headline claim fails, and `0` on success.

---

## Controls Created by setup_controls.py

| Control | Type | Scope | What it does |
|---------|------|-------|-------------|
| `steer-max-guests` | STEER | LLM output (post) | Guides agent to reduce guest count to <= 10 and inform the user |
| `deny-no-payment` | DENY | Tool input (pre) on `confirm_booking` | Blocks booking confirmation without payment |

---

## Why steering is bounded

Steering is a retry loop, and every retry loop needs a stop condition.

The `steer-max-guests` control matches a guest count above 10 in the model's output. The trouble is that the agent's compliant reply usually restates the total, as in "splitting your 15 guests across 2 rooms", which matches the same pattern and fires the control again. Agent Control's regex evaluator runs on RE2, which supports no lookahead, so the pattern cannot be narrowed to exclude the agent's own correction.

Unbounded, this livelocks. Measured behaviour with no cap, from `logs/05-baseline-asshipped.log`:

```
🔄 Steered: 9 time(s)          # steering fired on the agent's own corrections
Tool #1 .. Tool #17            # book_hotel called 17 times; the demo expects 2
```

After nine injected steering messages the model concluded it was under a prompt-injection attack, said so in its output, refused to split the booking, and booked all 15 guests into a single room. That is the exact operation the control existed to prevent, so the guardrail failed open.

`MAX_STEERS` in `demo_hooks_vs_control.py` bounds the loop to one corrective nudge. The general lesson: **cap steer retries, and assert on your own state rather than on the model's wording.** A guardrail that nags indefinitely eventually gets ignored.

---

## Expected Output

Generated from a real run. Timings and token counts vary between runs.

```
Approach                                Time              Outcome
-----------------------------------------------------------------
Hooks (cancel_tool)                    5.8s           no-booking
Agent Control (steer)                  8.6s       split-bookings

✅ CLAIM HOLDS — hooks hard-blocked; Agent Control steered
   the agent to self-correct and complete the booking.
```

Test 2 detail from the same run:

```
🔄 Steered: 1 time(s)
📒 Bookings created: 2 — guests per booking: [10, 5]
```

The pass condition is checked against the booking ledger in `tools.py`, not against the model's prose. Test 1 passes when the hook blocks and no over-limit booking reaches the ledger. Test 2 passes when two or more bookings exist, none exceeds 10 guests, and they sum to the 15 guests requested.

---

## Cleanup

Stop the Agent Control server following the [shutdown instructions](https://docs.agentcontrol.dev/).

---

## Files

| File | Purpose |
|------|---------|
| `tools.py` | Booking tools, clean, with no validation logic |
| `controls.yaml` | Control definitions, the source of truth for what `setup_controls.py` registers |
| `setup_controls.py` | Creates the steer and deny controls on the Agent Control server |
| `demo_hooks_vs_control.py` | Runs both approaches on the same query and compares results |
| `test_hooks_vs_control.ipynb` | Interactive notebook version |
| `requirements.txt` | Dependencies |

---

## References

### Research
- [ATA: Autonomous Trustworthy Agents (2024)](https://arxiv.org/html/2510.16381v1) — Guardrail failure patterns in AI agents
- [Enhancing LLMs through Neuro-Symbolic Integration](https://arxiv.org/pdf/2504.07640v1) — Combining neural + symbolic reasoning

### Strands Agents
- [Strands Agents with Agent Control](https://strandsagents.com/blog/strands-agents-with-agent-control/) — Blog announcement
- [Agent Control Plugin](https://strandsagents.com/docs/community/plugins/agent-control/) — Strands integration docs
- [Strands Hooks](https://strandsagents.com/docs/user-guide/concepts/agents/hooks/) — `BeforeToolCallEvent`, `cancel_tool`
- [Strands Steering](https://strandsagents.com/docs/user-guide/concepts/plugins/steering/) — `Guide`, `Proceed`, `SteeringHandler`
- [Strands Model Providers](https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/) — Swap to Amazon Bedrock, Anthropic, Ollama

### Agent Control
- [Agent Control GitHub](https://github.com/agentcontrol/agent-control) — Open source, Apache 2.0
- [Agent Control Docs](https://docs.agentcontrol.dev/) — Server setup and API reference

---

## Frequently Asked Questions

### What is the difference between Agent Control and Amazon Bedrock AgentCore?

They are different products. **Agent Control** is an open-source guardrail server that evaluates agent actions and returns steer/deny decisions — it runs locally or on any infrastructure. **Amazon Bedrock AgentCore** is an AWS managed service for hosting and deploying agents in production with MCP routing, observability, and scaling. Demo 05 uses Agent Control for steering; [Demo 06](../06-agentcore-cdk-demo/) uses Amazon Bedrock AgentCore for production deployment.

### When should I use steering (Agent Control) instead of blocking (hooks)?

Use **hooks** (blocking) when the violation is a hard constraint that cannot be self-corrected — for example, confirming a booking without payment. Use **steering** (Agent Control) when the agent can adjust and complete the task — for example, reducing 15 guests to the maximum of 10 and informing the user. Steering reduces user friction because the task completes instead of failing.

### Can I use the steering pattern with other agent frameworks?

Yes. The steer-instead-of-block pattern is framework-agnostic. Agent Control integrates as a plugin with Strands Agents, but the concept — intercepting LLM output, evaluating it against rules, and injecting corrective guidance — can be implemented in LangGraph, CrewAI, AutoGen, or any framework that supports middleware or output hooks.

---

## Navigation

- **Previous:** [Demo 04 - Neurosymbolic Guardrails](../04-neurosymbolic-demo/)
- **Next:** [Demo 06 - Amazon Bedrock AgentCore Production](../06-agentcore-cdk-demo/) — Deploy all techniques to production on AWS

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.
