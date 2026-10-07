[< Back to Main README](../README.md)

# Agent Steering: Self-Correct Instead of Hard-Block

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Strands Agents](https://img.shields.io/badge/Strands_Agents-1.58.0-00B4D8.svg?style=flat)](https://strandsagents.com)
[![Amazon Bedrock](https://img.shields.io/badge/Amazon-Bedrock-FF9900.svg?style=flat&logo=amazon-aws)](https://aws.amazon.com/bedrock/)

**A guardrail can react to a bad request in two ways: say *no* (hard block, the task fails) or say *here's how* (steering, the agent fixes the request and finishes). This demo contrasts a blocking lifecycle hook with native Strands steering, using one real rule: a room holds at most 10 guests. Ask to book 15 and watch each approach.**

![Two stacked bands over the same request to book 15 guests where the hotel allows 10 per room. In the hard block band the hook sees 15 over 10, the tool is canceled with no guidance and the booking never happens, so the rule holds but the task dies. In the steering band a guide tells the agent to split into 10 plus 5, the agent books two rooms and all 15 guests are placed, so the rule holds and the task completes](../images/steering-flow.png)

## The Problem

A hard guardrail stops the harm but strands the user. If the rule is "max 10 guests per room" and someone asks to book 15, a blocking hook simply cancels the tool call. The agent is told *rejected* with no path forward, the booking never happens, and the only "recovery" is the model guessing a fix on its own, unguided.

Demo 04 showed hooks that **block**. The gap: blocking is the right move for a hard safety stop, but for a *recoverable* violation it throws away a completable task.

## The Solution

Native Strands **steering** reads the real tool input before the call runs and, instead of cancelling, returns a `Guide` that tells the agent *how* to comply. The agent self-corrects and completes the task.

```python
from strands.vended_plugins.steering import SteeringHandler, Guide, Proceed, ToolSteeringAction

MAX_PER_ROOM = 10

class MaxGuestsSteering(SteeringHandler):
    name = "max-guests"
    async def steer_before_tool(self, *, agent, tool_use, **kwargs) -> ToolSteeringAction:
        if tool_use.get("name") != "book_hotel":
            return Proceed()
        guests = (tool_use.get("input") or {}).get("guests", 1)
        if guests and guests > MAX_PER_ROOM:
            rooms = split(guests)                 # 15 -> [10, 5], 23 -> [10, 10, 3]
            return Guide(reason=f"...split into {len(rooms)} rooms of {rooms} and book each.")
        return Proceed()
```

Key points, and what the old version got wrong:

| Old (broken) | This demo |
|---|---|
| Required an external **Agent Control server** + `agent-control-sdk` | **Native Strands steering**, no server, nothing to run |
| Prompt forced the LLM to **narrate** the guest count so a **regex** could catch it | Reads the **real structured tool input** `tool_use["input"]["guests"]` |
| Split **hardcoded** to 10 + 5 | **Generic** `ceil(guests / 10)`: 15 → [10, 5], 23 → [10, 10, 3] |
| Deny-no-payment control that never checked payment | Removed; the rule here is a real, verifiable limit |

### Rules can live outside the code (optional)

The same handler can load its limit from an external source (`rules.json`, a config service) instead of a constant, so business rules live outside code **without** needing an Agent Control server. If you *do* use Agent Control, it is strictly optional: this demo runs fully without it.

```python
rules = json.load(open("rules.json"))     # {"max_guests_per_room": 10}
RuleSteering(rules)                         # same native steering, cap from config
```

## Architecture

- **Steering** (`MaxGuestsSteering`), a `SteeringHandler` plugin that guides `book_hotel` to split oversize parties. Reads structured tool input; no regex, no server.
- **Blocking hook** (`BlockOversizeRoomHook`), a `BeforeToolCallEvent` hook that `cancel_tool`s the oversize call, shown only as the contrast.
- **Booking store** (`BookingStore`), reservations go to a local JSON file through an `@tool` class method (`book_hotel` has a `guests` parameter so steering can read it). Swappable to Amazon DynamoDB by reimplementing `_load`/`_save`; the agent code is unchanged. Bookings never live in `agent.state`.
- **Neo4j** (`HotelCheck.hotel_exists`), **read-only**: confirms a hotel exists before booking. Neo4j never writes.
- **Model**, Amazon Bedrock (Claude Sonnet 4) by default.

## Files

| File | Purpose |
|---|---|
| `test_hooks_vs_control.ipynb` | The walkthrough: Part 1 blocking hook, Part 2 native steering, Part 3 rule-from-config |
| `steering_tools.py` | `MaxGuestsSteering`, `RuleSteering`, `BlockOversizeRoomHook`, read-only `HotelCheck` |
| `booking_store.py` | JSON booking store (swappable to DynamoDB) |
| `rules.json` | External rule source for `RuleSteering` |
| `chat.py` | Small REPL to try steering yourself |
| `.env.example` | Neo4j + region config (no secrets) |

## Run It

```bash
cd 05-steering-demo
uv venv && uv pip install -r requirements.txt
cp .env.example .env   # set NEO4J_PASSWORD

# Notebook
AWS_PROFILE=<profile> AWS_REGION=us-east-1 jupyter notebook test_hooks_vs_control.ipynb

# Or the REPL
AWS_PROFILE=<profile> AWS_REGION=us-east-1 python chat.py
```

Prerequisite: a running Neo4j with the demo-01 hotel graph (see [`01-graphrag-demo`](../01-graphrag-demo/)).

### What you'll see

```
[STEERING] 🧭 15 > 10/room → split into [10, 5]
...
✅ 2 rooms booked:
  BK0001: 10 guests at Cliffside Resort
  BK0002: 5 guests at Cliffside Resort
total guests: 15
```

The agent asked to book all 15 in one room, steering guided it to split, and the booking **completed**, instead of failing.

## When to Block vs Steer

- **Block** when the action must not happen at all (unauthorized payment, policy-prohibited operation). A hard stop is correct.
- **Steer** when the request is *recoverable*, the intent is fine, only the shape is wrong (too many guests for one room). Guide the agent to the compliant version and let it finish.

## What's Next

These five defenses run locally. In **[Demo 06, Amazon Bedrock AgentCore](../06-agentcore-boto3-demo/)** you take the same patterns to production with AgentCore Runtime, AWS Lambda, and Amazon DynamoDB.

---

## Navigation

- **Previous:** [Demo 04 - Neurosymbolic Guardrails](../04-neurosymbolic-demo/)
- **Next:** [Demo 06 - Amazon Bedrock AgentCore](../06-agentcore-boto3-demo/): take all five techniques to production.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.
