# Module 7: AgentCore Long-Term Memory

Tests cross-session memory recall with Amazon Bedrock AgentCore.

## Prerequisites

**Module 6 must be completed first.** This module reuses:
- AgentCore Gateway (HotelBookingGateway)
- Lambda tools (7 booking functions)
- IAM role (workshop-AgentCoreExecutionRole)

## What This Module Does

1. Creates an AgentCore Memory resource with two strategies:
   - `UserPreferences` (hotel preferences: stars, cities)
   - `UserFacts` (user information: name, loyalty number)

2. Deploys a second agent with `memory_mode="STM_AND_LTM"`
   - Code: `booking_agent_with_memory.py`
   - Memory integration via `AgentCoreMemorySessionManager`

3. Tests memory across sessions:
   - **Session A:** User provides name and preferences
   - **Wait 60s:** AgentCore extracts strategies asynchronously
   - **Session B:** New session, same actor → agent recalls from LTM

## Files

| File | Purpose |
|------|---------|
| `test_memory.ipynb` | Notebook to create memory and test cross-session recall |
| `booking_agent_with_memory.py` | Strands agent with memory integration |
| `agent_requirements.txt` | Python dependencies |

## Key Differences from Module 6

| Module 6 | Module 7 |
|----------|----------|
| `memory_mode="STM_ONLY"` | `memory_mode="STM_AND_LTM"` |
| No Memory resource | AgentCore Memory with strategies |
| `booking_agent.py` | `booking_agent_with_memory.py` |
| Session-scoped memory | Cross-session memory |

## Actor ID

Memory is scoped by **actor ID** (format: `user-{8-char-uuid}`). Same actor ID across sessions → shared memory.

Actor ID is passed via custom HTTP header:
```python
'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id': user_id
```

## Run the Demo

```bash
cd 07-agentcore-memory-demo
jupyter notebook test_memory.ipynb
```

Execute all cells. The notebook:
1. Recovers Module 6 resources
2. Creates Memory resource
3. Deploys memory-enabled agent
4. Tests STM (same session)
5. Waits 60s for strategy extraction
6. Tests LTM (different session, same actor)

## Expected Results

**Session A (STM):**
```
User: My name is Alex and I prefer 4-star hotels in Paris.
Agent: I've noted your preferences, Alex.

User: What's my loyalty number?
Agent: Your loyalty number is HOTEL-12345.
```

**Session B (LTM, 60s later):**
```
User: Do you remember me? What's my name?
Agent: Yes, I remember you! Your name is Alex.

User: Find me a hotel based on my preferences
Agent: Based on your preference for 4-star hotels in Paris, I recommend...
```

## Cleanup

To delete the Memory resource:
```python
agentcore_control.delete_memory(memoryIdentifier=MEMORY_ID)
```

To delete the memory-enabled agent:
```python
agentcore.delete_agent_runtime(agentRuntimeIdentifier=MEMORY_RUNTIME_ID)
```
