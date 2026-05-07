# Module 7: Production Deploy with Memory

Deploy AgentCore agent with long-term memory that recalls user preferences across sessions.

## Prerequisites

**Module 6 must be completed first.** This module reuses:
- AgentCore Gateway (`HotelBookingGateway`)
- Lambda tools (8 booking + graph functions)
- DynamoDB tables (Hotels, Bookings, SteeringRules)
- IAM role (`workshop-AgentCoreExecutionRole`)

## What This Module Does

1. **Creates AgentCore Memory resource** with two strategies:
   - `UserPreferences` — hotel preferences (stars, cities)
   - `UserFacts` — user information (name, loyalty number)

2. **Deploys a second agent** with `memory_mode="STM_AND_LTM"`:
   - Code: `booking_agent_with_memory.py`
   - Memory integration via `AgentCoreMemorySessionManager`
   - Same infrastructure as Module 6 (Gateway + Lambdas)

3. **Tests cross-session memory recall**:
   - **Session A:** User shares name and preferences
   - **Wait 60s:** AgentCore extracts strategies asynchronously
   - **Session B:** New session, same actor → agent recalls from long-term memory

## Files

| File | Purpose |
|------|---------|
| `deploy_memory_agent.ipynb` | Create Memory resource, deploy agent, test cross-session recall |
| `booking_agent_with_memory.py` | Strands agent with AgentCore Memory integration |
| `agent_requirements.txt` | Python dependencies (strands-agents, bedrock-agentcore-starter-toolkit) |

## Key Differences from Module 6

| Module 6 | Module 7 |
|----------|----------|
| Runtime memory only (ephemeral) | AgentCore Memory with `STM_AND_LTM` |
| No Memory resource | AgentCore Memory with strategies |
| `booking_agent.py` | `booking_agent_with_memory.py` |
| Conversation buffer (lost on session end) | Persistent memory (recalled across sessions) |

**Memory types explained:**
- **Runtime memory** (Module 6): Temporary conversation buffer maintained by Strands Agent. Lost when session ends.
- **STM (Short-Term Memory)**: Session-scoped memory managed by AgentCore. Lost when session ends.
- **LTM (Long-Term Memory)**: Persistent memory managed by AgentCore. Extracts strategies asynchronously and recalls across sessions.

## How Memory Works

**AgentCore Memory** stores extracted strategies from conversations and recalls them across sessions. The memory-enabled agent integrates with AgentCore Memory via the Strands Agent `session_manager` parameter:

```python
from strands_agents.models import Agent
from bedrock_agentcore_starter_toolkit import AgentCoreMemoryConfig, AgentCoreMemorySessionManager

# Configure memory retrieval
memory_config = AgentCoreMemoryConfig(
    memory_id=MEMORY_ID,              # AgentCore Memory resource ID
    session_id=context.session_id,     # Current session
    actor_id=actor_id,                 # User identifier
    retrieval_config={
        f"/users/{actor_id}/facts": RetrievalConfig(top_k=3, relevance_score=0.5),
        f"/users/{actor_id}/preferences": RetrievalConfig(top_k=3, relevance_score=0.5)
    }
)

# Create session manager
session_manager = AgentCoreMemorySessionManager(memory_config, region)

# Pass to Strands Agent
agent = Agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    hooks=hooks,
    session_manager=session_manager  # ← Enables AgentCore Memory integration
)
```

**Actor ID** scopes memory by user (format: `user-{8-char-uuid}`). Same actor ID across sessions → shared memory. Passed via custom HTTP header:
```python
'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id': user_id
```

## Run the Demo

Open `deploy_memory_agent.ipynb` and execute all cells. The notebook:
1. Recovers Module 6 resources (Gateway, IAM role)
2. Creates AgentCore Memory resource with strategies
3. Deploys memory-enabled agent with `memory_mode="STM_AND_LTM"`
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
