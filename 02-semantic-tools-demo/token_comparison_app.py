# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Token Comparison App - Measures token savings in semantic tool discovery"""
import sys
import io
from dotenv import load_dotenv
load_dotenv()

from strands import Agent
from enhanced_tools import ALL_TOOLS
from registry import build_index, search_tools, swap_tools

# Model configuration — Amazon Bedrock (default, requires AWS credentials)
# Strands Agents uses Bedrock by default. No extra import needed.
# To use a specific Bedrock model, pass the model ID as a string:
#   MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
#
# To use a different provider (e.g., OpenAI), install the extra and configure:
#   pip install "strands-agents[openai]"
#   from strands.models.openai import OpenAIModel
#   MODEL = OpenAIModel(model_id="gpt-4o-mini")
#   (requires OPENAI_API_KEY env var — get one at https://platform.openai.com/api-keys)
#
# See all providers: https://strandsagents.com/docs/user-guide/concepts/model-providers/

PROMPT = "You are a travel assistant. Use the correct tool to answer questions."

TESTS = [
    ("What's the weather in Paris?", "get_weather"),
    ("Find flights from NYC to London", "search_flights"),
    ("Book a hotel in Rome for John", "book_hotel"),
]

def run_query_with_tokens(agent, query):
    """Run query and extract token usage from agent trace"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        response = agent(query)
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
    
    # Try to get tokens from agent's last message
    tokens = {'input': 0, 'output': 0, 'total': 0}
    if hasattr(agent, 'messages') and len(agent.messages) > 0:
        last_msg = agent.messages[-1]
        # Check different possible locations for usage data
        if hasattr(last_msg, 'usage') and last_msg.usage:
            tokens['input'] = getattr(last_msg.usage, 'input_tokens', 0)
            tokens['output'] = getattr(last_msg.usage, 'output_tokens', 0)
            tokens['total'] = tokens['input'] + tokens['output']
        elif hasattr(last_msg, 'response_metadata'):
            usage = last_msg.response_metadata.get('usage', {})
            tokens['input'] = usage.get('input_tokens', 0)
            tokens['output'] = usage.get('output_tokens', 0)
            tokens['total'] = tokens['input'] + tokens['output']
    
    # Count tools in system prompt as proxy for input tokens if no usage data
    if tokens['total'] == 0:
        # Estimate: each tool description ~50 tokens
        num_tools = len(agent.tool_registry.get_all_tools_config())
        
        # Count conversation history (messages accumulate in memory agent)
        history_tokens = 0
        if hasattr(agent, 'messages') and len(agent.messages) > 2:
            # Each previous turn adds ~100 tokens (user query + assistant response)
            history_tokens = (len(agent.messages) - 2) * 100
        
        tokens['input'] = num_tools * 50 + 100 + history_tokens  # tools + prompt + history
        tokens['output'] = 50  # estimated response
        tokens['total'] = tokens['input'] + tokens['output']
        tokens['estimated'] = True
    else:
        tokens['estimated'] = False
    
    return tokens

print("="*70)
print("TOKEN COMPARISON: Traditional vs Semantic Tool Discovery")
print("="*70)

build_index(ALL_TOOLS)

# Test 1: Traditional
print(f"\n[1/3] Traditional - {len(ALL_TOOLS)} tools every query...")
trad_tokens = []
for query, _ in TESTS:
    agent = Agent(tools=ALL_TOOLS, system_prompt=PROMPT)
    tokens = run_query_with_tokens(agent, query)
    trad_tokens.append(tokens)
    est = " (est)" if tokens.get('estimated') else ""
    print(f"  {query[:40]:40} | {tokens['total']:5} tokens{est}")

# Test 2: Semantic
print("\n[2/3] Semantic - Top-3 tools per query...")
sem_tokens = []
for query, _ in TESTS:
    selected = search_tools(query, top_k=3)
    agent = Agent(tools=selected, system_prompt=PROMPT)
    tokens = run_query_with_tokens(agent, query)
    sem_tokens.append(tokens)
    est = " (est)" if tokens.get('estimated') else ""
    print(f"  {query[:40]:40} | {tokens['total']:5} tokens{est}")

# Test 3: Semantic + Memory
print("\n[3/3] Semantic + Memory - Single agent, swap tools...")
initial_tools = search_tools(TESTS[0][0], top_k=3)
memory_agent = Agent(tools=initial_tools, system_prompt=PROMPT)
mem_tokens = []

for query, _ in TESTS:
    selected = search_tools(query, top_k=3)
    swap_tools(memory_agent, selected)
    tokens = run_query_with_tokens(memory_agent, query)
    mem_tokens.append(tokens)
    est = " (est)" if tokens.get('estimated') else ""
    print(f"  {query[:40]:40} | {tokens['total']:5} tokens{est}")

# Results
trad_total = sum(t['total'] for t in trad_tokens)
sem_total = sum(t['total'] for t in sem_tokens)
mem_total = sum(t['total'] for t in mem_tokens)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nTotal tokens:")
print(f"  Traditional:     {trad_total:6} tokens")
if trad_total > 0:
    savings_sem = trad_total - sem_total
    savings_mem = trad_total - mem_total
    print(f"  Semantic:        {sem_total:6} tokens ({100*savings_sem/trad_total:+.1f}%)")
    print(f"  Semantic+Memory: {mem_total:6} tokens ({100*savings_mem/trad_total:+.1f}%)")
    
    print(f"\n{'Query':<45} {'Trad':>8} {'Sem':>8} {'Mem':>8} {'Saved':>8}")
    print("-"*70)
    for i, (query, _) in enumerate(TESTS):
        t = trad_tokens[i]['total']
        s = sem_tokens[i]['total']
        m = mem_tokens[i]['total']
        print(f"{query[:44]:<45} {t:8} {s:8} {m:8} {t-m:8}")
    
    print(f"\n✅ Key Finding:")
    print(f"   • Traditional sends {len(ALL_TOOLS)} tools every query")
    print(f"   • Semantic sends only 3 tools per query")
    print(f"   • Semantic saves {savings_sem} tokens ({100*savings_sem/trad_total:.1f}%) vs Traditional")
    print(f"   • Memory approach accumulates conversation history")
    print(f"   • Memory uses {mem_total - sem_total} MORE tokens than Semantic (conversation context)")
    print(f"   • But still saves {savings_mem} tokens ({100*savings_mem/trad_total:.1f}%) vs Traditional")
    
    if trad_tokens[0].get('estimated'):
        print(f"\n⚠️  Note: Token counts are estimated")
        print(f"   Traditional: ~{len(ALL_TOOLS)} tools × 50 tokens/tool = ~{len(ALL_TOOLS)*50} tokens/query")
        print(f"   Semantic: ~3 tools × 50 tokens/tool = ~150 tokens/query")
        print(f"   Memory: ~150 tokens + conversation history (~100 tokens/turn)")
else:
    print(f"  Semantic:        {sem_total:6} tokens")
    print(f"  Semantic+Memory: {mem_total:6} tokens")
    print("\n⚠️  No token data captured from model.")
