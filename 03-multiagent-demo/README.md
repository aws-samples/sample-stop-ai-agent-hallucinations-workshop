[< Back to Main README](../README.md)

# Multi-Agent Validation: Catching Confident Guesses with a Model Consortium

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Strands Agents](https://img.shields.io/badge/Strands_Agents-1.58-00B4D8.svg?style=flat)](https://strandsagents.com)
[![Amazon Bedrock](https://img.shields.io/badge/Amazon-Bedrock-FF9900.svg?style=flat&logo=amazon-aws)](https://aws.amazon.com/bedrock/)

> Some questions have a single correct answer in the database. For those, use Graph-RAG (Demo 01) or a deterministic hook (Demo 04), a second agent that re-checks a Cypher result is theatre. But many real questions are *judgements* with no single database answer ("which hotel is best for a honeymoon?"), and that is exactly where a model guesses, confidently. This demo asks the **same** question to several **heterogeneous** Amazon Bedrock models and measures their agreement: consensus means confident, divergence flags a likely hallucination.

Based on research: [Teaming LLMs to Detect and Mitigate Hallucinations (consortium consistency)](https://arxiv.org/abs/2510.19507).

![Three models, Claude Haiku, Nova Pro and Nova Lite, answer one judgement question in parallel and a node clusters the answers to measure agreement. When they agree there is one cluster with entropy 0 and the verdict is CONFIDENT. When they diverge there are three clusters with high entropy and the verdict is LIKELY HALLUCINATION. Agreement across independent models is the signal, not any single answer](../images/multiagent-consortium.png)

## The Problem

A single model answering a subjective question gives you one confident sentence and no signal about whether it's grounded or invented. On a judgement with no deterministic answer, "confident" and "correct" are not the same thing, and a lone model has no way to tell you which one you got.

Re-checking that answer with a second agent over the **same** data does not help: both agents share the same evidence and often the same biases, so they converge. For a deterministic source this is pure overhead, Graph-RAG or a hook catches violations more cheaply. Multi-agent validation earns its cost somewhere else.

## The Solution: Consortium Consistency

Consortium consistency ([arXiv:2510.19507](https://arxiv.org/abs/2510.19507)) asks the **same** question to several **heterogeneous** models, different training and architecture, so they don't share the same blind spots, then clusters their answers and measures agreement:

- **High agreement (low entropy)** → the models converge on shared evidence → **confident, likely reliable.**
- **Low agreement (high entropy)** → the models are each guessing from their own priors → **likely hallucination, flag for review.**

A model that hallucinates on its own is simply out-voted by the others.

```python
from multiagent_tools import hotel_context, ask_consortium, consortium_verdict

context = hotel_context()                       # read-only Neo4j: hotels + ratings
answers = ask_consortium("Which hotel is best for a luxury traveler?", context)
verdict = consortium_verdict(answers)
# verdict["clusters"], verdict["agreement"], verdict["entropy"], verdict["verdict"]
```

### The consortium

Three models from **two different families** (so the heterogeneity is real, not one model sampled three times):

| Role | Model | Family |
|------|-------|--------|
| `claude-haiku` | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | Anthropic |
| `nova-pro` | `amazon.nova-pro-v1:0` | Amazon |
| `nova-lite` | `us.amazon.nova-2-lite-v1:0` | Amazon |

> **No Llama.** Llama models can't be enabled in the Workshop Studio environment, so the consortium uses Claude + Nova, which are available there. Swap in any heterogeneous set you have access to.

### How agreement is measured

Each model is asked to name exactly one hotel from the context list. Because models phrase the same pick differently ("Ubud Retreat" vs "AnyCompany Ubud Retreat (Bali, 4.9)"), each answer is **canonicalized to the known hotel name it mentions** before clustering, so formatting differences don't look like disagreement. The verdict then reports the vote clusters, the agreement ratio, and a normalized entropy; one cluster → `CONFIDENT`, multiple clusters → `LIKELY HALLUCINATION (models disagree, verify)`.

### Neo4j's role here (read-only)

Neo4j supplies the hotel **context** the models reason over, a compact list of hotels with city and rating, pulled with read-only Cypher. Every demo touches the graph, but here it is the shared evidence, **not** the validator: the task has no single Cypher answer, so there is nothing deterministic to validate against. Neo4j is never written to.

## When to Use Consortium vs Hooks vs Graph-RAG

This is the honest boundary, pick the technique by the shape of the question:

| Question shape | Example | Best technique |
|----------------|---------|----------------|
| Deterministic lookup / aggregation | "How many hotels have a pool?" | **Graph-RAG** (Demo 01), one exact Cypher answer |
| A business rule that must hold | "Book 15 guests in one room" | **Neurosymbolic hook** (Demo 04), cheap, deterministic enforcement |
| Subjective / parametric judgement | "Which hotel suits a honeymoon?" | **Consortium** (this demo), no single answer; agreement is the signal |

Consortium costs several model calls per question, so reserve it for the subjective questions where it actually earns its cost. For deterministic sources, a hook or Graph-RAG is cheaper and more reliable.

## Quick Start

### Prerequisites

- Python 3.9+
- An AWS account with [Amazon Bedrock](https://aws.amazon.com/bedrock/) access and **Claude Haiku + Nova Pro + Nova Lite enabled**. **No external API key is needed.**
- A running Neo4j with the hotel graph (built in [Demo 01](../01-graphrag-demo/); inside Workshop Studio it is restored from a dump). It provides the read-only hotel context.

### Install

```bash
cd 03-multiagent-demo
uv venv && uv pip install -r requirements.txt
cp ../01-graphrag-demo/.env .env   # or set NEO4J_* yourself
```

### Run

```bash
# Notebook (recommended), open in VS Code, Kiro, or Jupyter
test_multiagent_hallucinations.ipynb

# Or the interactive REPL
AWS_PROFILE=<profile> AWS_REGION=us-east-1 python chat.py
```

### What you'll see

A **grounded** judgement (the ratings point to one clear answer) → the models agree → low entropy → `CONFIDENT`:

```
Which hotel is the single best choice overall for a luxury traveler?
clusters: {'AnyCompany ... Retreat': 3} | agreement: 1.0 | entropy: 0.0
=> CONFIDENT (consensus)
```

A **speculative** judgement (nothing in the data decides it) → the models diverge → high entropy → `LIKELY HALLUCINATION`:

```
Which hotel would a quirky artist secretly prefer?
clusters: {'A': 1, 'B': 1, 'C': 1} | agreement: 0.33 | entropy: 1.0
=> LIKELY HALLUCINATION (models disagree, verify)
```

## Files

| File | What it is |
|------|------------|
| `test_multiagent_hallucinations.ipynb` | The walkthrough: the consortium + Neo4j context (Part 1), a grounded question where models agree (Part 2), a speculative one where they diverge (Part 3). Logic is inline to teach it. |
| `multiagent_tools.py` | `hotel_context` (read-only Neo4j), `ask_consortium` (same question to each heterogeneous model), `consortium_verdict` (cluster + agreement + entropy). Copied so `chat.py` can import them. |
| `chat.py` | REPL that runs your own judgement question through the consortium. |

## Frequently Asked Questions

### Why not an Executor → Validator → Critic swarm over Neo4j?

Because over a deterministic source it's theatre: a Cypher query already returns the truth, and a second agent that queries the same thing just agrees. Modern models (even small ones) don't reliably hallucinate on simple, clear data, so there's nothing for the validator to catch, and a hook (Demo 04) catches genuine rule violations more cheaply. Consortium consistency moves multi-agent validation to where it actually helps: subjective questions with no single answer.

### What does high entropy actually tell me?

That the models are each answering from their own priors rather than from shared evidence, the exact situation where a single confident answer would be an undetected hallucination. It's a signal to verify (or to fall back to a human), not a final answer by itself.

### Does this increase latency and cost?

Yes, it's several model calls per question instead of one. That's why the "when to use" table matters: reserve the consortium for subjective judgements, and use Graph-RAG or a hook for everything with a deterministic answer.

### Can I use this with other frameworks?

Yes. Ask the same prompt to several heterogeneous models, cluster semantically-equivalent answers, and measure agreement, the pattern is framework-agnostic. This demo expresses it with Strands `Agent` + `BedrockModel`.

## References

- [Teaming LLMs to Detect and Mitigate Hallucinations (consortium consistency)](https://arxiv.org/abs/2510.19507)
- [Markov Chain Multi-Agent Debate](https://arxiv.org/html/2406.03075v1)
- [Strands Multi-Agent Documentation](https://strandsagents.com/docs/learning/multi-agent-patterns-graph-workflows/)

---

## Navigation

- **Previous:** [Demo 02 - Semantic Tool Selection](../02-semantic-tools-demo/)
- **Next:** [Demo 04 - Neurosymbolic Guardrails](../04-neurosymbolic-demo/): enforce business rules the LLM cannot bypass.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.
