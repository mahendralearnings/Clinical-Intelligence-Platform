# Clinical Intelligence Platform

An enterprise-grade **healthcare RAG + Agentic AI platform**, built end-to-end with production engineering discipline — not a tutorial demo.

It answers clinical questions strictly from a trusted document library (SOPs, drug manuals, trial summaries), refuses to answer when the information isn't there (no hallucination), reasons through multi-step questions with tool-using agents, coordinates a multi-agent team, and defends itself against prompt-injection attacks — all with evaluation, observability, and 70+ automated tests.

> **Why this project exists:** to demonstrate, in one coherent system, the full modern GenAI engineering stack — RAG, agents, multi-agent orchestration, evaluation, observability, and guardrails — with the kind of production discipline (clean architecture, swappable providers, a real test suite, cost-awareness) that separates a shipped system from a demo.

---

## Table of Contents
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [The one pattern that repeats everywhere](#the-one-pattern-that-repeats-everywhere)
- [Tech stack](#tech-stack)
- [Features by phase](#features-by-phase)
- [Getting started](#getting-started)
- [Running the demo UI](#running-the-demo-ui)
- [Testing philosophy](#testing-philosophy)
- [Key engineering lessons](#key-engineering-lessons)
- [What's deliberately out of scope](#whats-deliberately-out-of-scope)

---

## What it does

Ask it a clinical question, and depending on the mode you choose:

- **Plain RAG** — finds the most relevant document passages and generates a cited, grounded answer. Asks something out-of-scope? It says *"I don't know based on the available documents"* instead of making something up.
- **Agent (LangGraph)** — a single reasoning agent that can call tools (document search + a safe calculator) across multiple steps. Example: *"What's the max daily metformin dose, split across 3 doses?"* → it searches for the dose, then calculates, then answers.
- **Crew (CrewAI)** — a 3-agent team (Researcher → Compliance Reviewer → Writer) that hands work down a chain, each specialist doing one job.

All three run behind a JWT-authenticated FastAPI backend with role-based access control, and can be explored through a simple Streamlit demo UI.

---

## Architecture

Every layer depends only on the layer above it. The `domain` layer (the rules) knows nothing about AWS, files, or the web — which is what makes every external piece swappable.

```
┌─────────────────────────────────────────────────────────────┐
│  api/          FastAPI routes — the "front door" (HTTP)       │
│                /auth  /rag  /agent  /crew  /observability     │
└───────────────────────────┬─────────────────────────────────┘
                            │ depends on
┌───────────────────────────▼─────────────────────────────────┐
│  services/     Orchestration — RAG, retrieval, guardrails     │
│                (the "managers" that coordinate the work)      │
└───────────────────────────┬─────────────────────────────────┘
                            │ depends on
┌───────────────────────────▼─────────────────────────────────┐
│  infrastructure/  Real workers — AWS Bedrock, OpenAI,         │
│                   Anthropic, JSON vector store, file I/O      │
└───────────────────────────┬─────────────────────────────────┘
                            │ implements interfaces from
┌───────────────────────────▼─────────────────────────────────┐
│  domain/       The rulebook — interfaces + models ONLY.       │
│                No AWS, no files, no web. Pure Python.         │
└─────────────────────────────────────────────────────────────┘

        Agents live alongside, using the same services:
        agents/tools  →  agents/graphs (LangGraph)  →  agents/crews (CrewAI)
```

**The RAG request flow, end to end:**

```
Question
  → [Guardrail] prompt-injection check + PII redaction
  → [Retrieval] embed question → cosine similarity search → top-K chunks (min score 0.5)
  → [If no chunk passes threshold] return "I don't know" (no LLM call — saves cost)
  → [Generation] strict grounded prompt → LLM (temperature 0.0)
  → Answer + source citations
  → [Log] append to JSONL for observability + evaluation
```

---

## The one pattern that repeats everywhere

If you read only one thing, read this. **Every feature follows the same 4-layer shape:**

| Layer | Job | Example |
|-------|-----|---------|
| `domain/` | Define **what** must happen (interfaces, no real code) | `LLMProvider`, `VectorStore` interfaces |
| `services/` | **Orchestrate** the work, following the interfaces | `RagQueryService` |
| `infrastructure/` | The **real** implementation (AWS, files, APIs) | `BedrockLLMProvider`, `JsonVectorStore` |
| `api/` | Expose it over **HTTP** | `POST /rag/query` |

**Why this matters:** because the service layer only depends on interfaces, swapping AWS Bedrock for OpenAI or Anthropic means writing **one new adapter class** — with zero changes to business logic. This wasn't just a design goal; it was used for real when AWS Bedrock hit a rate limit mid-development and the system was switched to OpenAI in minutes.

---

## Tech stack

- **Language:** Python 3.12
- **API:** FastAPI, Pydantic, JWT auth, bcrypt, role-based access control
- **LLMs (swappable):** AWS Bedrock (Claude), Anthropic API, OpenAI — one interface, three adapters
- **Retrieval:** Bedrock/Cohere embeddings, JSON vector store with cosine similarity
- **Agents:** LangGraph (single-agent ReAct), CrewAI (multi-agent sequential)
- **Evaluation:** golden dataset + rule-based checks + LLM-as-judge
- **Observability:** custom metrics endpoint + LangSmith distributed tracing
- **Guardrails:** rule-based prompt-injection detection + PII redaction
- **UI:** Streamlit demo calling the real backend
- **Tooling:** uv, ruff, mypy (strict), pytest, Git

---

## Features by phase

The project was built in deliberate, verifiable phases:

| Phase | Feature |
|-------|---------|
| 0 | Foundation — config, structured logging, error handling, health checks |
| 1 | Auth — JWT + bcrypt + role-based access control (4 clinical personas) |
| 2 | Document chunking — hybrid structure-aware + recursive splitting |
| 3 | Embeddings + retrieval — swappable providers, cosine similarity search |
| 4 | LLM gateway — 3 swappable providers, retry/backoff, temperature control |
| 5 | RAG — retrieval + strictly-grounded generation with citations |
| 6 | Evaluation — golden dataset, rule-based + LLM-as-judge |
| 7 | Observability — metrics endpoint + LangSmith tracing |
| 8 | Agents — LangGraph ReAct agent with search + calculator tools |
| 9 | Multi-agent — CrewAI 3-agent pipeline (Researcher → Reviewer → Writer) |
| 10 | Guardrails — prompt-injection detection + PII redaction |
| 11 | MCP server — search exposed as a standard Model Context Protocol tool |

---

## Getting started

**Prerequisites:** Python 3.12, [uv](https://github.com/astral-sh/uv), and API access to at least one LLM provider (AWS Bedrock, OpenAI, or Anthropic).

```bash
# 1. Clone and enter
git clone https://github.com/mahendralearnings/Clinical-Intelligence-Platform.git
cd Clinical-Intelligence-Platform

# 2. Create the environment and install
uv venv --python 3.12
uv sync --extra dev

# 3. Configure secrets (copy the example, then fill in your keys)
cp .env.example .env
# edit .env — set LLM_PROVIDER_TYPE and the matching API key

# 4. Run the tests (no API keys or network needed — uses fakes)
uv run pytest -v

# 5. Start the API
uv run uvicorn clinical_platform.main:app --reload
# open http://127.0.0.1:8000/docs
```

> **Note:** the default test suite uses fake providers, so it runs fast, free, and offline. Real-provider integration tests are opt-in via environment variables (e.g. `BEDROCK_INTEGRATION=1`).

---

## Running the demo UI

With the API running in one terminal, start the Streamlit UI in another:

```bash
uv run streamlit run ui_demo.py
```

Log in, pick a mode (Plain RAG / Agent / Crew), and ask a question. The Agent and Crew modes show each reasoning step / each specialist's output visually.

---

## Testing philosophy

This project follows a **testing pyramid**:

- **Fast, free, offline unit tests** using fakes / test doubles for every unit of logic — these run on every change.
- **A small number of real integration tests**, guarded behind environment variables (`BEDROCK_INTEGRATION=1`, `AGENT_INTEGRATION=1`, `CREW_INTEGRATION=1`), run deliberately rather than automatically.

This keeps the suite fast and free while still proving real-world correctness where it matters. Working with **non-deterministic** systems (LLMs give different outputs for the same input) is exactly why evaluation and careful test design are first-class concerns here, not afterthoughts.

---

## Key engineering lessons

Real problems hit and solved during the build (documented because debugging is a core skill, not an embarrassment):

- **Swappable design paid off for real:** when AWS Bedrock hit a daily rate limit mid-build, switching to OpenAI took one adapter and one config change — zero business-logic changes.
- **Giving an agent tools doesn't mean it uses them:** the LangGraph agent initially answered dosage questions from general knowledge instead of searching. Fixed with an explicit tool-use-first system prompt — the same "strict grounding" principle as RAG.
- **Evaluation tooling needs its own evaluation:** the LLM-as-judge scored correct answers low — because it wasn't given the retrieved context to check against, and its prompt didn't treat a correct refusal as valid. A reminder never to trust an automated score blindly.
- **Agents need loop limits:** an unbounded agent loop can silently rack up API calls. An explicit recursion limit turns a runaway cost into a fast, clear failure.
- **Observability is a debugging tool, not just monitoring:** distributed tracing was used to prove a "hanging" agent was actually hitting provider rate limits, not stuck in a logic loop.

---

## What's deliberately out of scope

Knowing what *not* to build is part of the engineering:

- **Re-ranking, hybrid search, HyDE** — known advanced retrieval techniques, deliberately deferred. The score-threshold approach is sufficient at this scale; these would be added only if evaluation metrics showed a real need.
- **A production vector database (pgvector, OpenSearch, Pinecone)** — a JSON vector store is used behind the `VectorStore` interface, so migrating is a one-adapter change. Kept simple intentionally.
- **AI-based guardrails** — rule-based pattern matching is the correct, fast, free first line of defense; an AI-based second layer would be added only if needed.

---

*Built as a portfolio project demonstrating production-grade GenAI engineering. Feedback and questions welcome via issues.*