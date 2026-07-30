# Clinical Intelligence Platform

Enterprise-grade Generative AI platform for pharma/healthcare knowledge 
work — clinical trials, drug manuals, FDA guidelines, SOPs, research 
papers — built as a depth-first portfolio project for Senior AI Engineer 
/ GenAI Solution Architect interviews.

Serves four personas — **Researchers**, **Doctors**, **Compliance 
Officers**, **Clinical Operations** — each with scoped retrieval and 
permissions.

## Philosophy

This project prioritizes **depth over width**. Every phase documents 
not just what was built, but the alternatives considered and why one 
was chosen over another — the reasoning is the point, not just the code.

## Architecture

Clean/hexagonal architecture. Domain and service layers have zero 
dependency on FastAPI, LangChain, or any specific LLM/vector-DB SDK — 
those are adapters, swappable behind interfaces (`LLMProvider`, 
`VectorStore`). Full rationale: [`docs/architecture/overview.md`](docs/architecture/overview.md)

## Build Order (Phases)

Built one phase at a time, each with its own design doc under `docs/phases/`.

| Phase | Status |
|---|---|
| `00-platform-foundation` | ✅ done |
| `01-auth-minimal` | in progress |
| `02-ingestion-chunking` | planned |
| `03-retrieval-embeddings` | planned |
| `04-llm-gateway` | planned |
| `05-rag-query-service` | planned |
| `06-guardrails` | planned |
| `07-agents-langgraph` | planned |
| `08-mcp-integration` | planned |
| `09-bedrock-native` | planned |

## Local Development

\`\`\`bash
cp .env.example .env
uv sync --extra dev
uv run uvicorn clinical_platform.api.main:app --reload
\`\`\`

## Testing

\`\`\`bash
uv run pytest
uv run ruff check src tests
uv run mypy src
\`\`\`

## Tech Stack

- **Language:** Python 3.12
- **API:** FastAPI
- **LLMs:** Claude, OpenAI, Bedrock (provider-agnostic gateway)
- **Vector Search:** pgvector, with AWS Bedrock Knowledge Base as an 
  AWS-native alternative (Phase 9)
- **Orchestration:** LangGraph (agents), MCP (tool interoperability)
- **Cloud:** AWS (Bedrock)