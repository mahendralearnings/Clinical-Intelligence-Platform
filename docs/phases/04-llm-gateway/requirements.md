# Phase 4 — LLM Gateway: Requirements (Piece 2 — Core Interface)

## Context

Phase 3 proved that AWS Bedrock is reachable and that Cohere embeddings
work end-to-end. The `test_bedrock_connection.py` script also confirmed
that Claude 3 Haiku responds correctly via boto3's `converse()` API.

Phase 4 adds the **LLM Gateway** — the abstraction layer that lets the
rest of the application call a language model without knowing whether it
is Claude, GPT-4, or a local model. This is Piece 2: the minimal core
interface + one concrete implementation. Retries, timeouts, streaming,
and API routing come in Piece 3.

---

## Functional Requirements

### FR-1 LLMProvider interface
`domain/ports.py` shall gain a new `LLMProvider` Protocol with a single
method:

```python
def generate(self, prompt: str) -> str: ...
```

It takes a plain string prompt and returns a plain string response.
No streaming, no message history, no system prompt parameter yet — those
are Piece 3 concerns.

### FR-2 Bedrock Claude implementation
`infrastructure/llm_providers/bedrock_llm_provider.py` shall implement
`LLMProvider` using boto3's `converse()` API with model ID
`anthropic.claude-3-haiku-20240307-v1:0`.

The implementation must:
- Wrap the prompt in the `converse()` message format proven in the test script.
- Extract the text reply from `result["output"]["message"]["content"][0]["text"]`.
- Catch all boto3/botocore exceptions and re-raise as `LLMError` (defined
  in `domain/ports.py`).
- Create the boto3 client lazily (same pattern as `BedrockEmbeddingProvider`).

### FR-3 FakeLLMProvider test double
A `FakeLLMProvider` class in the test file shall:
- Satisfy the `LLMProvider` Protocol structurally (no inheritance).
- Return a configurable fixed string (default: `"fake response"`).
- Make zero AWS calls.

### FR-4 Pytest tests
`tests/test_llm_provider.py` shall contain:
- `test_fake_provider_satisfies_protocol` — `isinstance(fake, LLMProvider)` is `True`.
- `test_fake_provider_returns_non_empty_string` — `generate()` returns a non-empty `str`.
- `test_fake_provider_returns_configured_response` — custom string comes back unchanged.
- One optional integration test guarded by `BEDROCK_INTEGRATION=1` that
  calls the real Claude model and asserts the response is a non-empty string.

---

## Out of Scope (Piece 2)

- Retries, exponential backoff, timeout configuration (Piece 3)
- Streaming / async generation (Piece 3)
- System prompt, temperature, max_tokens parameters (Piece 3)
- Wiring into any API route (Phase 5)
- Multiple model support / model selection logic (Piece 3)
- Token counting or cost tracking (Phase 5+)
