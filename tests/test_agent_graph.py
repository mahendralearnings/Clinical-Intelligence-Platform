"""
Tests for the agent's decision loop.

Three layers, cheapest/fastest first:
  1. Tools work correctly and safely, on their own - no AI needed.
  2. Routing decision (should_continue) works correctly - no AI needed.
  3. Full loop works end-to-end using a FREE fake LLM - no real API calls.
  4. (Optional, real AI, real cost) the full graph actually solves a
     multi-step question - guarded behind an env var, run rarely.
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from clinical_platform.agents.tools.clinical_tools import calculate, set_retrieval_service


# --- Layer 1: Tool tests (no AI needed at all) ---

def test_calculate_handles_simple_division() -> None:
    result = calculate.invoke({"expression": "2550 / 3"})
    assert result == "850.0"


def test_calculate_rejects_unsafe_input() -> None:
    result = calculate.invoke({"expression": "__import__('os').system('echo hacked')"})
    assert "Could not calculate" in result


def test_search_tool_reports_when_service_not_configured() -> None:
    set_retrieval_service(None)
    from clinical_platform.agents.tools.clinical_tools import search_clinical_documents

    result = search_clinical_documents.invoke({"query": "anything"})
    assert "not configured" in result


# --- Layer 2: Routing logic tests (no AI needed) ---

def _make_state(messages: list) -> dict:
    return {"messages": messages}


def test_should_continue_routes_to_tools_when_tool_call_present() -> None:
    from clinical_platform.agents.graphs.clinical_agent_graph import should_continue

    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "call_1"}],
    )
    state = _make_state([HumanMessage(content="what is 1+1?"), ai_message])

    assert should_continue(state) == "tools"


def test_should_continue_ends_when_no_tool_call() -> None:
    from clinical_platform.agents.graphs.clinical_agent_graph import should_continue
    from langgraph.graph import END

    ai_message = AIMessage(content="The answer is 2.")
    state = _make_state([HumanMessage(content="what is 1+1?"), ai_message])

    assert should_continue(state) == END


# --- Layer 3: Full loop test using a FREE fake LLM (no real API calls) ---

class FakeToolCallingLLM:
    """
    Pretends to be an LLM that can call tools - returns a SCRIPTED
    sequence of responses, one per call, so we can test the FULL
    agent -> tool -> agent -> tool -> agent loop for FREE.
    """

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self._call_count = 0

    def invoke(self, messages):
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


def test_full_graph_loop_with_fake_llm_searches_then_calculates() -> None:
    from clinical_platform.agents.graphs.clinical_agent_graph import build_agent_graph

    fake_llm = FakeToolCallingLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_clinical_documents", "args": {"query": "max metformin dose"}, "id": "call_1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "calculate", "args": {"expression": "2550 / 3"}, "id": "call_2"}],
            ),
            AIMessage(content="The dose per administration would be 850.0 mg."),
        ]
    )

    graph = build_agent_graph(llm_with_tools=fake_llm)

    result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is the maximum daily metformin dose, divided evenly across 3 doses?"
            )
        ]
    },
    config={"recursion_limit": 6},  # safety net: max 6 steps, then fail fast instead of looping forever
)

    final_answer = result["messages"][-1].content
    assert "850" in final_answer
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 2


# --- Layer 4: Optional real integration test (real cost, run rarely) ---

@pytest.mark.skipif(
    os.getenv("AGENT_INTEGRATION") != "1",
    reason="Set AGENT_INTEGRATION=1 to run a real OpenAI-powered agent test",
)
def test_real_agent_solves_multistep_dosage_question() -> None:
    from pathlib import Path

    from clinical_platform.agents.graphs.clinical_agent_graph import build_agent_graph
    from clinical_platform.core.config import get_settings
    from clinical_platform.infrastructure.embedding_providers.bedrock_embedding_provider import (
        BedrockEmbeddingProvider,
    )
    from clinical_platform.infrastructure.vector_stores.json_vector_store import JsonVectorStore
    from clinical_platform.services.retrieval_service import RetrievalService

    settings = get_settings()

    embedder = BedrockEmbeddingProvider(region=settings.bedrock_region)
    store = JsonVectorStore(store_path=Path(settings.vector_store_path))
    retrieval = RetrievalService(embedder=embedder, store=store)
    set_retrieval_service(retrieval)

    graph = build_agent_graph(api_key=settings.openai_api_key)

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the maximum daily metformin dose, divided evenly across 3 doses?"
                )
            ]
        }
    )

    for msg in result["messages"]:
        print(f"\n--- {type(msg).__name__} ---")
        print(msg.content if msg.content else getattr(msg, "tool_calls", ""))

    final_answer = result["messages"][-1].content
    assert "850" in final_answer