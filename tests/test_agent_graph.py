"""
Tests for the agent's decision loop - using fakes, no real OpenAI calls.
Split into 3 concerns:
  1. The tools work correctly and safely, on their own.
  2. The routing decision (should_continue) works correctly, on its own.
  3. (Optional, real AI) the full graph actually solves a multi-step
     question - guarded behind an env var, same pattern as Phase 3/4.
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from clinical_platform.agents.tools.clinical_tools import calculate, set_retrieval_service


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


@pytest.mark.skipif(
    os.getenv("AGENT_INTEGRATION") != "1",
    reason="Set AGENT_INTEGRATION=1 to run a real OpenAI-powered agent test",
)
def test_real_agent_solves_multistep_dosage_question() -> None:
    from clinical_platform.agents.graphs.clinical_agent_graph import build_agent_graph
    from clinical_platform.core.config import get_settings

    settings = get_settings()
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
    final_answer = result["messages"][-1].content
    assert "850" in final_answer