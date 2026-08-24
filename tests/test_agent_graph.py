import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from clinical_platform.agents.tools.clinical_tools import calculate, set_retrieval_service

from langchain_core.messages import AIMessage, HumanMessage

class FakeToolCallingLLM:
    """
    Pretends to be an LLM that can call tools - returns a SCRIPTED
    sequence of responses, one per call, so we can test the FULL
    agent -> tool -> agent -> tool -> agent loop for FREE, with zero
    real API calls. This is what should have existed from the start.
    """

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self._call_count = 0

    def invoke(self, messages):
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


def test_full_graph_loop_with_fake_llm_searches_then_calculates() -> None:
    """
    Proves the ENTIRE agent loop works correctly - search, then
    calculate, then final answer - without spending a single real
    API call. This is the test we should run constantly while
    debugging, saving the real integration test for a final,
    occasional check only.
    """
    from clinical_platform.agents.graphs.clinical_agent_graph import build_agent_graph

    fake_llm = FakeToolCallingLLM(
        responses=[
            # Turn 1: agent decides to search
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_clinical_documents", "args": {"query": "max metformin dose"}, "id": "call_1"}
                ],
            ),
            # Turn 2: agent decides to calculate, using a fake found fact
            AIMessage(
                content="",
                tool_calls=[{"name": "calculate", "args": {"expression": "2550 / 3"}, "id": "call_2"}],
            ),
            # Turn 3: agent gives the final answer
            AIMessage(content="The dose per administration would be 850.0 mg."),
        ]
    )

    graph = build_agent_graph(llm_with_tools=fake_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the max dose, split into 3 doses?")]}
    )

    final_answer = result["messages"][-1].content
    assert "850" in final_answer
    # Also prove BOTH tools actually got called, not just the final answer being right
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 2