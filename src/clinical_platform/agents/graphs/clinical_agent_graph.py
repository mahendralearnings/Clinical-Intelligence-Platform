"""
The agent's decision loop, built with LangGraph.

Design: a 2-node loop.
  - "agent" node: asks the LLM what to do next (answer, or call a tool)
  - "tools" node: actually runs whichever tool the LLM asked for
A conditional edge decides, after each "agent" turn, whether to loop
back to "tools" again or stop because the LLM gave a final answer.
"""

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from clinical_platform.agents.tools.clinical_tools import calculate, search_clinical_documents


class AgentState(TypedDict):
    """The 'memory' carried through every step of the loop."""

    messages: Annotated[list[BaseMessage], lambda old, new: old + new]


TOOLS = [search_clinical_documents, calculate]


def agent_node(state: AgentState, llm_with_tools) -> dict:
    """Ask the LLM: given everything so far, what do you want to do next?"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """After the agent speaks: did it ask for a tool, or is it done?"""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_agent_graph(api_key: str | None = None, model_id: str = "gpt-4o-mini", llm_with_tools=None):
    """
    llm_with_tools can be injected directly for testing (bypassing the
    real OpenAI connection) - same dependency-injection pattern used
    everywhere else in this project.
    """
    if llm_with_tools is None:
        llm = ChatOpenAI(api_key=api_key, model=model_id, temperature=0.0)
        llm_with_tools = llm.bind_tools(TOOLS)

    def _agent_node(state: AgentState) -> dict:
        return agent_node(state, llm_with_tools)

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()