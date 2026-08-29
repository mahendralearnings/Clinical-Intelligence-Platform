"""
POST /agent/query - wires the LangGraph agent (Phase 8) into a real
HTTP endpoint, so it can be called from Swagger or a UI.
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from clinical_platform.agents.graphs.clinical_agent_graph import build_agent_graph
from clinical_platform.agents.tools.clinical_tools import set_retrieval_service
from clinical_platform.api.middleware.auth_dependencies import CurrentUser
from clinical_platform.core.config import Settings, get_settings
from clinical_platform.infrastructure.embedding_providers.bedrock_embedding_provider import (
    BedrockEmbeddingProvider,
)
from clinical_platform.infrastructure.vector_stores.json_vector_store import JsonVectorStore
from clinical_platform.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentQueryRequest(BaseModel):
    question: str


class AgentStep(BaseModel):
    step_type: str
    content: str


class AgentQueryResponse(BaseModel):
    final_answer: str
    steps: list[AgentStep]


@router.post("/query", response_model=AgentQueryResponse)
def query_agent(
    request: AgentQueryRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    _: CurrentUser,
) -> AgentQueryResponse:
    embedder = BedrockEmbeddingProvider(region=settings.bedrock_region)
    store = JsonVectorStore(store_path=Path(settings.vector_store_path))
    retrieval = RetrievalService(embedder=embedder, store=store)
    set_retrieval_service(retrieval)

    graph = build_agent_graph(api_key=settings.openai_api_key)

    result = graph.invoke(
        {"messages": [HumanMessage(content=request.question)]},
        config={"recursion_limit": 12},
    )

    steps = []
    for msg in result["messages"][1:]:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage":
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    steps.append(AgentStep(step_type="tool_call", content=f"{tc['name']}({tc['args']})"))
            elif msg.content:
                steps.append(AgentStep(step_type="ai_message", content=msg.content))
        elif msg_type == "ToolMessage":
            steps.append(AgentStep(step_type="tool_result", content=str(msg.content)[:300]))

    return AgentQueryResponse(final_answer=result["messages"][-1].content, steps=steps)