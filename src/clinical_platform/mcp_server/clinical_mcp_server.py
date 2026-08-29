from pathlib import Path

from mcp.server.fastmcp import FastMCP
from clinical_platform.core.config import get_settings
from clinical_platform.infrastructure.embedding_providers.bedrock_embedding_provider import (
    BedrockEmbeddingProvider,
)
from clinical_platform.infrastructure.vector_stores.json_vector_store import JsonVectorStore
from clinical_platform.services.retrieval_service import RetrievalService


mcp=FastMCP("Clinical-Intelligence_platform")


def _build_retrieval() -> RetrievalService:
    settings=get_settings
    
    embedder=BedrockEmbeddingProvider(region=settings.bedrock_region)
    
    
    store = JsonVectorStore(
         store_path=Path(settings.vector_store_path)
        )
    return RetrievalService(
            embedder=embedder,
            store=store 
        )
    
@mcp.tool()
###i want to expose my tool as an MCP Tool
def search_clinical_documents(query: str,top_k: int = 3) -> str:
    
        """
        Search the clinical document library and return
        relevant document excerpts.
        """
        retrieval = _build_retrieval()
        results = retrieval.search(query=query, top_k=top_k, min_score=0.5)
        if not results:
            return "No relevant documents found for this query."

        lines = []
        for chunk, score in results:
            lines.append(
                f"[Source: {chunk.metadata.source} | Section: {chunk.metadata.section_title} "
                f"| Score: {score:.2f}]\n{chunk.content}"
            )
        return "\n\n".join(lines)

if __name__ == "__main__":
    mcp.run()